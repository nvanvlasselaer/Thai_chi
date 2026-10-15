"""Whole-recording alignment of the novice and trained performances.

Both participants perform the same form, but not on the same clock: the
recordings start at different moments and nobody keeps an exact tempo.  Pairing
events by their order in each recording goes wrong as soon as a detector finds
a movement in one recording and not the other -- on these recordings every
trunk-rotation pair and half the sequence segments ended up comparing
different movements.  Matching short windows by DTW (the v1 method) fails the
other way: 2-4 s of Tai Chi is not distinctive enough to be found reliably.

The whole form is distinctive.  :func:`align` warps one complete recording onto
the other with dynamic time warping on slow, whole-body channels -- chest yaw,
trunk-pelvis yaw and both knees' flexion, at 5 Hz -- and every event is then
paired through that map.  Checked against the four single-leg stances, whose
pairing is unambiguous, the map puts each novice stance within 0.1-0.7 s of its
trained partner.  On these recordings the offset is almost constant, about
-3.6 s: both performers followed the same instruction video.

Two recordings need not cover the same stretch of the form: one may start
later, or stop before the form ends.  So the warping is done in two steps.
First :func:`straight_line` finds where the recordings correspond as a whole --
the offset and tempo ratio of the straight map along which they correlate
best.  Then the DTW searches a band around that line, and its ends are left
free over whatever part of either recording the line puts outside the other,
plus a margin (:func:`unmatched_ends`), so the quiet standing either side and a
recording that stops early do not bend the map.  (The band used to be centred
on the diagonal from start to end, which assumes both recordings span the same
part of the form: a novice recording that stopped 55 s before the end of the
form put its single-leg stances 18-36 s from their partners.)

When a sensor is missing in either recording, the channels that need it are
left out of both, and pelvis yaw carries the turns when the chest is missing
(:func:`channels_for`).  Simulated on the June pair, every reduced set kept all
38 curated pairs the same movement; without the chest the map stays within
0.6 s of the full one at the 95th percentile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import correlate

from analysis.coverage import has
from analysis.kinematics import LoadedTrial
from analysis.signals import highpass_detrend, lowpass_signal


@dataclass(frozen=True)
class AlignmentParams:
    fs_hz: float = 5.0
    """Rate the channels are compared at; the form is carried by motion well below 1 Hz."""
    band_frac: float = 0.15
    """Largest drift allowed from the straight line (:func:`straight_line`), as a fraction of the recording."""
    yaw_cutoff_hz: float = 0.5
    knee_cutoff_hz: float = 1.0
    free_end_s: float = 30.0
    """How much of either recording's start or end the path may skip, beyond what the straight line
    already puts outside the other recording (:func:`unmatched_ends`)."""
    max_tempo_ratio: float = 1.25
    """The straight line's slope is searched between the inverse of this and this: how much faster one
    performance may be than the other as a whole.  Local differences are the DTW's to absorb."""
    min_overlap_frac: float = 0.5
    """The straight line must lay at least this share of the shorter recording over the other one."""


@dataclass
class Alignment:
    """A monotone map between the two recordings' clocks."""

    novice_s: np.ndarray
    """Novice times, on the alignment grid."""
    trained_s: np.ndarray
    """The trained time each novice time corresponds to."""
    cost: float
    """Mean distance per step along the path, in z-scored channel units."""
    params: AlignmentParams
    channels: tuple[str, ...] = ()
    """The channels matched on (:func:`channels_for`); empty means all of :data:`FULL_CHANNELS`."""
    line: tuple[float, float, float] = (1.0, 0.0, float("nan"))
    """``(slope, offset_s, r)``: the straight map ``trained = slope * novice + offset`` the DTW searched
    around, and the mean correlation of the two recordings along it (:func:`straight_line`)."""

    def to_trained(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.interp(t, self.novice_s, self.trained_s)

    def to_novice(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.interp(t, self.trained_s, self.novice_s)

    def map(self, t: float | np.ndarray, frm: str, to: str) -> float | np.ndarray:
        """Carry a time from one role's clock to the other's."""
        if frm == to:
            return t
        return self.to_trained(t) if frm == "Novice" else self.to_novice(t)

    @property
    def median_offset_s(self) -> float:
        """Typical novice-minus-trained time of the same moment of the form."""
        return float(np.median(self.novice_s - self.trained_s))

    def digest(self) -> str:
        payload = {"params": asdict(self.params), "cost": round(self.cost, 6),
                   "map": np.round(self.trained_s[:: int(self.params.fs_hz)], 3).tolist()}
        if self.channels and self.channels != FULL_CHANNELS:
            payload["channels"] = list(self.channels)
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


CHANNEL_SENSORS = {
    "chest yaw": ("chestbone",),
    "trunk-pelvis yaw": ("lumbar", "chestbone"),
    "left knee": ("lthigh", "ltibia"),
    "right knee": ("rthigh", "rtibia"),
    "pelvis yaw": ("lumbar",),
}
FULL_CHANNELS = ("chest yaw", "trunk-pelvis yaw", "left knee", "right knee")
"""The channels matched on when both recordings have every sensor."""


def channels_for(*sensor_sets) -> tuple[str, ...]:
    """The channels every recording can provide, in :data:`FULL_CHANNELS` order.

    Pelvis yaw carries the turns instead of the chest yaw when a recording has
    no chest sensor: on the June pair that keeps the map within 0.6 s of the
    full one (95th percentile), against 1.0 s on the knees alone.
    """
    common = set.intersection(*(set(sensors) for sensors in sensor_sets))
    chosen = [name for name in FULL_CHANNELS if has(common, *CHANNEL_SENSORS[name])]
    if "chest yaw" not in chosen and has(common, *CHANNEL_SENSORS["pelvis yaw"]):
        chosen.insert(0, "pelvis yaw")
    return tuple(chosen)


def _channel(kin, fs: float, name: str, params: AlignmentParams) -> tuple[np.ndarray, float]:
    if name == "chest yaw":
        return highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs), params.yaw_cutoff_hz
    if name == "trunk-pelvis yaw":
        return highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs), params.yaw_cutoff_hz
    if name == "pelvis yaw":
        return highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs), params.yaw_cutoff_hz
    knee = kin.left_knee_deg if name == "left knee" else kin.right_knee_deg
    return knee[:, 0], params.knee_cutoff_hz


def alignment_features(loaded: LoadedTrial, params: AlignmentParams,
                       channels: tuple[str, ...] = FULL_CHANNELS) -> tuple[np.ndarray, np.ndarray]:
    """The channels the recordings are matched on, z-scored, on a uniform grid.

    Chest yaw and trunk-pelvis yaw carry the turns of the form, both knees the
    steps and the single-leg passage.  Yaw is de-drifted as everywhere else;
    the low cutoffs keep only the choreography, not how each person executes it.
    """
    kin, fs = loaded.kin, loaded.fs
    grid = np.arange(kin.t[0], kin.t[-1], 1.0 / params.fs_hz)
    columns = []
    for signal, cutoff in (_channel(kin, fs, name, params) for name in channels):
        values = np.interp(grid, kin.t, lowpass_signal(signal, fs, cutoff_hz=cutoff))
        spread = np.std(values)
        columns.append((values - np.mean(values)) / (spread if spread > 0 else 1.0))
    return grid, np.column_stack(columns)


TEMPO_STEPS = 41
"""Slopes :func:`straight_line` tries, evenly spaced on a log scale (about 1 % apart at the default range)."""


def _pearson_by_lag(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The channels' mean Pearson correlation of ``x[k]`` with ``y[k + lag]``, at every lag.

    Returns ``(lags, r, overlap)``: the lags from ``1 - len(x)`` to
    ``len(y) - 1``, the correlation over the samples the two series share at
    each lag, and how many samples that is.  Each channel is centred and scaled
    over the shared samples only, so a stretch of quiet standing does not
    correlate with another just for sitting at the same level.
    """
    n, m = len(x), len(y)
    lags = np.arange(1 - n, m)
    k0, k1 = np.maximum(0, -lags), np.minimum(n, m - lags)
    count = (k1 - k0).astype(float)
    total = np.zeros(len(lags))
    for xc, yc in zip(x.T, y.T):
        sxy = correlate(yc, xc, mode="full", method="fft")  # sum over k of y[k + lag] * x[k]
        px, pxx = np.concatenate([[0.0], np.cumsum(xc)]), np.concatenate([[0.0], np.cumsum(xc * xc)])
        py, pyy = np.concatenate([[0.0], np.cumsum(yc)]), np.concatenate([[0.0], np.cumsum(yc * yc)])
        sx, sxx = px[k1] - px[k0], pxx[k1] - pxx[k0]
        sy, syy = py[k1 + lags] - py[k0 + lags], pyy[k1 + lags] - pyy[k0 + lags]
        spread = np.sqrt(np.clip(count * sxx - sx * sx, 0.0, None) * np.clip(count * syy - sy * sy, 0.0, None))
        total += np.divide(count * sxy - sx * sy, spread, out=np.zeros(len(lags)), where=spread > 0)
    return lags, total / x.shape[1], count


def straight_line(grid_a: np.ndarray, features_a: np.ndarray, grid_b: np.ndarray, features_b: np.ndarray,
                  params: AlignmentParams) -> tuple[float, float, float]:
    """The straight map ``t_b = slope * t_a + offset`` along which two recordings correspond best.

    For each slope between ``1 / max_tempo_ratio`` and ``max_tempo_ratio``,
    recording a is resampled at that tempo and slid along recording b, and at
    every shift that lays at least ``min_overlap_frac`` of the shorter recording
    over the other, the channels' correlations over the overlap are averaged
    (:func:`_pearson_by_lag`).  Returns ``(slope, offset_s, r)`` at the best
    shift.  Neither recording's start or end enters it, so it holds whether the
    two cover the same stretch of the form or not.
    """
    dt = 1.0 / params.fs_hz
    best = (-np.inf, 1.0, float(grid_b[0] - grid_a[0]))
    for slope in np.geomspace(1.0 / params.max_tempo_ratio, params.max_tempo_ratio, TEMPO_STEPS):
        times = grid_a[0] + np.arange(int((grid_a[-1] - grid_a[0]) * slope / dt) + 1) * dt / slope
        stretched = np.column_stack([np.interp(times, grid_a, column) for column in features_a.T])
        lags, r, count = _pearson_by_lag(stretched, features_b)
        r = np.where(count >= params.min_overlap_frac * min(len(stretched), len(features_b)), r, -np.inf)
        k = int(np.argmax(r))
        if r[k] > best[0]:
            best = (float(r[k]), float(slope), float(grid_b[0] + lags[k] * dt - slope * grid_a[0]))
    r, slope, offset = best
    return slope, offset, r


def unmatched_ends(grid_a: np.ndarray, grid_b: np.ndarray, slope: float, offset: float
                   ) -> tuple[float, float, float, float]:
    """How much of each recording the straight line puts outside the other, in seconds.

    ``(start of a, start of b, end of a, end of b)``: a recording that starts
    earlier in the form, or runs on after the other has stopped, has nothing to
    be matched with there, so the DTW's free ends are widened by this much.
    """
    first, last = slope * grid_a[0] + offset, slope * grid_a[-1] + offset  # a's ends on b's clock
    return (max(0.0, (grid_b[0] - first) / slope), max(0.0, first - grid_b[0]),
            max(0.0, (last - grid_b[-1]) / slope), max(0.0, grid_b[-1] - last))


def dtw_path(a: np.ndarray, b: np.ndarray, band: int, free: int | tuple[int, int, int, int],
             centre: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Banded DTW between two multichannel series with open ends.

    ``free`` is how many samples the path may skip at the ends: one number for
    both ends of both series, or ``(start of a, start of b, end of a, end of
    b)``.  ``centre`` gives, for each sample of a, the sample of b the band is
    centred on -- the straight line of :func:`straight_line`; without it the
    band follows the diagonal from both starts to both ends.  Returns the path
    as ``(i, j)`` pairs and its mean step cost.
    """
    n, m = len(a), len(b)
    start_a, start_b, end_a, end_b = (free,) * 4 if np.isscalar(free) else free
    if centre is None:
        band = max(band, abs(n - m) + 1)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, : start_b + 1] = 0.0
    cost[: start_a + 1, 0] = 0.0
    for i in range(1, n + 1):
        middle = int(round(i * m / n)) if centre is None else int(centre[i - 1]) + 1
        j0, j1 = max(1, middle - band), min(m, middle + band)
        local = np.linalg.norm(b[j0 - 1:j1] - a[i - 1], axis=1)
        row, above = cost[i], cost[i - 1]
        for offset, j in enumerate(range(j0, j1 + 1)):
            row[j] = local[offset] + min(above[j], above[j - 1], row[j - 1])

    # Best end on the last row or the last column, within the free margin.
    ends = [(n, j) for j in range(max(1, m - end_b), m + 1)] + [(i, m) for i in range(max(1, n - end_a), n + 1)]
    i, j = min(ends, key=lambda cell: cost[cell])
    total = cost[i, j]
    path = [(i - 1, j - 1)]
    while i > 1 and j > 1:
        step = int(np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else ((i - 1, j) if step == 1 else (i, j - 1))
        if not np.isfinite(cost[i, j]) or cost[i, j] == 0.0:
            path.append((i - 1, j - 1))
            break
        path.append((i - 1, j - 1))
    path = np.array(path[::-1])
    return path, float(total / max(1, len(path)))


def align(novice: LoadedTrial, trained: LoadedTrial, params: AlignmentParams | None = None,
          channels: tuple[str, ...] | None = None) -> Alignment:
    """Warp the novice recording onto the trained one (about 0.2 s), on the channels both have.

    First the straight line along which the two correspond (:func:`straight_line`),
    then the DTW in a band around it, free at either end over what the line puts
    outside the other recording plus ``free_end_s``.
    """
    params = params or AlignmentParams()
    channels = channels or channels_for(novice.kin.sensors, trained.kin.sensors)
    if not channels:
        raise ValueError("The two recordings share no sensor the alignment can use.")
    grid_n, features_n = alignment_features(novice, params, channels)
    grid_t, features_t = alignment_features(trained, params, channels)
    slope, offset, r = straight_line(grid_n, features_n, grid_t, features_t, params)
    band = int(params.band_frac * max(len(grid_n), len(grid_t)))
    free = tuple(int(round((params.free_end_s + extra) * params.fs_hz))
                 for extra in unmatched_ends(grid_n, grid_t, slope, offset))
    centre = np.rint((slope * grid_n + offset - grid_t[0]) * params.fs_hz).astype(int)
    path, mean_cost = dtw_path(features_n, features_t, band, free, centre)

    # One trained time per novice sample on the path; the median where the path
    # runs vertically.  Beyond the path's ends the local offset is carried on.
    novice_idx = np.unique(path[:, 0])
    trained_idx = np.array([np.median(path[path[:, 0] == i, 1]) for i in novice_idx])
    trained_idx = np.maximum.accumulate(trained_idx)
    novice_s = grid_n[novice_idx]
    trained_s = np.interp(trained_idx, np.arange(len(grid_t)), grid_t)
    start_offset = novice_s[0] - trained_s[0]
    end_offset = novice_s[-1] - trained_s[-1]
    novice_s = np.concatenate([[grid_n[0] - 1e3], novice_s, [grid_n[-1] + 1e3]])
    trained_s = np.concatenate([[grid_n[0] - 1e3 - start_offset], trained_s, [grid_n[-1] + 1e3 - end_offset]])
    return Alignment(novice_s=novice_s, trained_s=trained_s, cost=mean_cost, params=params,
                     channels=() if channels == FULL_CHANNELS else channels, line=(slope, offset, r))


_cache: dict[tuple, Alignment] = {}


def for_trials(trials: dict[str, LoadedTrial], params: AlignmentParams | None = None) -> Alignment | None:
    """The alignment of the two loaded recordings, or None with fewer than two or no channel in common.

    Cached per process by the two files' content and the channels matched on,
    so the dashboard, the seeders and validation share one computation.
    """
    if "Novice" not in trials or "Trained" not in trials:
        return None
    params = params or AlignmentParams()
    channels = channels_for(trials["Novice"].kin.sensors, trials["Trained"].kin.sensors)
    if not channels:
        return None
    key = (trials["Novice"].recording.sha256(), trials["Trained"].recording.sha256(), params, channels)
    if key not in _cache:
        _cache[key] = align(trials["Novice"], trials["Trained"], params, channels)
    return _cache[key]


# ---------------------------------------------------------------------------
# Matching events through the map
# ---------------------------------------------------------------------------


def overlap(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """``(overlap / shorter, intersection over union)`` of two time spans."""
    shared = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (shared / shorter if shorter > 0 else 0.0), (shared / union if union > 0 else 0.0)


def mapped_overlap(alignment: Alignment, novice: tuple[float, float],
                   trained: tuple[float, float]) -> tuple[float, float]:
    """:func:`overlap` of a novice window, carried onto the trained clock, with a trained one."""
    carried = (float(alignment.to_trained(novice[0])), float(alignment.to_trained(novice[1])))
    return overlap(carried, trained)


def match_windows(
    alignment: Alignment, novice: list[tuple[float, float]], trained: list[tuple[float, float]],
    min_iou: float = 0.5,
) -> list[tuple[int, int]]:
    """Pairs ``(novice index, trained index)`` of windows that cover the same movement.

    A pair must be each other's best match by intersection over union, and
    reach ``min_iou``.  IoU rather than containment, because containment scores
    a long window around a short one as a perfect match.
    """
    if not novice or not trained:
        return []
    scores = np.array([[mapped_overlap(alignment, a, b)[1] for b in trained] for a in novice])
    pairs = []
    for i in range(len(novice)):
        j = int(np.argmax(scores[i]))
        if scores[i, j] >= min_iou and int(np.argmax(scores[:, j])) == i:
            pairs.append((i, j))
    return pairs


def match_landmarks(
    alignment: Alignment, novice: list[tuple[float, int]], trained: list[tuple[float, int]]
) -> list[tuple[int, int]]:
    """Pairs of turning points ``(time_s, direction)`` that mark the same moment.

    Each novice landmark is carried onto the trained clock and matched to the
    nearest trained landmark turning the same way, if that one is closer than
    half the spacing to its neighbours -- otherwise the match would be a guess
    -- and if the novice landmark is also the trained one's nearest.  Matches
    are kept in order on both clocks.
    """
    if not novice or not trained:
        return []
    times_t = np.array([t for t, _ in trained])
    spacing = np.diff(times_t)
    carried = [float(alignment.to_trained(t)) for t, _ in novice]

    def nearest(k_novice: int) -> int | None:
        direction = novice[k_novice][1]
        options = [k for k, (_, d) in enumerate(trained) if d == direction]
        if not options:
            return None
        return min(options, key=lambda k: abs(times_t[k] - carried[k_novice]))

    def reverse_nearest(k_trained: int) -> int:
        direction = trained[k_trained][1]
        options = [k for k, (_, d) in enumerate(novice) if d == direction]
        return min(options, key=lambda k: abs(carried[k] - times_t[k_trained]))

    pairs: list[tuple[int, int]] = []
    for k_novice in range(len(novice)):
        k_trained = nearest(k_novice)
        if k_trained is None:
            continue
        left = spacing[k_trained - 1] if k_trained > 0 else np.inf
        right = spacing[k_trained] if k_trained < len(spacing) else np.inf
        if abs(times_t[k_trained] - carried[k_novice]) >= 0.5 * min(left, right):
            continue
        if reverse_nearest(k_trained) != k_novice:
            continue
        if pairs and (k_trained <= pairs[-1][1]):
            continue
        pairs.append((k_novice, k_trained))
    return pairs
