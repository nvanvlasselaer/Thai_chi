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

The ends are left free, so neither recording has to start or stop at the same
point of the form, and the quiet standing either side does not bend the map.

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

from analysis.coverage import has
from analysis.kinematics import LoadedTrial
from analysis.signals import highpass_detrend, lowpass_signal


@dataclass(frozen=True)
class AlignmentParams:
    fs_hz: float = 5.0
    """Rate the channels are compared at; the form is carried by motion well below 1 Hz."""
    band_frac: float = 0.15
    """Largest drift allowed from the diagonal, as a fraction of the recording."""
    yaw_cutoff_hz: float = 0.5
    knee_cutoff_hz: float = 1.0
    free_end_s: float = 30.0
    """How much of either recording's start or end the path may skip."""


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


def dtw_path(a: np.ndarray, b: np.ndarray, band: int, free: int) -> tuple[np.ndarray, float]:
    """Banded DTW between two multichannel series with open ends.

    The path may begin anywhere in the first ``free`` samples of either series
    and end anywhere in the last ``free``.  Returns the path as ``(i, j)`` pairs
    and its mean step cost.
    """
    n, m = len(a), len(b)
    band = max(band, abs(n - m) + 1)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, : free + 1] = 0.0
    cost[: free + 1, 0] = 0.0
    for i in range(1, n + 1):
        centre = int(round(i * m / n))
        j0, j1 = max(1, centre - band), min(m, centre + band)
        local = np.linalg.norm(b[j0 - 1:j1] - a[i - 1], axis=1)
        row, above = cost[i], cost[i - 1]
        for offset, j in enumerate(range(j0, j1 + 1)):
            row[j] = local[offset] + min(above[j], above[j - 1], row[j - 1])

    # Best end on the last row or the last column, within the free margin.
    ends = [(n, j) for j in range(max(1, m - free), m + 1)] + [(i, m) for i in range(max(1, n - free), n + 1)]
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
    """Warp the novice recording onto the trained one (about 0.2 s), on the channels both have."""
    params = params or AlignmentParams()
    channels = channels or channels_for(novice.kin.sensors, trained.kin.sensors)
    if not channels:
        raise ValueError("The two recordings share no sensor the alignment can use.")
    grid_n, features_n = alignment_features(novice, params, channels)
    grid_t, features_t = alignment_features(trained, params, channels)
    band = int(params.band_frac * max(len(grid_n), len(grid_t)))
    free = int(params.free_end_s * params.fs_hz)
    path, mean_cost = dtw_path(features_n, features_t, band, free)

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
                     channels=() if channels == FULL_CHANNELS else channels)


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
