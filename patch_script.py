import re
from pathlib import Path

path = Path("analysis/tai_chi_kinematic_analysis.py")
content = path.read_text()

# Replace select_novice_event
old_select = """def select_novice_event(kin: Kinematics, fs: float) -> tuple[int, int, int, str]:
    yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
    chest_omega = lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=6.0)
    ignore = int(15 * fs)
    window = int(8 * fs)
    step = int(1 * fs)
    best_score = -np.inf
    best_start = ignore
    for start in range(ignore, len(yaw) - window - ignore, step):
        end = start + window
        yaw_range = np.ptp(yaw[start:end])
        omega_peak = np.percentile(chest_omega[start:end], 95)
        score = yaw_range * np.log1p(omega_peak)
        if score > best_score:
            best_score = score
            best_start = start
    best_end = best_start + window
    peak = best_start + int(np.argmax(np.abs(yaw[best_start:best_end] - np.median(yaw[best_start:best_end]))))
    rationale = (
        "Selected an 8 s trunk-rotation transition because the novice trial showed the largest "
        f"combined trunk-pelvis relative-yaw excursion and chest angular-velocity peak in this window "
        f"({np.ptp(yaw[best_start:best_end]):.1f} deg yaw range)."
    )
    return best_start, best_end, peak, rationale"""

new_select = """def select_novice_events(kin: Kinematics, fs: float, num_events: int = 3) -> tuple[list[tuple[int, int, int]], str]:
    yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
    chest_omega = lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=6.0)
    ignore = int(15 * fs)
    window = int(8 * fs)
    step = int(1 * fs)
    
    scores = []
    for start in range(ignore, len(yaw) - window - ignore, step):
        end = start + window
        yaw_range = np.ptp(yaw[start:end])
        omega_peak = np.percentile(chest_omega[start:end], 95)
        score = yaw_range * np.log1p(omega_peak)
        scores.append((score, start, end))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    
    selected_events = []
    for score, start, end in scores:
        overlap = False
        for sel_start, sel_end, _ in selected_events:
            if not (end < sel_start or start > sel_end):
                overlap = True
                break
        if not overlap:
            peak = start + int(np.argmax(np.abs(yaw[start:end] - np.median(yaw[start:end]))))
            selected_events.append((start, end, peak))
        if len(selected_events) == num_events:
            break
            
    selected_events.sort(key=lambda x: x[0])
    rationale = f"Selected {num_events} 8 s trunk-rotation transitions based on combined trunk-pelvis relative-yaw excursion and chest angular-velocity peaks."
    return selected_events, rationale"""
content = content.replace(old_select, new_select)

# Replace make_traceability_figure definition
old_fig_def = """def make_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    metrics: pd.DataFrame,
    windows: dict[str, tuple[int, int, int, int]],
    fs: float,
) -> None:"""
new_fig_def = """def make_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    metrics: pd.DataFrame,
    windows: dict[str, list[tuple[int, int, int, int, float]]],
    fs: float,
) -> None:"""
content = content.replace(old_fig_def, new_fig_def)

# Replace the loop in make_traceability_figure
old_fig_loop = """    for ax, label, kin in [(axes[0], "Novice", novice), (axes[1], "Trained", trained)]:
        event_start, event_end, stab_start, stab_end = windows[label]
        t = kin.t
        yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
        roll = lowpass_signal(kin.trunk_rel_euler_deg[:, 0], fs, cutoff_hz=4.0)
        pitch = lowpass_signal(kin.trunk_rel_euler_deg[:, 1], fs, cutoff_hz=4.0)
        yaw_vel = np.gradient(yaw, 1.0 / fs)
        
        ax.plot(t, yaw, color="#1f77b4", lw=1.2, label="rel yaw")
        ax.plot(t, roll, color="#9467bd", lw=1.2, label="rel roll")
        ax.plot(t, pitch, color="#8c564b", lw=1.2, label="rel pitch")
        
        ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event")
        ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization")
        event_yaw = yaw[event_start:event_end]
        peak_idx = event_start + int(np.argmax(np.abs(event_yaw - np.median(event_yaw))))
        ax.scatter([t[peak_idx]], [yaw[peak_idx]], s=30, color="#d1495b", zorder=4, label="yaw peak")
        
        ax2 = ax.twinx()
        ax2.plot(t, yaw_vel, color="#e377c2", lw=1.0, linestyle="--", label="yaw vel")
        ax2.set_ylabel("vel (deg/s)", color="#e377c2", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#e377c2", labelsize=8)
        
        ax.set_ylabel(f"{label}\\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False, ncol=4)"""

new_fig_loop = """    for ax, label, kin in [(axes[0], "Novice", novice), (axes[1], "Trained", trained)]:
        t = kin.t
        yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
        roll = lowpass_signal(kin.trunk_rel_euler_deg[:, 0], fs, cutoff_hz=4.0)
        pitch = lowpass_signal(kin.trunk_rel_euler_deg[:, 1], fs, cutoff_hz=4.0)
        yaw_vel = np.gradient(yaw, 1.0 / fs)
        
        ax.plot(t, yaw, color="#1f77b4", lw=1.2, label="rel yaw")
        ax.plot(t, roll, color="#9467bd", lw=1.2, label="rel roll")
        ax.plot(t, pitch, color="#8c564b", lw=1.2, label="rel pitch")
        
        ax2 = ax.twinx()
        ax2.plot(t, yaw_vel, color="#e377c2", lw=1.0, linestyle="--", label="yaw vel")
        ax2.set_ylabel("vel (deg/s)", color="#e377c2", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#e377c2", labelsize=8)
        
        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization" if i == 0 else "")
            if label == "Novice":
                ax.scatter([t[peak]], [yaw[peak]], s=30, color="#d1495b", zorder=4, label="yaw peak" if i == 0 else "")
        
        ax.set_ylabel(f"{label}\\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False, ncol=4)"""

content = content.replace(old_fig_loop, new_fig_loop)

# Fix metric aggregation in bar chart
old_novice_vals = 'novice_vals = [metrics.loc[metrics.trial == "Novice", m].iloc[0] for m in metric_names]'
new_novice_vals = 'novice_vals = [metrics.loc[metrics.trial == "Novice", m].mean() for m in metric_names]'
content = content.replace(old_novice_vals, new_novice_vals)

old_trained_vals = 'trained_vals = [metrics.loc[metrics.trial == "Trained", m].iloc[0] for m in metric_names]'
new_trained_vals = 'trained_vals = [metrics.loc[metrics.trial == "Trained", m].mean() for m in metric_names]'
content = content.replace(old_trained_vals, new_trained_vals)


# Replace write_report definition
old_report_def = """def write_report(
    novice_trial: TrialData,
    trained_trial: TrialData,
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    windows: dict[str, tuple[int, int, int, int]],
    novice_rationale: str,
    trained_dtw_distance: float,
) -> None:"""
new_report_def = """def write_report(
    novice_trial: TrialData,
    trained_trial: TrialData,
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    windows: dict[str, list[tuple[int, int, int, int, float]]],
    novice_rationale: str,
    trained_dtw_distance: float,
) -> None:"""
content = content.replace(old_report_def, new_report_def)

# Update write_report body
old_report_body = """    n = metrics.set_index("trial").to_dict(orient="index")
    novice_event = windows["Novice"]
    trained_event = windows["Trained"]"""
new_report_body = """    n = metrics.groupby("trial").mean().to_dict(orient="index")
    novice_event = windows["Novice"][0]
    trained_event = windows["Trained"][0]"""
content = content.replace(old_report_body, new_report_body)


# Replace main
old_main = """    novice_start, novice_end, novice_peak, rationale = select_novice_event(novice_kin, novice_trial.fs)
    trained_start, trained_end, dtw_dist = find_trained_match(
        novice_kin, trained_kin, novice_trial.fs, novice_start, novice_end
    )
    novice_stab_start, novice_stab_end = find_stabilization(novice_kin, novice_trial.fs, novice_end)
    trained_stab_start, trained_stab_end = find_stabilization(trained_kin, trained_trial.fs, trained_end)

    windows = {
        "Novice": (novice_start, novice_end, novice_stab_start, novice_stab_end),
        "Trained": (trained_start, trained_end, trained_stab_start, trained_stab_end),
    }
    pd.DataFrame(
        [
            {
                "trial": label,
                "event_start_s": start / FS,
                "event_end_s": end / FS,
                "event_peak_s": (novice_peak / FS if label == "Novice" else np.nan),
                "stabilization_start_s": stab_start / FS,
                "stabilization_end_s": stab_end / FS,
            }
            for label, (start, end, stab_start, stab_end) in windows.items()
        ]
    ).to_csv(OUTPUT_DIR / "event_windows.csv", index=False)

    metrics = pd.DataFrame(
        [
            compute_metrics("Novice", novice_kin, novice_trial.fs, novice_start, novice_end, novice_stab_start, novice_stab_end),
            compute_metrics("Trained", trained_kin, trained_trial.fs, trained_start, trained_end, trained_stab_start, trained_stab_end),
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "balance_metrics.csv", index=False)

    make_traceability_figure(novice_kin, trained_kin, metrics, windows, novice_trial.fs)
    make_orientation_validation_figure(novice_kin, trained_kin, novice_trial.fs)
    write_report(novice_trial, trained_trial, inventory, metrics, windows, rationale, dtw_dist)"""

new_main = """    novice_events, rationale = select_novice_events(novice_kin, novice_trial.fs, 3)
    
    all_metrics = []
    windows = {"Novice": [], "Trained": []}
    dtw_dists = []
    
    for idx, (novice_start, novice_end, novice_peak) in enumerate(novice_events):
        trained_start, trained_end, dtw_dist = find_trained_match(
            novice_kin, trained_kin, novice_trial.fs, novice_start, novice_end
        )
        dtw_dists.append(dtw_dist)
        
        novice_stab_start, novice_stab_end = find_stabilization(novice_kin, novice_trial.fs, novice_end)
        trained_stab_start, trained_stab_end = find_stabilization(trained_kin, trained_trial.fs, trained_end)
        
        windows["Novice"].append((novice_start, novice_end, novice_stab_start, novice_stab_end, novice_peak))
        windows["Trained"].append((trained_start, trained_end, trained_stab_start, trained_stab_end, np.nan))
        
        all_metrics.append(compute_metrics("Novice", novice_kin, novice_trial.fs, novice_start, novice_end, novice_stab_start, novice_stab_end))
        all_metrics.append(compute_metrics("Trained", trained_kin, trained_trial.fs, trained_start, trained_end, trained_stab_start, trained_stab_end))

    event_rows = []
    for label, events in windows.items():
        for i, (start, end, stab_start, stab_end, peak) in enumerate(events):
            event_rows.append({
                "trial": label,
                "event_index": i + 1,
                "event_start_s": start / FS,
                "event_end_s": end / FS,
                "event_peak_s": (peak / FS if label == "Novice" else np.nan),
                "stabilization_start_s": stab_start / FS,
                "stabilization_end_s": stab_end / FS,
            })
    pd.DataFrame(event_rows).to_csv(OUTPUT_DIR / "event_windows.csv", index=False)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUTPUT_DIR / "balance_metrics.csv", index=False)

    make_traceability_figure(novice_kin, trained_kin, metrics, windows, novice_trial.fs)
    make_orientation_validation_figure(novice_kin, trained_kin, novice_trial.fs)
    avg_dtw_dist = sum(dtw_dists) / len(dtw_dists)
    write_report(novice_trial, trained_trial, inventory, metrics, windows, rationale, avg_dtw_dist)"""

content = content.replace(old_main, new_main)

path.write_text(content)
