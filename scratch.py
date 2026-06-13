import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

DATA_DIR = Path("data")
FS = 370.3704

sys.path.append(".")
from analysis.tai_chi_kinematic_analysis import parse_trigno_csv, madgwick_imu, quat_inverse, quat_multiply, quat_to_euler_deg

novice = parse_trigno_csv(DATA_DIR / "IMU_Trial_1_RC_Novice.csv", "Novice")

def find_neutral_pose_window(trial, min_duration_s=2.0):
    lumbar_acc_x = trial.data["lumbar"]["acc_x_g"].to_numpy()
    lumbar_gyro = trial.data["lumbar"][["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()
    lumbar_omega = np.linalg.norm(lumbar_gyro, axis=1)
    
    search_window = int(10 * trial.fs)
    search_window = min(search_window, len(lumbar_acc_x))
    peak_idx = int(np.nanargmax(np.abs(lumbar_acc_x[:search_window])))
    
    settle_time = int(0.5 * trial.fs)
    start_search = peak_idx + settle_time
    window_size = int(trial.fs * min_duration_s)
    
    if start_search + window_size >= len(lumbar_omega):
        return 0, len(lumbar_omega)
    
    search_end = min(len(lumbar_omega), start_search + int(15 * trial.fs))
    omega_search = lumbar_omega[start_search:search_end]
    
    rolling_mean_omega = pd.Series(omega_search).rolling(window_size).mean().to_numpy()
    best_end_rel = int(np.nanargmin(rolling_mean_omega))
    
    best_end = start_search + best_end_rel
    best_start = best_end - window_size + 1
    
    return best_start, best_end

def mean_quaternion(quats: np.ndarray) -> np.ndarray:
    q = np.mean(quats, axis=0)
    return q / np.linalg.norm(q)

n = min(len(df) for df in novice.data.values())
df = novice.data["lumbar"].iloc[:n]
acc = df[["acc_x_g", "acc_y_g", "acc_z_g"]].to_numpy()
gyro = df[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()

raw_quats = madgwick_imu(acc, gyro, novice.fs)
neutral_start, neutral_end = find_neutral_pose_window(novice)
print(f"Neutral window: {neutral_start/novice.fs:.2f} s to {neutral_end/novice.fs:.2f} s")

q_offset = mean_quaternion(raw_quats[neutral_start:neutral_end])
print(f"q_offset: {q_offset}")

q_offset_inv = quat_inverse(q_offset)
print(f"q_offset_inv: {q_offset_inv}")

# multiply live sensor by inverse of offset
calibrated_quats = quat_multiply(raw_quats, q_offset_inv)

# let's look at the euler angles in the neutral pose before and after calibration
eulers_raw = quat_to_euler_deg(raw_quats[neutral_start:neutral_end])
eulers_calibrated = quat_to_euler_deg(calibrated_quats[neutral_start:neutral_end])

print(f"Raw mean euler: {np.mean(eulers_raw, axis=0)}")
print(f"Calibrated mean euler: {np.mean(eulers_calibrated, axis=0)}")
