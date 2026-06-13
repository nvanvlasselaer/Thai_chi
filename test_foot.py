from analysis.tai_chi_kinematic_analysis import parse_trigno_csv
from pathlib import Path
trial = parse_trigno_csv(Path("data/IMU_Trial_1_RC_Novice.csv"), "Novice")
print("Lfoot neutral acc:")
print(trial.data["lfoot"].iloc[1500:2000][["acc_x_g", "acc_y_g", "acc_z_g"]].mean())
