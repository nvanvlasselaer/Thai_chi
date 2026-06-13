from analysis.tai_chi_kinematic_analysis import parse_trigno_csv
from pathlib import Path

trial = parse_trigno_csv(Path("data/IMU_Trial_1_RC_Novice.csv"), "Novice")
print("Lumbar neutral acc:")
print(trial.data["lumbar"].iloc[1500:2000][["acc_x_g", "acc_y_g", "acc_z_g"]].mean())
print("Chestbone neutral acc:")
print(trial.data["chestbone"].iloc[1500:2000][["acc_x_g", "acc_y_g", "acc_z_g"]].mean())
print("Lhumerus neutral acc:")
print(trial.data["lhumerus"].iloc[1500:2000][["acc_x_g", "acc_y_g", "acc_z_g"]].mean())
print("Rhumerus neutral acc:")
print(trial.data["rhumerus"].iloc[1500:2000][["acc_x_g", "acc_y_g", "acc_z_g"]].mean())
