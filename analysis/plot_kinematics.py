import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your CSV file here
CSV_PATH = "outputs/kinematic_variables_novice_50hz.csv"  # Change this to your CSV file path

def load_data(filepath):
    """Load the CSV file"""
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def plot_angle_groups(data):
    """Plot angles in logical groups"""
    # Define angle groups
    groups = {
        'Trunk-Pelvis Angles': ['trunk_pelvis_roll_deg', 'trunk_pelvis_pitch_deg', 'trunk_pelvis_yaw_deg'],
        'Lumbar Angles': ['lumbar_roll_deg', 'lumbar_pitch_deg', 'lumbar_yaw_deg'],
        'Chest Angles': ['chest_yaw_deg'],
        'Angular Velocities': ['lumbar_omega_dps', 'chest_omega_dps'],
        'left_knee Angles': ['left_knee_roll_est_deg', 'left_knee_pitch_est_deg', 'left_knee_yaw_est_deg'],
        'right_knee Angles': ['right_knee_roll_est_deg', 'right_knee_pitch_est_deg', 'right_knee_yaw_est_deg'],
        'left_hip Angles': ['left_hip_roll_est_deg', 'left_hip_pitch_est_deg', 'left_hip_yaw_est_deg'],
        'right_hip Angles': ['right_hip_roll_est_deg', 'right_hip_pitch_est_deg', 'right_hip_yaw_est_deg'],
        'left_shoulder Angles': ['left_shoulder_roll_est_deg', 'left_shoulder_pitch_est_deg', 'left_shoulder_yaw_est_deg'],
        'right_shoulder Angles': ['right_shoulder_roll_est_deg', 'right_shoulder_pitch_est_deg', 'right_shoulder_yaw_est_deg'], 
    }
    # Filter groups to only include columns that exist in the data
    available_groups = {}
    for group_name, columns in groups.items():
        available_columns = [col for col in columns if col in data.columns]
        if available_columns:
            available_groups[group_name] = available_columns
    
    # Create subplots
    n_groups = len(available_groups)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    fig.suptitle('Angle Groups', fontsize=14, fontweight='bold')
    
    # Flatten axes for easy iteration
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Plot each group
    for idx, (group_name, columns) in enumerate(available_groups.items()):
        ax = axes[idx]
        
        for col in columns:
            ax.plot(data['time_s'], data[col], linewidth=1.5, label=col.replace('_', ' ').title())
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)' if 'dps' not in group_name.lower() else 'Angular Velocity (dps)')
        ax.set_title(group_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    # Load the data
    data = load_data(CSV_PATH)
    
    if data is None:
        return
    
    
    print("\nPlotting angle groups...")
    plot_angle_groups(data)
    

if __name__ == "__main__":
    main()