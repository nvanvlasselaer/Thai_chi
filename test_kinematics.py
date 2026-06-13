import numpy as np
from scipy.spatial.transform import Rotation as sT

def get_nominal_sensor_quaternion(sensor_name: str) -> np.ndarray:
    if sensor_name in ["chestbone", "lulna", "rulna"]:
        R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    elif sensor_name == "lumbar":
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    elif sensor_name in ["lhumerus", "lhand", "lthigh", "ltibia"]:
        R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    elif sensor_name in ["rhumerus", "rhand", "rthigh", "rtibia"]:
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    elif sensor_name in ["lfoot", "rfoot"]:
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        R = np.eye(3)
    q = sT.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

print("Lumbar nom:", get_nominal_sensor_quaternion("lumbar"))
print("Chest nom:", get_nominal_sensor_quaternion("chestbone"))
