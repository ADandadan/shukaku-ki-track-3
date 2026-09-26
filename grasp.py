import numpy as np

def compute_grasp_pose(target_point_body: np.ndarray, stand_off_m: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes SE(3) end-effector target matrix and a standoff approach pose.
    Assumes arm reaches forward (-Z or +X depending on your body frame definition).
    """
    # Define tool orientation matrix (R_target) relative to robot body frame
    # Example: Tool Z-axis aligned towards target, Y-axis aligned with gravity
    z_axis = target_point_body / np.linalg.norm(target_point_body)
    y_axis = np.array([0, 0, -1], dtype=np.float64) # pointing down
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    R_target = np.column_stack((x_axis, y_axis, z_axis))
    
    # 1. Standoff pose (10 cm back along approach vector)
    p_approach = target_point_body - (stand_off_m * z_axis)
    
    # 2. Final grasp pose
    p_grasp = target_point_body.copy()
    
    return create_se3_matrix(R_target, p_approach), create_se3_matrix(R_target, p_grasp)

def create_se3_matrix(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T