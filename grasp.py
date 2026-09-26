import numpy as np

# Measure this once on the real arm (a pose that's worked in testing), then hardcode it.
# This matches what both reference projects converged on — don't solve orientation dynamically.
FIXED_R_TARGET = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)  # placeholder — replace with your actual measured/tested orientation


def create_se3_matrix(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def compute_grasp_pose(
    target_point_body: np.ndarray,
    stand_off_m: float = 0.10,
    rotation: np.ndarray = FIXED_R_TARGET,
):
    """Fixed-orientation grasp pose: only position varies with the target."""
    target = np.asarray(target_point_body, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_point_body must contain three finite values")
    if np.linalg.norm(target) <= 1e-9:
        raise ValueError("target_point_body must not be the origin")
    if not np.isfinite(stand_off_m) or stand_off_m <= 0:
        raise ValueError("stand_off_m must be finite and positive")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
        raise ValueError("rotation must be a proper orthonormal rotation matrix")

    approach_dir = target / np.linalg.norm(target)
    p_approach = target - (stand_off_m * approach_dir)
    p_grasp = target.copy()
    return create_se3_matrix(rotation, p_approach), create_se3_matrix(rotation, p_grasp)


"""
def execute_grasp_sequence(self, target_point_body: np.ndarray, detection=None, frame_age_s: float = 0.0):
    ok, reason = safety_check(detection, target_point_body, frame_age_s=frame_age_s)
    if not ok:
        print(f"[PICKER] Safety check failed: {reason} — aborting grasp")
        self.state = RobotState.SEARCH
        return

    self.state = RobotState.GRASP
    T_standoff, T_grasp = compute_grasp_pose(target_point_body, stand_off_m=self.config.standoff_dist_m)

    q_current = self.sdk.get_current_joint_positions()
    success, q_standoff = self.ik.solve_ik(T_standoff, q_init=q_current)
    if not success:
        print("[ERROR] Standoff IK failed — resetting to search")
        self.state = RobotState.SEARCH  # <- was missing, left it stuck in GRASP
        return

    self.sdk.send_joint_trajectory(q_current, q_standoff, duration_s=1.5)

    success, q_grasp = self.ik.solve_ik(T_grasp, q_init=q_standoff)
    if not success:
        print("[ERROR] Grasp IK failed — resetting to search")
        self.state = RobotState.SEARCH
        return

    self.sdk.send_joint_trajectory(q_standoff, q_grasp, duration_s=0.8)
    self.sdk.set_gripper_state(open_amount=0.0)
    time.sleep(0.5)
    self.execute_basket_deposit(q_grasp)
"""