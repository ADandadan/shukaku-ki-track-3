import time
import enum
import numpy as np
from dataclasses import dataclass


class RobotState(enum.Enum):
    SEARCH = 0
    APPROACH = 1
    ALIGN = 2
    GRASP = 3
    DEPOSIT = 4


@dataclass
class PickConfig:
    max_grasp_reach_m: float = 0.60
    standoff_dist_m: float = 0.12
    ideal_grasp_dist_m: float = 0.45
    basket_drop_joints: list = None


class OkraPickerStateMachine:
    def __init__(self, ik_solver, sdk_controller, dimos_client):
        self.ik = ik_solver
        self.sdk = sdk_controller
        self.dimos = dimos_client
        self.config = PickConfig(basket_drop_joints=[-0.5, 0.2, 0.0, 1.5, 0.0, 0.0, 0.0])
        self.state = RobotState.SEARCH

    def process_frame(self, target_point_body, frame_age_s: float, min_confidence_ok: bool = True):
        if target_point_body is None or not min_confidence_ok or frame_age_s > 1.0:
            self.state = RobotState.SEARCH
            return

        x, y, z = target_point_body
        dist_2d = float(np.linalg.norm([x, y]))

        if self.state in (RobotState.SEARCH, RobotState.APPROACH):
            if dist_2d > self.config.max_grasp_reach_m:
                self.execute_dimos_approach(x, y, dist_2d)
            else:
                self.state = RobotState.ALIGN

        if self.state == RobotState.ALIGN:
            self.execute_grasp_sequence(target_point_body)

    def execute_dimos_approach(self, target_x, target_y, dist_2d):
        self.state = RobotState.APPROACH
        walk_dist = dist_2d - self.config.ideal_grasp_dist_m
        heading = np.arctan2(target_y, target_x)
        dx, dy = walk_dist * np.cos(heading), walk_dist * np.sin(heading)
        print(f"[dimOS] walking rel (dx={dx:.2f}, dy={dy:.2f})")
        self.dimos.call("move_to", x=float(dx), y=float(dy), relative=True)
        time.sleep(2.0)

    def execute_grasp_sequence(self, target_point_body):
        self.state = RobotState.GRASP
        T_standoff, T_grasp = compute_grasp_pose(target_point_body, self.config.standoff_dist_m)

        q_current = self.sdk.get_current_joint_positions()
        ok, q_standoff = self.ik.solve_ik(T_standoff, q_init=q_current)
        if not ok:
            print("[ERROR] standoff IK failed, back to search")
            self.state = RobotState.SEARCH   # <- the fix: don't get stuck
            return
        self.sdk.send_joint_trajectory(q_current, q_standoff, duration_s=1.5)

        ok, q_grasp = self.ik.solve_ik(T_grasp, q_init=q_standoff)
        if not ok:
            print("[ERROR] grasp IK failed, back to search")
            self.state = RobotState.SEARCH   # <- same fix here
            return
        self.sdk.send_joint_trajectory(q_standoff, q_grasp, duration_s=0.8)
        self.sdk.set_gripper_state(open_amount=0.0)
        time.sleep(0.5)
        self.execute_basket_deposit(q_grasp)

    def execute_basket_deposit(self, q_from):
        self.state = RobotState.DEPOSIT
        q_lift = q_from.copy()
        q_lift[1] -= 0.2
        self.sdk.send_joint_trajectory(q_from, q_lift, duration_s=1.0)

        q_basket = np.array(self.config.basket_drop_joints)
        self.sdk.send_joint_trajectory(q_lift, q_basket, duration_s=2.0)

        self.sdk.set_gripper_state(open_amount=1.0)
        time.sleep(0.5)
        self.sdk.reset_arm_to_home()
        self.state = RobotState.SEARCH