import time
import enum
import numpy as np
from dataclasses import dataclass

# DimOS skill execution client (dimos agent / skill RPC transport)
from dimos.skills import SkillClient # conceptual dimos RPC skill interface

class RobotState(enum.Enum):
    SEARCH = 0
    APPROACH = 1
    ALIGN = 2
    GRASP = 3
    DEPOSIT = 4

@dataclass
class PickConfig:
    max_grasp_reach_m: float = 0.60   # Max reach along forward/side axes (meters)
    standoff_dist_m: float = 0.12     # Standoff approach offset before closing hand
    ideal_grasp_dist_m: float = 0.45   # Target distance to step towards during APPROACH
    basket_drop_joints: list[float] = None # Joint config for dropping into basket

class OkraPickerStateMachine:
    def __init__(self, ik_solver, sdk_controller, dimos_client: SkillClient):
        self.ik = ik_solver
        self.sdk = sdk_controller
        self.dimos = dimos_client
        self.config = PickConfig(
            basket_drop_joints=[-0.5, 0.2, 0.0, 1.5, 0.0, 0.0, 0.0]
        )
        self.state = RobotState.SEARCH

    def process_frame(self, target_point_body: np.ndarray | None, frame_age_s: float):
        """Main execution iteration triggered on each perception update."""
        if target_point_body is None:
            self.state = RobotState.SEARCH
            return

        # 1. Measure 2D planar distance to target pod
        x, y, z = target_point_body
        dist_2d = float(np.linalg.norm([x, y]))

        # State Transitions
        if self.state == RobotState.SEARCH or self.state == RobotState.APPROACH:
            if dist_2d > self.config.max_grasp_reach_m:
                self.execute_dimos_approach(x, y, dist_2d)
            else:
                self.state = RobotState.ALIGN

        if self.state == RobotState.ALIGN:
            print(f"[PICKER] Target in arm reach ({dist_2d:.2f}m). Locking stance for IK...")
            self.execute_grasp_sequence(target_point_body)

    def execute_dimos_approach(self, target_x: float, target_y: float, dist_2d: float):
        """Commands dimOS locomotion to walk within reach range of the target."""
        self.state = RobotState.APPROACH
        
        # Calculate relative walking vector: step forward until dist_2d = ideal_grasp_dist_m
        walk_dist = dist_2d - self.config.ideal_grasp_dist_m
        heading_angle = np.arctan2(target_y, target_x)
        
        dx = walk_dist * np.cos(heading_angle)
        dy = walk_dist * np.sin(heading_angle)

        print(f"[dimOS] Target out of reach ({dist_2d:.2f}m). Walking rel (dx={dx:.2f}m, dy={dy:.2f}m)...")
        
        # Execute relative movement skill via dimOS CLI / RPC wrapper
        self.dimos.call("move_to", x=float(dx), y=float(dy), relative=True)
        
        # Allow robot time to complete step sequence and balance
        time.sleep(2.0)

    def execute_grasp_sequence(self, target_point_body: np.ndarray):
        """Executes arm reach, pinch, and basket deposit using Pinocchio IK & SDK 2."""
        self.state = RobotState.GRASP
        
        # Step 1: Compute SE(3) Matrices
        T_standoff, T_grasp = compute_grasp_pose(
            target_point_body, stand_off_m=self.config.standoff_dist_m
        )

        # Step 2: Solve IK for Standoff
        q_current = self.sdk.get_current_joint_positions()
        success, q_standoff = self.ik.solve_ik(T_standoff, q_init=q_current)
        if not success:
            print("[ERROR] Standoff IK failed! Re-evaluating...")
            return

        # Step 3: Move to Standoff
        self.sdk.send_joint_trajectory(q_current, q_standoff, duration_s=1.5)

        # Step 4: Solve & Move to Final Grasp Pose
        success, q_grasp = self.ik.solve_ik(T_grasp, q_init=q_standoff)
        if success:
            self.sdk.send_joint_trajectory(q_standoff, q_grasp, duration_s=0.8)
            
            # Step 5: Actuate End-Effector (Close Gripper)
            self.sdk.set_gripper_state(open_amount=0.0)
            time.sleep(0.5)

            # Step 6: Deposit into Off-Hand Basket
            self.execute_basket_deposit(q_grasp)

    def execute_basket_deposit(self, q_from: np.ndarray):
        """Moves picking arm across torso to drop okra into the basket carried by left hand."""
        self.state = RobotState.DEPOSIT
        print("[PICKER] Moving to basket drop-off zone...")
        
        # 1. Lift pod upwards to clear foliage
        q_lift = q_from.copy()
        q_lift[1] -= 0.2  # Pitch arm up
        self.sdk.send_joint_trajectory(q_from, q_lift, duration_s=1.0)

        # 2. Move to predefined basket position above off-hand
        q_basket = np.array(self.config.basket_drop_joints)
        self.sdk.send_joint_trajectory(q_lift, q_basket, duration_s=2.0)

        # 3. Release gripper
        self.sdk.set_gripper_state(open_amount=1.0)
        time.sleep(0.5)

        # 4. Return to home ready stance
        self.sdk.reset_arm_to_home()
        self.state = RobotState.SEARCH