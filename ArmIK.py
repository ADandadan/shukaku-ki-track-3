import pinocchio as pin
import numpy as np

class G1ArmIK:
    def __init__(self, urdf_path: str, ee_frame_name: str = "right_hand_tcp"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
    def solve_ik(
        self, 
        target_se3: np.ndarray, 
        q_init: np.ndarray, 
        max_iter: int = 100, 
        eps: float = 1e-4, 
        dt: float = 1e-1, 
        damp: float = 1e-6
    ) -> tuple[bool, np.ndarray]:
        """Solves IK for target SE(3) frame matrix relative to body base."""
        q = q_init.copy()
        oMdes = pin.SE3(target_se3[:3, :3], target_se3[:3, 3])
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            dMi = oMdes.actInv(self.data.oMf[self.ee_frame_id])
            err = pin.log(dMi).vector # 6D Spatial error
            
            if np.linalg.norm(err) < eps:
                return True, q
                
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )
            v = - J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * dt)
            
            # Enforce joint position limits
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            
        return False, q