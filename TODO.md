# Deployment checklist

- [x] Okra segmentation, depth unprojection, freshness/confidence/range checks, and guarded pick sequencing.
- [x] Dex1-1 DDS position client using the official `rt/dex1/{side}/cmd` and `/state` topics.
- [ ] Implement `OKRA_CAMERA_CLIENT` for the deployed RGB-D camera; frames must be RGB, depth in meters, and carry Unix-epoch capture timestamps.
- [ ] Measure and configure `OKRA_CAMERA_TO_BODY` and `OKRA_GRASP_ROTATION` for the installed camera and gripper. Placeholder geometry is rejected for live operation.
- [ ] Implement `OKRA_ARM_BACKEND` with tested Cartesian `move_to_pose(target_se3, duration_s=...)`, joint-limit enforcement, and a verified emergency stop path.
- [ ] Test the complete sequence in Development Mode with the robot secured and a soft target before enabling autonomous operation.
