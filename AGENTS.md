# Okra Picking Project Handoff

## Goal

Detect okra from an RGB-D camera, estimate its position in the G1 body frame, and perform a guarded single-item pick with the Dex1-1 gripper. Run application code from an external PC in the appropriate G1 mode (Development Mode by default). Do not install or modify software on the G1 internal PC.

The user currently has no robot access. Do not attempt SSH, connect to the robot, or run hardware tests unless the user later confirms access and explicitly requests them.

## Deferred SSH Access

When the user has the robot physically available and explicitly authorizes access:

1. Connect the external development PC to the G1 LAN port with a wired Ethernet cable.
2. Confirm the robot is in the appropriate development mode and that the PC is on the robot network.
3. From the external PC, connect to the internal PC:

   ```bash
   ssh unitree@192.168.123.164
   ```

4. Enter the robot's configured SSH password when prompted. Do not store the password in this repository, scripts, environment files, or chat logs.
5. Use SSH for inspection and deployment diagnostics only. Do not install new software on the internal PC or modify its existing programs.
6. Exit the session with `exit` when inspection is complete.

Do not run this command, probe the address, or perform any robot-side checks until the user confirms that robot access is available and requests the operation. Run the okra application from the external PC; SSH is not required for the normal application path.

## Current Implementation

- `okra_pipeline.py` extracts segmentation-mask centroids, selects detections, filters depth samples, unprojects pixels, transforms points, checks confidence/freshness/range, loads local weights first, and supports image-only dry runs.
- `robot.py` is the entry point. It supports local-image detection and an RGB-D streaming loop. Live operation uses explicit camera and arm plugins and requires calibrated transforms. Motion requires `--enable-motion`.
- `movement.py` implements one guarded pick sequence: open gripper, approach pose, grasp pose, close gripper, and retreat. Unsafe detections and missing grasp calibration issue no motion commands.
- `grasp.py` creates approach/grasp poses and validates target points and rotation matrices. Its default rotation remains a placeholder for offline use; live picking must supply a measured `OKRA_GRASP_ROTATION`.
- `dex1_gripper.py` controls the configured Dex1 side over the official serial2dds DDS topics. It expects the service to be running and publishes a position target while checking state feedback.
- `test_okra_pipeline.py` contains perception and fake-backend pick tests.

The bundled Unitree Python SDK has arm gestures and low-level examples, but this project does not yet contain an RGB-D camera client or a tested Cartesian arm trajectory backend. `dim.py` is not wired into `robot.py`; do not treat it as a functioning locomotion or arm driver. The current picker does not deposit into a basket or move the robot base.

## Live Integration Contracts

Before live operation, implement and bench-test the deployment-specific components:

- `OKRA_CAMERA_CLIENT=python.module:factory`: the factory is called with `host=...`; the client implements `connect()`, `get_intrinsics()`, `get_aligned_frame()`, and `close()`. Intrinsics expose `fx`, `fy`, `cx`, `cy`. Each frame call returns `(rgb_frame, aligned_depth_m, capture_timestamp_unix_seconds)`. Depth must be in meters and aligned to the RGB frame.
- `OKRA_CAMERA_TO_BODY`: JSON 4x4 calibrated homogeneous transform from the camera optical frame (x right, y down, z forward) to the body frame. The rotation must be a proper orthonormal matrix.
- `OKRA_GRASP_ROTATION`: JSON 3x3 measured end-effector orientation in the body frame. It must be a proper orthonormal rotation matrix.
- `OKRA_ARM_BACKEND=python.module:factory`: a zero-argument factory returning a tested Cartesian arm controller with `move_to_pose(target_se3, duration_s=...)`. That controller must enforce joint limits and the robot's control-mode requirements, and provide a verified emergency stop path.
- `UNITREE_NETWORK_INTERFACE` or `--network-interface`: the network interface connected to the G1 DDS network. Install the bundled Unitree SDK Python package and its Cyclone DDS requirements on the external PC. The Dex1 serial2dds service must already be installed, calibrated, and running on the robot.
- `OKRA_DEX1_SIDE`: `left` or `right` (defaults to `left`). `OKRA_DEX1_CLOSED_RAD` and `OKRA_DEX1_OPEN_RAD` default to `0.0` and `5.5`; set these to verified positions for the calibrated gripper.

`OKRA_CAMERA_TO_BODY` and `OKRA_GRASP_ROTATION` are not safely inferable from the repository. Do not substitute guessed values to enable motion.

## Running Without Hardware

Run local detection without camera or robot transports:

```powershell
python robot.py --image image.png
```

The checked-in `best.pt` is used when present. Otherwise the model is downloaded from the configured Hugging Face repository, so the relevant Python dependencies and network access are needed.

For a future live deployment, only after all contracts above are implemented and verified:

```powershell
python robot.py --enable-motion --network-interface <g1-network-interface>
```

Without `--enable-motion`, the streaming path is perception-only. Do not run the live command until the RGB-D alignment, coordinate calibration, arm workspace, Dex1 direction/range, control mode, and emergency-stop behavior have been validated.

## Verification

- Syntax checks with `python -m py_compile` passed for the touched Python modules.
- Hardware-free smoke checks passed for stale-frame rejection, missing grasp-calibration rejection, transform parsing, and the fake-backend pick sequence.
- `git diff --check` passed.
- `pytest` was unavailable in the development environment (`No module named pytest`); run `python -m pytest -q test_okra_pipeline.py` once pytest is installed.
- No SSH or robot-side checks were performed.

## Next Steps

1. Implement the RGB-D client and verify its intrinsics, depth units/alignment, and timestamp clock.
2. Measure the camera extrinsics and gripper orientation; record verified matrices outside source code and configure the environment variables.
3. Implement the Cartesian arm backend against the actual G1 model/control mode, with joint-limit checks and an emergency stop.
4. Test the Dex1 DDS side and open/closed endpoints independently, then bench-test the full sequence with the robot secured and a soft target in Development Mode.
5. Add basket deposit/base approach only after the single-pick sequence is proven safe.
