"""Okra detection and guarded pick entry point.

Camera and motion transports are deployment-specific plugins because the bundled
SDK does not provide an RGB-D ImageClient or Cartesian arm trajectory API.
"""

import argparse
import importlib
import json
import os
import time

import numpy as np

from okra_pipeline import (
    CameraIntrinsics,
    SafetyConfig,
    _load_model,
    choose_detection,
    detections_from_result,
    median_depth,
    run_dry_run,
    safety_check,
    transform_point,
    unproject_pixel,
)
from movement import OkraPicker, PickConfig

def _load_factory(spec: str):
    """Load a deployment plugin written as ``python.module:factory``."""
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("plugin must use the format 'python.module:factory'")
    return getattr(importlib.import_module(module_name), factory_name)


def _load_matrix(name: str, shape: tuple[int, int]) -> np.ndarray:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to a calibrated {shape[0]}x{shape[1]} JSON matrix")
    try:
        matrix = np.asarray(json.loads(value), dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid JSON containing a numeric matrix") from error
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def get_frame_client(host: str):
    spec = os.environ.get("OKRA_CAMERA_CLIENT")
    if not spec:
        raise RuntimeError(
            "Set OKRA_CAMERA_CLIENT=module:factory to a client that implements "
            "connect(), get_intrinsics(), get_aligned_frame(), and close()."
        )
    client = _load_factory(spec)(host=host)
    client.connect()
    return client


def _as_intrinsics(value) -> CameraIntrinsics:
    try:
        if isinstance(value, CameraIntrinsics):
            intrinsics = value
        elif isinstance(value, dict):
            intrinsics = CameraIntrinsics(
                **{key: float(value[key]) for key in ("fx", "fy", "cx", "cy")}
            )
        else:
            intrinsics = CameraIntrinsics(
                fx=float(value.fx), fy=float(value.fy), cx=float(value.cx), cy=float(value.cy)
            )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("camera intrinsics must expose finite fx, fy, cx, and cy") from error
    if not all(np.isfinite(item) for item in vars(intrinsics).values()):
        raise ValueError("camera intrinsics must contain finite values")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    return intrinsics


def _make_picker(args) -> OkraPicker:
    rotation = _load_matrix("OKRA_GRASP_ROTATION", (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
        raise ValueError("OKRA_GRASP_ROTATION must be a proper orthonormal rotation matrix")
    spec = os.environ.get("OKRA_ARM_BACKEND")
    if not spec:
        raise RuntimeError(
            "Set OKRA_ARM_BACKEND=module:factory to a tested Cartesian arm controller "
            "implementing move_to_pose(target_se3, duration_s=...)."
        )
    arm = _load_factory(spec)()
    from dex1_gripper import Dex1GripperDDS

    gripper = Dex1GripperDDS(
        side=os.environ.get("OKRA_DEX1_SIDE", "left"),
        network_interface=args.network_interface,
        closed_position_rad=float(os.environ.get("OKRA_DEX1_CLOSED_RAD", "0.0")),
        open_position_rad=float(os.environ.get("OKRA_DEX1_OPEN_RAD", "5.5")),
    )

    class ArmAndGripperBackend:
        def move_to_pose(self, target_se3, *, duration_s):
            arm.move_to_pose(target_se3, duration_s=duration_s)

        def set_gripper(self, *, open_amount):
            gripper.set_gripper(open_amount=open_amount)

        def close(self):
            gripper.close()

    return OkraPicker(
        ArmAndGripperBackend(),
        PickConfig(
            safety=SafetyConfig(min_confidence=0.6, max_distance_m=1.0),
            grasp_rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Detect okra and optionally pick it up")
    parser.add_argument("--image", help="Run detection on a local image without robot hardware")
    parser.add_argument("--host", default="192.168.123.164", help="Robot image-server address")
    parser.add_argument(
        "--network-interface",
        default=os.environ.get("UNITREE_NETWORK_INTERFACE"),
        help="Network interface connected to the G1 DDS network",
    )
    parser.add_argument("--enable-motion", action="store_true", help="Allow the configured motion backend to move")
    parser.add_argument("--repo-id", default="Kota0612/okra11n-seg-v5")
    parser.add_argument("--weights", default="best.pt", help="Local weights path or model-repository filename")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    print("[INFO] Loading okra segmentation model...")
    model = _load_model(args.repo_id, args.weights)
    if args.image:
        return run_dry_run(args.image, model, args.confidence)

    camera_to_body = _load_matrix("OKRA_CAMERA_TO_BODY", (4, 4))
    if not np.allclose(camera_to_body[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("OKRA_CAMERA_TO_BODY must be a homogeneous rigid transform")
    if not np.allclose(
        camera_to_body[:3, :3].T @ camera_to_body[:3, :3], np.eye(3), atol=1e-3
    ) or not np.isclose(np.linalg.det(camera_to_body[:3, :3]), 1.0, atol=1e-3):
        raise ValueError("OKRA_CAMERA_TO_BODY rotation must be proper and orthonormal")

    picker = None
    if args.enable_motion:
        picker = _make_picker(args)
    else:
        print("[INFO] Motion disabled; use --enable-motion after configuring a tested backend.")
    print("[INFO] Connecting to robot's image_server...")
    client = get_frame_client(args.host)
    try:
        intrinsics = _as_intrinsics(client.get_intrinsics())
        print(
            f"[INFO] Camera Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
            f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}"
        )

        target_latched = False
        while True:
            frame, depth_m, capture_ts = client.get_aligned_frame()
            if frame is None or depth_m is None or capture_ts is None:
                continue
            depth_m = np.asarray(depth_m)
            if depth_m.ndim != 2:
                raise ValueError("camera depth must be a 2D array in meters")
            if getattr(frame, "shape", None) is None or frame.shape[:2] != depth_m.shape:
                raise ValueError("camera RGB and aligned depth dimensions must match")
            try:
                capture_ts = float(capture_ts)
            except (TypeError, ValueError) as error:
                raise ValueError("camera capture timestamp must be Unix seconds") from error

            results = model.predict(
                source=frame, imgsz=640, conf=args.confidence, verbose=False
            )
            detections = [d for result in results for d in detections_from_result(result)]
            selected = choose_detection(detections, minimum_confidence=0.5)

            if selected is None:
                target_latched = False
                print("[INFO] No okra detection")
                continue

            if target_latched:
                print("[INFO] Waiting for the picked target to leave the camera view")
                continue

            d = median_depth(depth_m, selected.centroid_u, selected.centroid_v, radius_px=3)
            if d is None:
                print("[WARN] No valid depth near the detected okra")
                continue

            point_cam = unproject_pixel(selected.centroid_u, selected.centroid_v, d, intrinsics)
            point_body = transform_point(point_cam, camera_to_body)

            # Camera plugins must provide Unix epoch seconds for capture_ts.
            frame_age = time.time() - capture_ts
            ok, reason = safety_check(
                selected,
                point_body,
                frame_age_s=frame_age,
                config=SafetyConfig(min_confidence=0.6, max_distance_m=1.0),
            )

            print(
                f"conf={selected.confidence:.2f} | "
                f"u={selected.centroid_u:.1f}, v={selected.centroid_v:.1f} | "
                f"depth={d:.3f}m | "
                f"point_body=[{point_body[0]:.3f}, {point_body[1]:.3f}, {point_body[2]:.3f}] | "
                f"ok={ok} ({reason})"
            )

            if ok and picker is not None:
                picked, pick_reason = picker.pick(
                    point_body, selected, frame_age_s=frame_age
                )
                print(f"[PICK] {pick_reason}")
                target_latched = picked

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        client.close()
        if picker is not None:
            picker.backend.close()


if __name__ == "__main__":
    raise SystemExit(main())