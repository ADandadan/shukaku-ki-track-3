"""Hardware-independent okra perception and camera-to-body geometry helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    centroid_u: float
    centroid_v: float
    confidence: float
    area_px: float
    index: int


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SafetyConfig:
    min_confidence: float = 0.60
    max_distance_m: float = 2.0
    max_frame_age_s: float = 0.25


def detections_from_result(result: Any) -> list[Detection]:
    """Extract valid polygon centroids from one Ultralytics segmentation result."""
    if result.masks is None:
        return []

    detections: list[Detection] = []
    for index, polygon in enumerate(result.masks.xy):
        points = np.asarray(polygon, dtype=np.float32)
        if len(points) < 3:
            continue
        moments = cv2.moments(points)
        area = float(moments["m00"])
        if area <= 1e-6:
            continue
        detections.append(
            Detection(
                centroid_u=moments["m10"] / area,
                centroid_v=moments["m01"] / area,
                confidence=float(result.boxes.conf[index]),
                area_px=area,
                index=index,
            )
        )
    return detections


def choose_detection(
    detections: Iterable[Detection], *, minimum_confidence: float = 0.0
) -> Detection | None:
    """Choose the highest-confidence valid detection, breaking ties by area."""
    candidates = [
        detection
        for detection in detections
        if detection.confidence >= minimum_confidence
    ]
    return max(candidates, key=lambda item: (item.confidence, item.area_px), default=None)


def median_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    radius_px: int = 2,
) -> float | None:
    """Return the median finite, positive depth in a square pixel neighborhood."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")

    center_u, center_v = round(u), round(v)
    height, width = depth_m.shape
    u0, u1 = max(0, center_u - radius_px), min(width, center_u + radius_px + 1)
    v0, v1 = max(0, center_v - radius_px), min(height, center_v + radius_px + 1)
    if u0 >= u1 or v0 >= v1:
        return None

    values = depth_m[v0:v1, u0:u1]
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.median(valid)) if valid.size else None


def unproject_pixel(
    u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Unproject a pixel into the camera optical frame (x right, y down, z forward)."""
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be finite and positive")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    return np.array(
        [
            (u - intrinsics.cx) * depth_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def transform_point(point_camera: Sequence[float], camera_to_body: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous camera-to-body transform."""
    transform = np.asarray(camera_to_body, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("camera_to_body must have shape (4, 4)")
    point = np.asarray(point_camera, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_camera must contain three finite values")
    homogeneous = transform @ np.append(point, 1.0)
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-12:
        raise ValueError("camera_to_body produced an invalid homogeneous point")
    return homogeneous[:3] / homogeneous[3]


def safety_check(
    detection: Detection,
    point_body: Sequence[float],
    *,
    frame_age_s: float,
    config: SafetyConfig = SafetyConfig(),
) -> tuple[bool, str]:
    """Validate freshness, confidence, and target distance before motion is enabled."""
    if not np.isfinite(frame_age_s) or frame_age_s < 0:
        return False, "invalid frame age"
    if frame_age_s > config.max_frame_age_s:
        return False, "stale frame"
    if detection.confidence < config.min_confidence:
        return False, "confidence below grasp threshold"

    point = np.asarray(point_body, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False, "invalid body-frame point"
    distance = float(np.linalg.norm(point))
    if distance <= 0 or distance > config.max_distance_m:
        return False, "target distance outside safety limit"
    return True, "ok"


def _load_model(repo_id: str, filename: str) -> Any:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(weights)


def run_dry_run(image_path: Path, model: Any, confidence: float) -> int:
    """Detect and print 2D centroids; no robot or camera transport is contacted."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"unable to read image: {image_path}")
    results = model.predict(source=image, imgsz=640, conf=confidence, verbose=False)
    detections = [detection for result in results for detection in detections_from_result(result)]
    selected = choose_detection(detections)
    if selected is None:
        print("No valid okra detection")
        return 1
    print(
        f"okra index={selected.index} centroid=({selected.centroid_u:.1f}, "
        f"{selected.centroid_v:.1f}) confidence={selected.confidence:.3f} "
        f"area_px={selected.area_px:.1f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hardware-free okra detection dry-run")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--repo-id", default="Kota0612/okra11n-seg-v5", help="Hugging Face model repository"
    )
    parser.add_argument(
        "--weights",
        default="output/okra_finetune_v5/weights/best.pt",
        help="Weights path inside the model repository",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    return run_dry_run(args.image, _load_model(args.repo_id, args.weights), args.conf)


if __name__ == "__main__":
    raise SystemExit(main())
