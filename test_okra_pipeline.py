import numpy as np
import pytest

from okra_pipeline import (
    CameraIntrinsics,
    Detection,
    SafetyConfig,
    median_depth,
    safety_check,
    transform_point,
    unproject_pixel,
)


def test_median_depth_ignores_invalid_neighbors():
    depth = np.array([[np.nan, 2.0, 100.0], [0.0, 2.2, 2.1]])
    assert median_depth(depth, 1, 1, radius_px=1) == pytest.approx(2.15)


def test_unproject_center_pixel():
    point = unproject_pixel(320, 240, 2.0, CameraIntrinsics(500, 500, 320, 240))
    np.testing.assert_allclose(point, [0, 0, 2])


def test_transform_point_applies_translation():
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(transform_point([.1, .2, .3], transform), [1.1, 2.2, 3.3])


def test_safety_check_rejects_stale_frame():
    detection = Detection(10, 20, 0.9, 100, 0)
    ok, reason = safety_check(
        detection,
        [0, 0, 1],
        frame_age_s=0.3,
        config=SafetyConfig(max_frame_age_s=0.25),
    )
    assert not ok
    assert reason == "stale frame"
