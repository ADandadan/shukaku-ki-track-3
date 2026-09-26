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
from movement import OkraPicker, PickConfig, PickerState


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


def test_picker_rejects_stale_frame_without_motion():
    calls = []

    class Backend:
        def move_to_pose(self, target_se3, *, duration_s):
            calls.append("move")

        def set_gripper(self, *, open_amount):
            calls.append("gripper")

    picker = OkraPicker(Backend(), sleep=lambda _: None)
    ok, reason = picker.pick(
        [0, 0, 0.4], Detection(10, 10, 0.9, 100, 0), frame_age_s=1.0
    )

    assert not ok
    assert reason == "stale frame"
    assert calls == []
    assert picker.state is PickerState.SEARCH


def test_picker_requires_calibrated_rotation():
    calls = []

    class Backend:
        def move_to_pose(self, target_se3, *, duration_s):
            calls.append("move")

        def set_gripper(self, *, open_amount):
            calls.append("gripper")

    picker = OkraPicker(Backend(), sleep=lambda _: None)
    ok, reason = picker.pick(
        [0, 0, 0.4], Detection(10, 10, 0.9, 100, 0), frame_age_s=0.01
    )

    assert not ok
    assert reason == "grasp rotation is not calibrated"
    assert calls == []


def test_picker_runs_open_approach_grasp_close_retreat():
    calls = []

    class Backend:
        def move_to_pose(self, target_se3, *, duration_s):
            calls.append(("move", duration_s))

        def set_gripper(self, *, open_amount):
            calls.append(("gripper", open_amount))

    rotation = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    picker = OkraPicker(
        Backend(), PickConfig(grasp_rotation=rotation), sleep=lambda _: None
    )
    ok, reason = picker.pick(
        [0, 0, 0.4], Detection(10, 10, 0.9, 100, 0), frame_age_s=0.01
    )

    assert ok, reason
    assert calls == [
        ("gripper", 1.0),
        ("move", picker.config.approach_duration_s),
        ("move", picker.config.grasp_duration_s),
        ("gripper", 0.0),
        ("move", picker.config.retreat_duration_s),
    ]
    assert picker.state is PickerState.SEARCH
