"""Hardware-independent grasp sequencing for a configured arm/gripper backend."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from grasp import compute_grasp_pose
from okra_pipeline import Detection, SafetyConfig, safety_check


class MotionBackend(Protocol):
	"""Required contract for a robot-specific arm and Dex1 driver."""

	def move_to_pose(self, target_se3: np.ndarray, *, duration_s: float) -> None: ...

	def set_gripper(self, *, open_amount: float) -> None: ...


class PickerState(enum.Enum):
	SEARCH = "search"
	APPROACH = "approach"
	GRASP = "grasp"
	RETREAT = "retreat"
	ERROR = "error"


@dataclass(frozen=True)
class PickConfig:
	safety: SafetyConfig = SafetyConfig()
	standoff_m: float = 0.10
	grasp_rotation: tuple[tuple[float, float, float], ...] | None = None
	approach_duration_s: float = 1.5
	grasp_duration_s: float = 0.8
	retreat_duration_s: float = 1.0
	gripper_settle_s: float = 0.5


class OkraPicker:
	"""Execute one guarded pick using a robot-specific ``MotionBackend``."""

	def __init__(
		self,
		backend: MotionBackend,
		config: PickConfig = PickConfig(),
		*,
		sleep: Callable[[float], None] = time.sleep,
	) -> None:
		if not callable(getattr(backend, "move_to_pose", None)) or not callable(
			getattr(backend, "set_gripper", None)
		):
			raise TypeError("motion backend must implement move_to_pose and set_gripper")
		if config.standoff_m <= 0:
			raise ValueError("standoff_m must be positive")
		self.backend = backend
		self.config = config
		self._sleep = sleep
		self.state = PickerState.SEARCH

	def pick(
		self,
		target_point_body: Sequence[float],
		detection: Detection,
		*,
		frame_age_s: float,
	) -> tuple[bool, str]:
		"""Perform a single pick; reject unsafe detections without issuing commands."""
		allowed, reason = safety_check(
			detection,
			target_point_body,
			frame_age_s=frame_age_s,
			config=self.config.safety,
		)
		if not allowed:
			self.state = PickerState.SEARCH
			return False, reason
		if self.config.grasp_rotation is None:
			self.state = PickerState.SEARCH
			return False, "grasp rotation is not calibrated"

		target = np.asarray(target_point_body, dtype=np.float64)
		approach_pose, grasp_pose = compute_grasp_pose(
			target,
			stand_off_m=self.config.standoff_m,
			rotation=np.asarray(self.config.grasp_rotation, dtype=np.float64),
		)

		try:
			self.state = PickerState.APPROACH
			self.backend.set_gripper(open_amount=1.0)
			self.backend.move_to_pose(
				approach_pose, duration_s=self.config.approach_duration_s
			)
			self.state = PickerState.GRASP
			self.backend.move_to_pose(grasp_pose, duration_s=self.config.grasp_duration_s)
			self.backend.set_gripper(open_amount=0.0)
			self._sleep(self.config.gripper_settle_s)
			self.state = PickerState.RETREAT
			self.backend.move_to_pose(
				approach_pose, duration_s=self.config.retreat_duration_s
			)
		except Exception:
			self.state = PickerState.ERROR
			raise

		self.state = PickerState.SEARCH
		return True, "pick sequence completed"
