"""DDS position control for a calibrated Unitree Dex1-1 gripper."""

from __future__ import annotations

import time
import math


class Dex1GripperDDS:
    """Control one Dex1-1 hand through the official serial2dds service topics."""

    def __init__(
        self,
        *,
        side: str,
        network_interface: str,
        closed_position_rad: float = 0.0,
        open_position_rad: float = 5.5,
        kp: float = 5.0,
        kd: float = 0.05,
        state_timeout_s: float = 3.0,
        command_timeout_s: float = 6.0,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not network_interface:
            raise ValueError("network_interface is required for Unitree DDS")
        if not 0.0 <= closed_position_rad <= 5.62:
            raise ValueError("closed_position_rad must be between 0 and 5.62 rad")
        if not 0.0 <= open_position_rad <= 5.62:
            raise ValueError("open_position_rad must be between 0 and 5.62 rad")
        if closed_position_rad == open_position_rad:
            raise ValueError("open and closed gripper positions must differ")
        if not all(math.isfinite(value) for value in (kp, kd, state_timeout_s, command_timeout_s)):
            raise ValueError("gains and timeouts must be finite")
        if kp < 0 or kd < 0 or state_timeout_s <= 0 or command_timeout_s <= 0:
            raise ValueError("gains must be non-negative and timeouts must be positive")

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            MotorCmd_,
            MotorCmds_,
            MotorStates_,
        )

        ChannelFactoryInitialize(0, network_interface)
        self._command_type = MotorCmd_
        self._commands_type = MotorCmds_
        self._publisher = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
        self._subscriber = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
        self._publisher.Init()
        self._subscriber.Init()
        self._closed_position = float(closed_position_rad)
        self._open_position = float(open_position_rad)
        self._kp = float(kp)
        self._kd = float(kd)
        self._command_timeout_s = float(command_timeout_s)
        self._position = self._wait_for_state(state_timeout_s)

    def _wait_for_state(self, timeout_s: float) -> float:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._subscriber.Read(timeout=0.1)
            if message is not None and message.states:
                return float(message.states[0].q)
        self.close()
        raise TimeoutError("no Dex1 state received; check the service and DDS interface")

    def set_gripper(self, *, open_amount: float) -> None:
        if not 0.0 <= open_amount <= 1.0:
            raise ValueError("open_amount must be between 0.0 (closed) and 1.0 (open)")
        target = self._closed_position + open_amount * (
            self._open_position - self._closed_position
        )
        command = self._command_type(
            mode=1,
            q=target,
            dq=0.0,
            tau=0.0,
            kp=self._kp,
            kd=self._kd,
            reserve=[0, 0, 0],
        )
        message = self._commands_type(cmds=[command])

        deadline = time.monotonic() + self._command_timeout_s
        while time.monotonic() < deadline:
            if not self._publisher.Write(message):
                raise RuntimeError("failed to publish Dex1 command")
            state = self._subscriber.Read(timeout=0.02)
            if state is not None and state.states:
                self._position = float(state.states[0].q)
                if abs(self._position - target) <= 0.12:
                    return
        raise TimeoutError(f"Dex1 did not reach target position {target:.2f} rad")

    def close(self) -> None:
        publisher = getattr(self, "_publisher", None)
        subscriber = getattr(self, "_subscriber", None)
        if publisher is not None:
            publisher.Close()
            self._publisher = None
        if subscriber is not None:
            subscriber.Close()
            self._subscriber = None