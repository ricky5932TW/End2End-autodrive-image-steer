"""Virtual Xbox controller output via vgamepad."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ControllerOutput:
    steer: float
    accel: float
    brake: float


class NullController:
    def __init__(self) -> None:
        self.last = ControllerOutput(0.0, 0.0, 0.0)

    def apply(self, steer: float, accel: float, brake: float) -> ControllerOutput:
        self.last = ControllerOutput(steer, accel, brake)
        return self.last

    def reset(self) -> None:
        self.last = ControllerOutput(0.0, 0.0, 0.0)


class GamepadController:
    def __init__(
        self,
        max_accel: float = 0.75,
        steer_smoothing: float = 0.35,
        trigger_smoothing: float = 0.25,
        steer_deadzone: float = 0.02,
    ) -> None:
        try:
            import vgamepad as vg
        except ImportError as exc:
            raise RuntimeError("vgamepad is not installed in this Python environment") from exc

        try:
            self.gamepad = vg.VX360Gamepad()
        except Exception as exc:
            raise RuntimeError(
                "Could not create VX360Gamepad. Install or repair the ViGEmBus driver, "
                "then rerun the vgamepad smoke test."
            ) from exc

        self.max_accel = clamp(max_accel, 0.0, 1.0)
        self.steer_smoothing = clamp(steer_smoothing, 0.0, 1.0)
        self.trigger_smoothing = clamp(trigger_smoothing, 0.0, 1.0)
        self.steer_deadzone = clamp(steer_deadzone, 0.0, 1.0)
        self._steer = 0.0
        self._accel = 0.0
        self._brake = 0.0
        self.reset()

    def apply(self, steer: float, accel: float, brake: float) -> ControllerOutput:
        target_steer = clamp(float(steer), -1.0, 1.0)
        target_accel = clamp(float(accel), 0.0, self.max_accel)
        target_brake = clamp(float(brake), 0.0, 1.0)

        if abs(target_steer) < self.steer_deadzone:
            target_steer = 0.0

        self._steer = self._blend(self._steer, target_steer, self.steer_smoothing)
        self._accel = self._blend(self._accel, target_accel, self.trigger_smoothing)
        self._brake = self._blend(self._brake, target_brake, self.trigger_smoothing)

        self.gamepad.left_joystick_float(x_value_float=self._steer, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=self._accel)
        self.gamepad.left_trigger_float(value_float=self._brake)
        self.gamepad.update()
        return ControllerOutput(self._steer, self._accel, self._brake)

    def reset(self) -> None:
        self._steer = 0.0
        self._accel = 0.0
        self._brake = 0.0
        self.gamepad.reset()
        self.gamepad.update()

    @staticmethod
    def _blend(current: float, target: float, alpha: float) -> float:
        return current + (target - current) * alpha
