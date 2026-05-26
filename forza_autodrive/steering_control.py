"""Closed-loop steering correction for model targets and UDP feedback."""

from __future__ import annotations

from dataclasses import dataclass
import time


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SteeringControlOutput:
    target: float
    measurement: float | None
    filtered_measurement: float | None
    error: float | None
    correction: float
    command: float


class OneDimensionalKalmanFilter:
    """Tiny scalar Kalman filter for noisy normalized steering feedback."""

    def __init__(
        self,
        process_noise: float = 0.002,
        measurement_noise: float = 0.03,
        initial_uncertainty: float = 1.0,
    ) -> None:
        self.process_noise = max(1e-9, float(process_noise))
        self.measurement_noise = max(1e-9, float(measurement_noise))
        self.initial_uncertainty = max(1e-9, float(initial_uncertainty))
        self.estimate: float | None = None
        self.uncertainty = self.initial_uncertainty

    def reset(self) -> None:
        self.estimate = None
        self.uncertainty = self.initial_uncertainty

    def update(self, measurement: float) -> float:
        measurement = clamp(float(measurement), -1.0, 1.0)
        if self.estimate is None:
            self.estimate = measurement
            return measurement

        self.uncertainty += self.process_noise
        gain = self.uncertainty / (self.uncertainty + self.measurement_noise)
        self.estimate += gain * (measurement - self.estimate)
        self.uncertainty *= 1.0 - gain
        return clamp(self.estimate, -1.0, 1.0)


class SteeringPidController:
    """PID wrapper that treats model steering as target and UDP steer as feedback."""

    def __init__(
        self,
        kp: float = 0.45,
        ki: float = 0.0,
        kd: float = 0.04,
        correction_limit: float = 0.35,
        correction_rate_limit: float = 2.0,
        integral_limit: float = 0.5,
        use_kalman: bool = False,
        kalman_process_noise: float = 0.002,
        kalman_measurement_noise: float = 0.03,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.correction_limit = max(0.0, float(correction_limit))
        self.correction_rate_limit = max(0.0, float(correction_rate_limit))
        self.integral_limit = max(0.0, float(integral_limit))
        self.kalman = (
            OneDimensionalKalmanFilter(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise,
            )
            if use_kalman
            else None
        )
        self._integral = 0.0
        self._previous_error: float | None = None
        self._correction = 0.0
        self._last_update_at: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None
        self._correction = 0.0
        self._last_update_at = None
        if self.kalman is not None:
            self.kalman.reset()

    def update(
        self,
        target: float,
        measurement: float | None,
    ) -> SteeringControlOutput:
        target = clamp(float(target), -1.0, 1.0)
        if measurement is None:
            self.reset()
            return SteeringControlOutput(
                target=target,
                measurement=None,
                filtered_measurement=None,
                error=None,
                correction=0.0,
                command=target,
            )

        measurement = clamp(float(measurement), -1.0, 1.0)
        filtered_measurement = (
            self.kalman.update(measurement) if self.kalman is not None else measurement
        )
        error = target - filtered_measurement
        dt = self._elapsed_s()

        self._integral = clamp(
            self._integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        self._previous_error = error

        correction = self.kp * error + self.ki * self._integral + self.kd * derivative
        correction = clamp(correction, -self.correction_limit, self.correction_limit)
        if self.correction_rate_limit > 0.0:
            max_delta = self.correction_rate_limit * dt
            correction = clamp(
                correction,
                self._correction - max_delta,
                self._correction + max_delta,
            )
        self._correction = correction
        command = clamp(target + correction, -1.0, 1.0)

        return SteeringControlOutput(
            target=target,
            measurement=measurement,
            filtered_measurement=filtered_measurement,
            error=error,
            correction=correction,
            command=command,
        )

    def _elapsed_s(self) -> float:
        now = time.monotonic()
        if self._last_update_at is None:
            dt = 1.0 / 60.0
        else:
            dt = now - self._last_update_at
        self._last_update_at = now
        return clamp(dt, 1.0 / 120.0, 0.2)
