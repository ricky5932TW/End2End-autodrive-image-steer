"""Forza Data Out UDP listener and 324-byte Dash packet parser."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from .config import TELEMETRY_COLUMNS

DASH_PACKET_SIZE = 324


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u8(data: bytes, offset: int) -> int:
    return struct.unpack_from("<B", data, offset)[0]


def _s8(data: bytes, offset: int) -> int:
    return struct.unpack_from("<b", data, offset)[0]


def parse_dash_packet(data: bytes) -> dict[str, float | int]:
    """Parse the 324-byte Forza Dash packet observed in the training data."""

    if len(data) != DASH_PACKET_SIZE:
        raise ValueError(f"expected {DASH_PACKET_SIZE} bytes, got {len(data)}")

    return {
        #"IsRaceOn": _i32(data, 0),
        #"TimestampMS": _u32(data, 4),
        #"CurrentEngineRpm": _f32(data, 16),
        #"AccelerationX": _f32(data, 20),
        #"AccelerationY": _f32(data, 24),
        #"AccelerationZ": _f32(data, 28),
        #"VelocityX": _f32(data, 32),
        #"VelocityY": _f32(data, 36),
        #"VelocityZ": _f32(data, 40),
        ##"AngularVelocityY": _f32(data, 48),
        #"AngularVelocityZ": _f32(data, 52),
        #"Yaw": _f32(data, 56),
        #"Pitch": _f32(data, 60),
        #"Roll": _f32(data, 64),
        #"NormalizedSuspensionTravelFrontLeft": _f32(data, 68),
        #"NormalizedSuspensionTravelFrontRight": _f32(data, 72),
        #"NormalizedSuspensionTravelRearLeft": _f32(data, 76),
        #"NormalizedSuspensionTravelRearRight": _f32(data, 80),
        #"TireSlipRatioFrontLeft": _f32(data, 84),
        #"TireSlipRatioFrontRight": _f32(data, 88),
        #"TireSlipRatioRearLeft": _f32(data, 92),
        #"TireSlipRatioRearRight": _f32(data, 96),
        #"WheelRotationSpeedFrontLeft": _f32(data, 100),
        #"WheelRotationSpeedFrontRight": _f32(data, 104),
        #"WheelRotationSpeedRearLeft": _f32(data, 108),
        #"WheelRotationSpeedRearRight": _f32(data, 112),
        "Speed": _f32(data, 256),
        #"Power": _f32(data, 260),
        #"Torque": _f32(data, 264),
        "Gear": _u8(data, 319),
        "Steer": _s8(data, 320),
    }


def telemetry_vector(
    row: dict[str, Any],
    columns: tuple[str, ...] = TELEMETRY_COLUMNS,
) -> np.ndarray:
    return np.array([float(row.get(column, 0.0) or 0.0) for column in columns], dtype=np.float32)


@dataclass(frozen=True)
class TelemetrySnapshot:
    row: dict[str, float | int]
    received_at: float
    packet_count: int

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.received_at


class TelemetryReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        socket_timeout_s: float = 0.2,
    ) -> None:
        self.host = host
        self.port = port
        self.socket_timeout_s = socket_timeout_s
        self.packet_count = 0
        self.parse_errors = 0
        self._latest: TelemetrySnapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="forza-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self, max_age_s: float | None = None) -> TelemetrySnapshot | None:
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return None
        if max_age_s is not None and snapshot.age_s > max_age_s:
            return None
        return snapshot

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.socket_timeout_s)
        sock.bind((self.host, self.port))

        while not self._stop.is_set():
            try:
                data, _addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                row = parse_dash_packet(data)
            except ValueError:
                self.parse_errors += 1
                continue

            self.packet_count += 1
            snapshot = TelemetrySnapshot(
                row=row,
                received_at=time.monotonic(),
                packet_count=self.packet_count,
            )
            with self._lock:
                self._latest = snapshot
