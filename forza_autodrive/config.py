"""Shared constants for Forza capture, telemetry, and model IO."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_PATH = Path(
    r"best_model.pth"
)

ACTION_COLUMNS = ("Steer", "Accel", "Brake")

TELEMETRY_COLUMNS = (
    "TimestampMS",
    "Speed",
    "CurrentEngineRpm",
    "Gear",
    "IsRaceOn",
    "AccelerationX",
    "AccelerationY",
    "AccelerationZ",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "AngularVelocityX",
    "AngularVelocityY",
    "AngularVelocityZ",
    "Yaw",
    "Pitch",
    "Roll",
    "NormalizedSuspensionTravelFrontLeft",
    "NormalizedSuspensionTravelFrontRight",
    "NormalizedSuspensionTravelRearLeft",
    "NormalizedSuspensionTravelRearRight",
    "TireSlipRatioFrontLeft",
    "TireSlipRatioFrontRight",
    "TireSlipRatioRearLeft",
    "TireSlipRatioRearRight",
    "WheelRotationSpeedFrontLeft",
    "WheelRotationSpeedFrontRight",
    "WheelRotationSpeedRearLeft",
    "WheelRotationSpeedRearRight",
    "Power",
    "Torque",
)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
REFERENCE_IMAGE_HEIGHT = 1080
TRAIN_CROP_TOP = 450
TRAIN_CROP_BOTTOM = 730


def crop_box_for_size(width: int, height: int) -> tuple[int, int, int, int]:
    top = int(round(height * TRAIN_CROP_TOP / REFERENCE_IMAGE_HEIGHT))
    bottom = int(round(height * TRAIN_CROP_BOTTOM / REFERENCE_IMAGE_HEIGHT))
    return 0, top, width, bottom


CROP_BOX = crop_box_for_size(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
