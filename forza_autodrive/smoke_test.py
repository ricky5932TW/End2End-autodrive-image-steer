"""Smoke tests for imports, preprocessing, model inference, and vgamepad."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "forza_autodrive"

from .config import CROP_BOX, DEFAULT_MODEL_PATH, TELEMETRY_COLUMNS
from .controller import GamepadController
from .model import checkpoint_telemetry_dim, load_model
from .preprocess import preprocess_image_file
from .telemetry import telemetry_vector

DEFAULT_SAMPLE_IMAGE = Path(
    r"D:\pyprojects\End-to-End-Learning-for-Self-Driving-Cars-in-Forza\sessions\20260523_004430\images\000100.jpg"
)
DEFAULT_SAMPLE_DATASET = Path(
    r"D:\pyprojects\End-to-End-Learning-for-Self-Driving-Cars-in-Forza\sessions\20260523_004430\dataset.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=Path, default=DEFAULT_SAMPLE_IMAGE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_SAMPLE_DATASET)
    parser.add_argument("--skip-vgamepad", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_first_valid_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("is_valid") == "1" and float(row.get("Speed", 0.0) or 0.0) > 0:
                return row
    raise ValueError(f"no valid moving row found in {path}")


def main() -> int:
    args = build_parser().parse_args()

    import cv2
    import dxcam
    import PIL
    import torchvision
    import vgamepad

    print("imports ok")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} cuda_version={torch.version.cuda}")
    print(f"torchvision={torchvision.__version__}")
    print(f"PIL={PIL.__version__} cv2={cv2.__version__} dxcam={dxcam.__file__}")
    print(f"vgamepad={vgamepad.__file__}")

    tensor = preprocess_image_file(args.image).unsqueeze(0)
    print(f"preprocess shape={tuple(tensor.shape)}")
    expected_shape = (1, 3, CROP_BOX[3] - CROP_BOX[1], CROP_BOX[2] - CROP_BOX[0])
    if tuple(tensor.shape) != expected_shape:
        raise AssertionError(f"unexpected preprocess shape: {tuple(tensor.shape)}")

    checkpoint_dim = checkpoint_telemetry_dim(args.model)
    print(f"checkpoint telemetry_dim={checkpoint_dim}")
    if checkpoint_dim not in (0, len(TELEMETRY_COLUMNS)):
        raise AssertionError(
            f"checkpoint expects {checkpoint_dim}, runtime has {len(TELEMETRY_COLUMNS)}"
        )

    if checkpoint_dim == 0:
        telemetry = torch.empty((1, 0), dtype=torch.float32)
    else:
        row = load_first_valid_row(args.dataset)
        telemetry = torch.from_numpy(telemetry_vector(row)).unsqueeze(0)
    model = load_model(args.model, device=args.device)
    with torch.inference_mode():
        output = model(tensor.to(args.device), telemetry.to(args.device)).cpu()
    print(f"model output shape={tuple(output.shape)} values={output.squeeze(0).tolist()}")
    if tuple(output.shape) != (1, 3):
        raise AssertionError(f"unexpected model output shape: {tuple(output.shape)}")

    if args.skip_vgamepad:
        print("vgamepad smoke skipped")
    else:
        controller = GamepadController()
        controller.reset()
        print("vgamepad smoke ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
