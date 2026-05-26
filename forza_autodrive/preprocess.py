"""Image capture and preprocessing matching the training notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageGrab
import torch

from .config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGENET_MEAN, IMAGENET_STD, crop_box_for_size


def image_size(image: Image.Image | np.ndarray) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return width, height
    return image.size


def to_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def resize_to_training_frame(image: Image.Image | np.ndarray) -> Image.Image:
    image = to_rgb_image(image)
    if image.size == (IMAGE_WIDTH, IMAGE_HEIGHT):
        return image

    try:
        import cv2

        source_width, source_height = image.size
        interpolation = (
            cv2.INTER_AREA
            if source_width >= IMAGE_WIDTH and source_height >= IMAGE_HEIGHT
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            np.asarray(image),
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=interpolation,
        )
        return Image.fromarray(resized)
    except ImportError:
        resampling = (
            Image.Resampling.BOX
            if image.width >= IMAGE_WIDTH and image.height >= IMAGE_HEIGHT
            else Image.Resampling.BILINEAR
        )
        return image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), resampling)


def model_input_image(image: Image.Image | np.ndarray) -> Image.Image:
    image = resize_to_training_frame(image)
    image = image.crop(crop_box_for_size(IMAGE_WIDTH, IMAGE_HEIGHT))
    return image


def preprocess_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    image = model_input_image(image)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image_file(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_image(image)


def save_preprocess_debug(
    image: Image.Image | np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = to_rgb_image(image)
    resized = resize_to_training_frame(raw)
    model_input = model_input_image(raw)

    raw_path = output_dir / f"{prefix}_raw.jpg"
    resized_path = output_dir / f"{prefix}_resized.jpg"
    model_path = output_dir / f"{prefix}_model_input.jpg"

    raw.save(raw_path, quality=90)
    resized.save(resized_path, quality=95)
    model_input.save(model_path, quality=95)

    return {
        "raw_path": raw_path,
        "resized_path": resized_path,
        "model_path": model_path,
        "raw_size": raw.size,
        "resized_size": resized.size,
        "model_size": model_input.size,
    }


class FrameGrabber:
    def __init__(
        self,
        monitor: int = 0,
        region: tuple[int, int, int, int] | None = None,
        backend: str = "auto",
    ) -> None:
        self.requested_backend = backend.lower()
        self.backend = self.requested_backend
        self.region = region
        self.camera = None
        self.fallback_reason: str | None = None
        self.last_frame_size: tuple[int, int] | None = None

        if self.requested_backend not in {"auto", "dxcam", "imagegrab"}:
            raise ValueError(
                f"unknown capture backend {backend!r}; use auto, dxcam, or imagegrab"
            )

        if self.requested_backend in {"auto", "dxcam"}:
            try:
                import dxcam

                self.camera = dxcam.create(output_idx=monitor, output_color="RGB")
                if self.camera is None:
                    raise RuntimeError(f"could not create dxcam camera for monitor {monitor}")
                self.backend = "dxcam"
                return
            except Exception as exc:
                if self.requested_backend == "dxcam":
                    raise RuntimeError(
                        "dxcam capture initialization failed. Try "
                        "--capture-backend imagegrab, or run the game in borderless/windowed mode."
                    ) from exc
                self.fallback_reason = f"dxcam unavailable: {exc}"

        self.backend = "imagegrab"

    def grab_frame(self) -> Image.Image | np.ndarray | None:
        if self.backend == "dxcam":
            try:
                frame = self.camera.grab(region=self.region)
            except Exception as exc:
                if self.requested_backend == "dxcam":
                    raise
                self.fallback_reason = f"dxcam grab failed: {exc}"
                self.backend = "imagegrab"
                frame = None
            if frame is None and self.backend == "dxcam":
                return None
        if self.backend == "imagegrab":
            frame = ImageGrab.grab(bbox=self.region, all_screens=self.region is not None)
        if frame is None:
            return None
        self.last_frame_size = image_size(frame)
        return frame

    def grab_tensor(self) -> torch.Tensor | None:
        frame = self.grab_frame()
        if frame is None:
            return None
        return preprocess_image(frame).unsqueeze(0)
