"""Model architecture matching the trained Forza controller checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


FEATURE_FLAT_DIM = 576 * 4 * 12
MODEL_STEER_RANGE = 127.0


class ForzaModel(nn.Module):
    def __init__(self, telemetry_dim, dropout=0.4):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        self.features = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 12))

        # mobilenet_v3_small 最後 feature channels 通常是 576
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576 * 4 * 12 + telemetry_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

    def forward(self, video, telemetry):
        x = self.features(video)
        x = self.spatial_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, telemetry], dim=1)
        steer = torch.tanh(self.head(x)) * 127.0

        accel = torch.full_like(steer, 0.5)
        brake = torch.full_like(steer, 0.0)
        return torch.cat([steer, accel, brake], dim=1)


def _load_state_dict(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mobilenet_v3_small_feature_dim() -> int:
    return FEATURE_FLAT_DIM


def checkpoint_telemetry_dim(path: str | Path) -> int:
    state_dict = _load_state_dict(Path(path))
    weight = state_dict.get("head.1.weight")
    if weight is None:
        raise KeyError("checkpoint is missing head.1.weight")

    feature_dim = _mobilenet_v3_small_feature_dim()
    telemetry_dim = int(weight.shape[1]) - feature_dim
    if telemetry_dim < 0:
        raise ValueError(
            f"checkpoint head.1.weight has input dim {weight.shape[1]}, "
            f"which is smaller than the feature dim {feature_dim}"
        )
    return telemetry_dim


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ForzaModel:
    checkpoint_path = Path(path)
    telemetry_dim = checkpoint_telemetry_dim(checkpoint_path)
    model = ForzaModel(telemetry_dim=telemetry_dim)
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
