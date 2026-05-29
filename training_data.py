from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image


DEFAULT_DATA_ROOTS = (
    Path("20260520_165701"),
    Path("20260523_200621"),
    Path("20260524_162634"),
    Path("sessions"),
)
DEFAULT_ACTION_COLUMNS = ("Steer", "Accel", "Brake")


def _as_roots(data_roots: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(data_roots, (str, Path)):
        return [Path(data_roots)]
    return [Path(root) for root in data_roots]


def _session_id(session_dir: Path) -> str:
    try:
        return session_dir.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return session_dir.as_posix()


def discover_session_dirs(data_roots: str | Path | Iterable[str | Path]) -> list[Path]:
    session_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in _as_roots(data_roots):
        if not root.exists():
            raise FileNotFoundError(root)

        candidates = [root]
        if root.is_dir():
            candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))

        for session_dir in candidates:
            resolved = session_dir.resolve()
            if resolved in seen:
                continue
            if (session_dir / "dataset.csv").exists():
                session_dirs.append(session_dir)
                seen.add(resolved)
    return session_dirs


def _required_columns(action_columns: Sequence[str], telemetry_columns: Sequence[str], extra: set[str]) -> set[str]:
    return set(action_columns) | set(telemetry_columns) | extra


def _load_image_session(
    session_dir: Path,
    action_columns: Sequence[str],
    telemetry_columns: Sequence[str],
) -> pd.DataFrame:
    csv_path = session_dir / "dataset.csv"
    table = pd.read_csv(csv_path)
    required = _required_columns(action_columns, telemetry_columns, {"frame_id", "image_path", "is_valid", "Speed"})
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"{csv_path} missing columns: {missing}")

    table["sample_source"] = "image"
    table["session_id"] = _session_id(session_dir)
    table["session_dir"] = str(session_dir)
    table["image_file"] = table["image_path"].map(lambda value: str(session_dir / str(value)))
    return table


def apply_steer_filter(dataset: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("STEER_FILTER_ALPHA must be in (0, 1]")

    filtered = dataset.copy()
    filtered["Steer_raw"] = filtered["Steer"]
    filtered["Steer"] = (
        filtered.groupby("session_id", group_keys=False)["Steer"]
        .transform(lambda values: values.ewm(alpha=alpha, adjust=False).mean())
        .clip(-127.0, 127.0)
    )

    steer_delta = (filtered["Steer"] - filtered["Steer_raw"]).abs()
    print(
        f"Steer EMA filter: alpha={alpha:.2f}, "
        f"mean |delta|={steer_delta.mean():.3f}, "
        f"raw std={filtered['Steer_raw'].std(ddof=0):.3f}, "
        f"filtered std={filtered['Steer'].std(ddof=0):.3f}"
    )
    return filtered


def load_dataset_table(
    data_roots: str | Path | Iterable[str | Path] = DEFAULT_DATA_ROOTS,
    action_columns: Sequence[str] = DEFAULT_ACTION_COLUMNS,
    telemetry_columns: Sequence[str] = (),
    steer_filter_enabled: bool = True,
    steer_filter_alpha: float = 1.0,
) -> pd.DataFrame:
    session_dirs = discover_session_dirs(data_roots)
    if not session_dirs:
        roots = ", ".join(str(root) for root in _as_roots(data_roots))
        raise FileNotFoundError(f"No dataset.csv image sessions found under: {roots}")

    tables = [_load_image_session(session_dir, action_columns, telemetry_columns) for session_dir in session_dirs]
    dataset = pd.concat(tables, ignore_index=True, sort=False)
    rows_before_filter = len(dataset)

    numeric_columns = list(dict.fromkeys(["is_valid", "Speed", "frame_id", *action_columns, *telemetry_columns]))
    for column in numeric_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.dropna(subset=numeric_columns)
    dataset = dataset[(dataset["is_valid"] == 1) & (dataset["Speed"] > 0)].copy()
    if dataset.empty:
        raise ValueError("Dataset is empty after filtering is_valid == 1 and Speed > 0")

    missing_images = [path for path in dataset["image_file"] if not Path(path).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing image files, first few: {missing_images[:5]}")

    dataset = dataset.reset_index(drop=True)
    if steer_filter_enabled:
        dataset = apply_steer_filter(dataset, alpha=steer_filter_alpha)
    else:
        dataset["Steer_raw"] = dataset["Steer"]

    print(f"Loaded {len(tables)} dataset.csv image sessions")
    print(f"Rows before filter: {rows_before_filter}")
    print(f"Rows after Speed > 0 and is_valid == 1: {len(dataset)}")
    print("Rows by session:")
    print(dataset.groupby("session_id").size().rename("rows").to_string())
    return dataset


def _crop_image(image: Image.Image, crop_box: tuple[int, int, int, int] | None) -> Image.Image:
    if crop_box is None:
        return image
    left, top, right, bottom = crop_box
    width, height = image.size
    left = max(0, min(int(left), width))
    right = max(0, min(int(right), width))
    top = max(0, min(int(top), height))
    bottom = max(0, min(int(bottom), height))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box: {(left, top, right, bottom)} for image {width}x{height}")
    return image.crop((left, top, right, bottom))


def load_row_image(row: pd.Series, crop_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    with Image.open(row["image_file"]) as image_file:
        image = image_file.convert("RGB")
    return _crop_image(image, crop_box)
