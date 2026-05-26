"""Export small README media from local recordings and notebook outputs."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

from PIL import Image
import imageio_ffmpeg


MAX_VIDEO_BYTES = 10 * 1024 * 1024
MAX_POSTER_BYTES = 500 * 1024
MAX_ASSET_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class ClipSpec:
    name: str
    source: str
    start: str
    duration_s: float
    poster_at_s: float


DEFAULT_CLIPS = (
    ClipSpec(
        name="live-debug",
        source="Forza debug 2026-05-26 11-34-01.mp4",
        start="00:00:05",
        duration_s=10.0,
        poster_at_s=3.0,
    ),
    ClipSpec(
        name="long-run",
        source="ScreenRecording_05-25-2026 23-07-27_1.MP4",
        start="00:00:20",
        duration_s=12.0,
        poster_at_s=4.0,
    ),
    ClipSpec(
        name="short-control",
        source="ScreenRecording_05-25-2026 22-39-39_1.MP4",
        start="00:00:01",
        duration_s=8.0,
        poster_at_s=2.0,
    ),
)


def parse_time(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def run_ffmpeg(args: list[str], *, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        if not quiet:
            print(result.stdout)
            print(result.stderr)
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}")
    return result


def probe_duration_s(ffmpeg: str, path: Path) -> float | None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def scale_filter(max_width: int) -> str:
    return f"fps=24,scale='min({max_width},iw)':-2"


def encode_clip(
    ffmpeg: str,
    source: Path,
    output: Path,
    start_s: float,
    duration_s: float,
) -> None:
    attempts = (
        {"width": 1280, "crf": 28},
        {"width": 960, "crf": 30},
        {"width": 720, "crf": 32},
        {"width": 640, "crf": 34},
    )
    for attempt in attempts:
        output.unlink(missing_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration_s:.3f}",
            "-an",
            "-vf",
            scale_filter(attempt["width"]),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(attempt["crf"]),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        run_ffmpeg(cmd)
        size = output.stat().st_size
        print(
            f"video {output.name}: {size / 1024 / 1024:.2f} MiB "
            f"(width<={attempt['width']}, crf={attempt['crf']})"
        )
        if size <= MAX_VIDEO_BYTES:
            return
    raise RuntimeError(f"{output} is still larger than {MAX_VIDEO_BYTES} bytes")


def export_poster(
    ffmpeg: str,
    source: Path,
    output: Path,
    at_s: float,
    max_width: int = 1280,
) -> None:
    tmp = output.with_suffix(".tmp.jpg")
    tmp.unlink(missing_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-ss",
        f"{at_s:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({max_width},iw)':-2",
        "-q:v",
        "4",
        str(tmp),
    ]
    run_ffmpeg(cmd)
    compress_jpeg(tmp, output, MAX_POSTER_BYTES)
    tmp.unlink(missing_ok=True)
    print(f"poster {output.name}: {output.stat().st_size / 1024:.0f} KiB")


def compress_jpeg(input_path: Path, output_path: Path, max_bytes: int) -> None:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        for quality in (88, 82, 76, 70, 64, 58, 52, 46):
            output_path.unlink(missing_ok=True)
            image.save(output_path, format="JPEG", quality=quality, optimize=True)
            if output_path.stat().st_size <= max_bytes:
                return
        width, height = image.size
        scale = min(1.0, (max_bytes / max(output_path.stat().st_size, 1)) ** 0.5)
        resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        resized.save(output_path, format="JPEG", quality=58, optimize=True)


def export_clips(source_dir: Path, out_dir: Path, tail_trim_s: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for spec in DEFAULT_CLIPS:
        source = source_dir / spec.source
        if not source.exists():
            raise FileNotFoundError(source)

        start_s = parse_time(spec.start)
        duration_s = max(1.0, spec.duration_s - tail_trim_s)
        source_duration = probe_duration_s(ffmpeg, source)
        if source_duration is not None:
            available_s = source_duration - start_s - tail_trim_s
            if available_s <= 0:
                raise ValueError(
                    f"{spec.source} does not have enough duration before the final tail trim"
                )
            duration_s = min(duration_s, available_s)
        if duration_s < 1.0:
            raise ValueError(f"{spec.source} does not have enough duration after tail trim")

        clip_path = out_dir / f"{spec.name}.mp4"
        poster_path = out_dir / f"{spec.name}-poster.jpg"
        poster_s = start_s + min(spec.poster_at_s, max(0.0, duration_s - 0.25))
        print(
            f"exporting {spec.name}: start={start_s:.2f}s duration={duration_s:.2f}s "
            f"poster={poster_s:.2f}s"
        )
        encode_clip(ffmpeg, source, clip_path, start_s, duration_s)
        export_poster(ffmpeg, source, poster_path, poster_s)


def notebook_pngs(notebook_path: Path) -> dict[int, bytes]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    outputs: dict[int, bytes] = {}
    for cell_index in (14, 15):
        cell = notebook["cells"][cell_index]
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if not png:
                continue
            if isinstance(png, list):
                png = "".join(png)
            outputs[cell_index] = base64.b64decode(png)
    missing = {14, 15} - set(outputs)
    if missing:
        raise ValueError(f"missing notebook image/png outputs for cells {sorted(missing)}")
    return outputs


def content_bands(image: Image.Image) -> list[tuple[int, int]]:
    gray = image.convert("L")
    mask = gray.point(lambda pixel: 255 if pixel < 248 else 0)
    width, height = mask.size
    rows: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y in range(height):
        has_content = mask.crop((0, y, width, y + 1)).getbbox() is not None
        if has_content and not in_band:
            in_band = True
            start = y
        elif not has_content and in_band:
            rows.append((start, y))
            in_band = False
    if in_band:
        rows.append((start, height))

    merged: list[tuple[int, int]] = []
    for start, end in rows:
        if not merged or start - merged[-1][1] > 28:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], end)
    return [(max(0, start - 8), min(height, end + 8)) for start, end in merged if end - start > 20]


def pick_evenly(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[index] for index in indexes]


def export_gradcam_montage(
    png_bytes: bytes,
    output_path: Path,
    *,
    sample_count: int = 5,
    max_width: int = 1120,
) -> None:
    source_path = output_path.with_suffix(".source.png")
    source_path.write_bytes(png_bytes)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        bands = content_bands(image)
        if not bands:
            raise ValueError(f"could not detect content bands in {output_path.name}")
        selected = pick_evenly(bands, sample_count)
        crops = [image.crop((0, start, image.width, end)) for start, end in selected]

    gap = 16
    total_height = sum(crop.height for crop in crops) + gap * (len(crops) - 1)
    montage = Image.new("RGB", (max(crop.width for crop in crops), total_height), "white")
    y = 0
    for crop in crops:
        montage.paste(crop, (0, y))
        y += crop.height + gap

    if montage.width > max_width:
        height = round(montage.height * max_width / montage.width)
        montage = montage.resize((max_width, height), Image.Resampling.LANCZOS)
    tmp = output_path.with_suffix(".tmp.jpg")
    montage.save(tmp, format="JPEG", quality=90, optimize=True)
    compress_jpeg(tmp, output_path, max_bytes=900 * 1024)
    tmp.unlink(missing_ok=True)
    source_path.unlink(missing_ok=True)
    print(f"gradcam {output_path.name}: {output_path.stat().st_size / 1024:.0f} KiB")


def export_gradcam_assets(notebook_path: Path, out_dir: Path) -> None:
    pngs = notebook_pngs(notebook_path)
    export_gradcam_montage(pngs[14], out_dir / "scorecam-final-layer.jpg")
    export_gradcam_montage(pngs[15], out_dir / "scorecam-shallow-layer.jpg")


def assert_asset_sizes(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise RuntimeError(f"{path} is larger than GitHub's 100 MiB file limit")
        if path.suffix.lower() == ".mp4" and size > MAX_VIDEO_BYTES:
            raise RuntimeError(f"{path} is larger than the README video budget")
        if path.name.endswith("-poster.jpg") and size > MAX_POSTER_BYTES:
            raise RuntimeError(f"{path} is larger than the poster budget")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("media"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--notebook", type=Path, default=Path("load_data.ipynb"))
    parser.add_argument(
        "--tail-trim-s",
        type=float,
        default=1.0,
        help="seconds trimmed from each requested segment and from the source tail",
    )
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_video:
        export_clips(args.source_dir, args.out_dir, tail_trim_s=max(1.0, args.tail_trim_s))
    if not args.skip_gradcam:
        export_gradcam_assets(args.notebook, args.out_dir)

    assert_asset_sizes(args.out_dir.rglob("*"))
    print(f"done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
