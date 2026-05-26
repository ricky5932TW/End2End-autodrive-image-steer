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
MAX_SVG_BYTES = 250 * 1024


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


def write_svg(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"svg {path.name}: {path.stat().st_size / 1024:.0f} KiB")


def export_training_diagrams(out_dir: Path) -> None:
    training_pipeline_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="360" viewBox="0 0 1120 360" role="img" aria-labelledby="title desc">
  <title id="title">Training pipeline for the Forza image-to-steering model</title>
  <desc id="desc">A deterministic diagram showing data collection, filtering, crop, augmentation, balanced sampling, MobileNetV3-Small, and steering loss.</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="#4b5563"/>
    </marker>
    <style>
      .bg { fill: #f8fafc; }
      .title { font: 700 25px Arial, sans-serif; fill: #111827; }
      .subtitle { font: 15px Arial, sans-serif; fill: #4b5563; }
      .box { fill: #ffffff; stroke: #cbd5e1; stroke-width: 2; rx: 8; }
      .box-accent { fill: #ecfeff; stroke: #0891b2; stroke-width: 2; rx: 8; }
      .box-warm { fill: #fff7ed; stroke: #ea580c; stroke-width: 2; rx: 8; }
      .label { font: 700 15px Arial, sans-serif; fill: #111827; }
      .small { font: 13px Arial, sans-serif; fill: #475569; }
      .metric { font: 700 13px Arial, sans-serif; fill: #0f766e; }
      .arrow { stroke: #4b5563; stroke-width: 2; fill: none; marker-end: url(#arrow); }
    </style>
  </defs>
  <rect class="bg" x="0" y="0" width="1120" height="360" rx="14"/>
  <text class="title" x="40" y="48">Training Pipeline</text>
  <text class="subtitle" x="40" y="74">Forza gameplay turns into a steering-supervised visual controller, with augmentation and resampling to reduce straight-driving bias.</text>

  <rect class="box" x="40" y="118" width="135" height="116"/>
  <text class="label" x="60" y="148">Collect</text>
  <text class="small" x="60" y="174">8 sessions</text>
  <text class="small" x="60" y="196">screen + UDP</text>
  <text class="metric" x="60" y="218">96,182 rows</text>

  <path class="arrow" d="M180 176 H218"/>
  <rect class="box" x="225" y="118" width="135" height="116"/>
  <text class="label" x="245" y="148">Filter</text>
  <text class="small" x="245" y="174">Speed &gt; 0</text>
  <text class="small" x="245" y="196">valid rows only</text>
  <text class="metric" x="245" y="218">72,942 rows</text>

  <path class="arrow" d="M365 176 H403"/>
  <rect class="box" x="410" y="118" width="135" height="116"/>
  <text class="label" x="430" y="148">Preprocess</text>
  <text class="small" x="430" y="174">EMA alpha 0.70</text>
  <text class="small" x="430" y="196">crop road band</text>
  <text class="metric" x="430" y="218">(0,75,320,122)</text>

  <path class="arrow" d="M550 176 H588"/>
  <rect class="box-accent" x="595" y="118" width="135" height="116"/>
  <text class="label" x="615" y="148">Augment</text>
  <text class="small" x="615" y="174">flip + steer sign</text>
  <text class="small" x="615" y="196">color, affine</text>
  <text class="small" x="615" y="218">noise + erasing</text>

  <path class="arrow" d="M735 176 H773"/>
  <rect class="box-warm" x="780" y="118" width="135" height="116"/>
  <text class="label" x="800" y="148">Resample</text>
  <text class="small" x="800" y="174">&gt; 3 std actions</text>
  <text class="small" x="800" y="196">512 / 1024 batch</text>
  <text class="metric" x="800" y="218">rare rows boosted</text>

  <path class="arrow" d="M920 176 H958"/>
  <rect class="box" x="965" y="118" width="115" height="116"/>
  <text class="label" x="985" y="148">Train</text>
  <text class="small" x="985" y="174">MobileNetV3</text>
  <text class="small" x="985" y="196">AdamW + cosine</text>
  <text class="metric" x="985" y="218">steering loss</text>

  <rect class="box" x="225" y="270" width="690" height="48"/>
  <text class="small" x="248" y="300">90/10 train-validation split, ImageNet normalization, early stopping, best checkpoint saved as best_model.pth.</text>
</svg>"""

    resampling_balance_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360" role="img" aria-labelledby="title desc">
  <title id="title">Resampling reduces straight-driving bias</title>
  <desc id="desc">A bar chart comparing rare high-steering action rows in the raw train split with their share in a balanced training batch.</desc>
  <defs>
    <style>
      .bg { fill: #f8fafc; }
      .title { font: 700 25px Arial, sans-serif; fill: #111827; }
      .subtitle { font: 15px Arial, sans-serif; fill: #4b5563; }
      .axis { stroke: #94a3b8; stroke-width: 2; }
      .label { font: 700 16px Arial, sans-serif; fill: #111827; }
      .small { font: 13px Arial, sans-serif; fill: #475569; }
      .pct { font: 700 22px Arial, sans-serif; fill: #0f766e; }
      .common { fill: #cbd5e1; }
      .rare { fill: #f97316; }
      .batch { fill: #06b6d4; }
      .legend { font: 13px Arial, sans-serif; fill: #334155; }
    </style>
  </defs>
  <rect class="bg" x="0" y="0" width="960" height="360" rx="14"/>
  <text class="title" x="44" y="48">Resampling Strategy</text>
  <text class="subtitle" x="44" y="74">Most driving frames are straight or low-steering. Each training batch deliberately contains many high-action examples.</text>

  <line class="axis" x1="110" y1="282" x2="850" y2="282"/>
  <line class="axis" x1="110" y1="110" x2="110" y2="282"/>

  <text class="label" x="168" y="312">Raw train split</text>
  <text class="small" x="168" y="332">5,060 / 65,648 rows over 3 std</text>
  <rect class="common" x="190" y="128" width="170" height="154" rx="8"/>
  <rect class="rare" x="190" y="269" width="170" height="13" rx="4"/>
  <text class="pct" x="242" y="116">7.7%</text>
  <text class="small" x="210" y="158">common steering</text>
  <text class="small" x="215" y="263">rare action rows</text>

  <text class="label" x="568" y="312">Training batch</text>
  <text class="small" x="568" y="332">512 / 1024 rows sampled from outliers</text>
  <rect class="common" x="590" y="128" width="170" height="154" rx="8"/>
  <rect class="batch" x="590" y="205" width="170" height="77" rx="8"/>
  <text class="pct" x="640" y="116">50%</text>
  <text class="small" x="608" y="158">regular rows</text>
  <text class="small" x="612" y="250">rare-action rows</text>

  <rect class="rare" x="112" y="42" width="14" height="14" rx="3"/>
  <text class="legend" x="134" y="54">Raw rare-action share</text>
  <rect class="batch" x="290" y="42" width="14" height="14" rx="3"/>
  <text class="legend" x="312" y="54">Batch rare-action share</text>
</svg>"""

    write_svg(out_dir / "training-pipeline.svg", training_pipeline_svg)
    write_svg(out_dir / "resampling-balance.svg", resampling_balance_svg)


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
        if path.suffix.lower() == ".svg" and size > MAX_SVG_BYTES:
            raise RuntimeError(f"{path} is larger than the SVG budget")


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
    export_training_diagrams(args.out_dir)

    assert_asset_sizes(args.out_dir.rglob("*"))
    print(f"done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
