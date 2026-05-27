"""Export small README media from local recordings and notebook outputs."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Iterable


MAX_VIDEO_BYTES = 10 * 1024 * 1024
MAX_GIF_BYTES = 10 * 1024 * 1024
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


def gif_scale_filter(max_width: int, fps: int) -> str:
    return f"fps={fps},scale='min({max_width},iw)':-2:flags=lanczos"


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


def export_gif_preview(ffmpeg: str, source: Path, output: Path) -> None:
    attempts = (
        {"width": 640, "fps": 12},
        {"width": 480, "fps": 12},
        {"width": 420, "fps": 10},
        {"width": 360, "fps": 10},
        {"width": 320, "fps": 8},
    )
    palette = output.with_name(f"{output.stem}-palette.png")
    for attempt in attempts:
        output.unlink(missing_ok=True)
        palette.unlink(missing_ok=True)
        filters = gif_scale_filter(attempt["width"], attempt["fps"])
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-i",
                str(source),
                "-vf",
                f"{filters},palettegen=stats_mode=diff",
                "-update",
                "1",
                str(palette),
            ]
        )
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-i",
                str(source),
                "-i",
                str(palette),
                "-lavfi",
                f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-loop",
                "0",
                str(output),
            ]
        )
        size = output.stat().st_size
        print(
            f"gif {output.name}: {size / 1024 / 1024:.2f} MiB "
            f"(width<={attempt['width']}, fps={attempt['fps']})"
        )
        if size <= MAX_GIF_BYTES:
            palette.unlink(missing_ok=True)
            return
    palette.unlink(missing_ok=True)
    raise RuntimeError(f"{output} is still larger than {MAX_GIF_BYTES} bytes")


def compress_jpeg(input_path: Path, output_path: Path, max_bytes: int) -> None:
    from PIL import Image

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
    import imageio_ffmpeg

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
        gif_path = out_dir / f"{spec.name}-preview.gif"
        poster_s = start_s + min(spec.poster_at_s, max(0.0, duration_s - 0.25))
        print(
            f"exporting {spec.name}: start={start_s:.2f}s duration={duration_s:.2f}s "
            f"poster={poster_s:.2f}s"
        )
        encode_clip(ffmpeg, source, clip_path, start_s, duration_s)
        export_poster(ffmpeg, source, poster_path, poster_s)
        export_gif_preview(ffmpeg, clip_path, gif_path)


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
    from PIL import Image

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
    plot_left = 110
    plot_right = 850
    plot_top = 120
    plot_bottom = 320
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    def loss_weight(abs_steer: float) -> float:
        log_value = math.log(abs_steer) if abs_steer > 0 else float("-inf")
        return 1.0 + max(log_value, 0.01)

    def curve_point(abs_steer: float) -> tuple[float, float]:
        weight = loss_weight(abs_steer)
        x = plot_left + (abs_steer / 127.0) * plot_width
        y = plot_bottom - ((weight - 1.0) / 5.0) * plot_height
        return x, y

    curve_samples = [0.0, 1.0, *[float(value) for value in range(2, 128, 2)], 127.0]
    curve_points = [curve_point(abs_steer) for abs_steer in curve_samples]
    curve_path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in curve_points)
    area_path = (
        f"M {plot_left:.1f} {plot_bottom:.1f} L "
        + " L ".join(f"{x:.1f} {y:.1f}" for x, y in curve_points)
        + f" L {plot_right:.1f} {plot_bottom:.1f} Z"
    )
    marker_points = {
        name: (*curve_point(abs_steer), loss_weight(abs_steer))
        for name, abs_steer in (("zero", 0.0), ("ten", 10.0), ("sixty_four", 64.0), ("max", 127.0))
    }

    training_pipeline_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="400" viewBox="0 0 1120 400" role="img" aria-labelledby="title desc">
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
  <rect class="bg" x="0" y="0" width="1120" height="400" rx="14"/>
  <text class="title" x="40" y="48">Training Pipeline</text>
  <text class="subtitle" x="40" y="74">Forza gameplay becomes steering-supervised training data.</text>
  <text class="subtitle" x="40" y="96">Augmentation and balanced sampling reduce straight-driving bias before MobileNetV3 learns steering.</text>

  <rect class="box" x="40" y="136" width="135" height="126"/>
  <text class="label" x="60" y="166">Collect</text>
  <text class="small" x="60" y="194">8 sessions</text>
  <text class="small" x="60" y="216">screen + UDP</text>
  <text class="metric" x="60" y="240">96,182 rows</text>

  <path class="arrow" d="M180 199 H218"/>
  <rect class="box" x="225" y="136" width="135" height="126"/>
  <text class="label" x="245" y="166">Filter</text>
  <text class="small" x="245" y="194">Speed &gt; 0</text>
  <text class="small" x="245" y="216">valid rows only</text>
  <text class="metric" x="245" y="240">72,942 rows</text>

  <path class="arrow" d="M365 199 H403"/>
  <rect class="box" x="410" y="136" width="135" height="126"/>
  <text class="label" x="430" y="166">Preprocess</text>
  <text class="small" x="430" y="194">EMA alpha 0.70</text>
  <text class="small" x="430" y="216">crop road band</text>
  <text class="metric" x="430" y="240">(0,75,320,122)</text>

  <path class="arrow" d="M550 199 H588"/>
  <rect class="box-accent" x="595" y="136" width="135" height="126"/>
  <text class="label" x="615" y="166">Augment</text>
  <text class="small" x="615" y="194">flip + steer sign</text>
  <text class="small" x="615" y="216">color, affine</text>
  <text class="small" x="615" y="240">noise + erasing</text>

  <path class="arrow" d="M735 199 H773"/>
  <rect class="box-warm" x="780" y="136" width="135" height="126"/>
  <text class="label" x="800" y="166">Resample</text>
  <text class="small" x="800" y="194">&gt; 3 std actions</text>
  <text class="small" x="800" y="216">512 / 1024 batch</text>
  <text class="metric" x="800" y="240">rare rows boosted</text>

  <path class="arrow" d="M920 199 H958"/>
  <rect class="box" x="965" y="136" width="115" height="126"/>
  <text class="label" x="985" y="166">Train</text>
  <text class="small" x="985" y="194">MobileNetV3</text>
  <text class="small" x="985" y="216">AdamW + cosine</text>
  <text class="metric" x="985" y="240">steering loss</text>

  <rect class="box" x="225" y="306" width="690" height="58"/>
  <text class="small" x="248" y="331">90/10 train-validation split, ImageNet normalization, and early stopping.</text>
  <text class="small" x="248" y="351">Best checkpoint is saved as best_model.pth.</text>
</svg>"""

    resampling_balance_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="400" viewBox="0 0 960 400" role="img" aria-labelledby="title desc">
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
  <rect class="bg" x="0" y="0" width="960" height="400" rx="14"/>
  <text class="title" x="44" y="48">Resampling Strategy</text>
  <text class="subtitle" x="44" y="74">Most driving frames are straight or low-steering.</text>
  <text class="subtitle" x="44" y="96">Balanced batches keep high-action examples visible to the optimizer.</text>

  <line class="axis" x1="150" y1="286" x2="810" y2="286"/>
  <line class="axis" x1="150" y1="128" x2="150" y2="286"/>

  <text class="pct" x="248" y="124">7.7%</text>
  <rect class="common" x="210" y="136" width="160" height="150" rx="8"/>
  <rect class="rare" x="210" y="274" width="160" height="12" rx="4"/>
  <text class="small" x="226" y="166">common steering</text>
  <text class="small" x="228" y="268">rare action rows</text>
  <text class="label" x="185" y="316">Raw train split</text>
  <text class="small" x="160" y="338">5,060 / 65,648 rows over 3 std</text>

  <text class="pct" x="646" y="124">50%</text>
  <rect class="common" x="590" y="136" width="160" height="150" rx="8"/>
  <rect class="batch" x="590" y="211" width="160" height="75" rx="8"/>
  <text class="small" x="610" y="166">regular rows</text>
  <text class="small" x="606" y="254">rare-action rows</text>
  <text class="label" x="568" y="316">Training batch</text>
  <text class="small" x="540" y="338">512 / 1024 rows sampled from outliers</text>

  <rect class="rare" x="264" y="362" width="14" height="14" rx="3"/>
  <text class="legend" x="286" y="374">Raw rare-action share</text>
  <rect class="batch" x="488" y="362" width="14" height="14" rx="3"/>
  <text class="legend" x="510" y="374">Batch rare-action share</text>
</svg>"""

    loss_weight_curve_svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-labelledby="title desc">
  <title id="title">Steering loss weight curve</title>
  <desc id="desc">A line chart showing how absolute steering target increases the weighted MSE loss multiplier from 1.01x near zero steering to 5.84x at full steering.</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="#4b5563"/>
    </marker>
    <style>
      .bg { fill: #f8fafc; }
      .title { font: 700 25px Arial, sans-serif; fill: #111827; }
      .subtitle { font: 15px Arial, sans-serif; fill: #4b5563; }
      .axis { stroke: #94a3b8; stroke-width: 2; }
      .grid { stroke: #e2e8f0; stroke-width: 1; }
      .tick { font: 12px Arial, sans-serif; fill: #64748b; }
      .label { font: 700 15px Arial, sans-serif; fill: #111827; }
      .small { font: 13px Arial, sans-serif; fill: #475569; }
      .formula { font: 700 14px Arial, sans-serif; fill: #0f766e; }
      .area { fill: #cffafe; opacity: 0.85; }
      .curve { stroke: #0891b2; stroke-width: 4; fill: none; stroke-linejoin: round; stroke-linecap: round; }
      .dot { fill: #f97316; stroke: #ffffff; stroke-width: 3; }
      .callout { fill: #ffffff; stroke: #cbd5e1; stroke-width: 2; rx: 8; }
      .arrow { stroke: #4b5563; stroke-width: 1.8; fill: none; marker-end: url(#arrow); }
    </style>
  </defs>
  <rect class="bg" x="0" y="0" width="960" height="420" rx="14"/>
  <text class="title" x="44" y="48">Steering Loss Weight</text>
  <text class="subtitle" x="44" y="74">The weighted MSE keeps straight-driving errors visible, then raises the cost of missing larger steering targets.</text>
  <text class="formula" x="44" y="98">weight = 1 + max(log(abs(target_steer)), 0.01)</text>

  <line class="grid" x1="110" y1="320" x2="850" y2="320"/>
  <line class="grid" x1="110" y1="280" x2="850" y2="280"/>
  <line class="grid" x1="110" y1="240" x2="850" y2="240"/>
  <line class="grid" x1="110" y1="200" x2="850" y2="200"/>
  <line class="grid" x1="110" y1="160" x2="850" y2="160"/>
  <line class="grid" x1="110" y1="120" x2="850" y2="120"/>
  <line class="axis" x1="110" y1="320" x2="850" y2="320"/>
  <line class="axis" x1="110" y1="120" x2="110" y2="320"/>

  <text class="tick" x="82" y="324">1x</text>
  <text class="tick" x="82" y="284">2x</text>
  <text class="tick" x="82" y="244">3x</text>
  <text class="tick" x="82" y="204">4x</text>
  <text class="tick" x="82" y="164">5x</text>
  <text class="tick" x="82" y="124">6x</text>
  <text class="tick" x="106" y="342">0</text>
  <text class="tick" x="164" y="342">10</text>
  <text class="tick" x="278" y="342">30</text>
  <text class="tick" x="476" y="342">64</text>
  <text class="tick" x="838" y="342">127</text>
  <text class="label" x="386" y="382">abs(target_steer)</text>
  <text class="label" transform="translate(40 248) rotate(-90)">loss multiplier</text>

  <path class="area" d="@@AREA_PATH@@"/>
  <path class="curve" d="@@CURVE_PATH@@"/>

  <circle class="dot" cx="@@ZERO_X@@" cy="@@ZERO_Y@@" r="6"/>
  <circle class="dot" cx="@@TEN_X@@" cy="@@TEN_Y@@" r="6"/>
  <circle class="dot" cx="@@SIXTY_FOUR_X@@" cy="@@SIXTY_FOUR_Y@@" r="6"/>
  <circle class="dot" cx="@@MAX_X@@" cy="@@MAX_Y@@" r="6"/>

  <rect class="callout" x="132" y="274" width="158" height="54"/>
  <text class="small" x="146" y="296">0 steering</text>
  <text class="formula" x="146" y="316">1.01x floor</text>
  <path class="arrow" d="M132 306 L116 319"/>

  <rect class="callout" x="188" y="212" width="168" height="54"/>
  <text class="small" x="202" y="234">10 steering</text>
  <text class="formula" x="202" y="254">3.30x multiplier</text>
  <path class="arrow" d="M188 242 L170 229"/>

  <rect class="callout" x="510" y="154" width="168" height="54"/>
  <text class="small" x="524" y="176">64 steering</text>
  <text class="formula" x="524" y="196">5.16x multiplier</text>
  <path class="arrow" d="M510 184 L486 155"/>

  <rect class="callout" x="690" y="104" width="180" height="54"/>
  <text class="small" x="704" y="126">127 steering</text>
  <text class="formula" x="704" y="146">5.84x multiplier</text>
  <path class="arrow" d="M850 150 L850 128"/>

  <text class="small" x="278" y="404">The multiplier is applied to squared steering error before taking the batch mean.</text>
</svg>"""
    for token, value in {
        "@@AREA_PATH@@": area_path,
        "@@CURVE_PATH@@": curve_path,
        "@@ZERO_X@@": f"{marker_points['zero'][0]:.1f}",
        "@@ZERO_Y@@": f"{marker_points['zero'][1]:.1f}",
        "@@TEN_X@@": f"{marker_points['ten'][0]:.1f}",
        "@@TEN_Y@@": f"{marker_points['ten'][1]:.1f}",
        "@@SIXTY_FOUR_X@@": f"{marker_points['sixty_four'][0]:.1f}",
        "@@SIXTY_FOUR_Y@@": f"{marker_points['sixty_four'][1]:.1f}",
        "@@MAX_X@@": f"{marker_points['max'][0]:.1f}",
        "@@MAX_Y@@": f"{marker_points['max'][1]:.1f}",
    }.items():
        loss_weight_curve_svg = loss_weight_curve_svg.replace(token, value)

    write_svg(out_dir / "training-pipeline.svg", training_pipeline_svg)
    write_svg(out_dir / "resampling-balance.svg", resampling_balance_svg)
    write_svg(out_dir / "loss-weight-curve.svg", loss_weight_curve_svg)


def assert_asset_sizes(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise RuntimeError(f"{path} is larger than GitHub's 100 MiB file limit")
        if path.suffix.lower() == ".mp4" and size > MAX_VIDEO_BYTES:
            raise RuntimeError(f"{path} is larger than the README video budget")
        if path.suffix.lower() == ".gif" and size > MAX_GIF_BYTES:
            raise RuntimeError(f"{path} is larger than the README GIF budget")
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
