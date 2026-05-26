"""Run live Forza AI driving through a virtual controller."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "forza_autodrive"

from .config import DEFAULT_MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, TELEMETRY_COLUMNS
from .controller import GamepadController, NullController
from .model import MODEL_STEER_RANGE, checkpoint_telemetry_dim, load_model
from .preprocess import FrameGrabber, model_input_image, preprocess_image, save_preprocess_debug, to_rgb_image
from .steering_control import SteeringControlOutput, SteeringPidController, clamp
from .telemetry import TelemetryReceiver, telemetry_vector


def parse_region(values: list[int] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("region requires left top right bottom")
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("region right/bottom must be greater than left/top")
    return left, top, right, bottom


def normalized_udp_steer(snapshot) -> float | None:
    if snapshot is None:
        return None
    try:
        return max(-1.0, min(1.0, float(snapshot.row["Steer"]) / 127.0))
    except (KeyError, TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=9999)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "dxcam", "imagegrab"),
        default="auto",
        help="screen capture backend; auto falls back from dxcam to Pillow ImageGrab",
    )
    parser.add_argument("--max-accel", type=float, default=0.35)
    parser.add_argument(
        "--steer-scale",
        type=float,
        default=2.0,
        help="multiply normalized model steering before controller output; negative values invert steering",
    )
    parser.add_argument("--steer-smoothing", type=float, default=1.)
    parser.add_argument(
        "--steer-feedback",
        choices=("off", "pid"),
        default="pid",
        help="closed-loop steering correction using UDP Steer as feedback",
    )
    parser.add_argument("--steer-pid-kp", type=float, default=0.1)
    parser.add_argument("--steer-pid-ki", type=float, default=0.8)
    parser.add_argument("--steer-pid-kd", type=float, default=0.01)
    parser.add_argument("--steer-pid-correction-limit", type=float, default=0.5)
    parser.add_argument("--steer-pid-correction-rate-limit", type=float, default=0.5)
    parser.add_argument("--steer-pid-integral-limit", type=float, default=1)
    parser.add_argument(
        "--steer-kalman",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="filter UDP Steer before PID correction",
    )
    parser.add_argument("--steer-kalman-process-noise", type=float, default=0.002)
    parser.add_argument("--steer-kalman-measurement-noise", type=float, default=0.03)
    parser.add_argument("--trigger-smoothing", type=float, default=1.)
    parser.add_argument("--telemetry-timeout", type=float, default=0.5)
    parser.add_argument("--debug-frame-dir", type=Path)
    parser.add_argument("--debug-frame-count", type=int, default=3)
    parser.add_argument("--debug-frame-interval", type=float, default=1.0)
    parser.add_argument(
        "--debug-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live debug window; use --no-debug-window to disable",
    )
    parser.add_argument("--debug-window-cam-interval", type=float, default=0.2)
    parser.add_argument(
        "--debug-window-cam-layer",
        type=int,
        default=1,
        help="index into model.features used for GradCAM++",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-armed", action="store_true")
    parser.add_argument("--no-controller", action="store_true")
    return parser


def install_hotkeys(armed: threading.Event, stop: threading.Event) -> None:
    import keyboard

    def toggle_armed() -> None:
        if armed.is_set():
            armed.clear()
            print("\nAI control disarmed")
        else:
            armed.set()
            print("\nAI control armed")

    def emergency_stop() -> None:
        print("\nEmergency stop requested")
        stop.set()

    keyboard.add_hotkey("f9", toggle_armed)
    keyboard.add_hotkey("f8", emergency_stop)


class DebugWindow:
    def __init__(
        self,
        model: torch.nn.Module,
        cam_interval: float = 0.2,
        cam_layer: int = -1,
    ) -> None:
        import cv2
        import numpy as np
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        class FixedTelemetryWrapper(torch.nn.Module):
            def __init__(self, wrapped_model: torch.nn.Module) -> None:
                super().__init__()
                self.model = wrapped_model
                self.telemetry: torch.Tensor | None = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.telemetry is None:
                    raise RuntimeError("debug telemetry is not set")
                telemetry = self.telemetry
                if telemetry.shape[0] != x.shape[0]:
                    telemetry = telemetry.expand(x.shape[0], -1)
                return self.model(x, telemetry)

        self.cv2 = cv2
        self.np = np
        self.show_cam_on_image = show_cam_on_image
        self.wrapper = FixedTelemetryWrapper(model)
        try:
            target_layer = model.features[cam_layer]
        except IndexError as exc:
            layer_count = len(model.features)
            raise ValueError(
                f"--debug-window-cam-layer must be in [{-layer_count}, {layer_count - 1}], "
                f"got {cam_layer}"
            ) from exc
        self.cam_label = f"gradcam++ f[{cam_layer}]"
        self.cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[target_layer])
        self.target = [ClassifierOutputTarget(0)]
        self.cam_interval = max(cam_interval, 0.0)
        self.last_cam_at = 0.0
        self.last_cam_rgb: np.ndarray | None = None
        self.window_name = "Forza debug"
        self.panel_size = (320, 180)
        self.canvas_size = (self.panel_size[0] * 2, self.panel_size[1] * 2)
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, *self.canvas_size)
        self.show_status("starting debug window")

    def update(
        self,
        raw_frame,
        frame_tensor: torch.Tensor,
        telemetry: torch.Tensor,
        steer_value: float,
        udp_steer_value: float | None,
    ) -> bool:
        processed_rgb = self._processed_rgb(raw_frame)
        now = time.monotonic()
        if self.last_cam_rgb is None or now - self.last_cam_at >= self.cam_interval:
            self.last_cam_rgb = self._gradcam_rgb(frame_tensor, telemetry, processed_rgb)
            self.last_cam_at = now

        raw_panel = self._panel(self._rgb_array(to_rgb_image(raw_frame)), "raw")
        processed_panel = self._panel(processed_rgb, "model input")
        cam_panel = self._panel(self.last_cam_rgb, self.cam_label)
        bar_panel = self._steer_bars(steer_value, udp_steer_value)
        canvas = self.np.vstack(
            (
                self.np.hstack((raw_panel, processed_panel)),
                self.np.hstack((cam_panel, bar_panel)),
            )
        )
        return self._show_canvas(canvas)

    def show_status(self, message: str) -> bool:
        width, height = self.canvas_size
        canvas = self.np.full((height, width, 3), 245, dtype=self.np.uint8)
        self.cv2.putText(
            canvas,
            message,
            (24, height // 2 - 12),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            canvas,
            "press q to close",
            (24, height // 2 + 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 80, 80),
            1,
            self.cv2.LINE_AA,
        )
        return self._show_canvas(canvas)

    def _show_canvas(self, canvas) -> bool:
        self.cv2.imshow(self.window_name, self.cv2.cvtColor(canvas, self.cv2.COLOR_RGB2BGR))
        key = self.cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        try:
            return self.cv2.getWindowProperty(self.window_name, self.cv2.WND_PROP_VISIBLE) >= 1
        except self.cv2.error:
            return False

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except self.cv2.error:
            pass

    def _processed_rgb(self, raw_frame):
        return self._rgb_array(model_input_image(raw_frame))

    def _gradcam_rgb(
        self,
        frame_tensor: torch.Tensor,
        telemetry: torch.Tensor,
        base_rgb: "np.ndarray",
    ):
        input_tensor = frame_tensor.detach().clone().requires_grad_(True)
        self.wrapper.telemetry = telemetry.detach()
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=self.target)[0]
        return self.show_cam_on_image(base_rgb.astype("float32") / 255.0, grayscale_cam, use_rgb=True)

    def _rgb_array(self, image):
        return self.np.asarray(image.convert("RGB"), dtype=self.np.uint8)

    def _panel(self, image_rgb, label: str):
        panel = self.cv2.resize(image_rgb, self.panel_size, interpolation=self.cv2.INTER_AREA)
        self.cv2.putText(
            panel,
            label,
            (8, 22),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        return panel

    def _steer_bars(self, ai_steer: float, udp_steer: float | None):
        width, height = self.panel_size
        panel = self.np.full((height, width, 3), 245, dtype=self.np.uint8)
        self.cv2.putText(panel, "steer", (8, 22), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        self._draw_bar(panel, 70, ai_steer, "AI", (60, 120, 220))
        self._draw_bar(panel, 125, udp_steer, "UDP", (220, 80, 80))
        return panel

    def _draw_bar(self, panel, y: int, value: float | None, label: str, color):
        width = panel.shape[1]
        center = width // 2
        bar_half = width // 2 - 24
        self.cv2.putText(panel, label, (8, y + 7), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        self.cv2.line(panel, (center - bar_half, y), (center + bar_half, y), (70, 70, 70), 2)
        self.cv2.line(panel, (center, y - 28), (center, y + 28), (120, 120, 120), 1)
        if value is None:
            self.cv2.putText(panel, "n/a", (center - 20, y + 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
            return
        steer_value = max(-1.0, min(1.0, float(value)))
        x = int(center + steer_value * bar_half)
        self.cv2.rectangle(panel, (min(center, x), y - 16), (max(center, x), y + 16), color, -1)
        self.cv2.putText(
            panel,
            f"{steer_value:+.2f}",
            (center - 32, y + 25),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            self.cv2.LINE_AA,
        )


def main() -> int:
    args = build_parser().parse_args()
    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    expected_dim = len(TELEMETRY_COLUMNS)
    checkpoint_dim = checkpoint_telemetry_dim(model_path)
    if checkpoint_dim not in (0, expected_dim):
        raise ValueError(
            f"checkpoint expects {checkpoint_dim} telemetry values, "
            f"but this runtime provides {expected_dim}: {TELEMETRY_COLUMNS}"
        )

    device = torch.device(args.device)
    model = load_model(model_path, device=device)
    print(f"Loaded model: {model_path}")
    print(f"Device: {device}")
    if checkpoint_dim == 0:
        print(f"Telemetry columns available: {expected_dim}; checkpoint does not use telemetry input")
    else:
        print(f"Telemetry columns: {expected_dim} (action columns excluded)")
    requires_telemetry = checkpoint_dim > 0
    debug_window = (
        DebugWindow(
            model,
            cam_interval=args.debug_window_cam_interval,
            cam_layer=args.debug_window_cam_layer,
        )
        if args.debug_window
        else None
    )
    if debug_window is not None:
        print(
            "Debug window enabled: "
            f"layer={args.debug_window_cam_layer} "
            f"cam_interval={args.debug_window_cam_interval:.3f}s"
        )

    receiver = TelemetryReceiver(host=args.udp_host, port=args.udp_port)
    receiver.start()
    grabber = FrameGrabber(
        monitor=args.monitor,
        region=parse_region(args.region),
        backend=args.capture_backend,
    )
    print(f"Capture backend: {grabber.backend}")
    if grabber.fallback_reason:
        print(f"Capture fallback: {grabber.fallback_reason}")

    controller = (
        NullController()
        if args.no_controller
        else GamepadController(
            max_accel=args.max_accel,
            steer_smoothing=args.steer_smoothing,
            trigger_smoothing=args.trigger_smoothing,
        )
    )
    steer_pid = (
        SteeringPidController(
            kp=args.steer_pid_kp,
            ki=args.steer_pid_ki,
            kd=args.steer_pid_kd,
            correction_limit=args.steer_pid_correction_limit,
            correction_rate_limit=args.steer_pid_correction_rate_limit,
            integral_limit=args.steer_pid_integral_limit,
            use_kalman=args.steer_kalman,
            kalman_process_noise=args.steer_kalman_process_noise,
            kalman_measurement_noise=args.steer_kalman_measurement_noise,
        )
        if args.steer_feedback == "pid"
        else None
    )
    if steer_pid is not None:
        print(
            "Steering feedback: "
            f"PID kp={args.steer_pid_kp:.3f} ki={args.steer_pid_ki:.3f} "
            f"kd={args.steer_pid_kd:.3f} correction_limit={args.steer_pid_correction_limit:.3f} "
            f"correction_rate_limit={args.steer_pid_correction_rate_limit:.3f}/s "
            f"kalman={'on' if args.steer_kalman else 'off'}"
        )

    armed = threading.Event()
    stop = threading.Event()
    if args.start_armed:
        armed.set()

    try:
        install_hotkeys(armed, stop)
        print("Hotkeys: F9 arm/disarm, F8 emergency stop, Ctrl+C exit")
    except Exception as exc:
        print(f"Could not install global hotkeys: {exc}")
        print("Ctrl+C will still reset the controller and exit.")

    frame_interval = 1.0 / max(args.fps, 0.1)
    last_status = 0.0
    logged_frame_size: tuple[int, int] | None = None
    debug_frames_saved = 0
    last_debug_frame = 0.0

    try:
        while not stop.is_set():
            loop_started = time.monotonic()
            snapshot = receiver.latest(max_age_s=args.telemetry_timeout)
            raw_frame = grabber.grab_frame()
            frame_tensor = preprocess_image(raw_frame).unsqueeze(0) if raw_frame is not None else None

            if raw_frame is not None:
                frame_size = grabber.last_frame_size
                if frame_size is not None and frame_size != logged_frame_size:
                    frame_width, frame_height = frame_size
                    model_height, model_width = frame_tensor.shape[-2:]
                    expected_aspect = IMAGE_WIDTH / IMAGE_HEIGHT
                    actual_aspect = frame_width / frame_height if frame_height else 0.0
                    aspect_note = ""
                    if frame_height and abs(actual_aspect - expected_aspect) / expected_aspect > 0.02:
                        aspect_note = (
                            f" WARNING aspect={actual_aspect:.3f}, expected={expected_aspect:.3f}"
                        )
                    print(
                        f"Captured frame: raw={frame_width}x{frame_height} "
                        f"resized={IMAGE_WIDTH}x{IMAGE_HEIGHT} "
                        f"model_input={model_width}x{model_height}{aspect_note}"
                    )
                    logged_frame_size = frame_size

                now = time.monotonic()
                if (
                    args.debug_frame_dir is not None
                    and debug_frames_saved < max(args.debug_frame_count, 0)
                    and (
                        debug_frames_saved == 0
                        or now - last_debug_frame >= max(args.debug_frame_interval, 0.0)
                    )
                ):
                    info = save_preprocess_debug(
                        raw_frame,
                        args.debug_frame_dir,
                        f"frame_{debug_frames_saved:03d}",
                    )
                    print(
                        "Saved debug frame: "
                        f"raw={info['raw_size']} resized={info['resized_size']} "
                        f"model_input={info['model_size']} path={info['model_path']}"
                    )
                    debug_frames_saved += 1
                    last_debug_frame = now

            missing_telemetry = requires_telemetry and snapshot is None
            if missing_telemetry or frame_tensor is None:
                if armed.is_set():
                    controller.reset()
                if steer_pid is not None:
                    steer_pid.reset()
                if debug_window is not None:
                    message = "waiting for telemetry" if missing_telemetry else "waiting for frame"
                    if not debug_window.show_status(message):
                        stop.set()
                        break
                now = time.monotonic()
                if now - last_status > 1.0:
                    print(
                        "Waiting for "
                        f"{'telemetry' if missing_telemetry else 'frame'}... "
                        f"packets={receiver.packet_count} parse_errors={receiver.parse_errors}"
                    )
                    last_status = now
                time.sleep(0.05)
                continue

            if requires_telemetry:
                telemetry = torch.from_numpy(telemetry_vector(snapshot.row)).unsqueeze(0)
            else:
                telemetry = torch.empty((1, 0), dtype=torch.float32)
            frame_tensor = frame_tensor.to(device, non_blocking=True)
            telemetry = telemetry.to(device, non_blocking=True)

            with torch.inference_mode():
                output = model(frame_tensor, telemetry).squeeze(0).detach().cpu().tolist()

            steer, accel, brake = output
            normalized_steer = steer / MODEL_STEER_RANGE
            scaled_steer = normalized_steer * args.steer_scale
            udp_steer_for_debug = normalized_udp_steer(snapshot)
            target_steer = clamp(float(scaled_steer), -1.0, 1.0)
            command_steer = scaled_steer
            steer_feedback_output: SteeringControlOutput | None = None
            if armed.is_set():
                if steer_pid is not None:
                    steer_feedback_output = steer_pid.update(target_steer, udp_steer_for_debug)
                    command_steer = steer_feedback_output.command
                sent = controller.apply(command_steer, accel, brake)
                state = "ARMED"
            else:
                controller.reset()
                if steer_pid is not None:
                    steer_pid.reset()
                sent = controller.last if isinstance(controller, NullController) else None
                state = "DISARMED"

            if debug_window is not None:
                steer_for_debug = (
                    sent.steer
                    if sent is not None
                    else max(-1.0, min(1.0, float(scaled_steer)))
                )
                if not debug_window.update(
                    raw_frame,
                    frame_tensor,
                    telemetry,
                    steer_for_debug,
                    udp_steer_for_debug,
                ):
                    stop.set()

            now = time.monotonic()
            if now - last_status > 0.5:
                steer_text = f"{steer:+.1f}/{normalized_steer:+.2f}"
                if args.steer_scale != 1.0:
                    steer_text = f"{steer:+.1f}/{normalized_steer:+.2f}->{scaled_steer:+.2f}"
                if sent is None:
                    sent_text = "neutral"
                else:
                    sent_text = (
                        f"sent=({sent.steer:+.2f}, {sent.accel:.2f}, {sent.brake:.2f})"
                    )
                feedback_text = ""
                if steer_feedback_output is not None:
                    filtered = steer_feedback_output.filtered_measurement
                    filtered_text = "n/a" if filtered is None else f"{filtered:+.2f}"
                    error = steer_feedback_output.error
                    error_text = "n/a" if error is None else f"{error:+.2f}"
                    feedback_text = (
                        f" pid=(target={steer_feedback_output.target:+.2f} "
                        f"udp={filtered_text} err={error_text} "
                        f"corr={steer_feedback_output.correction:+.2f} "
                        f"cmd={steer_feedback_output.command:+.2f})"
                    )
                if snapshot is None:
                    telemetry_status = f"telemetry=none packets={receiver.packet_count}"
                else:
                    telemetry_status = (
                        f"speed={float(snapshot.row.get('Speed', 0.0)):.1f} "
                        f"age={snapshot.age_s:.3f}s packets={snapshot.packet_count}"
                    )
                print(
                    f"{state} pred=({steer_text}, {accel:.2f}, {brake:.2f}) "
                    f"{sent_text}{feedback_text} {telemetry_status}"
                )
                last_status = now

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, frame_interval - elapsed))
    except KeyboardInterrupt:
        print("\nCtrl+C received")
    finally:
        if debug_window is not None:
            debug_window.close()
        controller.reset()
        receiver.stop()
        print("Controller reset; telemetry stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
