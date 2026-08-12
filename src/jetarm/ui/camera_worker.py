# camera_worker.py
# Owns the camera (single owner) and continuously updates the latest frame in memory.

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

# -----------------------------
# Shared state (read by server)
# -----------------------------
latest_frame: Optional[np.ndarray] = None
latest_frame_lock = threading.Lock()

_camera_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_is_running_lock = threading.Lock()


_V4L2_DEFAULT_RE = re.compile(
    r"^\s*([a-zA-Z0-9_]+).*?\bdefault=(-?\d+)\b"
)

# Restore only image-appearance controls. Do not modify unrelated PTZ, privacy,
# LED, or codec controls exposed by some UVC cameras.
_V4L2_IMAGE_CONTROL_ORDER = (
    "auto_exposure",
    "exposure_auto",
    "exposure_auto_priority",
    "exposure_dynamic_framerate",
    "white_balance_automatic",
    "auto_white_balance",
    "white_balance_temperature_auto",
    "focus_automatic_continuous",
    "focus_auto",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "red_balance",
    "blue_balance",
    "white_balance_temperature",
    "gamma",
    "gain",
    "power_line_frequency",
    "sharpness",
    "backlight_compensation",
    "exposure_time_absolute",
    "exposure_absolute",
    "focus_absolute",
)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(f"[CAMERA] Invalid {name}; using {default}")
        return default


def _parse_v4l2_defaults(output: str) -> dict[str, int]:
    defaults = {}
    for line in output.splitlines():
        match = _V4L2_DEFAULT_RE.match(line)
        if match is not None:
            defaults[match.group(1)] = int(match.group(2))
    return defaults


def _reset_camera_controls(cam_index: int) -> int:
    """Restore supported image controls to the V4L2 driver's defaults."""

    v4l2_ctl = shutil.which("v4l2-ctl")
    if v4l2_ctl is None:
        print("[CAMERA] v4l2-ctl unavailable; camera defaults were not reset")
        return 0

    device = f"/dev/video{cam_index}"
    listed = subprocess.run(
        [v4l2_ctl, "--device", device, "--list-ctrls"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        detail = listed.stderr.strip() or "unknown V4L2 error"
        print(f"[CAMERA] Could not read driver defaults for {device}: {detail}")
        return 0

    defaults = _parse_v4l2_defaults(listed.stdout)
    restored = 0
    for control_name in _V4L2_IMAGE_CONTROL_ORDER:
        if control_name not in defaults:
            continue
        result = subprocess.run(
            [
                v4l2_ctl,
                "--device",
                device,
                f"--set-ctrl={control_name}={defaults[control_name]}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            restored += 1

    print(f"[CAMERA] Restored {restored} image controls to driver defaults")
    return restored


def _camera_loop(
    cam_index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> None:
    """
    Background thread function.
    Opens the camera once and continually updates latest_frame.
    """
    # UVC image controls persist beyond the process that changed them. Restore
    # the camera's declared defaults before OpenCV takes ownership so every
    # branch starts from the same unenhanced feed.
    _reset_camera_controls(cam_index)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}")

    # Optional settings (safe to ignore if driver doesn't support)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(fps))

    try:
        # Warm-up frames (helps exposure/auto-focus settle)
        for _ in range(_env_int("JETARM_CAMERA_WARMUP_FRAMES", 30)):
            if _stop_event.is_set():
                break
            cap.read()
            time.sleep(0.02)

        while not _stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Write newest frame safely
            with latest_frame_lock:
                global latest_frame
                latest_frame = frame

            # Avoid pegging CPU
            time.sleep(0.005)
    finally:
        cap.release()


def start_camera(
    cam_index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> None:
    """
    Starts the camera thread if it isn't already running.
    Safe to call multiple times.
    """
    global _camera_thread
    with _is_running_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            return  # already running

        _stop_event.clear()
        _camera_thread = threading.Thread(
            target=_camera_loop,
            args=(cam_index, width, height, fps),
            daemon=True,
        )
        _camera_thread.start()


def stop_camera(timeout_s: float = 2.0) -> None:
    """
    Signals the camera thread to stop and waits briefly.
    """
    global _camera_thread

    with _is_running_lock:
        _stop_event.set()
        t = _camera_thread
        _camera_thread = None

    if t is not None and t.is_alive():
        t.join(timeout=timeout_s)

    # Clear last frame so callers can treat it as "not ready"
    with latest_frame_lock:
        global latest_frame
        latest_frame = None


def get_latest_frame_copy() -> Optional[np.ndarray]:
    """
    Returns a safe copy of the latest frame, or None if not ready.
    Copy is important: server can encode while camera keeps updating.
    """
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()
