# ui_server/camera_worker.py
# Owns the camera (single owner) and continuously updates the latest frame in memory.

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


def _camera_loop(cam_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
    """
    Background thread function.
    Opens the camera once and continually updates latest_frame.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}")

    # Optional settings (safe to ignore if driver doesn't support)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    try:
        # Warm-up frames (helps exposure/auto-focus settle)
        for _ in range(5):
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


def start_camera(cam_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
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
            args=(cam_index, width, height),
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
