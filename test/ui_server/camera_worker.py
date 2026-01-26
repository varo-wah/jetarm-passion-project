import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


latest_frame: Optional[np.ndarray] = None
latest_frame_lock = threading.Lock()

_camera_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_is_running_lock = threading.Lock()


ROI_X0_FRAC = 0.18
ROI_X1_FRAC = 0.82
ROI_Y0_FRAC = 0.05
ROI_Y1_FRAC = 0.62

ROI_MODE = "mask"       # "mask" | "box" | "crop" | "off"
ROI_DRAW_BOX = True


CAP_BUFFERSIZE = 1
READ_SLEEP_S = 0.005
WARMUP_FRAMES = 5

CAM_BRIGHTNESS = None
CAM_CONTRAST = None
CAM_SATURATION = None


def _roi_bounds(h: int, w: int) -> Tuple[int, int, int, int]:
    x0 = int(w * ROI_X0_FRAC)
    x1 = int(w * ROI_X1_FRAC)
    y0 = int(h * ROI_Y0_FRAC)
    y1 = int(h * ROI_Y1_FRAC)

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def _apply_roi(frame: np.ndarray) -> np.ndarray:
    mode = (ROI_MODE or "off").lower()
    if mode == "off":
        return frame

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(h, w)

    if mode == "crop":
        return frame[y0:y1, x0:x1]

    out = frame.copy()

    if mode == "mask":
        masked = np.zeros_like(out)
        masked[y0:y1, x0:x1] = out[y0:y1, x0:x1]
        out = masked

    if ROI_DRAW_BOX and mode in ("mask", "box"):
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(
            out,
            "DETECTION ROI",
            (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return out


def _apply_optional_camera_settings(cap: cv2.VideoCapture) -> None:
    if CAP_BUFFERSIZE is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(CAP_BUFFERSIZE))

    if CAM_BRIGHTNESS is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(CAM_BRIGHTNESS))
    if CAM_CONTRAST is not None:
        cap.set(cv2.CAP_PROP_CONTRAST, float(CAM_CONTRAST))
    if CAM_SATURATION is not None:
        cap.set(cv2.CAP_PROP_SATURATION, float(CAM_SATURATION))


def _camera_loop(cam_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    _apply_optional_camera_settings(cap)

    try:
        for _ in range(WARMUP_FRAMES):
            if _stop_event.is_set():
                break
            cap.read()
            time.sleep(0.02)

        while not _stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_out = _apply_roi(frame)

            with latest_frame_lock:
                global latest_frame
                latest_frame = frame_out

            time.sleep(READ_SLEEP_S)
    finally:
        cap.release()


def start_camera(cam_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
    global _camera_thread
    with _is_running_lock:
        if _camera_thread is not None and _camera_thread.is_alive():
            return

        _stop_event.clear()
        _camera_thread = threading.Thread(
            target=_camera_loop,
            args=(cam_index, width, height),
            daemon=True,
        )
        _camera_thread.start()


def stop_camera(timeout_s: float = 2.0) -> None:
    global _camera_thread
    with _is_running_lock:
        _stop_event.set()
        t = _camera_thread
        _camera_thread = None

    if t is not None and t.is_alive():
        t.join(timeout=timeout_s)

    with latest_frame_lock:
        global latest_frame
        latest_frame = None


def get_latest_frame_copy() -> Optional[np.ndarray]:
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()