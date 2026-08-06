"""Optional display-only adjustment; detector input remains unmodified."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(minimum, min(value, maximum))


SATURATION_GAIN = _env_float("JETARM_DISPLAY_SATURATION", 1.00, 0.50, 2.00)
SHADOW_GAMMA = _env_float("JETARM_DISPLAY_GAMMA", 1.00, 0.40, 1.60)


def enhance_display_frame(frame: np.ndarray) -> np.ndarray:
    """Return an adjusted copy, or an exact natural copy by default."""

    if frame is None or frame.size == 0:
        return frame
    if SATURATION_GAIN == 1.0 and SHADOW_GAMMA == 1.0:
        return frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURATION_GAIN, 0, 255)

    normalized_value = hsv[:, :, 2] / 255.0
    hsv[:, :, 2] = np.clip(
        np.power(normalized_value, SHADOW_GAMMA) * 255.0,
        0,
        255,
    )
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
