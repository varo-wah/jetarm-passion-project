"""Lighting-tolerant LEGO color classification for detection bounding boxes."""

from __future__ import annotations

import cv2
import numpy as np


MIN_SATURATION = 45
MIN_VALUE = 35
MIN_COLOR_EVIDENCE = 0.015
MIN_CLASS_SHARE = 0.45
MIN_DOMINANCE_RATIO = 1.10

# OpenCV hue ranges from 0 to 179. Small gaps deliberately keep ambiguous
# orange/yellow and cyan pixels from forcing a potentially unsafe bin choice.
HUE_RANGES = {
    "RED": ((0, 18), (165, 179)),
    "GREEN": ((25, 90),),
    "BLUE": ((95, 145),),
}


def _bounded_roi(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return np.empty((0, 0, 3), dtype=np.uint8)

    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(int(x), frame_width))
    y1 = max(0, min(int(y), frame_height))
    x2 = max(x1, min(int(x + width), frame_width))
    y2 = max(y1, min(int(y + height), frame_height))
    return frame[y1:y2, x1:x2]


def _hue_mask(hue: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for lower, upper in ranges:
        mask |= (hue >= lower) & (hue <= upper)
    return mask


def detect_color(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    """Classify a centered object as RED, GREEN, BLUE, or NEUTRAL.

    YOLO boxes usually include some table around the LEGO piece. Votes are
    therefore weighted by saturation, brightness, and distance from the box
    center instead of using the most common hue across the entire rectangle.
    """

    roi = _bounded_roi(frame, x, y, width, height)
    if roi.size == 0:
        return "NEUTRAL"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    roi_height, roi_width = hue.shape
    yy, xx = np.mgrid[0:roi_height, 0:roi_width]
    center_x = (roi_width - 1) / 2.0
    center_y = (roi_height - 1) / 2.0
    norm_x = (xx - center_x) / max(roi_width / 2.0, 1.0)
    norm_y = (yy - center_y) / max(roi_height / 2.0, 1.0)
    radius_squared = (norm_x * norm_x) + (norm_y * norm_y)
    spatial_weight = np.exp(-2.5 * radius_squared)

    core = radius_squared <= 0.50
    core_value = value[core]
    core_saturation = saturation[core]
    if core_value.size == 0 or float(np.median(core_value)) < MIN_VALUE:
        return "NEUTRAL"

    neutral_core = (core_saturation < 35) & (core_value >= 55)
    if float(neutral_core.mean()) >= 0.65:
        return "NEUTRAL"

    adaptive_value_floor = max(
        MIN_VALUE,
        float(np.percentile(value, 35)) * 0.60,
    )
    colored = (saturation >= MIN_SATURATION) & (value >= adaptive_value_floor)

    saturation_strength = (saturation.astype(np.float32) / 255.0) ** 2
    value_strength = value.astype(np.float32) / 255.0
    weights = spatial_weight * saturation_strength * value_strength * colored

    evidence = float(weights.sum()) / max(float(spatial_weight.sum()), 1.0)
    if evidence < MIN_COLOR_EVIDENCE:
        return "NEUTRAL"

    scores = {
        name: float(weights[_hue_mask(hue, ranges)].sum())
        for name, ranges in HUE_RANGES.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1]
    classified_score = sum(scores.values())

    if classified_score <= 0 or best_score / classified_score < MIN_CLASS_SHARE:
        return "NEUTRAL"
    if second_score > 0 and best_score / second_score < MIN_DOMINANCE_RATIO:
        return "NEUTRAL"
    return best_name
