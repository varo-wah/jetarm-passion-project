import threading
import time

import cv2

from jetarm.config.detection_roi import ROI_DRAW_BOX, roi_bounds_from_shape
from jetarm.vision.yolo_detector import choose_target, detect_objects

YOLO_MAX_FPS = 5.0
YOLO_INTERVAL_S = 1.0 / YOLO_MAX_FPS

_inference_lock = threading.Lock()
_cache_lock = threading.Lock()
_last_yolo_time = 0.0
_last_annotated_frame = None


def draw_detection(frame, detection, is_target):
    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]
    center_x = detection["center_x"]
    center_y = detection["center_y"]
    robot_x = detection["robot_x"]
    robot_y = detection["robot_y"]
    angle = detection["angle"]
    confidence = detection["confidence"]
    color = detection.get("color", "NEUTRAL")

    box_color = (0, 0, 255) if is_target else (0, 255, 0)
    text_color = (0, 0, 255) if is_target else (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

    if is_target:
        cv2.circle(frame, (center_x, center_y), 14, (0, 0, 255), 2)

    text_x = x1
    text_y = max(20, y1 - 54)

    labels = [
        f"{color} conf={confidence}",
        f"robot=({robot_x:.2f},{robot_y:.2f})",
        f"angle={angle:.1f}",
        f"px=({center_x},{center_y})",
    ]

    for index, label in enumerate(labels):
        cv2.putText(
            frame,
            label,
            (text_x, text_y + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
        )


def draw_detections(frame, detections, target):
    for detection in detections:
        draw_detection(frame, detection, target is not None and detection == target)

    if target is None:
        status = "TARGET: none"
    else:
        status = (
            f"TARGET: x={target['robot_x']:.2f}, "
            f"y={target['robot_y']:.2f}, "
            f"angle={target['angle']:.1f}, "
            f"color={target.get('color', 'NEUTRAL')}"
        )

    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    if ROI_DRAW_BOX:
        x0, y0, x1, y1 = roi_bounds_from_shape(frame.shape)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "YOLO DETECTION ROI",
            (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return frame


def print_detections(detections):
    if not detections:
        print("No YOLO detections")
        return

    for detection in detections:
        print(
            f"x={detection['robot_x']:.2f}, "
            f"y={detection['robot_y']:.2f}, "
            f"angle={detection['angle']:.1f}, "
            f"color={detection.get('color', 'NEUTRAL')}, "
            f"confidence={detection['confidence']:.2f}"
        )


def _status_frame(frame, text):
    out = frame.copy()
    cv2.putText(out, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if ROI_DRAW_BOX:
        x0, y0, x1, y1 = roi_bounds_from_shape(out.shape)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out


def _get_cached_frame(frame):
    global _last_yolo_time

    with _cache_lock:
        if _last_annotated_frame is None:
            return None, _last_yolo_time
        if _last_annotated_frame.shape != frame.shape:
            return None, _last_yolo_time
        return _last_annotated_frame.copy(), _last_yolo_time


def _set_cached_frame(frame, timestamp):
    global _last_annotated_frame, _last_yolo_time

    with _cache_lock:
        _last_annotated_frame = frame.copy()
        _last_yolo_time = timestamp


def annotate_yolo_frame(frame):
    if frame is None:
        return frame

    now = time.monotonic()
    cached_frame, last_yolo_time = _get_cached_frame(frame)

    if cached_frame is not None and now - last_yolo_time < YOLO_INTERVAL_S:
        return cached_frame

    if not _inference_lock.acquire(blocking=False):
        if cached_frame is not None:
            return cached_frame
        return _status_frame(frame, "YOLO busy")

    try:
        now = time.monotonic()
        cached_frame, last_yolo_time = _get_cached_frame(frame)
        if cached_frame is not None and now - last_yolo_time < YOLO_INTERVAL_S:
            return cached_frame

        source = frame.copy()
        detections = detect_objects(source)
        target = choose_target(detections)
        annotated_frame = draw_detections(source, detections, target)
        _set_cached_frame(annotated_frame, time.monotonic())
        return annotated_frame
    finally:
        _inference_lock.release()
