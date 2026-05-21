"""
Reusable YOLO detector module for JetArm.

Responsibilities:
- Load YOLO model once
- Run YOLO inference on a provided frame
- Extract structured detection data
- Convert pixel center coordinates to robot coordinates
- Estimate object angle inside the YOLO bounding box
- Select the best reachable target
- Provide old Vision_Scanner-compatible output

Does NOT:
- Open the camera
- Display OpenCV windows
- Move the robot
- Import Class_Execution
- Control servos or gripper
"""

from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

from jetarm.config.yolo_config import LEGO_YOLO_MODEL_PATH
from jetarm.vision.coordinatelogic import detect_color, pixel_to_robot

# ============================================================
# 1. PATH SETUP
# ============================================================

MODEL_PATH = LEGO_YOLO_MODEL_PATH


# ============================================================
# 2. MODEL CACHE
# ============================================================

_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

        print(f"[YOLO] Loading model from: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
        print("[YOLO] Model loaded successfully.")

    return _model


# ============================================================
# 3. ANGLE ESTIMATION
# ============================================================

def estimate_angle_from_yolo_box(frame, x1, y1, x2, y2) -> float:
    image_height, image_width = frame.shape[:2]

    x1 = max(0, min(int(x1), image_width - 1))
    x2 = max(0, min(int(x2), image_width - 1))
    y1 = max(0, min(int(y1), image_height - 1))
    y2 = max(0, min(int(y2), image_height - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        5,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 50:
        return 0.0

    rect = cv2.minAreaRect(largest_contour)
    (_, _), (rect_width, rect_height), angle = rect

    if rect_width <= 0 or rect_height <= 0:
        return 0.0

    if rect_height > rect_width:
        angle += 90

    angle = angle % 180

    return round(float(angle), 1)


# ============================================================
# 4. YOLO RESULT EXTRACTION
# ============================================================

def extract_detections(frame, result) -> List[Dict[str, Any]]:
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        robot_x, robot_y = pixel_to_robot(center_x, center_y)
        angle = estimate_angle_from_yolo_box(frame, x1, y1, x2, y2)
        box_w = x2 - x1
        box_h = y2 - y1
        pad = int(min(box_w, box_h) * 0.08)
        pad = max(2, min(pad, 6))
        color = detect_color(
            frame,
            x1 + pad,
            y1 + pad,
            max(1, box_w - 2 * pad),
            max(1, box_h - 2 * pad),
        )

        detections.append({
            "class_id": class_id,
            "confidence": round(confidence, 2),
            "center_x": center_x,
            "center_y": center_y,
            "robot_x": round(float(robot_x), 2),
            "robot_y": round(float(robot_y), 2),
            "angle": round(float(angle), 1),
            "color": color,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return detections


# ============================================================
# 5. TARGET SELECTION
# ============================================================

def choose_target(
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.40,
) -> Optional[Dict[str, Any]]:
    valid_detections = []

    for detection in detections:
        confidence = detection["confidence"]
        robot_x = detection["robot_x"]
        robot_y = detection["robot_y"]

        if confidence < min_confidence:
            continue

        if not (-20 <= robot_x <= 20):
            continue

        if not (5 <= robot_y <= 28):
            continue

        valid_detections.append(detection)

    if not valid_detections:
        return None

    return max(valid_detections, key=lambda d: d["confidence"])


# ============================================================
# 6. PUBLIC DETECTION FUNCTIONS
# ============================================================

def detect_objects(frame) -> List[Dict[str, Any]]:
    model = load_model()
    results = model(frame, imgsz=416, verbose=False)

    if not results:
        return []

    return extract_detections(frame, results[0])


def detect_target(frame) -> Optional[Dict[str, Any]]:
    detections = detect_objects(frame)
    return choose_target(detections)


# ============================================================
# 7. VISION_SCANNER COMPATIBILITY ADAPTER
# ============================================================

def to_vision_scanner_format(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bricks = []

    for detection in detections:
        angle = detection.get("angle", 0.0)

        if angle is None:
            angle = 0.0

        bricks.append({
            "x": float(detection["robot_x"]),
            "y": float(detection["robot_y"]),
            "angle": float(angle),
            "color": detection.get("color", "NEUTRAL"),
        })

    return bricks


def detect_bricks_yolo(frame) -> List[Dict[str, Any]]:
    detections = detect_objects(frame)
    return to_vision_scanner_format(detections)
