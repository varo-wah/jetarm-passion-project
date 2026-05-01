"""
Reusable YOLO detector module for JetArm.

This module:
- Loads YOLO
- Runs detection on one frame
- Converts pixel center to robot coordinates
- Estimates object angle using OpenCV inside the YOLO box
- Chooses the best reachable target

It does NOT:
- Open camera
- Show OpenCV windows
- Move robot
- Import Class_Execution
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

INFERENCE_DIR = CURRENT_FILE.parent
ML_DIR = INFERENCE_DIR.parent
TEST_DIR = ML_DIR.parent

MODEL_PATH = ML_DIR / "models" / "yolo11n.pt"
EXHIBITION_PHASE_DIR = TEST_DIR / "exhibition_phase"

if str(EXHIBITION_PHASE_DIR) not in sys.path:
    sys.path.append(str(EXHIBITION_PHASE_DIR))

from coordinatelogic import pixel_to_robot  # noqa: E402


# ------------------------------------------------------------
# Global model cache
# ------------------------------------------------------------

_model = None


def load_model():
    """
    Load YOLO model once and reuse it.
    """
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

        print(f"[YOLO] Loading model from: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
        print("[YOLO] Model loaded successfully.")

    return _model


def estimate_angle_from_crop(frame, x1, y1, x2, y2) -> Optional[float]:
    """
    Estimate object rotation angle from inside a YOLO bounding box.

    YOLO finds the object.
    OpenCV estimates the object's rotation.

    Args:
        frame: Full OpenCV frame.
        x1, y1, x2, y2: YOLO bounding box coordinates.

    Returns:
        float or None: Angle from 0 to 180 degrees.
    """
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 50:
        return None

    rect = cv2.minAreaRect(largest_contour)
    (_, _), (w, h), angle = rect

    if w <= 0 or h <= 0:
        return None

    if h > w:
        angle += 90

    angle = angle % 180

    return round(float(angle), 1)


def extract_detections(frame, result) -> List[Dict[str, Any]]:
    """
    Convert YOLO result into structured detections.
    """
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

        angle = estimate_angle_from_crop(frame, x1, y1, x2, y2)

        detection = {
            "class_id": class_id,
            "confidence": confidence,
            "center_x": center_x,
            "center_y": center_y,
            "robot_x": float(robot_x),
            "robot_y": float(robot_y),
            "angle": angle,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        detections.append(detection)

    return detections


def choose_target(
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.40,
) -> Optional[Dict[str, Any]]:
    """
    Choose the highest-confidence reachable detection.
    """
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


def detect_objects(frame) -> List[Dict[str, Any]]:
    """
    Run YOLO detection on one provided frame.
    """
    model = load_model()

    results = model(frame, imgsz=416, verbose=False)

    if not results:
        return []

    return extract_detections(frame, results[0])


def detect_target(frame) -> Optional[Dict[str, Any]]:
    """
    Detect all objects and return the chosen target.
    """
    detections = detect_objects(frame)
    return choose_target(detections)


def to_vision_scanner_format(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert YOLO detection into old Vision_Scanner.py style format.

    Old format:
    {
        "x": robot_x,
        "y": robot_y,
        "angle": angle,
        "color": color
    }
    """
    return {
        "x": detection["robot_x"],
        "y": detection["robot_y"],
        "angle": detection["angle"] if detection["angle"] is not None else 0.0,
        "color": "YOLO",
    }

    def to_vision_scanner_format(detections):
        """
        Convert YOLO detections into the old coordinatelogic.detect_bricks(frame)
        output format used by Vision_Scanner.py.

        Old format:
        [
            {
                "x": robot_x,
                "y": robot_y,
                "angle": angle,
                "color": color
            }
        ]
        """
        bricks = []

        for det in detections:
            angle = det.get("angle")

            if angle is None:
                angle = 90.0

            bricks.append({
                "x": float(det["robot_x"]),
                "y": float(det["robot_y"]),
                "angle": float(angle),
                "color": "YOLO",
            })

        return bricks


    def detect_bricks_yolo(frame):
        """
        Bridge function that makes YOLO behave like the old detect_bricks(frame).

        This allows future Vision_Scanner.py integration without changing the
        expected brick data format.
        """
        detections = detect_objects(frame)
        return to_vision_scanner_format(detections)
        
def to_vision_scanner_format(detections):
    bricks = []

    for det in detections:
        angle = det.get("angle")

        if angle is None:
            angle = 90.0

        bricks.append({
            "x": float(det["robot_x"]),
            "y": float(det["robot_y"]),
            "angle": float(angle),
            "color": "YOLO",
        })

    return bricks


def detect_bricks_yolo(frame):
    detections = detect_objects(frame)
    return to_vision_scanner_format(detections)