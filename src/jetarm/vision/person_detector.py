"""
Person detector for user-friendly follow mode.

Uses the pretrained COCO YOLO model, not the trained LEGO model.
This module never opens a camera and never moves hardware.
"""

from typing import Any, Dict, List, Optional

from ultralytics import YOLO

from jetarm.config.yolo_config import YOLO11N_MODEL_PATH

PERSON_CLASS_ID = 0
MODEL_PATH = YOLO11N_MODEL_PATH

_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Person YOLO model not found: {MODEL_PATH}")

        print(f"[PERSON FOLLOW] Loading model from: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
        print("[PERSON FOLLOW] Model loaded successfully.")

    return _model


def extract_people(frame, result, min_confidence: float = 0.45) -> List[Dict[str, Any]]:
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        if class_id != PERSON_CLASS_ID:
            continue

        confidence = float(box.conf[0].cpu().numpy())
        if confidence < min_confidence:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        area = max(0, x2 - x1) * max(0, y2 - y1)

        detections.append({
            "class_id": class_id,
            "confidence": round(confidence, 2),
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return detections


def detect_people(frame, min_confidence: float = 0.45) -> List[Dict[str, Any]]:
    model = load_model()
    results = model(frame, imgsz=416, classes=[PERSON_CLASS_ID], verbose=False)

    if not results:
        return []

    return extract_people(frame, results[0], min_confidence=min_confidence)


def choose_person(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not detections:
        return None

    return max(detections, key=lambda d: d["area"] * d["confidence"])
