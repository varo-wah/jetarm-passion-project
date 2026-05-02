# ============================================================
# YOLO11n Live Detection Test for JetArm
# ------------------------------------------------------------
# Purpose:
# 1. Load YOLO11n from Machine_Learning_Model/models/
# 2. Read live camera frames
# 3. Run YOLO detection
# 4. Extract bounding box center points
# 5. Filter invalid boxes, such as full-table/background boxes
# 6. Convert valid pixel centers into robot coordinates
# 7. Display detection + robot coordinate overlay
#
# IMPORTANT:
# This file does NOT move the robot.
# It is only for YOLO perception/debug testing.
# ============================================================

from ultralytics import YOLO
import cv2
import sys
import time
from pathlib import Path


# ============================================================
# PATH SETUP
# ------------------------------------------------------------
# detect11n.py location:
#   test/Machine_Learning_Model/inference/detect11n.py
#
# PROJECT_ROOT points to:
#   test/
#
# EXHIBITION_PHASE points to:
#   test/exhibition_phase/
#
# This allows us to import:
#   coordinatelogic.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXHIBITION_PHASE = PROJECT_ROOT / "exhibition_phase"

if str(EXHIBITION_PHASE) not in sys.path:
    sys.path.append(str(EXHIBITION_PHASE))

from coordinatelogic import pixel_to_robot


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "yolo11n.pt"
IMAGE_SIZE = 416
CAMERA_INDEX = 0
FRAME_DELAY = 0.03

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

MIN_BOX_AREA_RATIO = 0.002
MAX_BOX_AREA_RATIO = 0.30
BORDER_BOX_AREA_RATIO = 0.20
BORDER_MARGIN_PX = 5


# ============================================================
# MODEL LOADING
# ------------------------------------------------------------
# Loads:
#   test/Machine_Learning_Model/models/yolo11n.pt
# ============================================================

def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME

    if model_path.exists():
        print(f"Loading model from: {model_path}")
        return YOLO(str(model_path))

    print(f"Model not found at: {model_path}")
    print(f"Place {MODEL_NAME} inside Machine_Learning_Model/models/")
    return None


# ============================================================
# BOX VALIDATION FILTER
# ------------------------------------------------------------
# Rejects:
# - tiny noise boxes
# - giant full-frame/table/background boxes
# - large boxes touching the frame border
#
# This is a temporary debug filter.
# The real long-term fix is custom YOLO training.
# ============================================================

def is_valid_box(det, frame_width, frame_height):
    box_w = det["x2"] - det["x1"]
    box_h = det["y2"] - det["y1"]

    if box_w <= 0 or box_h <= 0:
        return False

    box_area = box_w * box_h
    frame_area = frame_width * frame_height

    min_area = frame_area * MIN_BOX_AREA_RATIO
    max_area = frame_area * MAX_BOX_AREA_RATIO

    if box_area < min_area:
        return False

    if box_area > max_area:
        return False

    touches_border = (
        det["x1"] <= BORDER_MARGIN_PX or
        det["y1"] <= BORDER_MARGIN_PX or
        det["x2"] >= frame_width - BORDER_MARGIN_PX or
        det["y2"] >= frame_height - BORDER_MARGIN_PX
    )

    if touches_border and box_area > frame_area * BORDER_BOX_AREA_RATIO:
        return False

    return True


# ============================================================
# DETECTION EXTRACTION
# ------------------------------------------------------------
# Converts YOLO raw boxes into dictionaries containing:
# - class_id
# - confidence
# - pixel center
# - robot coordinates
# - bounding box corners
#
# Then filters out boxes that are probably background/table detections.
# ============================================================

def extract_detections(result, frame_width, frame_height):
    detections = []
    rejected = []

    if result.boxes is None:
        return detections, rejected

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        robot_x, robot_y = pixel_to_robot(center_x, center_y)

        detection = {
            "class_id": cls_id,
            "confidence": round(conf, 2),

            "center_x": center_x,
            "center_y": center_y,

            "robot_x": round(robot_x, 2),
            "robot_y": round(robot_y, 2),

            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        }

        if is_valid_box(detection, frame_width, frame_height):
            detections.append(detection)
        else:
            rejected.append(detection)

    return detections, rejected


# ============================================================
# DRAW VALID DETECTION OVERLAY
# ------------------------------------------------------------
# Adds:
# - green box around valid detection
# - yellow dot at center
# - pixel coordinate text
# - robot coordinate text
# ============================================================

def draw_valid_detections(frame, detections):
    for det in detections:
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        cx = det["center_x"]
        cy = det["center_y"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 255),
            -1
        )

        text = (
            f"ID:{det['class_id']} conf:{det['confidence']} "
            f"px=({cx},{cy}) robot=({det['robot_x']},{det['robot_y']})"
        )

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return frame


# ============================================================
# DRAW REJECTED BOX OVERLAY
# ------------------------------------------------------------
# Optional visual debug:
# - red boxes show detections rejected by the area/border filter
# - useful for verifying that the bad full-table box is being removed
# ============================================================

def draw_rejected_detections(frame, rejected):
    for det in rejected:
        cv2.rectangle(
            frame,
            (det["x1"], det["y1"]),
            (det["x2"], det["y2"]),
            (0, 0, 255),
            1
        )

        cv2.putText(
            frame,
            "REJECTED",
            (det["x1"], max(20, det["y1"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    return frame


# ============================================================
# MAIN LOOP
# ------------------------------------------------------------
# 1. Load YOLO11n
# 2. Open camera
# 3. Run live inference
# 4. Filter invalid boxes
# 5. Draw overlays
# 6. Print valid detections in terminal
#
# Press q to quit.
# ============================================================

def main():
    model = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Starting YOLO11n live detection... Press 'q' to quit.")
    print("Green boxes = valid detections")
    print("Red boxes = rejected detections")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera")
            break

        frame_height, frame_width = frame.shape[:2]

        results = model(frame, imgsz=IMAGE_SIZE, verbose=False)
        result = results[0]

        detections, rejected = extract_detections(
            result,
            frame_width,
            frame_height
        )

        debug_frame = frame.copy()

        debug_frame = draw_valid_detections(debug_frame, detections)
        debug_frame = draw_rejected_detections(debug_frame, rejected)

        status_text = f"valid={len(detections)} rejected={len(rejected)}"

        cv2.putText(
            debug_frame,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("YOLO11n Detection - Filtered", debug_frame)

        if detections:
            print("VALID:", detections)

        if rejected:
            print("REJECTED:", rejected)

        time.sleep(FRAME_DELAY)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    main()