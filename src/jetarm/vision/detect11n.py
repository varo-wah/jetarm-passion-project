# ============================================================
# YOLO11n Live Detection Test for JetArm
# ------------------------------------------------------------
# Purpose:
# 1. Load YOLO11n from models/pretrained/
# 2. Read live camera frames
# 3. Run YOLO detection
# 4. Extract bounding box center points
# 5. Convert pixel center into robot coordinates
# 6. Display detection + robot coordinate overlay
#
# IMPORTANT:
# This file does NOT move the robot.
# It is only for YOLO perception/debug testing.
# ============================================================

from ultralytics import YOLO
import cv2
import time

from jetarm.config.yolo_config import YOLO11N_MODEL_PATH
from jetarm.vision.coordinatelogic import pixel_to_robot


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "yolo11n.pt"
IMAGE_SIZE = 416
CAMERA_INDEX = 0
FRAME_DELAY = 0.03  # ~30 FPS cap; helps reduce CPU/RAM pressure


# ============================================================
# MODEL LOADING
# ------------------------------------------------------------
# Loads:
#   models/pretrained/yolo11n.pt
# ============================================================

def load_model():
    model_path = YOLO11N_MODEL_PATH

    if model_path.exists():
        print(f"Loading model from: {model_path}")
        return YOLO(str(model_path))

    print(f"Model not found at: {model_path}")
    print(f"Place {MODEL_NAME} inside models/pretrained/")
    return None


# ============================================================
# DETECTION EXTRACTION
# ------------------------------------------------------------
# Converts YOLO raw boxes into dictionaries containing:
# - class_id
# - confidence
# - pixel center
# - robot coordinates
# - bounding box corners
# ============================================================

def extract_detections(result):
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # YOLO box format: x1, y1, x2, y2
        # x1, y1 = top-left corner
        # x2, y2 = bottom-right corner
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # Center point in image pixel coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Convert pixel center into robot coordinates using existing calibration
        robot_x, robot_y = pixel_to_robot(center_x, center_y)

        detections.append({
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
        })

    return detections


# ============================================================
# DRAW DEBUG OVERLAY
# ------------------------------------------------------------
# Adds:
# - yellow dot at center of each YOLO box
# - pixel coordinate text
# - robot coordinate text
# ============================================================

def draw_extra_info(frame, detections):
    for det in detections:
        cx = det["center_x"]
        cy = det["center_y"]

        # Draw center point
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 255),
            -1
        )

        # Coordinate text shown above detection box
        text = f"px=({cx},{cy}) robot=({det['robot_x']},{det['robot_y']})"

        cv2.putText(
            frame,
            text,
            (det["x1"], max(20, det["y1"] - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return frame


# ============================================================
# MAIN LOOP
# ------------------------------------------------------------
# 1. Load YOLO11n
# 2. Open camera
# 3. Run live inference
# 4. Draw YOLO + coordinate overlays
# 5. Print detections in terminal
#
# Press q to quit.
# ============================================================

def main():
    model = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Reduce camera buffering / stale frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Starting YOLO11n live detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera")
            break

        # Run YOLO inference at smaller image size for JetArm stability
        results = model(frame, imgsz=IMAGE_SIZE, verbose=False)
        result = results[0]

        # Convert YOLO result into structured detection dictionaries
        detections = extract_detections(result)

        # YOLO default annotation: box + class + confidence
        annotated_frame = result.plot()

        # Custom overlay: center point + robot coordinate
        annotated_frame = draw_extra_info(annotated_frame, detections)

        cv2.imshow("YOLO11n Detection", annotated_frame)

        if detections:
            print(detections)

        # FPS limit to reduce system load
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
