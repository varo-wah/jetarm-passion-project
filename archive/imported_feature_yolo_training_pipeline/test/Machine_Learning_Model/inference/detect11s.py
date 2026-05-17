# ============================================================
# YOLO Live Detection Test for JetArm
# ------------------------------------------------------------
# Purpose of this file:
# 1. Load a YOLO model from Machine_Learning_Model/models/
# 2. Read live camera frames
# 3. Detect objects with YOLO
# 4. Extract useful detection data:
#    - class ID
#    - confidence
#    - pixel center
#    - bounding box corners
#    - robot X/Y coordinates using old calibration
# 5. Choose one "best target" for future robot integration
# 6. Display everything visually for debugging
#
# IMPORTANT:
# This file does NOT move the robot.
# It is only the YOLO perception/debug test file.
# ============================================================

from ultralytics import YOLO
import cv2
import sys
from pathlib import Path


# ============================================================
# PATH SETUP
# ------------------------------------------------------------
# detect.py is located here:
#   test/Machine_Learning_Model/inference/detect.py
#
# PROJECT_ROOT points to:
#   test/
#
# EXHIBITION_PHASE points to:
#   test/exhibition_phase/
#
# We add exhibition_phase to sys.path so Python can import:
#   coordinatelogic.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXHIBITION_PHASE = PROJECT_ROOT / "exhibition_phase"

sys.path.append(str(EXHIBITION_PHASE))

# Import your old coordinate conversion function.
# This converts image pixel coordinates into robot X/Y coordinates.
from coordinatelogic import pixel_to_robot


# ============================================================
# MODEL LOADING
# ------------------------------------------------------------
# Loads the YOLO model from:
#   Machine_Learning_Model/models/yolo11s.pt
#
# Using Path(__file__) avoids path confusion.
# It works even if you run the file from different directories.
# ============================================================

def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "yolo11s.pt"

    if model_path.exists():
        print(f"Loading model from: {model_path}")
        return YOLO(str(model_path))

    print(f"Model not found at: {model_path}")
    print("Download or place yolo11s.pt inside Machine_Learning_Model/models/")
    return None


# ============================================================
# DETECTION EXTRACTION
# ------------------------------------------------------------
# YOLO gives raw detection boxes.
#
# This function converts each YOLO box into a cleaner dictionary:
#   class_id     → YOLO class number
#   confidence   → YOLO confidence score
#   center_x/y   → center of the box in image pixels
#   robot_x/y    → converted robot coordinates
#   x1,y1,x2,y2  → box corners in image pixels
#
# NOTE:
# x1,y1 = top-left corner of box
# x2,y2 = bottom-right corner of box
# ============================================================

def extract_detections(result):
    detections = []

    # result.boxes contains all YOLO detections in the current frame
    for box in result.boxes:
        # Class ID: numerical label from YOLO
        cls_id = int(box.cls[0])

        # Confidence: how sure YOLO is about this detection
        conf = float(box.conf[0])

        # xyxy format gives bounding box corners:
        # x1, y1 = top-left
        # x2, y2 = bottom-right
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # Calculate center of bounding box in pixel coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Convert pixel center into robot coordinates using old calibration
        robot_x, robot_y = pixel_to_robot(center_x, center_y)

        # Store all useful detection information in one dictionary
        detections.append({
            "class_id": cls_id,
            "confidence": round(conf, 2),

            # Pixel-space target
            "center_x": center_x,
            "center_y": center_y,

            # Robot-space target
            "robot_x": round(robot_x, 2),
            "robot_y": round(robot_y, 2),

            # Bounding box corners
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        })

    return detections


# ============================================================
# TARGET SELECTION
# ------------------------------------------------------------
# YOLO can detect multiple objects.
# The robot eventually needs ONE target to act on.
#
# This function:
# 1. Removes weak detections below min_confidence
# 2. Removes detections outside rough reachable robot area
# 3. Chooses the highest-confidence valid target
#
# This does NOT move the robot.
# It only chooses the best detection.
# ============================================================

def choose_target(detections, min_confidence=0.40):
    valid_targets = []

    for det in detections:
        # Ignore weak detections
        if det["confidence"] < min_confidence:
            continue

        robot_x = det["robot_x"]
        robot_y = det["robot_y"]

        # Rough robot reach filter.
        # Tune these based on your real JetArm workspace.
        if robot_x < -20 or robot_x > 20:
            continue

        if robot_y < 5 or robot_y > 28:
            continue

        valid_targets.append(det)

    # If nothing passes the filters, return no target
    if not valid_targets:
        return None

    # Choose highest-confidence valid detection
    return max(valid_targets, key=lambda d: d["confidence"])


# ============================================================
# DRAWING / VISUAL DEBUGGING
# ------------------------------------------------------------
# YOLO's result.plot() draws:
#   - bounding box
#   - class name
#   - confidence
#
# This function adds our custom debugging info:
#   - yellow dot at detection center
#   - pixel coordinates
#   - robot coordinates
#   - red ring around selected target
# ============================================================

def draw_extra_info(frame, detections, target):
    for det in detections:
        cx = det["center_x"]
        cy = det["center_y"]

        # Draw yellow dot at the center of every detected box
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 255),
            -1
        )

        # Text showing pixel and robot coordinates
        text = f"px=({cx},{cy}) robot=({det['robot_x']},{det['robot_y']})"

        # Draw text above each box
        cv2.putText(
            frame,
            text,
            (det["x1"], max(20, det["y1"] - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    # If target selection found a valid target, highlight it
    if target:
        tx = target["center_x"]
        ty = target["center_y"]

        # Red ring means "this is the chosen target"
        cv2.circle(
            frame,
            (tx, ty),
            12,
            (0, 0, 255),
            2
        )

        # Display chosen target robot coordinates at top-left
        cv2.putText(
            frame,
            f"TARGET robot=({target['robot_x']},{target['robot_y']})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return frame


# ============================================================
# MAIN LOOP
# ------------------------------------------------------------
# This runs the live camera test:
# 1. Load YOLO model
# 2. Open camera
# 3. Read frames
# 4. Run YOLO
# 5. Extract detections
# 6. Choose one target
# 7. Draw visual debug info
# 8. Print target in terminal
#
# Press q to quit.
# ============================================================

def main():
    # Load model
    model = load_model()
    if model is None:
        return

    # Open default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Starting YOLO live detection... Press 'q' to quit.")

    while True:
        # Read one camera frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera")
            break

        # Run YOLO on the frame
        # verbose=False keeps terminal output cleaner
        results = model(frame, verbose=False)
        result = results[0]

        # Convert YOLO output into useful dictionaries
        detections = extract_detections(result)

        # Choose one best target from all detections
        target = choose_target(detections)

        # Draw YOLO's default boxes/class labels
        annotated_frame = result.plot()

        # Draw our extra center/robot-coordinate/debug info
        annotated_frame = draw_extra_info(annotated_frame, detections, target)

        # Show video feed
        cv2.imshow("YOLO Detection", annotated_frame)

        # Print chosen target only, not every random detection
        if target:
            print("TARGET:", target)
        elif detections:
            print("Detections found, but no valid target:", detections)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up camera and windows
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# RUN FILE
# ------------------------------------------------------------
# This ensures main() only runs when this file is executed directly:
#   python3 detect.py
#
# If this file is imported later, main() will not auto-run.
# ============================================================

if __name__ == "__main__":
    main()