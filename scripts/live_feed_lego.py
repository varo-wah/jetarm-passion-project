"""
Live LEGO YOLO feed for JetArm.

This script opens a camera feed, runs the active JetArm YOLO detector,
and displays detections. It does not move the robot.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jetarm.vision.yolo_detector import choose_target, detect_objects, load_model  # noqa: E402


FRAME_DELAY = 0.03
CAMERA_INDEX = 0


def draw_detections(frame, detections, target):
    for det in detections:
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        cx = det["center_x"]
        cy = det["center_y"]

        robot_x = det["robot_x"]
        robot_y = det["robot_y"]
        angle = det["angle"]
        confidence = det["confidence"]

        class_name = "lego_brick"
        is_target = target is not None and det == target

        box_color = (0, 0, 255) if is_target else (0, 255, 0)
        text_color = (0, 0, 255) if is_target else (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label_1 = f"{class_name} conf={confidence}"
        label_2 = f"robot=({robot_x},{robot_y}) angle={angle}"
        label_3 = f"px=({cx},{cy})"

        text_x = x1
        text_y = max(20, y1 - 45)

        cv2.putText(frame, label_1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        cv2.putText(frame, label_2, (text_x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        cv2.putText(frame, label_3, (text_x, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        if is_target:
            cv2.circle(frame, (cx, cy), 14, (0, 0, 255), 2)

    if target:
        label = f"TARGET: lego_brick robot=({target['robot_x']},{target['robot_y']}) angle={target['angle']}"
    else:
        label = "TARGET: none"

    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return frame


def main():
    load_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Starting live LEGO feed...")
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        detections = detect_objects(frame)
        target = choose_target(detections)

        annotated_frame = draw_detections(frame.copy(), detections, target)
        cv2.imshow("Live LEGO YOLO Feed", annotated_frame)

        if detections:
            print("DETECTIONS:", detections)

        time.sleep(FRAME_DELAY)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
