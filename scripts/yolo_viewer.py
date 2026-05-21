"""
Live YOLO viewer for JetArm LEGO detection.

This script opens the camera, runs the active YOLO detector, and displays
robot coordinates, angle, confidence, and a target marker. It does not move
the robot or import hardware control modules.
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


CAMERA_INDEX = 0
FRAME_DELAY = 0.03
PRINT_INTERVAL_S = 0.25
COLOR_LABEL = "YOLO"


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

    box_color = (0, 0, 255) if is_target else (0, 255, 0)
    text_color = (0, 0, 255) if is_target else (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

    if is_target:
        cv2.circle(frame, (center_x, center_y), 14, (0, 0, 255), 2)

    text_x = x1
    text_y = max(20, y1 - 54)

    labels = [
        f"{COLOR_LABEL} conf={confidence}",
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
            f"color={COLOR_LABEL}"
        )

    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
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
            f"color={COLOR_LABEL}, "
            f"confidence={detection['confidence']:.2f}"
        )


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

    print("Starting JetArm YOLO viewer...")
    print("Press q or Esc to quit.")

    last_print_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        detections = detect_objects(frame)
        target = choose_target(detections)
        annotated_frame = draw_detections(frame.copy(), detections, target)

        now = time.monotonic()
        if now - last_print_time >= PRINT_INTERVAL_S:
            print_detections(detections)
            last_print_time = now

        cv2.imshow("JetArm YOLO Viewer", annotated_frame)
        time.sleep(FRAME_DELAY)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
