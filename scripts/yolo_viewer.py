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

from jetarm.ui.yolo_overlay import draw_detections, print_detections  # noqa: E402
from jetarm.vision.yolo_detector import choose_target, detect_objects, load_model  # noqa: E402


CAMERA_INDEX = 0
FRAME_DELAY = 0.03
PRINT_INTERVAL_S = 0.25


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
