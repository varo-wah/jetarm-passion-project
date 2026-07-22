"""
Move JetArm to scan pose, capture one frame, and run the active YOLO detector.

This script does not pick, place, or move to detected objects.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jetarm.hardware.Class_Execution import camera  # noqa: E402
from jetarm.vision.yolo_detector import detect_bricks_yolo, detect_objects, detect_target  # noqa: E402


CAM_INDEX = 0
WARMUP_FRAMES = 5


def take_snapshot():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Camera failed to capture frame")

    return frame


def main():
    print("\n[TEST] Moving to scan position...")
    camera.scan_position()
    time.sleep(0.8)

    print("[TEST] Capturing frame...")
    frame = take_snapshot()

    print("[TEST] Running YOLO detector...")
    detections = detect_objects(frame)
    target = detect_target(frame)
    bricks = detect_bricks_yolo(frame)

    print("\nALL YOLO DETECTIONS:")
    for det in detections:
        print(det)

    print("\nCHOSEN TARGET:")
    print(target)

    print("\nVISION_SCANNER FORMAT:")
    for brick in bricks:
        print(brick)

    print("\n[TEST COMPLETE] No pick/drop executed.")


if __name__ == "__main__":
    main()
