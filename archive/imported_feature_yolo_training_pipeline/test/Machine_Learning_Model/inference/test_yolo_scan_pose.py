# ============================================================
# YOLO Scan-Pose Snapshot Test
# ------------------------------------------------------------
# Purpose:
# 1. Move JetArm camera to the same scan pose used by the old system
# 2. Capture one frame
# 3. Run YOLO detector
# 4. Print Vision_Scanner-compatible output
#
# IMPORTANT:
# This does NOT pick anything.
# This does NOT move to the object.
# This only tests YOLO from the real scan pose.
# ============================================================

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXHIBITION_PHASE = PROJECT_ROOT / "exhibition_phase"

sys.path.append(str(EXHIBITION_PHASE))

from Class_Execution import camera
from yolo_detector import detect_objects, detect_target, detect_bricks_yolo


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAM_INDEX = 0
WARMUP_FRAMES = 5


# ============================================================
# SNAPSHOT FUNCTION
# ------------------------------------------------------------
# Same idea as your old Vision_Scanner.py:
# Try direct camera capture.
# ============================================================

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


# ============================================================
# MAIN TEST
# ============================================================

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