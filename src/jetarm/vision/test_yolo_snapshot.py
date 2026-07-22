"""
Snapshot test for the reusable YOLO detector.

Purpose:
- Capture one camera frame
- Run YOLO detector
- Print detailed detections
- Print selected target
- Print old Vision_Scanner-compatible format

This does NOT:
- Open a debug window
- Move the robot
- Import Class_Execution
"""

import cv2

from jetarm.vision.yolo_detector import detect_bricks_yolo, detect_objects, detect_target


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture frame")
        return

    detections = detect_objects(frame)
    target = detect_target(frame)
    bricks = detect_bricks_yolo(frame)

    print("\nALL DETECTIONS:")
    if detections:
        for det in detections:
            print(det)
    else:
        print("No detections found.")

    print("\nTARGET:")
    print(target)

    print("\nVISION_SCANNER FORMAT:")
    if bricks:
        for brick in bricks:
            print(brick)
    else:
        print("No YOLO bricks found.")


if __name__ == "__main__":
    main()
