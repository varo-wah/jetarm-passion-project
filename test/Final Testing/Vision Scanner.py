# sorter.py
import cv2
from coordinatelogic import detect_bricks
from Class_Execution import ik, gripper, camera   # adjust import to your actual file/object names

BUCKET_X, BUCKET_Y, BUCKET_Z = -15, -3, 15

APPROACH_Z = 25
PICK_Z = 12
DROP_Z = 5

def take_snapshot():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Camera failed to capture frame")
    return frame

def pick_and_drop(brick):
    x = brick["x"]
    y = brick["y"]
    angle = brick["angle"]

    # Approach
    ik.move_to(x, y, APPROACH_Z)

    # Align wrist
    gripper.turn_wrist(angle)

    # Pick
    ik.move_to(x, y, PICK_Z)
    gripper.close_gripper()
    ik.move_to(x, y, APPROACH_Z)

    # Drop in bucket
    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)
    ik.move_to(BUCKET_X, BUCKET_Y, DROP_Z)
    gripper.open_gripper()
    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)

def main():
    # 1) Scan position
    camera.scan_position()

    # 2) Snapshot
    frame = take_snapshot()

    # 3) Detect bricks
    bricks = detect_bricks(frame)
    print(f"Detected {len(bricks)} bricks:", bricks)

    # 4) Pick all → same bucket
    for brick in bricks:
        pick_and_drop(brick)

if __name__ == "__main__":
    main()
