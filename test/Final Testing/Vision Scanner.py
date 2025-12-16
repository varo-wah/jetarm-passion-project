# sorter.py
import cv2
import time
from coordinatelogic import detect_bricks
from Class_Execution import ik, gripper, camera   # adjust import to your actual file/object names

BUCKET_X, BUCKET_Y, BUCKET_Z = 15, 0, 10

APPROACH_Z = 15
PICK_Z = 10.25
DROP_Z = 15

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
    print("\n" + "-"*52)
    print(f"🚀 APPROACHING  •  x={x:.1f}  y={y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Moving to safe height above the brick")
    print("-"*52)


    # Align wrist
    gripper.turn_wrist(angle)
    print(f"🧭 ALIGN WRIST  •  target angle={angle:.1f}°")

    time.sleep(2)

    # Pick
    ik.move_to(x, y, PICK_Z)
    print("\n" + "-"*52)
    print(f"⬇️ GOING DOWN   •  x={x:.1f}  y={y:.1f}  z={PICK_Z:.1f}")
    print("   • Descending to grasp height")
    print("-"*52)
    print("-"*52)
    print("-"*52)

    time.sleep(2)

    gripper.close_gripper()
    print("✊ GRIP         •  closing gripper")

    time.sleep(2)

    ik.move_to(x, y, APPROACH_Z)
    print("\n" + "-"*52)
    print(f"⬆️ LIFTING UP   •  x={x:.1f}  y={y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Lifting brick to safe travel height")
    print("-"*52)
    print("-"*52)
    print("-"*52)
    time.sleep(2)

    # Drop in bucket
    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)
    print("\n" + "-"*52)
    print(f"🪣 TO BUCKET    •  x={BUCKET_X:.1f}  y={BUCKET_Y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Traveling to drop zone")
    print("-"*52)
    print("-"*52)
    print("-"*52)
    time.sleep(2)

    ik.move_to(BUCKET_X, BUCKET_Y, DROP_Z)
    time.sleep(2)

    print(f"⬇️ DROP DOWN    •  z={DROP_Z:.1f}")
    print("🖐️ RELEASE      •  opening gripper")

    gripper.open_gripper()
    time.sleep(2)

    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)
    time.sleep(2)

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
        print(f"Moving to brick: {brick}")
        pick_and_drop(brick)

if __name__ == "__main__":
    main()
