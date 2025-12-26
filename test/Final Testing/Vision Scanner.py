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
    if not ik.move_to(x, y, APPROACH_Z):
        print("⏭ Skipping target (unreachable)")
        return
    print("\n" + "-"*52)
    print(f"🚀 APPROACHING  •  x={x:.1f}  y={y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Moving to safe height above the brick")
    print("-"*52)


    # Align wrist
    gripper.turn_wrist(angle)
    print(f"🧭 ALIGN WRIST  •  target angle={angle:.1f}°")

    time.sleep(0.5)

    # Pick
    if not ik.move_to(x, y, PICK_Z):
        print("⏭ Skipping target (unreachable)")
        return
    print("\n" + "-"*52)
    print(f"⬇️ GOING DOWN   •  x={x:.1f}  y={y:.1f}  z={PICK_Z:.1f}")
    print("   • Descending to grasp height")
    print("-"*52)
    print("-"*52)
    print("-"*52)

    time.sleep(1)

    gripper.close_gripper()
    print("✊ GRIP         •  closing gripper")

    time.sleep(2)

    if not ik.move_to(x, y, APPROACH_Z):
        print("⏭ Skipping target (unreachable)")
        return
    print("\n" + "-"*52)
    print(f"⬆️ LIFTING UP   •  x={x:.1f}  y={y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Lifting brick to safe travel height")
    print("-"*52)
    print("-"*52)
    print("-"*52)
    time.sleep(0.5)

    # Drop in bucket
    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)
    print("\n" + "-"*52)
    print(f"🪣 TO BUCKET    •  x={BUCKET_X:.1f}  y={BUCKET_Y:.1f}  z={APPROACH_Z:.1f}")
    print("   • Traveling to drop zone")
    print("-"*52)
    print("-"*52)
    print("-"*52)
    time.sleep(0.5)

    ik.move_to(BUCKET_X, BUCKET_Y, DROP_Z)
    time.sleep(0.5)

    print(f"⬇️ DROP DOWN    •  z={DROP_Z:.1f}")
    print("🖐️ RELEASE      •  opening gripper")

    gripper.open_gripper()
    time.sleep(0.5)

    ik.move_to(BUCKET_X, BUCKET_Y, APPROACH_Z)
    time.sleep(0.5)

def scan_once():
    camera.scan_position()
    time.sleep(0.6)  # let arm + camera settle before snapshot

    frame = take_snapshot()
    bricks = detect_bricks(frame)

    print("\n" + "="*52)
    print(f"📷 Scan complete: {len(bricks)} brick(s) detected")
    for i, b in enumerate(bricks, 1):
        print(f"  [{i}] x={b['x']:.1f}, y={b['y']:.1f}, angle={b['angle']:.1f}, color={b['color']}")
    print("="*52)

    return bricks

def choose_brick(bricks):
    # Pick the closest brick to the robot origin (usually safest/reachable)
    return min(bricks, key=lambda b: (b["x"]**2 + b["y"]**2))


def main():
    picked = 0
    MAX_PICKS = 50  # safety limit so it doesn't run forever

    while picked < MAX_PICKS:
        bricks = scan_once()

        if not bricks:
            print("\n✅ No bricks detected. Done.")
            break

        brick = choose_brick(bricks)
        print(f"\n🎯 Selected brick: x={brick['x']:.1f}, y={brick['y']:.1f}, angle={brick['angle']:.1f}, color={brick['color']}")

        pick_and_drop(brick)
        picked += 1

    print(f"\n🏁 Finished. Picked {picked} brick(s).")


if __name__ == "__main__":
    main()
