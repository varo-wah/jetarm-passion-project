# Vision_Scanner_updated.py
# ------------------------------------------------------------
# Scan → pick 1 → drop → rescan (repeat)
# Blue / Neutral bucket drop now uses final direct servo poses
# instead of inverse kinematics.
# ------------------------------------------------------------

import cv2
import time
from coordinatelogic import detect_bricks
from Class_Execution import ik, gripper, camera, Arm
import os
import numpy as np
import requests

UI_SERVER = os.environ.get("UI_SERVER", "http://127.0.0.1:8000")
FRAME_URL = f"{UI_SERVER}/api/frame.jpg"

# =========================
# SETTINGS (EDIT THESE)
# =========================
NEUTRAL_BUCKET_X, NEUTRAL_BUCKET_Y = 15, -8
RED_BUCKET_X, RED_BUCKET_Y = -15, -8
BLUE_BUCKET_X, BLUE_BUCKET_Y = 15, 3
GREEN_BUCKET_X, GREEN_BUCKET_Y = -15, 3
APPROACH_Z = 7.0
APPROACH_BUCKET = 13.0
PICK_Z = 3.0
DROP_Z = 10.0

MAX_PICKS = 50

# Timing (important: prevents command spam / "glitching")
MOVE_TIME = 1.0
SETTLE_TIME = 0.15
SCAN_SETTLE = 0.6
WRIST_SETTLE = 0.5
GRIP_SETTLE = 1.00
RELEASE_SETTLE = 1.00
SERVO_STEP_DELAY = 0.25

CAM_INDEX = 0
WARMUP_FRAMES = 5

# =========================
# DIRECT SERVO DROP POSES
# Final tracked positions from your manual testing:
# BLUE    = s1=310, s2=415, s4=150
# NEUTRAL = s1=220, s2=430, s3=350, s4=60
# Note: blue keeps servo 3 at the home position.
# =========================
BLUE_DROP_POSE = {
    1: 310,
    2: 415,
    3: 350,
    4: 170,
}

NEUTRAL_DROP_POSE = {
    1: 220,
    2: 430,
    3: 350,
    4: 120,
}


# =========================
# PRETTY PRINT HELPERS
# =========================
def bar(char="-", n=52):
    print(char * n)


def stage(title, detail=""):
    print("\n" + "-" * 52)
    print(title)
    if detail:
        print(detail)
    print("-" * 52)


def print_bricks(bricks):
    print("\n" + "=" * 52)
    print(f"📷 Scan complete: {len(bricks)} brick(s) detected")
    for i, b in enumerate(bricks, 1):
        print(f"  [{i}] x={b['x']:.2f}, y={b['y']:.2f}, angle={b['angle']:.1f}, color={b['color']}")
    print("=" * 52)


# =========================
# CAMERA
# =========================
def take_snapshot():
    """
    If UI server is running, fetch latest frame from it.
    Falls back to direct camera capture if UI is not reachable.
    """
    try:
        r = requests.get(FRAME_URL, timeout=1.0)
        if r.status_code == 200:
            data = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("UI returned invalid JPEG")
            return frame
    except Exception:
        pass

    cap = cv2.VideoCapture(CAM_INDEX)
    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Camera failed to capture frame")

    return frame


# =========================
# COLOR DETECTION
# =========================
def bucket_for_color(color):
    color = (color or "NEUTRAL").upper()

    if color == "RED":
        return RED_BUCKET_X, RED_BUCKET_Y
    if color == "GREEN":
        return GREEN_BUCKET_X, GREEN_BUCKET_Y
    if color == "BLUE":
        return BLUE_BUCKET_X, BLUE_BUCKET_Y

    return NEUTRAL_BUCKET_X, NEUTRAL_BUCKET_Y


# =========================
# MOTION WRAPPER
# =========================
def move_wait(x, y, z, label):
    stage(label, f"• Target: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    ok = ik.move_to(x, y, z)
    time.sleep(MOVE_TIME + SETTLE_TIME)
    if not ok:
        print("⏭ Skipping (unreachable / joint limit)")
    return ok


def servo_move_wait(servo_id, position, delay=SERVO_STEP_DELAY):
    Arm.moveJetArm(servo_id, position)
    time.sleep(delay)


def run_drop_pose(pose, label):
    stage(label, "• Using final direct Arm.moveJetArm() pose")
    for servo_id in (4, 3, 2, 1):
        if servo_id not in pose:
            continue
        position = pose[servo_id]
        print(f"• Servo {servo_id} -> {position}")
        servo_move_wait(servo_id, position)
    return True

def go_to_blue_box_servo():
    return run_drop_pose(BLUE_DROP_POSE, "🪣 TO BLUE BUCKET")


def go_to_neutral_box_servo():
    return run_drop_pose(NEUTRAL_DROP_POSE, "🪣 TO NEUTRAL BUCKET")


# =========================
# SCAN (ONE CYCLE)
# =========================
def scan_once():
    stage("🔎 SCANNING POSITION", "• Moving arm to scan pose")
    camera.scan_position()
    time.sleep(SCAN_SETTLE)

    frame = take_snapshot()
    bricks = detect_bricks(frame)

    print_bricks(bricks)
    return bricks



def choose_brick(bricks):
    return min(bricks, key=lambda b: (b["x"] ** 2 + b["y"] ** 2))


# =========================
# PICK + DROP (ONE BRICK)
# =========================
def pick_and_drop(brick):
    x = brick["x"]
    y = brick["y"]
    angle = brick["angle"]
    color = (brick.get("color") or "NEUTRAL").upper()
    bx, by = bucket_for_color(color)

    stage("🎯 SELECTED BRICK",
          f"• x={x:.2f}, y={y:.2f}, angle={angle:.1f}, color={color}")

    if not move_wait(x, y, APPROACH_Z, "🚀 APPROACHING"):
        return False

    print(f"🧭 ALIGN WRIST  • target angle={angle:.1f}°")
    gripper.turn_wrist(angle)
    time.sleep(WRIST_SETTLE)

    if not move_wait(x, y, PICK_Z, "⬇️ GOING DOWN"):
        return False

    print("✊ GRIP         • closing gripper")
    gripper.close_gripper()
    time.sleep(GRIP_SETTLE)

    if not move_wait(x, y, APPROACH_Z, "⬆️ LIFTING UP"):
        print("⚠️ Lift failed after grip — opening gripper for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    if not move_wait(0, 13, 14, "🔄 RETURNING TO CENTER"):
        print("⚠️ Could not return to center — opening gripper for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    if color == "BLUE":
        ok = go_to_blue_box_servo()
        if not ok:
            print("⚠️ Blue bucket path failed — releasing for safety")
            gripper.open_gripper()
            time.sleep(RELEASE_SETTLE)
            return False

        print("🖐️ RELEASE      • opening gripper")
        time.sleep(RELEASE_SETTLE)
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return True

    if color in ("BLACK", "NEUTRAL"):
        ok = go_to_neutral_box_servo()
        if not ok:
            print("⚠️ Neutral bucket path failed — releasing for safety")
            time.sleep(RELEASE_SETTLE)
            gripper.open_gripper()
            time.sleep(RELEASE_SETTLE)
            return False
        print("🖐️ RELEASE      • opening gripper")
        time.sleep(RELEASE_SETTLE)
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return True

    if not move_wait(bx, by, APPROACH_BUCKET, f"🪣 TO {color} BUCKET"):
        print("⚠️ Bucket approach unreachable — releasing for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    print("🖐️ RELEASE      • opening gripper")
    gripper.open_gripper()
    time.sleep(RELEASE_SETTLE)

    ik.move_to(bx, by, APPROACH_BUCKET)
    time.sleep(MOVE_TIME + SETTLE_TIME)

    return True


# =========================
# MAIN LOOP (RESCAN EACH PICK)
# =========================
def main():
    picked = 0

    while picked < MAX_PICKS:
        bricks = scan_once()

        if not bricks:
            stage("✅ DONE", "• No bricks detected")
            break

        brick = choose_brick(bricks)

        ok = pick_and_drop(brick)
        if ok:
            picked += 1
            print(f"✅ Picked count: {picked}")
        else:
            print("⏭ No pick this cycle (rescan next)")

    stage("🏁 FINISHED", f"• Total picked: {picked}")


if __name__ == "__main__":
    main()
