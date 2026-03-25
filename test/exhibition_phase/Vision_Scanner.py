# Vision_Scanner_updated.py
# ------------------------------------------------------------
# Scan → pick 1 → drop → rescan (repeat)
# Blue / Black bucket drop now uses direct servo positions
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
RELEASE_SETTLE = 0.35
SERVO_STEP_DELAY = 0.25

CAM_INDEX = 0
WARMUP_FRAMES = 5

# =========================
# DIRECT SERVO DROP POSES
# Final tracked positions from your manual testing:
# BLUE  = s1=340, s2=550, s3=159, s4=260
# BLACK = s1=230, s2=575, s3=200, s4=130
# =========================
BLUE_DROP_SEQUENCE = [
    (4, 250),
    (1, 350),
    (2, 530),
    (2, 550),
    (1, 345),
    (1, 340),
    (4, 260),
]

BLACK_DROP_SEQUENCE = [
    (4, 250),
    (1, 350),
    (2, 530),
    (2, 550),
    (1, 345),
    (1, 340),
    (4, 260),
    (1, 150),
    (1, 200),
    (1, 215),
    (3, 170),
    (3, 200),
    (4, 100),
    (4, 150),
    (2, 575),
    (4, 60),
    (4, 90),
    (4, 130),
    (1, 230),
]


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


def run_servo_sequence(sequence, label):
    stage(label, "• Using direct Arm.moveJetArm() drop sequence")
    for servo_id, position in sequence:
        print(f"• Servo {servo_id} -> {position}")
        servo_move_wait(servo_id, position)


def go_to_blue_box_servo():
    stage("🪣 TO BLUE BUCKET", "• Using manual servo path")
    if not move_wait(5, 8, 12, "📍 BLUE STAGING POSE"):
        return False
    run_servo_sequence(BLUE_DROP_SEQUENCE, "🔵 BLUE DROP SEQUENCE")
    return True


def go_to_black_box_servo():
    stage("🪣 TO BLACK BUCKET", "• Using manual servo path")
    if not move_wait(5, 8, 12, "📍 BLACK STAGING POSE"):
        return False
    run_servo_sequence(BLACK_DROP_SEQUENCE, "⚫ BLACK DROP SEQUENCE")
    return True


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
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return True

    if color in ("BLACK", "NEUTRAL"):
        ok = go_to_black_box_servo()
        if not ok:
            print("⚠️ Black bucket path failed — releasing for safety")
            gripper.open_gripper()
            time.sleep(RELEASE_SETTLE)
            return False

        print("🖐️ RELEASE      • opening gripper")
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
