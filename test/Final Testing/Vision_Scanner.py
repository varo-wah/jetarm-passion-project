# Vision_Scanner.py
# ------------------------------------------------------------
# Scan → pick 1 → drop → rescan (repeat)
# ------------------------------------------------------------

import cv2
import time
from coordinatelogic import detect_bricks
from Class_Execution import ik, gripper, camera

# =========================
# SETTINGS (EDIT THESE)
# =========================
NEUTRAL_BUCKET_X, NEUTRAL_BUCKET_Y = 20, 0
RED_BUCKET_X, RED_BUCKET_Y = 20, -10
BLUE_BUCKET_X, BLUE_BUCKET_Y = -20, -10
GREEN_BUCKET_X, GREEN_BUCKET_Y = -20, 0
APPROACH_Z = 15
APPROACH_BUCKET = 18
PICK_Z = 10
DROP_Z = 15

MAX_PICKS = 50

# Timing (important: prevents command spam / "glitching")
# Tune MOVE_TIME to match your servo motion duration (often ~1.0s).
MOVE_TIME = 1.0
SETTLE_TIME = 0.15
SCAN_SETTLE = 0.6
WRIST_SETTLE = 0.5
GRIP_SETTLE = 1.00
RELEASE_SETTLE = 0.35

CAM_INDEX = 0
WARMUP_FRAMES = 5


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
    cap = cv2.VideoCapture(CAM_INDEX)

    # Warm up exposure/autofocus
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

    # fallback
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
    # Closest to origin tends to be most reachable
    return min(bricks, key=lambda b: (b["x"] ** 2 + b["y"] ** 2))


# =========================
# PICK + DROP (ONE BRICK)
# =========================
def pick_and_drop(brick):
    x = brick["x"]
    y = brick["y"]
    angle = brick["angle"]
    bx, by = bucket_for_color(brick.get("color"))

    stage("🎯 SELECTED BRICK",
          f"• x={x:.2f}, y={y:.2f}, angle={angle:.1f}, color={brick['color']}")

    # 1) Approach above brick
    if not move_wait(x, y, APPROACH_Z, "🚀 APPROACHING"):
        return False

    # 2) Wrist align
    print(f"🧭 ALIGN WRIST  • target angle={angle:.1f}°")
    gripper.turn_wrist(angle)
    time.sleep(WRIST_SETTLE)

    # 3) Go down to pick
    if not move_wait(x, y, PICK_Z, "⬇️ GOING DOWN"):
        return False

    # 4) Grip
    print("✊ GRIP         • closing gripper")
    gripper.close_gripper()
    time.sleep(GRIP_SETTLE)

    # 5) Lift back up
    if not move_wait(x, y, APPROACH_Z, "⬆️ LIFTING UP"):
        print("⚠️ Lift failed after grip — opening gripper for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    # EXTRA) GO TO SCAN POSITION
    ik.move_to(0, 15, 23)
    time.sleep(GRIP_SETTLE)

    # 7) Move to bucket (approach)
    if not move_wait(bx, by, APPROACH_BUCKET, f"🪣 TO {brick.get('color', 'NEUTRAL')} BUCKET"):
        print("⚠️ Bucket approach unreachable — releasing for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    # 8) Drop down
    if not move_wait(bx, by, DROP_Z, "⬇️ DROP DOWN"):
        print("⚠️ Bucket drop unreachable — releasing for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    # 9) Release + lift off
    print("🖐️ RELEASE      • opening gripper")
    gripper.open_gripper()
    time.sleep(RELEASE_SETTLE)

    ik.move_to(bx, by, APPROACH_Z)
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
