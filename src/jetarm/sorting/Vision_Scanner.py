# Vision_Scanner.py
# ------------------------------------------------------------
# Scan → pick 1 → drop → rescan (repeat)
# ------------------------------------------------------------

import cv2
import time
import os
import numpy as np
import requests

from jetarm.hardware.Class_Execution import ik, gripper, camera
from jetarm.vision.yolo_detector import detect_bricks_yolo as detect_bricks
from jetarm.vision.wrist_safety import choose_safe_wrist_angle

ACTUATION_ENV = "JETARM_ENABLE_ACTUATION"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_ACTUATION = env_flag(ACTUATION_ENV, default=False)

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

# Cluster separation assist. When enabled, the scanner separates tight brick
# clusters before attempting a full pick/drop.
ENABLE_CLUSTER_SEPARATION = True
CLUSTER_DISTANCE_CM = 4.0
MAX_CLUSTER_SEPARATIONS = 8
SEPARATION_PUSH_CM = 3.0
SEPARATION_NUDGE_Z = PICK_Z + 1.5
SEPARATION_GRIP_SETTLE = 0.35
SEPARATION_X_LIMITS = (-20.0, 20.0)
SEPARATION_Y_LIMITS = (5.0, 28.0)

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
    """
    If UI server is running, fetch latest frame from it.
    Falls back to direct camera capture if UI is not reachable.
    """
    # 1) Try UI server first (preferred)
    try:
        r = requests.get(FRAME_URL, timeout=1.0)
        if r.status_code == 200:
            data = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("UI returned invalid JPEG")
            return frame
    except Exception:
        pass  # fallback below

    # 2) Fallback: direct camera (use only if UI is off)
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
    for brick in bricks:
        brick["frame_shape"] = frame.shape

    print_bricks(bricks)
    return bricks


def choose_brick(bricks):
    # Closest to origin tends to be most reachable
    return min(bricks, key=lambda b: (b["x"] ** 2 + b["y"] ** 2))


def closest_neighbor(brick, bricks):
    neighbors = []
    for other in bricks:
        if other is brick:
            continue

        dx = float(brick["x"]) - float(other["x"])
        dy = float(brick["y"]) - float(other["y"])
        distance = float(np.hypot(dx, dy))
        neighbors.append((distance, other))

    if not neighbors:
        return None, None

    return min(neighbors, key=lambda item: item[0])


def clamped_separation_point(brick, neighbor):
    dx = float(brick["x"]) - float(neighbor["x"])
    dy = float(brick["y"]) - float(neighbor["y"])
    length = float(np.hypot(dx, dy))

    if length < 0.01:
        dx = float(brick["x"])
        dy = float(brick["y"])
        length = float(np.hypot(dx, dy))

    if length < 0.01:
        dx, dy, length = 1.0, 0.0, 1.0

    ux = dx / length
    uy = dy / length

    push_x = float(brick["x"]) + ux * SEPARATION_PUSH_CM
    push_y = float(brick["y"]) + uy * SEPARATION_PUSH_CM

    push_x = max(SEPARATION_X_LIMITS[0], min(SEPARATION_X_LIMITS[1], push_x))
    push_y = max(SEPARATION_Y_LIMITS[0], min(SEPARATION_Y_LIMITS[1], push_y))

    return push_x, push_y


def separate_close_cluster(brick, neighbor, distance):
    x = float(brick["x"])
    y = float(brick["y"])
    detected_angle = float(brick.get("angle", 90.0))
    angle, edge_status = choose_safe_wrist_angle(brick, brick.get("frame_shape"))
    push_x, push_y = clamped_separation_point(brick, neighbor)

    if float(np.hypot(push_x - x, push_y - y)) < 0.5:
        print("[SEPARATION] Push vector too small after workspace clamp; skipping assist")
        return False

    stage(
        "[SEPARATION] CLOSE BRICKS DETECTED",
        (
            f"• target=({x:.2f}, {y:.2f}) "
            f"neighbor=({neighbor['x']:.2f}, {neighbor['y']:.2f}) "
            f"distance={distance:.2f}cm push_to=({push_x:.2f}, {push_y:.2f})"
        ),
    )

    if not move_wait(x, y, APPROACH_Z, "[SEPARATION] APPROACH TARGET"):
        return False

    print(
        "[SEPARATION] ALIGN WRIST  • "
        f"detected={detected_angle:.1f}° final={angle:.1f}° edge={edge_status}"
    )
    gripper.turn_wrist(angle)
    time.sleep(WRIST_SETTLE)

    print("[SEPARATION] CLOSE GRIPPER AS PUSH FINGER")
    gripper.close_gripper()
    time.sleep(SEPARATION_GRIP_SETTLE)

    if not move_wait(x, y, SEPARATION_NUDGE_Z, "[SEPARATION] LOWER TO NUDGE HEIGHT"):
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    if not move_wait(push_x, push_y, SEPARATION_NUDGE_Z, "[SEPARATION] PUSH AWAY FROM NEIGHBOR"):
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    if not move_wait(push_x, push_y, APPROACH_Z, "[SEPARATION] LIFT AFTER PUSH"):
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    print("[SEPARATION] OPEN GRIPPER AND RESCAN")
    gripper.open_gripper()
    time.sleep(RELEASE_SETTLE)
    camera.scan_position()
    time.sleep(SCAN_SETTLE)
    return True


def should_separate_before_pick(brick, bricks):
    if not ENABLE_CLUSTER_SEPARATION:
        return False, None, None

    distance, neighbor = closest_neighbor(brick, bricks)
    if neighbor is None or distance is None:
        return False, None, None

    return distance <= CLUSTER_DISTANCE_CM, neighbor, distance


# =========================
# PICK + DROP (ONE BRICK)
# =========================
def pick_and_drop(brick):
    x = brick["x"]
    y = brick["y"]
    detected_angle = brick["angle"]
    angle, edge_status = choose_safe_wrist_angle(brick, brick.get("frame_shape"))
    bx, by = bucket_for_color(brick.get("color"))

    stage("🎯 SELECTED BRICK",
          f"• x={x:.2f}, y={y:.2f}, angle={detected_angle:.1f}, color={brick['color']}")

    # 1) Approach above brick
    if not move_wait(x, y, APPROACH_Z, "🚀 APPROACHING"):
        return False

    # 2) Wrist align
    print(f"🧭 ALIGN WRIST  • detected={detected_angle:.1f}° final={angle:.1f}° edge={edge_status}")
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
    ik.move_to(0, 13, 14)
    time.sleep(GRIP_SETTLE)

    # 7) Move to bucket (approach)
    if not move_wait(bx, by, APPROACH_BUCKET, f"🪣 TO {brick.get('color', 'NEUTRAL')} BUCKET"):
        print("⚠️ Bucket approach unreachable — releasing for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    # 9) Release + lift off
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
    if not ENABLE_ACTUATION:
        stage(
            "SCANNER ACTUATION BLOCKED",
            f"Set {ACTUATION_ENV}=1 before launch to explicitly enable robot motion.",
        )
        return

    picked = 0
    separation_attempts = 0

    while picked < MAX_PICKS:
        bricks = scan_once()

        if not bricks:
            stage("✅ DONE", "• No bricks detected")
            break

        brick = choose_brick(bricks)

        should_separate, neighbor, distance = should_separate_before_pick(brick, bricks)
        if should_separate:
            if separation_attempts >= MAX_CLUSTER_SEPARATIONS:
                stage(
                    "[SEPARATION] LIMIT REACHED",
                    "• Continuing with normal pick/drop to avoid endless separation loops",
                )
            else:
                ok = separate_close_cluster(brick, neighbor, distance)
                if ok:
                    separation_attempts += 1
                    print(f"[SEPARATION] Completed assist count: {separation_attempts}")
                    continue
                print("[SEPARATION] Assist failed or skipped; attempting normal pick/drop")

        ok = pick_and_drop(brick)
        if ok:
            picked += 1
            separation_attempts = 0
            print(f"✅ Picked count: {picked}")
        else:
            print("⏭ No pick this cycle (rescan next)")

    stage("🏁 FINISHED", f"• Total picked: {picked}")


if __name__ == "__main__":
    main()
