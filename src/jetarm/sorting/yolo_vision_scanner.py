"""
YOLO-based Vision_Scanner variant for JetArm.

This module preserves the old scan-pick-drop scanner structure, but replaces
contour/color detection with the trained LEGO YOLO detector.
"""

import os
import time

import cv2
import numpy as np
import requests

from jetarm.hardware.Class_Execution import camera, gripper, ik
from jetarm.vision.yolo_detector import detect_bricks_yolo
from jetarm.vision.wrist_safety import choose_safe_wrist_angle


ACTUATION_ENV = "JETARM_ENABLE_ACTUATION"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Preview-only unless the operator explicitly opts in before starting the server.
ENABLE_PICK_AND_DROP = env_flag(ACTUATION_ENV, default=False)

UI_SERVER = os.environ.get("UI_SERVER", "http://127.0.0.1:8000")
FRAME_URL = f"{UI_SERVER}/api/frame.jpg"

# Same bucket/drop geometry as the old Vision_Scanner.
NEUTRAL_BUCKET_X, NEUTRAL_BUCKET_Y = 15, -8
RED_BUCKET_X, RED_BUCKET_Y = -15, -8
BLUE_BUCKET_X, BLUE_BUCKET_Y = 15, 3
GREEN_BUCKET_X, GREEN_BUCKET_Y = -15, 3
APPROACH_Z = 7.0
APPROACH_BUCKET = 13.0
PICK_Z = 3.0

MAX_PICKS = 50

# Same conservative motion timing as the old Vision_Scanner.
MOVE_TIME = 1.0
SETTLE_TIME = 0.15
SCAN_SETTLE = 0.6
WRIST_SETTLE = 0.5
GRIP_SETTLE = 1.00
RELEASE_SETTLE = 0.35

CAM_INDEX = 0
WARMUP_FRAMES = 5


def stage(title, detail=""):
    print("\n" + "-" * 52)
    print(title)
    if detail:
        print(detail)
    print("-" * 52)


def print_yolo_bricks(bricks):
    print(f"[YOLO SCANNER] Detections found: {len(bricks)}")
    for index, brick in enumerate(bricks, 1):
        print(
            "[YOLO SCANNER] "
            f"[{index}] x={brick['x']:.2f}, "
            f"y={brick['y']:.2f}, "
            f"angle={brick['angle']:.1f}, "
            f"color={brick['color']}"
        )


def take_snapshot():
    """
    Use the same camera approach as the old Vision_Scanner:
    prefer the UI server frame, then fall back to direct camera capture.
    """
    try:
        response = requests.get(FRAME_URL, timeout=1.0)
        if response.status_code == 200:
            data = np.frombuffer(response.content, dtype=np.uint8)
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


def bucket_for_color(color):
    color = (color or "NEUTRAL").upper()

    if color == "RED":
        return RED_BUCKET_X, RED_BUCKET_Y
    if color == "GREEN":
        return GREEN_BUCKET_X, GREEN_BUCKET_Y
    if color == "BLUE":
        return BLUE_BUCKET_X, BLUE_BUCKET_Y

    return NEUTRAL_BUCKET_X, NEUTRAL_BUCKET_Y


def move_wait(x, y, z, label):
    stage(label, f"Target: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    ok = ik.move_to(x, y, z)
    time.sleep(MOVE_TIME + SETTLE_TIME)
    if not ok:
        print("[YOLO SCANNER] Skipping: unreachable or joint limit")
    return ok


def scan_once(move_to_scan_pose=None):
    if move_to_scan_pose is None:
        move_to_scan_pose = ENABLE_PICK_AND_DROP

    if move_to_scan_pose:
        print("[YOLO SCANNER] Moving to scan pose")
        if not camera.scan_position():
            raise RuntimeError("Scanner motion was blocked before capture")
        time.sleep(SCAN_SETTLE)
    else:
        print("[YOLO SCANNER] Preview mode: leaving robot position unchanged")

    print("[YOLO SCANNER] Capturing frame")
    frame = take_snapshot()

    print("[YOLO SCANNER] Running YOLO detection")
    bricks = detect_bricks_yolo(frame)
    for brick in bricks:
        brick["frame_shape"] = frame.shape

    print_yolo_bricks(bricks)
    return bricks


def choose_brick(bricks):
    if not bricks:
        return None

    return bricks[0]


def print_selected_target(brick):
    if brick is None:
        print("[YOLO SCANNER] Selected target: none")
        return

    print(
        "[YOLO SCANNER] Selected target: "
        f"x={brick['x']:.2f}, "
        f"y={brick['y']:.2f}, "
        f"angle={brick['angle']:.1f}, "
        f"color={brick['color']}"
    )


def pick_and_drop(brick):
    x = brick["x"]
    y = brick["y"]
    detected_angle = brick["angle"]
    angle, edge_status = choose_safe_wrist_angle(brick, brick.get("frame_shape"))
    bx, by = bucket_for_color(brick.get("color"))

    stage(
        "[YOLO SCANNER] Selected brick",
        f"x={x:.2f}, y={y:.2f}, angle={detected_angle:.1f}, color={brick['color']}",
    )

    if not move_wait(x, y, APPROACH_Z, "[YOLO SCANNER] Approaching"):
        return False

    print(
        "[YOLO SCANNER] Aligning wrist: "
        f"detected={detected_angle:.1f} final={angle:.1f} edge={edge_status}"
    )
    gripper.turn_wrist(angle)
    time.sleep(WRIST_SETTLE)

    if not move_wait(x, y, PICK_Z, "[YOLO SCANNER] Going down"):
        return False

    print("[YOLO SCANNER] Closing gripper")
    gripper.close_gripper()
    time.sleep(GRIP_SETTLE)

    if not move_wait(x, y, APPROACH_Z, "[YOLO SCANNER] Lifting up"):
        print("[YOLO SCANNER] Lift failed after grip; opening gripper for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    ik.move_to(0, 13, 14)
    time.sleep(GRIP_SETTLE)

    if not move_wait(bx, by, APPROACH_BUCKET, f"[YOLO SCANNER] To {brick.get('color', 'NEUTRAL')} bucket"):
        print("[YOLO SCANNER] Bucket approach unreachable; opening gripper for safety")
        gripper.open_gripper()
        time.sleep(RELEASE_SETTLE)
        return False

    print("[YOLO SCANNER] Opening gripper")
    gripper.open_gripper()
    time.sleep(RELEASE_SETTLE)

    ik.move_to(bx, by, APPROACH_BUCKET)
    time.sleep(MOVE_TIME + SETTLE_TIME)

    return True


def main():
    if not ENABLE_PICK_AND_DROP:
        bricks = scan_once(move_to_scan_pose=False)
        target = choose_brick(bricks)
        print_selected_target(target)
        print(
            f"[YOLO SCANNER] Preview complete; actuation disabled. "
            f"Set {ACTUATION_ENV}=1 before server startup to opt in."
        )
        return

    picked = 0

    while picked < MAX_PICKS:
        bricks = scan_once()
        target = choose_brick(bricks)
        print_selected_target(target)

        if target is None:
            stage("[YOLO SCANNER] Done", "No YOLO bricks detected")
            break

        print("[YOLO SCANNER] Executing pick/drop")
        ok = pick_and_drop(target)

        if ok:
            picked += 1
            print(f"[YOLO SCANNER] Picked count: {picked}")
        else:
            print("[YOLO SCANNER] No pick this cycle; rescanning next")

    stage("[YOLO SCANNER] Finished", f"Total picked: {picked}")


if __name__ == "__main__":
    main()
