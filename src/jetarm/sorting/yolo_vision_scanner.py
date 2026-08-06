"""
YOLO-based Vision_Scanner variant for JetArm.

This module preserves the old scan-pick-drop scanner structure, but replaces
contour/color detection with the trained LEGO YOLO detector.
"""

import os
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from jetarm.control.limits import JointLimitsError
from jetarm.hardware.Class_Execution import camera, gripper, ik
from jetarm.vision.yolo_detector import detect_bricks_yolo
from jetarm.vision.wrist_safety import choose_safe_wrist_angle


ACTUATION_ENV = "JETARM_ENABLE_ACTUATION"
SCANNER_EVENT_TOKEN_ENV = "JETARM_SCANNER_EVENT_TOKEN"
SCANNER_EVENT_PATH = "/api/scanner/event"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Preview-only unless the operator explicitly opts in before starting the server.
ENABLE_PICK_AND_DROP = env_flag(ACTUATION_ENV, default=False)

UI_SERVER = os.environ.get("UI_SERVER", "http://127.0.0.1:8000")
FRAME_URL = f"{UI_SERVER}/api/frame.jpg"
EVENT_URL = f"{UI_SERVER}{SCANNER_EVENT_PATH}"

_last_abort_event = None
_last_preflight_skip = None

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


def build_event(
    *,
    severity,
    code,
    stage_name,
    summary,
    detail,
    action,
    object_state="not_gripped",
    requires_ack=None,
):
    if requires_ack is None:
        requires_ack = severity != "warning"
    return {
        "id": uuid.uuid4().hex,
        "severity": severity,
        "source": "scanner",
        "code": code,
        "stage": stage_name,
        "summary": summary,
        "detail": str(detail),
        "action": action,
        "object_state": object_state,
        "requires_ack": requires_ack,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def event_code_for_error(error):
    if isinstance(error, JointLimitsError):
        return "JOINT_LIMIT"
    if isinstance(error, ValueError):
        return "INVALID_TARGET"
    message = str(error).lower()
    if "camera" in message or "jpeg" in message or "frame" in message:
        return "CAMERA_CAPTURE_FAILED"
    if "blocked" in message or "paused" in message or "e-stop" in message:
        return "MOTION_BLOCKED"
    if isinstance(error, RuntimeError):
        return "SCANNER_RUNTIME_ERROR"
    return "SCANNER_UNEXPECTED_ERROR"


def remember_abort(*, error, stage_name, summary, action, object_state="not_gripped"):
    global _last_abort_event

    _last_abort_event = build_event(
        severity="abort",
        code=event_code_for_error(error),
        stage_name=stage_name,
        summary=summary,
        detail=error,
        action=action,
        object_state=object_state,
    )
    return _last_abort_event


def report_event(event):
    """Print the event locally and forward it to the authenticated UI server."""
    print(
        "ALERT "
        f"[{event['severity'].upper()}][SCANNER]"
        f"[{event['stage'].upper()}][{event['code']}] "
        f"{event['detail']} | object={event['object_state']}"
    )

    token = os.environ.get(SCANNER_EVENT_TOKEN_ENV)
    if not token:
        print("[YOLO SCANNER] Structured alert was not forwarded: event token unavailable")
        return False

    try:
        response = requests.post(
            EVENT_URL,
            json=event,
            headers={"X-JetArm-Scanner-Token": token},
            timeout=0.75,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as error:
        print(f"[YOLO SCANNER] Structured alert forwarding failed: {error}")
        return False


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


def move_wait(
    x,
    y,
    z,
    label,
    *,
    stage_name="motion",
    object_state="not_gripped",
):
    stage(label, f"Target: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    try:
        ok = ik.move_to(x, y, z)
    except (JointLimitsError, RuntimeError, ValueError) as error:
        remember_abort(
            error=error,
            stage_name=stage_name,
            summary=f"{label} rejected",
            action="Inspect the route and robot state before acknowledging, resuming, and retrying.",
            object_state=object_state,
        )
        print(f"[YOLO SCANNER] Motion rejected: {error}")
        return False
    if not ok:
        error = RuntimeError("Motion command was blocked or the target was unreachable")
        remember_abort(
            error=error,
            stage_name=stage_name,
            summary=f"{label} did not complete",
            action="Inspect the route and motion state before acknowledging, resuming, and retrying.",
            object_state=object_state,
        )
        print(f"[YOLO SCANNER] Motion rejected: {error}")
        return False
    time.sleep(MOVE_TIME + SETTLE_TIME)
    return True


def preflight_pick_and_drop(brick):
    """Validate the complete route before the gripper can acquire an object."""
    global _last_preflight_skip

    x = brick["x"]
    y = brick["y"]
    bx, by = bucket_for_color(brick.get("color"))
    route = (
        (x, y, APPROACH_Z, "target approach"),
        (x, y, PICK_Z, "target pickup"),
        (0, 13, 14, "transfer waypoint"),
        (bx, by, APPROACH_BUCKET, f"{brick.get('color', 'NEUTRAL')} bucket"),
    )
    approach_targets = None
    for route_x, route_y, route_z, label in route:
        try:
            targets = ik.plan_to(route_x, route_y, route_z)
            if label == "target approach":
                approach_targets = targets
        except (JointLimitsError, RuntimeError, ValueError) as error:
            _last_preflight_skip = build_event(
                severity="warning",
                code=event_code_for_error(error),
                stage_name=label.replace(" ", "_"),
                summary="Detected target skipped before motion",
                detail=error,
                action="Move the detected object inside the calibrated workspace and scan again.",
                requires_ack=False,
            )
            print(f"[YOLO SCANNER] Preflight rejected {label}: {error}")
            return False

    try:
        angle, _ = choose_safe_wrist_angle(brick, brick.get("frame_shape"))
        base_angle = (approach_targets[1] - ik.BASE_ZERO_OFFSET) * ik.DEG_PER_PULSE
        gripper.plan_wrist(angle, base_angle=base_angle)
        gripper.plan_gripper()
    except (JointLimitsError, RuntimeError, ValueError) as error:
        _last_preflight_skip = build_event(
            severity="warning",
            code=event_code_for_error(error),
            stage_name="gripper_route",
            summary="Detected target skipped before motion",
            detail=error,
            action="Reposition or rotate the object so the wrist and gripper route is calibrated.",
            requires_ack=False,
        )
        print(f"[YOLO SCANNER] Preflight rejected gripper route: {error}")
        return False
    return True


def scan_once(move_to_scan_pose=None):
    if move_to_scan_pose is None:
        move_to_scan_pose = ENABLE_PICK_AND_DROP

    if move_to_scan_pose:
        print("[YOLO SCANNER] Moving to scan pose")
        if not camera.scan_position():
            error = RuntimeError("Scanner motion was blocked before capture")
            remember_abort(
                error=error,
                stage_name="scan_pose",
                summary="Scanner could not enter the capture pose",
                action="Inspect motion authority and the scan-pose route before acknowledging and retrying.",
            )
            raise error
        time.sleep(SCAN_SETTLE)
    else:
        print("[YOLO SCANNER] Preview mode: leaving robot position unchanged")

    print("[YOLO SCANNER] Capturing frame")
    try:
        frame = take_snapshot()
    except (RuntimeError, ValueError) as error:
        remember_abort(
            error=error,
            stage_name="camera_capture",
            summary="Camera capture failed",
            action="Check the camera connection and ownership, then acknowledge and retry.",
        )
        raise

    print("[YOLO SCANNER] Running YOLO detection")
    try:
        bricks = detect_bricks_yolo(frame)
    except (RuntimeError, ValueError) as error:
        remember_abort(
            error=error,
            stage_name="yolo_detection",
            summary="YOLO detection failed",
            action="Inspect the model/runtime error, then acknowledge and retry.",
        )
        raise
    for brick in bricks:
        brick["frame_shape"] = frame.shape

    print_yolo_bricks(bricks)
    return bricks


def choose_brick(bricks):
    for brick in bricks:
        if not ENABLE_PICK_AND_DROP or preflight_pick_and_drop(brick):
            return brick
    return None


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


def run_gripper_step(
    operation,
    *,
    stage_name,
    summary,
    action,
    object_state,
):
    try:
        ok = operation()
    except (JointLimitsError, RuntimeError, ValueError) as error:
        remember_abort(
            error=error,
            stage_name=stage_name,
            summary=summary,
            action=action,
            object_state=object_state,
        )
        return False
    if not ok:
        remember_abort(
            error=RuntimeError("Gripper command was blocked by motion authority"),
            stage_name=stage_name,
            summary=summary,
            action=action,
            object_state=object_state,
        )
        return False
    return True


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

    if not move_wait(
        x,
        y,
        APPROACH_Z,
        "[YOLO SCANNER] Approaching",
        stage_name="target_approach",
    ):
        return False

    print(
        "[YOLO SCANNER] Aligning wrist: "
        f"detected={detected_angle:.1f} final={angle:.1f} edge={edge_status}"
    )
    if not run_gripper_step(
        lambda: gripper.turn_wrist(angle),
        stage_name="wrist_alignment",
        summary="Wrist alignment aborted",
        action="Inspect the target angle and wrist limits before acknowledging and retrying.",
        object_state="not_gripped",
    ):
        return False
    time.sleep(WRIST_SETTLE)

    if not move_wait(
        x,
        y,
        PICK_Z,
        "[YOLO SCANNER] Going down",
        stage_name="target_pickup",
    ):
        return False

    print("[YOLO SCANNER] Closing gripper")
    if not run_gripper_step(
        gripper.close_gripper,
        stage_name="gripper_close",
        summary="Gripper close aborted",
        action="Verify the object and gripper clearance before acknowledging and retrying.",
        object_state="not_gripped",
    ):
        return False
    time.sleep(GRIP_SETTLE)

    if not move_wait(
        x,
        y,
        APPROACH_Z,
        "[YOLO SCANNER] Lifting up",
        stage_name="target_lift",
        object_state="possibly_gripped",
    ):
        return False

    if not move_wait(
        0,
        13,
        14,
        "[YOLO SCANNER] Transfer waypoint",
        stage_name="transfer_waypoint",
        object_state="possibly_gripped",
    ):
        return False

    if not move_wait(
        bx,
        by,
        APPROACH_BUCKET,
        f"[YOLO SCANNER] To {brick.get('color', 'NEUTRAL')} bucket",
        stage_name="bucket_approach",
        object_state="possibly_gripped",
    ):
        return False

    print("[YOLO SCANNER] Opening gripper")
    if not run_gripper_step(
        gripper.open_gripper,
        stage_name="gripper_release",
        summary="Gripper release aborted",
        action="Treat the object as still held; inspect the gripper before acknowledging and resuming.",
        object_state="possibly_gripped",
    ):
        return False
    time.sleep(RELEASE_SETTLE)

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
            if bricks:
                skip = _last_preflight_skip or build_event(
                    severity="warning",
                    code="NO_SAFE_ROUTE",
                    stage_name="route_preflight",
                    summary="Detections skipped before motion",
                    detail="No detected object had a complete calibrated route",
                    action="Move the objects inside the calibrated workspace and scan again.",
                    requires_ack=False,
                )
                report_event(
                    build_event(
                        severity="warning",
                        code="NO_SAFE_ROUTE",
                        stage_name=skip["stage"],
                        summary="Detections skipped before motion",
                        detail=skip["detail"],
                        action=skip["action"],
                        requires_ack=False,
                    )
                )
                stage(
                    "[YOLO SCANNER] Done",
                    "Detections found, but none has a complete calibrated route",
                )
            else:
                stage("[YOLO SCANNER] Done", "No YOLO bricks detected")
            break

        print("[YOLO SCANNER] Executing pick/drop")
        ok = pick_and_drop(target)

        if ok:
            picked += 1
            print(f"[YOLO SCANNER] Picked count: {picked}")
        else:
            raise RuntimeError(
                "Pick/drop motion failed; scanner stopped without autonomous recovery"
            )

    stage("[YOLO SCANNER] Finished", f"Total picked: {picked}")


def run():
    try:
        main()
    except Exception as error:
        event = _last_abort_event or build_event(
            severity="fault",
            code=event_code_for_error(error),
            stage_name="scanner_main",
            summary="Scanner stopped unexpectedly",
            detail=error,
            action="Review the terminal details, acknowledge the alert, then Resume and retry.",
            object_state="unknown",
        )
        report_event(event)
        raise


if __name__ == "__main__":
    run()
