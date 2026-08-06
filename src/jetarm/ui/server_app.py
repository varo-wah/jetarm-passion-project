import os
import secrets
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Generator
import sys
import signal
import subprocess
import threading
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from jetarm.ui.camera_worker import get_latest_frame_copy, start_camera, stop_camera

# Robot control (manual moves/gripper/home)
from jetarm.hardware.Class_Execution import (
    ik, gripper, camera, ufm,
    stop_motion, estop_motion,
    pause_system, resume_system,
    clear_estop, motion_is_allowed, motion_safety_status,
    safe_shutdown,
    shutdown_control_client,
    hardware_is_available, hardware_unavailable_reason,
)
from jetarm.ui.viewer_overlay import annotate_frame
from jetarm.ui.yolo_overlay import annotate_yolo_frame

STREAM_SETTINGS = {
    "raw": {"fps": 12.0, "jpeg_quality": 65},
    "opencv": {"fps": 10.0, "jpeg_quality": 65},
    "yolo": {"fps": 8.0, "jpeg_quality": 60},
}

# -------------------------------------------------
# Joystick / Jog state
# -------------------------------------------------
joy_target = {"x": 0.0, "y": 15.0, "z": 13.0}
JOY_SPEED = 0.3  # cm per "tick" (frontend converts cm -> ticks)

# Optional safety limits (adjust for your rig)
JOY_LIMITS = {
    "x": (-30.0, 30.0),
    "y": (0.0, 40.0),
    "z": (0.0, 40.0),
}

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


app = FastAPI()

# Serve static files (index.html, etc.)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def on_startup() -> None:
    # Match scripts/yolo_viewer.py capture settings so website YOLO uses the
    # same frame geometry as the calibration/debug path.
    start_camera(cam_index=0, width=640, height=480, fps=15)
    if SCANNER_ACTUATION_REQUESTED and not HARDWARE_AVAILABLE:
        _status["state"] = "CONFIG_ERROR"
        _status["last_action"] = "hardware_preflight"
        error = _hardware_unavailable_error()
        _status["last_error"] = error
        _record_server_alert(
            severity="fault",
            code="HARDWARE_UNAVAILABLE",
            stage="startup_preflight",
            summary="Actuation hardware is unavailable",
            detail=error,
            action="Restore the ROS hardware connection, then restart the dashboard.",
        )


@app.on_event("shutdown")
def on_shutdown() -> None:
    shutdown_errors: list[str] = []

    def attempt(name: str, operation) -> None:
        try:
            operation()
        except Exception as exc:  # Continue releasing every subsystem on shutdown.
            shutdown_errors.append(f"{name}: {exc}")

    attempt("person follow stop", _request_person_follow_stop)
    attempt("controller pause", stop_motion)
    attempt("scanner stop", _stop_scanner_process)
    if SCANNER_ACTUATION_ENABLED:
        attempt("controller safe shutdown", safe_shutdown)
    attempt("camera stop", stop_camera)
    attempt("ROS client close", shutdown_control_client)

    if shutdown_errors:
        print("Shutdown completed with errors: " + "; ".join(shutdown_errors), file=sys.stderr)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _annotate_for_mode(frame, mode: str):
    if mode == "raw":
        return frame
    if mode == "yolo":
        return annotate_yolo_frame(frame)
    return annotate_frame(frame)


def _stream_settings_for_mode(mode: str):
    return STREAM_SETTINGS.get(mode, STREAM_SETTINGS["opencv"])


def mjpeg_generator(mode: str = "opencv") -> Generator[bytes, None, None]:
    """
    Streams frames as multipart/x-mixed-replace (MJPEG).
    Browser can display it in <img src="/video">.
    """
    boundary = b"--frame"
    settings = _stream_settings_for_mode(mode)
    frame_delay = 1.0 / settings["fps"]
    jpeg_quality = int(settings["jpeg_quality"])

    while True:
        frame = get_latest_frame_copy()
        if frame is None:
            time.sleep(0.05)
            continue

        # Apply overlay BEFORE encoding. If an overlay crashes, keep stream alive.
        try:
            frame = _annotate_for_mode(frame, mode)
        except Exception:
            pass

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            time.sleep(0.02)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield b"Content-Length: " + str(len(jpg)).encode("ascii") + b"\r\n\r\n"
        yield jpg.tobytes() + b"\r\n"
        time.sleep(frame_delay)


@app.get("/video")
def video() -> StreamingResponse:
    return video_opencv()


@app.get("/video/raw")
def video_raw() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/opencv")
def video_opencv() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator("opencv"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/yolo")
def video_yolo() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator("yolo"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# In-memory placeholders (v1). Later sorter process can update these.
_status = {
    "state": "IDLE",
    "fps": "--",
    "last_action": "--",
    "last_detection": "--",
    "sort_count": 0,
    "last_error": "--",
}

_active_alert: dict | None = None
_alert_history: deque[dict] = deque(maxlen=20)
_alert_lock = threading.Lock()

_scanner_proc: subprocess.Popen | None = None
_scanner_monitor_thread: threading.Thread | None = None
_scanner_autocycle_enabled = False
_scanner_lock = threading.Lock()
_motion_operation_lock = threading.RLock()
SCANNER_MODULE = "jetarm.sorting.yolo_vision_scanner"
SCANNER_ACTUATION_ENV = "JETARM_ENABLE_ACTUATION"
SCANNER_EVENT_TOKEN_ENV = "JETARM_SCANNER_EVENT_TOKEN"
SCANNER_RESTART_DELAY_SEC = 0.5
SCANNER_INTERRUPT_TIMEOUT_SEC = 3.0
SCANNER_TERMINATE_TIMEOUT_SEC = 2.0
SCANNER_KILL_TIMEOUT_SEC = 1.0
ALERT_SEVERITIES = {"warning", "abort", "estop", "fault"}
_scanner_event_token: str | None = None
_scanner_abort_received = False


def _alert_text(value, default: str, limit: int = 600) -> str:
    text = str(value).strip() if value is not None else ""
    return (text or default)[:limit]


def _normalize_alert(event: dict) -> dict:
    severity = _alert_text(event.get("severity"), "abort", 16).lower()
    if severity not in ALERT_SEVERITIES:
        severity = "abort"

    return {
        "id": _alert_text(event.get("id"), uuid.uuid4().hex, 64),
        "severity": severity,
        "source": _alert_text(event.get("source"), "system", 40),
        "code": _alert_text(event.get("code"), "UNSPECIFIED", 80),
        "stage": _alert_text(event.get("stage"), "unknown", 80),
        "summary": _alert_text(event.get("summary"), "Operation interrupted", 180),
        "detail": _alert_text(event.get("detail"), "No additional detail was provided"),
        "action": _alert_text(
            event.get("action"),
            "Inspect the robot state before acknowledging this alert.",
        ),
        "object_state": _alert_text(event.get("object_state"), "unknown", 40),
        "motion_state": _alert_text(event.get("motion_state"), "UNKNOWN", 40),
        "requires_ack": bool(event.get("requires_ack", severity != "warning")),
        "acknowledged": False,
        "timestamp": _alert_text(event.get("timestamp"), datetime.now().isoformat(), 64),
    }


def _record_alert(event: dict) -> dict:
    global _active_alert

    normalized = _normalize_alert(event)
    with _alert_lock:
        _active_alert = normalized
        _alert_history.append(normalized)

    _status["last_error"] = f"{normalized['summary']}: {normalized['detail']}"
    print(
        "ALERT "
        f"[{normalized['severity'].upper()}]"
        f"[{normalized['source'].upper()}]"
        f"[{normalized['stage'].upper()}]"
        f"[{normalized['code']}] "
        f"{normalized['detail']} | motion={normalized['motion_state']} "
        f"| object={normalized['object_state']}"
    )
    return normalized


def _record_server_alert(
    *,
    severity: str,
    code: str,
    stage: str,
    summary: str,
    detail: str,
    action: str,
    object_state: str = "not_applicable",
) -> dict:
    safety = motion_safety_status()
    return _record_alert(
        {
            "severity": severity,
            "source": "controller",
            "code": code,
            "stage": stage,
            "summary": summary,
            "detail": detail,
            "action": action,
            "object_state": object_state,
            "motion_state": safety.get("state", "UNKNOWN"),
            "requires_ack": severity != "warning",
        }
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SCANNER_ACTUATION_REQUESTED = _env_flag(SCANNER_ACTUATION_ENV, default=False)
HARDWARE_AVAILABLE = hardware_is_available()
HARDWARE_UNAVAILABLE_REASON = hardware_unavailable_reason()
SCANNER_ACTUATION_ENABLED = SCANNER_ACTUATION_REQUESTED and HARDWARE_AVAILABLE

PERSON_FOLLOW_CONFIDENCE = 0.45
PERSON_FOLLOW_DEADZONE_PX = 70
PERSON_FOLLOW_STEP_CM = 1.0
PERSON_FOLLOW_MAX_X_CM = 6.0
PERSON_FOLLOW_INTERVAL_SEC = 0.45
PERSON_FOLLOW_INVERT_X = False

_person_follow_thread: threading.Thread | None = None
_person_follow_stop = threading.Event()
_person_follow_x_cm = 0.0


def _scanner_is_running() -> bool:
    return _scanner_proc is not None and _scanner_proc.poll() is None


def _launch_scanner_process_unlocked() -> subprocess.Popen:
    global _scanner_event_token, _scanner_abort_received

    env = os.environ.copy()
    event_token = secrets.token_urlsafe(32)

    # IMPORTANT: force Vision_Scanner to use THIS server for frames (no camera conflict)
    env["UI_SERVER"] = "http://127.0.0.1:8000"
    env[SCANNER_ACTUATION_ENV] = "1" if SCANNER_ACTUATION_ENABLED else "0"
    env[SCANNER_EVENT_TOKEN_ENV] = event_token
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", SCANNER_MODULE],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            start_new_session=True,
        )
    except Exception:
        _scanner_event_token = None
        _scanner_abort_received = False
        raise

    _scanner_event_token = event_token
    _scanner_abort_received = False
    return proc


def _stop_scanner_process() -> bool:
    """Stop the isolated scanner process, escalating through INT/TERM/KILL."""
    global _scanner_proc, _scanner_autocycle_enabled
    global _scanner_event_token, _scanner_abort_received

    with _scanner_lock:
        _scanner_autocycle_enabled = False
        proc = _scanner_proc

    if proc is None or proc.poll() is not None:
        with _scanner_lock:
            if _scanner_proc is proc:
                _scanner_proc = None
                _scanner_event_token = None
                _scanner_abort_received = False
        return True

    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=SCANNER_INTERRUPT_TIMEOUT_SEC)
            return True
        except subprocess.TimeoutExpired:
            proc.terminate()

        try:
            proc.wait(timeout=SCANNER_TERMINATE_TIMEOUT_SEC)
            return True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=SCANNER_KILL_TIMEOUT_SEC)
            return True
    finally:
        with _scanner_lock:
            if _scanner_proc is proc:
                _scanner_proc = None
                _scanner_event_token = None
                _scanner_abort_received = False


def _scanner_autocycle_loop() -> None:
    global _scanner_proc, _scanner_autocycle_enabled, _scanner_event_token

    while True:
        with _scanner_lock:
            proc = _scanner_proc

        if proc is None:
            return

        exit_code = proc.wait()

        with _scanner_lock:
            if _scanner_proc is not proc:
                return
            _scanner_proc = None
            autocycle_enabled = _scanner_autocycle_enabled
            abort_received = _scanner_abort_received
            _scanner_event_token = None

        if exit_code != 0:
            with _scanner_lock:
                _scanner_autocycle_enabled = False
            stop_motion()
            _status["state"] = "SCANNER_ERROR"
            if abort_received:
                _status["last_action"] = "scanner_abort"
                with _alert_lock:
                    if _active_alert is not None:
                        _active_alert["exit_code"] = exit_code
                        _active_alert["motion_state"] = motion_safety_status().get(
                            "state", "PAUSED"
                        )
            else:
                _status["last_action"] = "scanner_exited"
                _record_server_alert(
                    severity="fault",
                    code="SCANNER_EXIT_NONZERO",
                    stage="scanner_process",
                    summary="Scanner process exited unexpectedly",
                    detail=f"YOLO scanner exited with code {exit_code}",
                    action="Review the terminal output, acknowledge this alert, then Resume and retry.",
                )
            return

        if not autocycle_enabled:
            if _status.get("state") == "SCANNER_RUNNING":
                _status["state"] = "IDLE"
            return

        if not SCANNER_ACTUATION_ENABLED or not motion_is_allowed():
            with _scanner_lock:
                _scanner_autocycle_enabled = False
            _status["state"] = "IDLE"
            _status["last_error"] = "Scanner auto-cycle blocked by safety configuration"
            return

        _status["state"] = "PRESSING_BUTTON"
        _status["last_action"] = "scanner_done_press_button"
        _status["last_error"] = "--"

        with _motion_operation_lock:
            if not _scanner_autocycle_enabled or not motion_is_allowed():
                with _scanner_lock:
                    _scanner_autocycle_enabled = False
                _status["state"] = "IDLE"
                _status["last_error"] = "Button press blocked by safety state"
                return
            ok = ufm.press_button()
        if not ok:
            with _scanner_lock:
                _scanner_autocycle_enabled = False
            _status["state"] = "IDLE"
            _status["last_error"] = "Button press failed after scanner finished; auto-cycle stopped"
            return

        time.sleep(SCANNER_RESTART_DELAY_SEC)

        with _motion_operation_lock:
            with _scanner_lock:
                if not _scanner_autocycle_enabled:
                    _status["state"] = "IDLE"
                    return

                _scanner_proc = _launch_scanner_process_unlocked()

        _status["state"] = "SCANNER_RUNNING"
        _status["last_action"] = "scanner_restart"
        _status["last_error"] = "--"


def _ensure_scanner_monitor() -> None:
    global _scanner_monitor_thread

    _scanner_monitor_thread = threading.Thread(
        target=_scanner_autocycle_loop,
        name="scanner-autocycle",
        daemon=True,
    )
    _scanner_monitor_thread.start()


def _request_scanner_autocycle_stop() -> None:
    global _scanner_autocycle_enabled

    with _scanner_lock:
        _scanner_autocycle_enabled = False


def _person_follow_is_running() -> bool:
    return _person_follow_thread is not None and _person_follow_thread.is_alive()


def _request_person_follow_stop() -> None:
    _person_follow_stop.set()


def _hardware_unavailable_error() -> str:
    detail = HARDWARE_UNAVAILABLE_REASON or "ROS hardware imports failed"
    return f"Robot hardware unavailable: {detail}"


def _manual_motion_conflict() -> str | None:
    if _scanner_is_running():
        return "Manual motion blocked while scanner is running"
    if _person_follow_is_running():
        return "Manual motion blocked while person follow is running"
    if SCANNER_ACTUATION_REQUESTED and not HARDWARE_AVAILABLE:
        return _hardware_unavailable_error()
    if not SCANNER_ACTUATION_ENABLED:
        return f"Robot actuation disabled; restart with {SCANNER_ACTUATION_ENV}=1 to enable motion"

    safety = motion_safety_status()
    if safety["estop_latched"]:
        return "Manual motion blocked: E-STOP is latched"
    if safety["paused"]:
        return "Manual motion blocked: system is paused"
    return None


def _motion_conflict_response() -> JSONResponse | None:
    error = _manual_motion_conflict()
    if error is None:
        return None
    _status["last_error"] = error
    return JSONResponse({"ok": False, "error": error}, status_code=409)


def _person_follow_loop() -> None:
    global _person_follow_x_cm

    from jetarm.vision.person_detector import choose_person, detect_people

    _person_follow_x_cm = 0.0
    _status["state"] = "PERSON_FOLLOW"
    _status["last_action"] = "person_follow_start"
    _status["last_error"] = "--"

    try:
        if not ufm.person_follow_pose(_person_follow_x_cm):
            _status["last_error"] = "Person follow failed to enter safe pose"
            return

        while not _person_follow_stop.is_set():
            if _scanner_is_running():
                _status["last_error"] = "Person follow stopped because scanner is running"
                return

            frame = get_latest_frame_copy()
            if frame is None:
                _status["last_detection"] = "person: no frame"
                _person_follow_stop.wait(0.1)
                continue

            people = detect_people(frame, min_confidence=PERSON_FOLLOW_CONFIDENCE)
            target = choose_person(people)
            if target is None:
                _status["last_detection"] = "person: none"
                _person_follow_stop.wait(PERSON_FOLLOW_INTERVAL_SEC)
                continue

            frame_width = frame.shape[1]
            center_x = int(target["center_x"])
            error_px = center_x - (frame_width / 2.0)
            confidence = float(target["confidence"])
            _status["last_detection"] = f"person x={center_x} err={error_px:.0f}px conf={confidence:.2f}"

            if abs(error_px) > PERSON_FOLLOW_DEADZONE_PX:
                direction = 1.0 if error_px > 0 else -1.0
                if PERSON_FOLLOW_INVERT_X:
                    direction *= -1.0

                _person_follow_x_cm += direction * PERSON_FOLLOW_STEP_CM
                _person_follow_x_cm = _clamp(
                    _person_follow_x_cm,
                    -PERSON_FOLLOW_MAX_X_CM,
                    PERSON_FOLLOW_MAX_X_CM,
                )

                ok = ufm.person_follow_pose(_person_follow_x_cm)
                if not ok:
                    _status["last_error"] = "Person follow motion failed or was blocked"
                    return

            _person_follow_stop.wait(PERSON_FOLLOW_INTERVAL_SEC)

    except Exception as exc:
        _status["last_error"] = str(exc)
    finally:
        if not _scanner_is_running() and _status.get("state") == "PERSON_FOLLOW":
            _status["state"] = "IDLE"
        _person_follow_stop.set()


@app.post("/api/joystick")
def joystick(cmd: dict = Body(...)):
    with _motion_operation_lock:
        conflict = _motion_conflict_response()
        if conflict is not None:
            return conflict

        dx = float(cmd.get("dx", 0))
        dy = float(cmd.get("dy", 0))
        dz = float(cmd.get("dz", 0))

        # Update target using server speed (cm per tick)
        joy_target["x"] += dx * JOY_SPEED
        joy_target["y"] += dy * JOY_SPEED
        joy_target["z"] += dz * JOY_SPEED

        # Optional clamping to workspace
        joy_target["x"] = _clamp(joy_target["x"], *JOY_LIMITS["x"])
        joy_target["y"] = _clamp(joy_target["y"], *JOY_LIMITS["y"])
        joy_target["z"] = _clamp(joy_target["z"], *JOY_LIMITS["z"])

        ok = ik.move_to(
            joy_target["x"],
            joy_target["y"],
            joy_target["z"]
        )

    if not ok:
        return JSONResponse({"ok": False, "error": "IK failed"}, status_code=400)

    return JSONResponse({"ok": True, "target": joy_target})


@app.post("/api/joystick/reset")
def joystick_reset():
    with _motion_operation_lock:
        conflict = _motion_conflict_response()
        if conflict is not None:
            return conflict

        # Reset to your preferred “safe jog pose”
        joy_target["x"] = 0.0
        joy_target["y"] = 15.0
        joy_target["z"] = 13.0

        # Clamp (in case you changed limits)
        joy_target["x"] = _clamp(joy_target["x"], *JOY_LIMITS["x"])
        joy_target["y"] = _clamp(joy_target["y"], *JOY_LIMITS["y"])
        joy_target["z"] = _clamp(joy_target["z"], *JOY_LIMITS["z"])

        ok = ik.move_to(joy_target["x"], joy_target["y"], joy_target["z"])
    if not ok:
        return JSONResponse({"ok": False, "error": "IK failed"}, status_code=400)

    return JSONResponse({"ok": True, "target": joy_target})


@app.get("/api/status")
def api_status():
    payload = dict(_status)
    payload["server_time"] = datetime.now().strftime("%H:%M:%S")
    payload["scanner_running"] = _scanner_is_running()
    payload["scanner_autocycle_enabled"] = _scanner_autocycle_enabled
    payload["scanner_actuation_enabled"] = SCANNER_ACTUATION_ENABLED
    payload["scanner_actuation_requested"] = SCANNER_ACTUATION_REQUESTED
    payload["hardware_available"] = HARDWARE_AVAILABLE
    payload["person_follow_running"] = _person_follow_is_running()
    payload["motion_safety"] = motion_safety_status()

    with _alert_lock:
        payload["active_alert"] = dict(_active_alert) if _active_alert is not None else None
        payload["alert_history"] = [dict(alert) for alert in reversed(_alert_history)]

    # Expose joystick config/state for the UI
    payload["joy_speed"] = JOY_SPEED
    payload["joy_target"] = dict(joy_target)

    return JSONResponse(payload)


@app.post("/api/scanner/event")
def scanner_event(
    event: dict = Body(...),
    x_jetarm_scanner_token: str | None = Header(default=None),
):
    """Accept an authenticated structured event from the active scanner child."""
    global _scanner_autocycle_enabled, _scanner_abort_received

    with _scanner_lock:
        expected_token = _scanner_event_token
        authorized = (
            isinstance(x_jetarm_scanner_token, str)
            and isinstance(expected_token, str)
            and secrets.compare_digest(x_jetarm_scanner_token, expected_token)
        )

    if not authorized:
        return JSONResponse({"ok": False, "error": "Invalid scanner event token"}, status_code=403)

    normalized = _normalize_alert(event)
    if normalized["severity"] in {"abort", "estop", "fault"}:
        with _scanner_lock:
            _scanner_autocycle_enabled = False
            _scanner_abort_received = True
        stop_motion()
        normalized["motion_state"] = motion_safety_status().get("state", "PAUSED")
        _status["state"] = "SCANNER_ERROR"
        _status["last_action"] = "scanner_abort"
    else:
        _status["last_action"] = "scanner_warning"

    recorded = _record_alert(normalized)
    return JSONResponse({"ok": True, "alert": recorded})


@app.post("/api/alerts/acknowledge")
def acknowledge_alert(payload: dict | None = Body(default=None)):
    """Acknowledge the active UI alert without changing motion authority."""
    global _active_alert

    requested_id = (payload or {}).get("id")
    with _alert_lock:
        if _active_alert is None:
            return JSONResponse({"ok": True, "acknowledged": False})
        if requested_id and requested_id != _active_alert["id"]:
            return JSONResponse(
                {"ok": False, "error": "Alert is no longer active"},
                status_code=409,
            )

        _active_alert["acknowledged"] = True
        _active_alert["acknowledged_at"] = datetime.now().isoformat()
        acknowledged_id = _active_alert["id"]
        _active_alert = None

    return JSONResponse({"ok": True, "acknowledged": True, "id": acknowledged_id})


@app.post("/api/cmd")
def api_cmd(cmd: dict = Body(...)):
    ctype = cmd.get("type")
    _status["last_action"] = ctype if ctype else "--"

    try:
        if ctype == "goto":
            with _motion_operation_lock:
                conflict = _motion_conflict_response()
                if conflict is not None:
                    return conflict

                x = float(cmd["x"])
                y = float(cmd["y"])
                z = float(cmd["z"])
                ok = ik.move_to(x, y, z)
            if not ok:
                _status["last_error"] = "IK failed / joint limit"
                _record_server_alert(
                    severity="warning",
                    code="MANUAL_TARGET_REJECTED",
                    stage="manual_goto",
                    summary="Manual target rejected before motion",
                    detail=f"Requested target x={x:.2f}, y={y:.2f}, z={z:.2f} failed IK or a joint limit.",
                    action="Choose a target inside the calibrated workspace and retry.",
                )
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)
            return JSONResponse({"ok": True})

        if ctype == "home":
            with _motion_operation_lock:
                conflict = _motion_conflict_response()
                if conflict is not None:
                    return conflict
                ok = camera.scan_position()
            if not ok:
                return JSONResponse({"ok": False, "error": "Scan pose failed or was blocked"}, status_code=400)
            return JSONResponse({"ok": True})

        if ctype == "open_gripper":
            with _motion_operation_lock:
                conflict = _motion_conflict_response()
                if conflict is not None:
                    return conflict
                ok = gripper.open_gripper()
            if not ok:
                return JSONResponse({"ok": False, "error": "Gripper command failed or was blocked"}, status_code=400)
            return JSONResponse({"ok": True})

        if ctype == "close_gripper":
            with _motion_operation_lock:
                conflict = _motion_conflict_response()
                if conflict is not None:
                    return conflict
                ok = gripper.close_gripper()
            if not ok:
                return JSONResponse({"ok": False, "error": "Gripper command failed or was blocked"}, status_code=400)
            return JSONResponse({"ok": True})

        friendly_commands = {
            "idle_pose": ufm.idle_pose,
            "look_around": ufm.look_around,
            "hello_wave": ufm.hello_wave,
            "curious_idle": ufm.curious_idle,
            "press_button": ufm.press_button,
        }
        if ctype in friendly_commands:
            with _motion_operation_lock:
                conflict = _motion_conflict_response()
                if conflict is not None:
                    return conflict
                ok = friendly_commands[ctype]()
            if not ok:
                _status["last_error"] = "User-friendly motion failed or was blocked"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)

            _status["state"] = "IDLE"
            return JSONResponse({"ok": True})

        if ctype == "stop":
            _request_person_follow_stop()
            _request_scanner_autocycle_stop()
            stop_motion()
            _stop_scanner_process()
            _status["state"] = "STOPPED"
            return JSONResponse({"ok": True})

        if ctype == "estop":
            estop_motion()
            _request_person_follow_stop()
            _request_scanner_autocycle_stop()
            _stop_scanner_process()
            _status["state"] = "ESTOP"
            _record_server_alert(
                severity="estop",
                code="ESTOP_LATCHED",
                stage="operator_command",
                summary="Emergency stop latched",
                detail="Software E-stop was activated and motion commands are blocked.",
                action="Inspect the robot, Clear E-Stop, acknowledge this alert, then press Resume separately.",
            )
            return JSONResponse({"ok": True})

        if ctype == "pause":
            pause_system()
            _request_person_follow_stop()
            _request_scanner_autocycle_stop()
            _stop_scanner_process()
            _status["state"] = "PAUSED"
            return JSONResponse({"ok": True})

        if ctype == "resume":
            ok = resume_system()
            if not ok:
                _status["last_error"] = "Cannot resume: E-STOP latched"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)
            _status["state"] = "IDLE"
            return JSONResponse({"ok": True})

        if ctype == "clear_estop":
            _request_person_follow_stop()
            _request_scanner_autocycle_stop()
            _stop_scanner_process()
            clear_estop()
            pause_system()
            _status["state"] = "PAUSED"
            _status["last_error"] = "--"
            return JSONResponse({
                "ok": True,
                "paused": True,
                "note": "E-STOP cleared; press Resume separately to permit motion",
            })

        return JSONResponse({"ok": False, "error": f"Unknown cmd type: {ctype}"}, status_code=400)

    except Exception as e:
        _status["last_error"] = str(e)
        _record_server_alert(
            severity="abort",
            code="COMMAND_EXCEPTION",
            stage=ctype or "unknown_command",
            summary="Controller command aborted",
            detail=str(e),
            action="Inspect the requested command and robot state before acknowledging and retrying.",
        )
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/person_follow/start")
def person_follow_start():
    global _person_follow_thread

    with _motion_operation_lock:
        if _scanner_is_running():
            _status["last_error"] = "Cannot start person follow while scanner is running"
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)
        if not SCANNER_ACTUATION_ENABLED:
            _status["last_error"] = (
                f"Person follow blocked; restart with {SCANNER_ACTUATION_ENV}=1 to enable motion"
            )
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)
        if not motion_is_allowed():
            _status["last_error"] = "Cannot start person follow while motion is paused or E-STOP is latched"
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)

        if _person_follow_is_running():
            return JSONResponse({"ok": True, "running": True, "note": "Person follow already running"})

        _person_follow_stop.clear()
        _person_follow_thread = threading.Thread(
            target=_person_follow_loop,
            name="person-follow",
            daemon=True,
        )
        _person_follow_thread.start()

    _status["state"] = "PERSON_FOLLOW"
    _status["last_action"] = "person_follow_start"
    _status["last_error"] = "--"
    return JSONResponse({"ok": True, "running": True})


@app.post("/api/person_follow/stop")
def person_follow_stop():
    global _person_follow_thread

    if not _person_follow_is_running():
        _person_follow_thread = None
        _status["last_action"] = "person_follow_stop"
        if not _scanner_is_running():
            _status["state"] = "IDLE"
        return JSONResponse({"ok": True, "running": False, "note": "Person follow not running"})

    _request_person_follow_stop()
    _person_follow_thread.join(timeout=1.5)
    if not _person_follow_thread.is_alive():
        _person_follow_thread = None

    _status["state"] = "IDLE"
    _status["last_action"] = "person_follow_stop"
    return JSONResponse({"ok": True, "running": _person_follow_is_running()})


@app.post("/api/scanner/start")
def scanner_start():
    global _scanner_proc, _scanner_autocycle_enabled

    with _motion_operation_lock:
        if SCANNER_ACTUATION_REQUESTED and not HARDWARE_AVAILABLE:
            _status["last_error"] = _hardware_unavailable_error()
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=503)
        if _person_follow_is_running():
            _status["last_error"] = "Stop person follow before starting scanner"
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)
        if SCANNER_ACTUATION_ENABLED and not motion_is_allowed():
            _status["last_error"] = "Cannot start actuating scanner while motion is paused or E-STOP is latched"
            return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)

        with _scanner_lock:
            _scanner_autocycle_enabled = SCANNER_ACTUATION_ENABLED

            # already running
            if _scanner_proc is not None and _scanner_proc.poll() is None:
                _status["last_action"] = "scanner_start"
                return JSONResponse({
                    "ok": True,
                    "running": True,
                    "autocycle": _scanner_autocycle_enabled,
                    "actuation_enabled": SCANNER_ACTUATION_ENABLED,
                    "note": "YOLO scanner already running",
                })

            _scanner_proc = _launch_scanner_process_unlocked()

    _ensure_scanner_monitor()

    _status["state"] = "SCANNER_RUNNING"
    _status["last_action"] = "scanner_start"
    _status["last_error"] = "--"
    return JSONResponse({
        "ok": True,
        "running": True,
        "autocycle": _scanner_autocycle_enabled,
        "actuation_enabled": SCANNER_ACTUATION_ENABLED,
        "mode": "actuating" if SCANNER_ACTUATION_ENABLED else "preview",
    })


@app.post("/api/scanner/stop")
def scanner_stop():
    was_running = _scanner_is_running()
    # Priority first: cancel controller motion before waiting on the worker.
    stop_motion()
    _stop_scanner_process()

    _status["state"] = "IDLE"
    _status["last_action"] = "scanner_stop"
    payload = {
        "ok": True,
        "running": False,
        "autocycle": False,
        "paused": True,
        "note": "Scanner stopped; press Resume before starting motion again",
    }
    if not was_running:
        payload["note"] = "YOLO scanner was not running; controller remains paused"
    return JSONResponse(payload)


@app.get("/api/frame.jpg")
def frame_jpg():
    frame = get_latest_frame_copy()
    if frame is None:
        return Response(status_code=503)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")
