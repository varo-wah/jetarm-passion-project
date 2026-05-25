import os
import time
from datetime import datetime
from typing import Generator
import sys
import signal
import subprocess
from pathlib import Path

import cv2
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from jetarm.ui.camera_worker import get_latest_frame_copy, start_camera

# Robot control (manual moves/gripper/home)
from jetarm.hardware.Class_Execution import (
    ik, gripper, camera, ufm,
    stop_motion, estop_motion,
    pause_system, resume_system
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

_scanner_proc: subprocess.Popen | None = None
SCANNER_MODULE = "jetarm.sorting.Vision_Scanner"


def _scanner_is_running() -> bool:
    return _scanner_proc is not None and _scanner_proc.poll() is None


@app.post("/api/joystick")
def joystick(cmd: dict = Body(...)):
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

    # Expose joystick config/state for the UI
    payload["joy_speed"] = JOY_SPEED
    payload["joy_target"] = dict(joy_target)

    return JSONResponse(payload)


@app.post("/api/cmd")
def api_cmd(cmd: dict = Body(...)):
    ctype = cmd.get("type")
    _status["last_action"] = ctype if ctype else "--"

    try:
        if ctype == "goto":
            x = float(cmd["x"])
            y = float(cmd["y"])
            z = float(cmd["z"])
            ok = ik.move_to(x, y, z)
            if not ok:
                _status["last_error"] = "IK failed / joint limit"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)
            return JSONResponse({"ok": True})

        if ctype == "home":
            camera.scan_position()
            return JSONResponse({"ok": True})

        if ctype == "open_gripper":
            gripper.open_gripper()
            return JSONResponse({"ok": True})

        if ctype == "close_gripper":
            gripper.close_gripper()
            return JSONResponse({"ok": True})

        friendly_commands = {
            "idle_pose": ufm.idle_pose,
            "look_around": ufm.look_around,
            "hello_wave": ufm.hello_wave,
            "curious_idle": ufm.curious_idle,
        }
        if ctype in friendly_commands:
            if _scanner_is_running():
                _status["last_error"] = "User-friendly mode blocked while scanner is running"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=409)

            ok = friendly_commands[ctype]()
            if not ok:
                _status["last_error"] = "User-friendly motion failed or was blocked"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)

            _status["state"] = "IDLE"
            return JSONResponse({"ok": True})

        if ctype == "stop":
            stop_motion()
            _status["state"] = "STOPPED"
            return JSONResponse({"ok": True})

        if ctype == "estop":
            estop_motion()
            _status["state"] = "ESTOP"
            return JSONResponse({"ok": True})

        if ctype == "pause":
            pause_system()
            _status["state"] = "PAUSED"
            return JSONResponse({"ok": True})

        if ctype == "resume":
            ok = resume_system()
            if not ok:
                _status["last_error"] = "Cannot resume: E-STOP latched"
                return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=400)
            _status["state"] = "IDLE"
            return JSONResponse({"ok": True})

        return JSONResponse({"ok": False, "error": f"Unknown cmd type: {ctype}"}, status_code=400)

    except Exception as e:
        _status["last_error"] = str(e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/scanner/start")
def scanner_start():
    global _scanner_proc

    # already running
    if _scanner_is_running():
        _status["last_action"] = "scanner_start"
        return JSONResponse({"ok": True, "running": True, "note": "Vision_Scanner already running"})

    env = os.environ.copy()

    # IMPORTANT: force Vision_Scanner to use THIS server for frames (no camera conflict)
    env["UI_SERVER"] = "http://127.0.0.1:8000"

    _scanner_proc = subprocess.Popen(
        [sys.executable, "-m", SCANNER_MODULE],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
    )

    _status["state"] = "SCANNER_RUNNING"
    _status["last_action"] = "scanner_start"
    _status["last_error"] = "--"
    return JSONResponse({"ok": True, "running": True})


@app.post("/api/scanner/stop")
def scanner_stop():
    global _scanner_proc

    if not _scanner_is_running():
        _scanner_proc = None
        _status["last_action"] = "scanner_stop"
        _status["state"] = "IDLE"
        return JSONResponse({"ok": True, "running": False, "note": "Vision_Scanner not running"})

    try:
        _scanner_proc.send_signal(signal.SIGINT)
        try:
            _scanner_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            _scanner_proc.terminate()
            _scanner_proc.wait(timeout=2.0)
    finally:
        _scanner_proc = None

    _status["state"] = "IDLE"
    _status["last_action"] = "scanner_stop"
    return JSONResponse({"ok": True, "running": False})


@app.get("/api/frame.jpg")
def frame_jpg():
    frame = get_latest_frame_copy()
    if frame is None:
        return Response(status_code=503)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")
