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


from ui_server.camera_worker import get_latest_frame_copy, start_camera
from ui_server.viewer_overlay import annotate_frame

# Robot control (manual moves/gripper/home)
from final_testing.Class_Execution import (
    ik, gripper, camera,
    stop_motion, estop_motion,
    pause_system, resume_system
)

joy_target = {"x": 0.0, "y": 15.0, "z": 20.0}
JOY_SPEED = 0.3  # cm per tick

app = FastAPI()

# Serve static files (index.html, etc.)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def on_startup() -> None:
    # Start camera thread once when server starts
    start_camera(cam_index=0)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def mjpeg_generator() -> Generator[bytes, None, None]:
    """
    Streams frames as multipart/x-mixed-replace (MJPEG).
    Browser can display it in <img src="/video">.
    """
    boundary = b"--frame"
    while True:
        frame = get_latest_frame_copy()
        if frame is None:
            time.sleep(0.05)
            continue

        # Apply overlay BEFORE encoding (and don't let overlay crash the stream)
        try:
            frame = annotate_frame(frame)
        except Exception:
            pass

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.02)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield b"Content-Length: " + str(len(jpg)).encode("ascii") + b"\r\n\r\n"
        yield jpg.tobytes() + b"\r\n"


@app.get("/video")
def video() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(),
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
SCANNER_PATH = Path(__file__).resolve().parents[1] / "final_testing" / "Vision_Scanner.py"

@app.post("/api/joystick")
def joystick(cmd: dict = Body(...)):
    dx = float(cmd.get("dx", 0))
    dy = float(cmd.get("dy", 0))
    dz = float(cmd.get("dz", 0))

    joy_target["x"] += dx * JOY_SPEED
    joy_target["y"] += dy * JOY_SPEED
    joy_target["z"] += dz * JOY_SPEED

    ok = ik.move_to(
        joy_target["x"],
        joy_target["y"],
        joy_target["z"]
    )

    if not ok:
        return JSONResponse({"ok": False, "error": "IK failed"}, status_code=400)

    return JSONResponse({"ok": True, "target": joy_target})

@app.get("/api/status")
def api_status():
    payload = dict(_status)
    payload["server_time"] = datetime.now().strftime("%H:%M:%S")
    payload["scanner_running"] = (_scanner_proc is not None and _scanner_proc.poll() is None)
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

        # placeholders for later
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
    if _scanner_proc is not None and _scanner_proc.poll() is None:
        _status["last_action"] = "scanner_start"
        return JSONResponse({"ok": True, "running": True, "note": "Vision_Scanner already running"})

    if not SCANNER_PATH.exists():
        _status["last_error"] = f"Vision_Scanner not found: {SCANNER_PATH}"
        return JSONResponse({"ok": False, "error": _status["last_error"]}, status_code=500)

    env = os.environ.copy()

    # IMPORTANT: force Vision_Scanner to use THIS server for frames (no camera conflict)
    # 127.0.0.1 is correct because Vision_Scanner runs on the same Jetson as this server.
    env["UI_SERVER"] = "http://127.0.0.1:8000"

    _scanner_proc = subprocess.Popen(
        [sys.executable, str(SCANNER_PATH)],
        cwd=str(SCANNER_PATH.parent),
        env=env,
    )

    _status["state"] = "SCANNER_RUNNING"
    _status["last_action"] = "scanner_start"
    _status["last_error"] = "--"
    return JSONResponse({"ok": True, "running": True})


@app.post("/api/scanner/stop")
def scanner_stop():
    global _scanner_proc

    if _scanner_proc is None or _scanner_proc.poll() is not None:
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
