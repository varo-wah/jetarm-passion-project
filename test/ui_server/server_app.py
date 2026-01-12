import os
import time
from datetime import datetime
from typing import Generator

import cv2
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ui_server.camera_worker import get_latest_frame_copy, start_camera
from ui_server.viewer_overlay import annotate_frame

# Robot control (manual moves/gripper/home)
from final_testing.Class_Execution import ik, gripper, camera

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


@app.get("/api/status")
def api_status():
    payload = dict(_status)
    payload["server_time"] = datetime.now().strftime("%H:%M:%S")
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
        if ctype in ("pause", "resume", "stop", "estop"):
            return JSONResponse({"ok": True, "note": "Not wired to sorter yet"})

        return JSONResponse({"ok": False, "error": f"Unknown cmd type: {ctype}"}, status_code=400)

    except Exception as e:
        _status["last_error"] = str(e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/frame.jpg")
def frame_jpg():
    frame = get_latest_frame_copy()
    if frame is None:
        return Response(status_code=503)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")
