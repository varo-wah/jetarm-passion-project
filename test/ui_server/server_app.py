# ui_server/server_app.py
import os
import time
from typing import Generator, Optional


import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ui_server.viewer_overlay import annotate_frame
from fastapi import Body
from fastapi.responses import JSONResponse
from datetime import datetime
from fastapi.responses import Response



from ui_server.camera_worker import start_camera, get_latest_frame_copy

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
        except Exception as e:
            # Optional: you can print once or log; for now keep stream alive
            # print(f"annotate_frame error: {e}")
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

# In-memory placeholders (v1). Later the sorter process will update these.
_status = {
    "state": "IDLE",
    "fps": "--",
    "last_action": "--",
    "last_detection": "--",
    "sort_count": 0,
    "last_error": "--",
}

_last_cmd = {"type": None}

@app.get("/api/status")
def api_status():
    # Minimal status payload for UI; replace with real telemetry later
    payload = dict(_status)
    payload["server_time"] = datetime.now().strftime("%H:%M:%S")
    return JSONResponse(payload)

@app.post("/api/cmd")
def api_cmd(cmd: dict = Body(...)):
    # Accept commands from the UI (pause/resume/stop/estop/goto/home)
    # For now: store last command and reflect it in status.
    ctype = cmd.get("type")
    _last_cmd["type"] = ctype
    _status["last_action"] = ctype if ctype else "--"
    return JSONResponse({"ok": True, "received": cmd})

@app.get("/api/frame.jpg")
def frame_jpg():
    frame = get_latest_frame_copy()
    if frame is None:
        return Response(status_code=503)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")
