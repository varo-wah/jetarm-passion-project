# ui_server/server_app.py
import os
import time
from typing import Generator, Optional

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ui_server.viewer_overlay import annotate_frame


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

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = annotate_frame(frame)
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
