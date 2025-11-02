from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

def find_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✅ Found camera at index {i}")
                return i
            else:
                print(f"⚠️ Camera {i} opened but no frame.")
        else:
            print(f"❌ Camera {i} not opened.")
    return -1

camera_index = find_camera_index()

# Lazily open the camera when stream requested (avoids stale handles)
def open_camera(idx):
    if idx < 0:
        return None
    cap = cv2.VideoCapture(idx)
    # (optional) set preferred resolution; adjust to a supported mode
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def gen_frames():
    cap = open_camera(camera_index)
    if cap is None or not cap.isOpened():
        print("❌ gen_frames: camera failed to open.")
        return
    frame_count = 0
    last_log = time.time()
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("⚠️ gen_frames: failed to read frame; stopping stream.")
            break

        # Periodic console log so we know it’s alive
        frame_count += 1
        now = time.time()
        if now - last_log >= 5:
            last_log = now

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("⚠️ gen_frames: JPEG encode failed; stopping.")
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    print("gen_frames: camera released.")

@app.route('/')
def index():
    if camera_index == -1:
        return "<h1>❌ No camera detected on Jetson</h1>"
    # Add width=auto to avoid Safari layout quirks
    return "<h1>JetArm Camera</h1><img src='/video_feed' style='max-width:100%;height:auto;'>"

@app.route('/video_feed')
def video_feed():
    if camera_index == -1:
        return "Camera not available", 503
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # threaded=True is default in debug; ensure multi-client
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)