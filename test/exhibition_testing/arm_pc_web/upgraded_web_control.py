from flask import Flask, request, jsonify, render_template_string
import controller_api as ctrl
import os, glob

# ---- CONFIG ----
SERVOS = [1, 2, 3, 4, 5, 10]       # 6 DOF + gripper
PULSE_MIN, PULSE_MAX = 0, 1000     # JetArm range
DEFAULT_PULSE = 500
MOVE_TIME_MS = 500
ACTIONS_DIR = os.path.join(os.path.dirname(__file__), "ActionGroups")

# Flask app
app = Flask(__name__)

# ---- UTILITIES ----
def list_actions():
    if not os.path.isdir(ACTIONS_DIR):
        return []
    actions = []
    for path in glob.glob(os.path.join(ACTIONS_DIR, "*")):
        name = os.path.splitext(os.path.basename(path))[0]
        actions.append(name)
    return sorted(actions)

# ---- API ROUTES ----
@app.route('/move_joint', methods=['POST'])
def move_joint():
    data = request.json
    sid = int(data['servo'])
    pulse = int(data['pulse'])
    pulse = max(PULSE_MIN, min(PULSE_MAX, pulse))  # clamp
    ctrl.move_joint(sid, pulse, MOVE_TIME_MS)
    return jsonify(ok=True, servo=sid, pulse=pulse)

@app.route('/run_action', methods=['POST'])
def run_action():
    name = request.json['name']
    ctrl.run_action_group(name)
    return jsonify(ok=True)

@app.route('/stop_action', methods=['POST'])
def stop_action():
    ctrl.stop_action_group()
    return jsonify(ok=True)

@app.route('/unload', methods=['POST'])
def unload():
    sid = int(request.json['servo'])
    ctrl.unload(sid)
    return jsonify(ok=True)

# ---- CAMERA STREAM ----
try:
    import cv2
    camera = cv2.VideoCapture(0)
except Exception:
    camera = None

def gen_frames():
    if camera is None:
        return
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return app.response_class(gen_frames(),
                              mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- UI ----
@app.route('/ui')
def ui():
    sliders_html = ""
    for sid in SERVOS:
        sliders_html += f"""
        <div style="margin-bottom:10px;">
          <label>Servo {sid}</label>
          <input type="range" min="{PULSE_MIN}" max="{PULSE_MAX}" value="{DEFAULT_PULSE}"
                 oninput="moveServo({sid}, this.value)">
          <span id="val{sid}">{DEFAULT_PULSE}</span>
          <button onclick="unloadServo({sid})">Unload</button>
        </div>"""

    actions = list_actions()
    action_options = "".join([f"<option value='{a}'>{a}</option>" for a in actions])

    html = f"""
    <html>
    <head>
      <title>JetArm Web Dashboard</title>
      <style>
        body {{ font-family: Arial; margin:20px; }}
        h1 {{ color: #222; }}
        .container {{ display: flex; gap: 20px; }}
        .panel {{ padding: 15px; border: 1px solid #ccc; border-radius: 8px; width:45%; }}
        button {{ margin-left:10px; }}
      </style>
    </head>
    <body>
      <h1>JetArm Web Control</h1>
      <div class="container">
        <div class="panel">
          <h2>Joint Control</h2>
          {sliders_html}
          <h3>Gripper</h3>
          <button onclick="moveServo(10, 200)">Open</button>
          <button onclick="moveServo(10, 800)">Close</button>
        </div>
        <div class="panel">
          <h2>Camera</h2>
          <img src="/video_feed" style="max-width:100%;border:1px solid #ccc;border-radius:8px;" />
          <h2 style="margin-top:20px;">Actions</h2>
          <select id="actionSelect">{action_options}</select>
          <button onclick="runAction()">Run</button>
          <button onclick="stopAction()">Stop</button>
        </div>
      </div>
      <script>
      function moveServo(id, pulse) {{
        document.getElementById('val'+id).innerText = pulse;
        fetch('/move_joint', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{servo:id, pulse:pulse}})
        }});
      }}
      function unloadServo(id) {{
        fetch('/unload', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{servo:id}})
        }});
      }}
      function runAction() {{
        const name = document.getElementById('actionSelect').value;
        fetch('/run_action', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{name:name}})
        }});
      }}
      function stopAction() {{
        fetch('/stop_action', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{}})
        }});
      }}
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/')
def index():
    return '<p>Go to <a href="/ui">/ui</a> for the JetArm Dashboard.</p>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)