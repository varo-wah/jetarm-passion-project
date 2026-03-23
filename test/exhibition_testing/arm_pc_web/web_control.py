from flask import Flask, request, jsonify, render_template_string
import controller_api as ctrl
import os

SERVOS = [1, 2, 3, 4, 5, 10]
PULSE_MIN, PULSE_MAX = 0, 1000
DEFAULT_PULSE = 500

app = Flask(__name__)

@app.route('/move_joint', methods=['POST'])
def move_joint():
    data = request.json
    sid = int(data['servo'])
    pulse = int(data['pulse'])
    ctrl.move_joint(sid, pulse)
    return jsonify(ok=True, servo=sid, pulse=pulse)

@app.route('/ui')
def ui():
    sliders = ""
    for sid in SERVOS:
        sliders += f"""
        <div>
            <label>Servo {sid}</label>
            <input type="range" min="{PULSE_MIN}" max="{PULSE_MAX}" value="{DEFAULT_PULSE}"
                   oninput="moveServo({sid}, this.value)">
            <span id="val{sid}">{DEFAULT_PULSE}</span>
        </div>
        """
    html = f"""
    <html>
    <head><title>JetArm Web Control</title></head>
    <body>
      <h1>JetArm Control Panel</h1>
      {sliders}
      <script>
      function moveServo(id, pulse) {{
        document.getElementById('val'+id).innerText = pulse;
        fetch('/move_joint', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{servo:id, pulse:pulse}})
        }});
      }}
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/')
def index():
    return '<p>Go to <a href="/ui">/ui</a> to control the JetArm.</p>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)