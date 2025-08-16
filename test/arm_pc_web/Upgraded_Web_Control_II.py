#!/usr/bin/env python3
from flask import Flask, render_template_string, request, jsonify

# ====== CAMERA STREAM =======
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

# ====== CONFIG ======
PULSE_MIN, PULSE_MAX = 0, 1000
LIVE_MOVE_TIME_MS = 200
APPLY_MOVE_TIME_MS = 2500  # 2.5 seconds for apply now

ARM_JOINTS = [
    {"name": "J1", "servo": 2, "min_deg": 0, "max_deg": 180, "length": 130},
    {"name": "J2", "servo": 3, "min_deg": 0, "max_deg": 180, "length": 110},
    {"name": "J3", "servo": 4, "min_deg": 0, "max_deg": 180, "length": 90},
]

# Extra servos: 5 (wrist), 10 (gripper)
EXTRA_SERVOS = [
    {"name": "Wrist", "servo": 5, "angle": 90},
    {"name": "Gripper", "servo": 10, "angle": 90},
]

USE_HW = True
try:
    import controller_api as ctrl
except Exception as e:
    print("[WARN] controller_api not found, using SIM mode:", e)
    USE_HW = False

def to_pulse(deg):
    return int(round(PULSE_MIN + (deg / 180.0) * (PULSE_MAX - PULSE_MIN)))

last_pulses = {
    "base": 500,
    2: 500,
    3: 500,
    4: 500,
    5: 500,
    10: 500
}

def hw_move_all(base_angle, joint_angles, extra_angles):
    pulses = {
        "base": to_pulse(base_angle),
        2: to_pulse(joint_angles[0]),
        3: to_pulse(joint_angles[1]),
        4: to_pulse(joint_angles[2]),
        5: to_pulse(extra_angles[0]),
        10: to_pulse(extra_angles[1])
    }

    # Calculate max delta among all servos
    max_delta = max(abs(pulses[k] - last_pulses[k]) for k in pulses)
    move_time = int((max_delta / 1000) * 5000)
    move_time = max(200, move_time)  # minimum 200ms

    # Update last pulses
    for k in pulses:
        last_pulses[k] = pulses[k]

    if not USE_HW:
        print(f"[SIM] Move {pulses} in {move_time}ms")
        return

    ctrl.move_joint(1, pulses["base"], move_time)
    ctrl.move_joint(2, pulses[2], move_time)
    ctrl.move_joint(3, pulses[3], move_time)
    ctrl.move_joint(4, pulses[4], move_time)
    ctrl.move_joint(5, pulses[5], move_time)
    ctrl.move_joint(10, pulses[10], move_time)

# ====== STATE ======
live_enabled = False

# ====== FLASK ======
app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>JetArm Control (With Gripper)</title>
<style>
  body { background:#f5f5f5; font-family:Arial,sans-serif; text-align:center; margin:0; }
  canvas { display:block; margin:16px auto; background:#f5f5f5; }
  #info { margin:12px 0 24px; font-size:14px; }
  button { padding:6px 12px; margin:8px; }
  #liveStatus.off { color:#b30000; font-weight:bold; }
  #liveStatus.on  { color:#0a7a0a; font-weight:bold; }
</style>
</head>
<body>
<h1>JetArm Control (With Gripper)</h1>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 40px; margin: 0 auto; max-width: 1900px;">

  <!-- Left side: Robot control -->
  <div style="width: 1000px; text-align:center;">
    <h1>JetArm Control (With Gripper)</h1>
    <div>
      <span id="liveStatus" class="off">LIVE CONTROL: OFF (SAFE)</span><br/>
      <button onclick="toggleLive()">Toggle Live Control</button>
      <button onclick="applyNow()">Apply to Robot Now</button>
      <button onclick="home()">Home (500 pulse each)</button>
    </div>
    <canvas id="cv" width="1000" height="700"></canvas>
    <div id="info">
      <div id="baseInfo"></div>
      <div id="angles"></div>
      <div id="pulses"></div>
      <div id="extras"></div>
    </div>
  </div>

  <!-- Right side: Camera -->
  <div style="width: 800px; padding: 10px; background: #fff; border: 1px solid #ccc; border-radius: 8px;">
    <h2>Camera</h2>
    <div style="width: 100%; height: 700px; display: flex; align-items: center; justify-content: center; background: #000;">
      <img src="/video_feed" style="width: 800px; height: auto; border: 1px solid #ccc; border-radius: 8px;" />
    </div>
  </div>

</div>

<script>
const ARM_JOINTS = {{ joints|tojson }};
const EXTRA_SERVOS = {{ extras|tojson }};
const PULSE_MIN = {{ pmin }};
const PULSE_MAX = {{ pmax }};
let LIVE_ENABLED = {{ live_init | lower }};

const state = {
  base: { angle: 90 },
  arm: ARM_JOINTS.map(j => ({ ...j, angle: 90, base:{x:0,y:0}, tip:{x:0,y:0}, cumAngle:90 })),
  extra: EXTRA_SERVOS.map(s => ({ ...s, angle: 90 })),
  armBase: { x: 400, y: 450 },
  arcRadius: 65,
  baseDial: { x: 800, y: 300, radius: 80 },
  extraControls: [
    { x: 250, y: 600, radius: 60 }, // Wrist (Servo 5)
    { x: 500, y: 600, radius: 60 }  // Gripper (Servo 10)
  ],
  dragging: null
};

function deg2rad(d){ return d * Math.PI / 180; }
function rad2deg(r){ return r * 180 / Math.PI; }
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function toPulse(d){ return Math.round(PULSE_MIN + (d/180)*(PULSE_MAX-PULSE_MIN)); }

function computeFK(){
  let bx = state.armBase.x;
  let by = state.armBase.y;
  let cumAngle = 90;
  for (let i=0; i<state.arm.length; i++){
    const j = state.arm[i];
    j.base.x = bx;
    j.base.y = by;
    cumAngle += (j.angle - 90);
    j.cumAngle = cumAngle;
    bx += Math.cos(deg2rad(cumAngle)) * j.length;
    by -= Math.sin(deg2rad(cumAngle)) * j.length;
    j.tip.x = bx;
    j.tip.y = by;
  }
}

function draw(){
  computeFK();
  const ctx = document.getElementById("cv").getContext("2d");
  ctx.clearRect(0,0,1000,700);

  // Base Dial
  ctx.strokeStyle="#333";
  ctx.lineWidth=3;
  ctx.beginPath();
  ctx.arc(state.baseDial.x, state.baseDial.y, state.baseDial.radius, 0, Math.PI*2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(state.baseDial.x, state.baseDial.y);
  ctx.lineTo(
    state.baseDial.x + state.baseDial.radius * Math.cos(deg2rad(state.base.angle)),
    state.baseDial.y - state.baseDial.radius * Math.sin(deg2rad(state.base.angle))
  );
  ctx.strokeStyle="blue";
  ctx.lineWidth=4;
  ctx.stroke();

  // Arm
  for (let i=0; i<state.arm.length; i++){
    const j = state.arm[i];
    ctx.strokeStyle = ["#444","#666","#999"][i%3];
    ctx.lineWidth = 16 - i*3;
    ctx.beginPath();
    ctx.moveTo(j.base.x, j.base.y);
    ctx.lineTo(j.tip.x, j.tip.y);
    ctx.stroke();

    ctx.strokeStyle="#00aaff";
    ctx.lineWidth=4;
    ctx.beginPath();
    ctx.arc(j.base.x, j.base.y, state.arcRadius, Math.PI, 0, false);
    ctx.stroke();

    const t = deg2rad(j.angle);
    const hx = j.base.x + state.arcRadius * Math.cos(t);
    const hy = j.base.y - state.arcRadius * Math.sin(t);
    ctx.fillStyle="blue";
    ctx.beginPath();
    ctx.arc(hx, hy, 6, 0, Math.PI*2);
    ctx.fill();
  }

  // Extra Controls (Wrist & Gripper)
  for (let i=0; i<state.extra.length; i++){
    const s = state.extra[i];
    const ec = state.extraControls[i];
    ctx.strokeStyle="#00aaff";
    ctx.lineWidth=4;
    ctx.beginPath();
    ctx.arc(ec.x, ec.y, ec.radius, Math.PI, 0, false);
    ctx.stroke();
    const t = deg2rad(s.angle);
    const hx = ec.x + ec.radius * Math.cos(t);
    const hy = ec.y - ec.radius * Math.sin(t);
    ctx.fillStyle="blue";
    ctx.beginPath();
    ctx.arc(hx, hy, 6, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle="#000";
    ctx.fillText(s.name, ec.x - 20, ec.y + ec.radius + 15);
  }

  updateInfo();
}

function updateInfo(){
  document.getElementById("baseInfo").innerHTML =
    `<b>Base:</b> ${state.base.angle.toFixed(1)}° (Pulse: ${toPulse(state.base.angle)})`;
  const a = state.arm.map((j,i)=>`${j.name}: ${j.angle.toFixed(1)}°`).join(" &nbsp; ");
  const p = state.arm.map((j,i)=>`${j.name}: ${toPulse(j.angle)}`).join(" &nbsp; ");
  const extras = state.extra.map(s=>`${s.name}: ${s.angle.toFixed(1)}° (Pulse: ${toPulse(s.angle)})`).join(" &nbsp; ");
  document.getElementById("angles").innerHTML = "<b>Arm Angles:</b> " + a;
  document.getElementById("pulses").innerHTML = "<b>Arm Pulses:</b> " + p;
  document.getElementById("extras").innerHTML = "<b>Gripper:</b> " + extras;
  const el = document.getElementById("liveStatus");
  el.className = LIVE_ENABLED ? "on" : "off";
  el.textContent = LIVE_ENABLED ? "LIVE CONTROL: ON" : "LIVE CONTROL: OFF (SAFE)";
}

function home(){
  state.base.angle = 90;
  state.arm.forEach(j => j.angle=90);
  state.extra.forEach(s => s.angle=90);
  draw();
  if (LIVE_ENABLED) pushAngles();
}

function mousePos(e){
  const r = cv.getBoundingClientRect();
  return { x:e.clientX-r.left, y:e.clientY-r.top };
}

function hitArc(mx,my){
  // Base dial
  const dx0 = mx - state.baseDial.x;
  const dy0 = my - state.baseDial.y;
  if (dx0*dx0 + dy0*dy0 <= (state.baseDial.radius+10)*(state.baseDial.radius+10))
    return {type:"base", idx:-1};

  // Extra controls
  for (let i=0; i<state.extraControls.length; i++){
    const ec = state.extraControls[i];
    const dx = mx - ec.x;
    const dy = my - ec.y;
    const d2 = dx*dx+dy*dy;
    if (d2 <= (ec.radius+10)*(ec.radius+10) && dy <= 0)
      return {type:"extra", idx:i};
  }

  // Arm arcs
  const r = state.arcRadius, tol=10;
  const rmin2 = (r-tol)*(r-tol);
  const rmax2 = (r+tol)*(r+tol);
  for (let i=state.arm.length-1; i>=0; i--){
    const j = state.arm[i];
    const dx=mx-j.base.x;
    const dy=my-j.base.y;
    const d2=dx*dx+dy*dy;
    if (d2>=rmin2 && d2<=rmax2 && dy<=0) return {type:"arm", idx:i};
  }
  return null;
}

function setAngleFromMouse(target, mx, my){
  if (target.type==="base") {
    let ang = rad2deg(Math.atan2(state.baseDial.y - my, mx - state.baseDial.x));
    ang = (ang + 360) % 360;
    if (ang>180) ang=180;
    state.base.angle = ang;
  } else if (target.type==="extra") {
    const ec = state.extraControls[target.idx];
    const dx = mx - ec.x;
    const dy = ec.y - my;
    let ang = rad2deg(Math.atan2(dy, dx));
    ang = clamp(ang, 0, 180);
    state.extra[target.idx].angle = ang;
  } else {
    const j = state.arm[target.idx];
    const dx = mx - j.base.x;
    const dy = j.base.y - my;
    let ang = rad2deg(Math.atan2(dy, dx));
    ang = clamp(ang, 0, 180);
    j.angle = ang;
  }
  if (LIVE_ENABLED) pushAngles();
}

function pushAngles() {
  fetch("/send_angles", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      base: state.base.angle,
      angles: state.arm.map(j => j.angle),
      extras: state.extra.map(s => s.angle)
    })
  });
}

function applyNow() {
  fetch("/apply_now", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      base: state.base.angle,
      angles: state.arm.map(j => j.angle),
      extras: state.extra.map(s => s.angle)
    })
  });
}

function toggleLive(){
  LIVE_ENABLED=!LIVE_ENABLED;
  fetch("/set_live", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({enabled: LIVE_ENABLED})
  }).then(()=>{ updateInfo(); if (LIVE_ENABLED) pushAngles(); });
}

const cv=document.getElementById("cv");
let dragTarget=null;
cv.addEventListener("mousedown",e=>{
  const m=mousePos(e);
  dragTarget = hitArc(m.x,m.y);
  if(dragTarget){ setAngleFromMouse(dragTarget,m.x,m.y); draw(); }
});
window.addEventListener("mousemove",e=>{
  if(!dragTarget) return;
  const m=mousePos(e);
  setAngleFromMouse(dragTarget,m.x,m.y);
  draw();
});
window.addEventListener("mouseup",()=>dragTarget=null);

draw();
updateInfo();
</script>
</body>
</html>
"""

@app.route('/video_feed')
def video_feed():
    return app.response_class(gen_frames(),
                              mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return render_template_string(
        HTML,
        joints=ARM_JOINTS,
        extras=EXTRA_SERVOS,
        pmin=PULSE_MIN,
        pmax=PULSE_MAX,
        live_init=str(False)
    )

@app.route("/set_live", methods=["POST"])
def set_live():
    global live_enabled
    data = request.get_json(force=True)
    live_enabled = bool(data.get("enabled", False))
    return jsonify(ok=True, live=live_enabled)

@app.route("/send_angles", methods=["POST"])
def send_angles():
    if not live_enabled:
        return jsonify(ok=False, reason="live_disabled")
    data = request.get_json(force=True)
    hw_move_all(data.get("base",90), data.get("angles",[]), data.get("extras",[]))
    return jsonify(ok=True)

@app.route("/apply_now", methods=["POST"])
def apply_now():
    data = request.get_json(force=True)
    hw_move_all(data.get("base",90), data.get("angles",[]), data.get("extras",[]))
    return jsonify(ok=True)

if __name__ == "__main__":
    print("Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)


