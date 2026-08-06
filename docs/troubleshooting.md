# Troubleshooting

- Use `./scripts/run_jetarm.sh --preview` or `./scripts/run_jetarm.sh --actuate`
  from the repository root.
- Never use a direct `PYTHONPATH=src` assignment in a ROS shell. It replaces
  the paths containing `rclpy` and JetArm message packages.
- If the actuation launcher reports missing ROS imports, open a JetArm ROS 2
  shell and rerun it. Do not install `rclpy` with pip.
- If startup reports another servo publisher, do not bypass the check. Inspect
  the listed node/process; centralized authority requires exactly one publisher.
- If startup reports `/dev/video0` is busy, stop the listed camera owner before
  retrying. The dashboard camera worker must be the device's only owner.
- The camera uses its driver defaults unless a `JETARM_CAMERA_*` override is
  explicitly provided. Keep the defaults for a natural image. If the raw feed
  is genuinely underexposed, tune `JETARM_CAMERA_BRIGHTNESS`,
  `JETARM_CAMERA_GAIN`, `JETARM_CAMERA_GAMMA`, or
  `JETARM_CAMERA_BACKLIGHT` individually in preview mode. The ROS
  `usb_cam_param.yaml` is not used by the dashboard's OpenCV capture path.
- Raw is always unmodified. OpenCV and YOLO display processing also defaults to
  natural output (`JETARM_DISPLAY_GAMMA=1.00` and
  `JETARM_DISPLAY_SATURATION=1.00`). Set display gamma below `1.00` only when a
  non-detection preview needs a modest shadow lift.
- If scanner preflight rejects a target, do not expand joint limits. Reposition
  the object or calibrate the bucket/arm geometry; no object has been gripped at
  the time of a preflight rejection.
- Read the dashboard alert by field: **Where** identifies the failed stage,
  **Code** classifies the fault, **Robot state** confirms whether motion is
  paused, **Object state** states whether an object may still be held, and
  **What to do** gives the recovery action. Acknowledging an alert never resumes
  motion or clears an E-stop.
- An amber `NO_SAFE_ROUTE` warning means detections were rejected before any
  pickup motion. A red abort/fault means the scanner auto-cycle was stopped and
  motion authority was paused. Preserve the copied diagnostics when reporting a
  repeatable failure.
- If pixel-to-robot conversion fails, verify the calibration files exist in `data/calibration/`.
