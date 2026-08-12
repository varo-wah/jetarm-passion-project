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
- On every camera start, the dashboard restores supported image controls to the
  V4L2 driver's defaults before OpenCV opens `/dev/video0`. If the startup log
  says `v4l2-ctl` is unavailable or the feed remains tinted, capture
  `v4l2-ctl -d /dev/video0 --all` instead of adding display filters.
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
