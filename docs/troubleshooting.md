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
- If pixel-to-robot conversion fails, verify the calibration files exist in `data/calibration/`.
