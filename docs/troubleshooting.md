# Troubleshooting

- Use `./scripts/run_jetarm.sh --preview` or `./scripts/run_jetarm.sh --actuate`
  from the repository root.
- Never use a direct `PYTHONPATH=src` assignment in a ROS shell. It replaces
  the paths containing `rclpy` and JetArm message packages.
- If the actuation launcher reports missing ROS imports, open a JetArm ROS 2
  shell and rerun it. Do not install `rclpy` with pip.
- If pixel-to-robot conversion fails, verify the calibration files exist in `data/calibration/`.
