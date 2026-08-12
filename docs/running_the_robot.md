# Running The Robot

Run the FastAPI dashboard in non-actuating preview mode:

```bash
./scripts/run_jetarm.sh --preview
```

Run one YOLO preview scan without moving the robot:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
JETARM_ENABLE_ACTUATION=0 \
python3 -m jetarm.sorting.yolo_vision_scanner
```

Enable robot motion only during supervised, staged hardware validation:

```bash
./scripts/run_jetarm.sh --actuate
```

The launcher preserves the JetArm ROS environment and checks `rclpy` plus
`ros_robot_controller_msgs` before starting actuation mode. Do not replace the
ROS path with a direct `PYTHONPATH=src` assignment.

Before opening `/dev/video0`, the dashboard restores supported brightness,
contrast, saturation, exposure, gain, gamma, focus, and white-balance controls
to the V4L2 driver's declared defaults. Raw, OpenCV, and YOLO views use the
camera frame directly; no display enhancement is applied.

The dashboard's stop, pause, E-stop, scanner, person-follow, and manual-motion
paths are software controls. They do not replace the robot's physical power
isolation or hardware emergency-stop procedure.
