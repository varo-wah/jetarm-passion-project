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
`ros_robot_controller_msgs` before starting actuation mode. It also stops the
vendor auto-start application, launches only the low-level
`ros_robot_controller` driver, refuses competing
servo publishers or camera owners, and keeps the ROS processes alive until the
dashboard has completed safe shutdown. Do not replace the ROS path with a direct
`PYTHONPATH=src` assignment.

The vendor service remains stopped after this launcher exits because restarting
it can command an initial pose. Reboot to restore the normal vendor application,
or restart it manually only after the servo power and workspace are safe.

The dashboard camera worker applies a repeatable low-light profile when it opens
`/dev/video0`: brightness `12`, gain `20`, gamma `140`, and backlight
compensation `2`. Override individual values before launching when lighting
conditions change:

```bash
JETARM_CAMERA_BRIGHTNESS=7 \
JETARM_CAMERA_GAIN=11 \
JETARM_CAMERA_GAMMA=120 \
JETARM_CAMERA_BACKLIGHT=1 \
./scripts/run_jetarm.sh --actuate
```

These settings belong to the dashboard's OpenCV camera owner. Editing the
vendor `usb_cam_param.yaml` does not configure this capture path.

Motion requests retain their requested duration when it is already safe. If a
request would exceed a calibrated joint velocity, the central controller
automatically extends only that trajectory to the shortest permitted duration.
The scanner also preflights the full target, transfer, bucket, wrist, and
gripper route before moving toward a detected object.

The dashboard's stop, pause, E-stop, scanner, person-follow, and manual-motion
paths are software controls. They do not replace the robot's physical power
isolation or hardware emergency-stop procedure.
