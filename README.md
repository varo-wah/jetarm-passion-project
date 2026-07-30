# jetarm-passion-project
William Alvaro Hartono's passion project (AI Sorting Machine). An exploration of HiWonder JetArm Jetson Nano Orin NX.

## Project Structure

```text
jetarm-passion-project/
├── src/jetarm/
│   ├── config/      # constants, paths, robot/camera/vision/YOLO settings
│   ├── hardware/    # ROS servo, arm/gripper control, and current IK logic
│   ├── vision/      # runtime camera detection, scanner, YOLO inference
│   ├── ml/          # YOLO training and dataset pipeline code
│   ├── sorting/     # high-level sorting workflows
│   └── ui/          # FastAPI dashboard, video stream, overlays
├── scripts/         # manual demos, calibration, camera/model checks
├── tests/           # automated tests
├── data/            # datasets and calibration matrices
├── models/          # pretrained, trained, and exported model files
├── docs/            # setup, architecture, calibration, troubleshooting
└── archive/         # preserved legacy experiments and prototypes
```

## Run Commands

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the FastAPI dashboard in the default non-actuating preview mode:

```bash
./scripts/run_jetarm.sh --preview
```

Run one YOLO preview scan directly without moving the robot:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
JETARM_ENABLE_ACTUATION=0 \
python3 -m jetarm.sorting.yolo_vision_scanner
```

Robot motion and scanner auto-cycle require an explicit opt-in before server
startup:

```bash
./scripts/run_jetarm.sh --actuate
```

Use actuation mode only after the E-stop path, workspace, calibration, and scan
pose have been checked on staged hardware. Resume never clears an E-stop latch;
clearing and resuming are deliberately separate operations.

Run manual tools:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/depth_viewer.py
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/run_viewer.py
```

ROS 2 packages such as `rclpy`, `sensor_msgs`, `cv_bridge`, and `ros_robot_controller_msgs` must come from the JetArm ROS environment, not standard `pip`.
Do not use a direct `PYTHONPATH=src` assignment in a ROS shell: it replaces
the ROS package paths. The launcher preserves them and verifies hardware imports
before actuation mode starts.

For now, IK stays in `src/jetarm/hardware/arm_controller.py` because it is still tightly connected to servo pulse conversion and arm movement. Split it into a separate `kinematics.py` module only when it becomes independently testable.
