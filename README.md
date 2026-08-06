# jetarm-passion-project
William Alvaro Hartono's passion project (AI Sorting Machine). An exploration of HiWonder JetArm Jetson Nano Orin NX.

## Project Structure

```text
jetarm-passion-project/
├── src/jetarm/
│   ├── config/      # constants, paths, robot/camera/vision/YOLO settings
│   ├── control/     # centralized ROS motion authority and clients
│   ├── hardware/    # arm/gripper compatibility API and current IK logic
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

Start the website in the default non-actuating preview mode:

```bash
./jetarm website
```

Then open `http://localhost:8000` on the JetArm. From another computer on the
same network, open `http://<JETARM_IP>:8000`.

Run one YOLO preview scan directly without moving the robot:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
JETARM_ENABLE_ACTUATION=0 \
python3 -m jetarm.sorting.yolo_vision_scanner
```

Robot motion and scanner auto-cycle require an explicit opt-in before server
startup:

```bash
./jetarm website --actuate
```

Both commands delegate to the same safety-aware launcher. The actuation path
stops Hiwonder's `start_app_node.service`, starts only the
low-level `ros_robot_controller` driver, verifies exclusive ownership of the
vendor servo command topic, and then starts `jetarm_control_node` plus the
dashboard. Actuation starts in
`BOOT_LOCKED`; press **Resume** once the workspace is clear. Stop or scanner stop
returns the controller to `PAUSED`, and clearing E-stop never resumes it.

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

Per-joint pulse ranges, speed limits, and the safe shutdown pose live in
`src/jetarm/config/joint_limits.yaml`. They are an initial software calibration
baseline and must be physically measured and signed off before Phase 0 is
declared complete.
