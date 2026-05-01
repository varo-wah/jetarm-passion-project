# jetarm-passion-project
William Alvaro Hartono's passion project (AI Sorting Machine). An exploration of HiWonder JetArm Jetson Nano Orin NX.

## Project Structure

```text
jetarm-passion-project/
├── src/jetarm/
│   ├── config/      # constants, paths, robot/camera/vision/YOLO settings
│   ├── hardware/    # ROS servo and arm/gripper control
│   ├── kinematics/  # future IK/FK/joint-limit modules
│   ├── vision/      # runtime camera detection, scanner, YOLO inference
│   ├── ml/          # YOLO training and dataset pipeline code
│   ├── sorting/     # high-level sorting workflows
│   ├── ui/          # FastAPI dashboard, video stream, overlays
│   └── utils/       # shared helpers
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

Run the FastAPI dashboard:

```bash
PYTHONPATH=src python3 -m uvicorn jetarm.ui.server_app:app --reload
```

Run the scanner directly:

```bash
PYTHONPATH=src python3 -m jetarm.vision.scanner
```

Run manual tools:

```bash
PYTHONPATH=src python3 scripts/depth_viewer.py
PYTHONPATH=src python3 scripts/run_viewer.py
```

ROS 2 packages such as `rclpy`, `sensor_msgs`, `cv_bridge`, and `ros_robot_controller_msgs` must come from the JetArm ROS environment, not standard `pip`.
