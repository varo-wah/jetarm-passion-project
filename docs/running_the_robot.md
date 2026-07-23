# Running The Robot

Run the FastAPI dashboard in non-actuating preview mode:

```bash
JETARM_ENABLE_ACTUATION=0 PYTHONPATH=src python3 -m uvicorn jetarm.ui.server_app:app --reload
```

Run one YOLO preview scan without moving the robot:

```bash
JETARM_ENABLE_ACTUATION=0 PYTHONPATH=src python3 -m jetarm.sorting.yolo_vision_scanner
```

Enable robot motion only during supervised, staged hardware validation:

```bash
JETARM_ENABLE_ACTUATION=1 PYTHONPATH=src python3 -m uvicorn jetarm.ui.server_app:app
```

The dashboard's stop, pause, E-stop, scanner, person-follow, and manual-motion
paths are software controls. They do not replace the robot's physical power
isolation or hardware emergency-stop procedure.
