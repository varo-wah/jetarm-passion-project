# Architecture

The runtime flow is:

camera -> vision -> sorting -> hardware -> UI

The `vision` package handles runtime perception. The `ml` package is reserved for YOLO dataset preparation, training, validation, and export.

Current IK logic lives in `hardware/arm_controller.py` because it remains coupled to servo pulse conversion and physical arm commands.
