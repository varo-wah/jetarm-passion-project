# Architecture

The runtime flow is:

camera -> vision -> sorting -> kinematics -> hardware -> UI

The `vision` package handles runtime perception. The `ml` package is reserved for YOLO dataset preparation, training, validation, and export.

