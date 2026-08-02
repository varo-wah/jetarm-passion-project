# Architecture

The runtime flow is:

camera -> vision -> sorting/UI -> control client -> jetarm_control_node -> vendor servo topic

The `vision` package handles runtime perception. The `ml` package is reserved for YOLO dataset preparation, training, validation, and export.

`jetarm_control_node` is the sole servo-topic publisher. It owns the shared ROS
state machine, per-joint validation, synchronized interpolation, goal
serialization, cancellation, and safe shutdown. Hardware compatibility modules
calculate IK and submit goals; they do not publish servo commands.

Priority is physical E-stop, software E-stop, pause/stop, shutdown, then normal
motion. The physical E-stop remains a hardware completion gate and must break
the 12 V STM32/servo command-power path independently of this software.
