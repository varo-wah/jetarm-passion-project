# Architecture

## Runtime flow

```text
camera worker -> vision -> sorting/UI -> control client
    -> jetarm_control_node -> low-level ros_robot_controller -> servos
```

The launcher starts only the low-level serial driver. It refuses actuation when
another publisher owns `/ros_robot_controller/bus_servo/set_position`, and it
checks controller readiness through ROS interface discovery rather than creating
repeated short-lived ROS clients.

## Motion ownership

`src/jetarm/control/control_node.py` is the sole servo-command publisher. It owns
the shared state machine, per-joint validation, synchronized interpolation,
safe-duration extension, goal serialization, cancellation, and shutdown. The
dashboard and scanner can only request motion through
`src/jetarm/control/client.py`.

Pure Cartesian planning lives in
`src/jetarm/hardware/inverse_kinematics.py`. It converts table-tip or wrist
targets into joint angles and validates the resulting pulses against the same
`joint_limits.yaml` used by the controller. `Class_Execution.py` remains a
motion-gated compatibility facade; it does not publish hardware topics.

## Perception and sorting

`src/jetarm/vision/` owns runtime perception. Color classification has one
canonical implementation in `color_detection.py`; the legacy and YOLO overlays
share it. Display appearance is isolated from detector input, and the default
camera path preserves natural driver pixels.

Before pickup, `sorting/yolo_vision_scanner.py` validates the complete target,
transfer, bucket, wrist, and gripper route. Scanner warnings and aborts use an
authenticated child-to-dashboard event channel. Abort, fault, and E-stop events
pause motion and stop auto-cycle; acknowledging an alert never resumes motion.

## Safety priority

Physical E-stop has priority over software E-stop, pause/stop, shutdown, and
normal motion. Software states start fail-closed in `BOOT_LOCKED` or `PAUSED`.
The physical E-stop remains a hardware completion gate and must interrupt the
12 V STM32/servo power or command path independently of Linux and ROS.
