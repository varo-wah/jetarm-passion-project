# Phase 0 Safety Status

## Software implemented on the feature branch

- ROS 2 action and services replace process-local safety state for actuating runs.
- `jetarm_control_node` is the only vendor servo-topic publisher.
- Multi-servo targets are validated and published as synchronized group frames.
- Six servo IDs have independent pulse ranges, speed limits, and safe-pose values.
- Stop and scanner stop pause the controller before terminating scanner work.
- Safe shutdown cancels work and uses the safe pose only from a known commanded state.
- E-stop clear always leaves the controller paused.

## Required JetArm validation

1. Start with `./scripts/run_jetarm.sh --actuate`; confirm the vendor app is
   stopped, the SDK is launcher-owned, and the dashboard remains paused.
2. Confirm `/jetarm_controller/follow_joint_trajectory` and all five controller services exist.
3. Confirm exactly one publisher exists on `/ros_robot_controller/bus_servo/set_position`.
4. Measure and update every range in `src/jetarm/config/joint_limits.yaml`.
5. Test Resume, low-speed safe pose, Pause, scanner Stop, E-stop, and shutdown in that order.

## Hardware blocker

Phase 0 is **not complete** until a maintained physical E-stop interrupts the
12 V STM32/servo power or command path independently of Linux and ROS.

The repository has no verified live servo-position feedback. Therefore the
first command after controller startup relies on the vendor controller's own
duration. Software can reject later frames but cannot guarantee interruption
of that already accepted first command. Keep actuation supervised and do not
claim immediate-stop certification until the physical E-stop is installed and
tested.
