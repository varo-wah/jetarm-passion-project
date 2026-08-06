# Phase 0 Safety Status

## Software implemented in the current integration checkout

- Actuating runs use one ROS 2 motion authority and one low-level serial driver.
- The launcher refuses competing servo publishers and camera owners.
- Controller readiness uses ROS action/service discovery; it no longer creates
  repeated short-lived `JetArmControlClient` processes.
- Motion starts `BOOT_LOCKED`; the process-local fallback starts paused.
- Pause, Stop, E-stop, and Clear E-stop report controller rejection instead of
  presenting false success in the dashboard.
- Six servo IDs have independent pulse, velocity, and safe-pose limits.
- Multi-servo targets use synchronized frames. Unsafe requested durations are
  extended to the shortest duration permitted by calibrated velocity limits.
- Pure IK planning validates table-tip and wrist targets before command
  submission. Scanner preflight validates the complete pick/drop route.
- Scanner aborts are authenticated, displayed with exact stage/reason/object
  state, stop auto-cycle, and pause motion. Alert acknowledgement never resumes.
- Natural camera-driver output is the default; exposure and display adjustments
  are explicit preview-mode overrides.
- Safe shutdown cancels work and uses the safe pose only from a known commanded
  state. E-stop clear always remains paused.

Local software validation on 2026-08-06: 75 tests passed. This is not physical
certification.

## Required JetArm validation

1. Run `./scripts/run_jetarm.sh --preview`; verify natural camera output, YOLO
   colors, alert rendering, and clean shutdown without servo power.
2. Install and test the independent physical E-stop before software actuation.
3. Start `./scripts/run_jetarm.sh --actuate`; confirm the vendor application is
   stopped, only the low-level driver is launcher-owned, and the dashboard shows
   `BOOT_LOCKED`.
4. Confirm the trajectory action, all five services, and exactly one publisher
   on `/ros_robot_controller/bus_servo/set_position`.
5. Measure and update every range, velocity, and safe pose in
   `src/jetarm/config/joint_limits.yaml`.
6. Test Resume, a low-speed known pose, Pause, scanner Stop, software E-stop,
   E-stop clear, shutdown, and measured stopping latency in that order.

## Hardware blocker

Phase 0 is **not complete** until a maintained physical E-stop interrupts the
12 V STM32/servo power or command path independently of Linux and ROS.

The repository has no verified live servo-position feedback. Therefore the
first command after controller startup relies on the vendor controller's own
duration. Software can reject later frames but cannot guarantee interruption
of that already accepted first command. Keep actuation supervised and do not
claim immediate-stop certification until the physical E-stop is installed and
tested.
