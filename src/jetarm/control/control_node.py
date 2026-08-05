"""ROS 2 node that is the sole publisher to the JetArm servo command topic."""

from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition
from std_msgs.msg import String
from std_srvs.srv import Trigger

from jetarm.control.authority import ControlState, MotionAuthority
from jetarm.control.limits import (
    JointLimitsError,
    fastest_safe_duration,
    limits_by_name,
    load_joint_limits,
)


ACTION_NAME = "/jetarm_controller/follow_joint_trajectory"
STATE_TOPIC = "/jetarm_controller/state"
VENDOR_COMMAND_TOPIC = "/ros_robot_controller/bus_servo/set_position"
FRAME_PERIOD_SECONDS = 0.05
MAX_VENDOR_FRAME_SECONDS = 0.10


class JetArmControlNode(Node):
    """Validate, serialize, synchronize, and pre-empt every servo command."""

    def __init__(self) -> None:
        super().__init__("jetarm_control_node")
        self._limits = load_joint_limits()
        self._limits_by_name = limits_by_name(self._limits)
        self._authority = MotionAuthority(self._limits)
        self._callback_group = ReentrantCallbackGroup()
        self._execution_lock = threading.RLock()
        self._last_positions: dict[int, int] = {}

        self._servo_publisher = self.create_publisher(
            ServosPosition,
            VENDOR_COMMAND_TOPIC,
            10,
        )
        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._state_publisher = self.create_publisher(String, STATE_TOPIC, state_qos)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
            execute_callback=self._execute_goal,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/jetarm_controller/pause",
            self._pause_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/jetarm_controller/resume",
            self._resume_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/jetarm_controller/estop",
            self._estop_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/jetarm_controller/clear_estop",
            self._clear_estop_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            "/jetarm_controller/safe_shutdown",
            self._safe_shutdown_callback,
            callback_group=self._callback_group,
        )
        self.create_timer(0.2, self._publish_state, callback_group=self._callback_group)
        self._publish_state()
        self.get_logger().info("JetArm controller ready in BOOT_LOCKED state")

    def _publish_state(self) -> None:
        self._state_publisher.publish(
            String(data=json.dumps(self._authority.snapshot().as_dict(), sort_keys=True))
        )

    def _set_trigger_response(self, response, success: bool, message: str):
        response.success = bool(success)
        response.message = message
        self._publish_state()
        return response

    def _pause_callback(self, request, response):
        del request
        ok = self._authority.pause()
        return self._set_trigger_response(response, ok, self._authority.snapshot().reason)

    def _resume_callback(self, request, response):
        del request
        ok = self._authority.resume()
        message = self._authority.snapshot().reason if ok else "Resume blocked by controller state"
        return self._set_trigger_response(response, ok, message)

    def _estop_callback(self, request, response):
        del request
        self._authority.estop()
        return self._set_trigger_response(response, True, "Software E-stop latched")

    def _clear_estop_callback(self, request, response):
        del request
        ok = self._authority.clear_estop()
        message = (
            "E-stop cleared; controller remains paused"
            if ok
            else "E-stop was not latched"
        )
        return self._set_trigger_response(response, ok, message)

    def _safe_shutdown_callback(self, request, response):
        del request
        snapshot = self._authority.snapshot()
        if snapshot.state in {
            ControlState.BOOT_LOCKED,
            ControlState.ESTOP_LATCHED,
            ControlState.FAULT,
            ControlState.SHUTTING_DOWN,
        }:
            if snapshot.state == ControlState.BOOT_LOCKED:
                self._authority.begin_shutdown()
            return self._set_trigger_response(
                response,
                True,
                f"Safe-pose motion skipped from {snapshot.state.value}",
            )

        self._authority.begin_shutdown()
        if not self._last_positions:
            return self._set_trigger_response(
                response,
                True,
                "No commanded pose is known; shutdown locked without movement",
            )
        safe_pose = {
            servo_id: joint.safe_pose_pulse
            for servo_id, joint in self._limits.items()
        }
        with self._execution_lock:
            self._move_to_targets(
                safe_pose,
                duration=1.2,
                generation=None,
                allow_shutdown=True,
            )
        return self._set_trigger_response(response, True, "Safe pose commanded; shutdown locked")

    def _goal_callback(self, goal_request) -> GoalResponse:
        snapshot = self._authority.snapshot()
        if not snapshot.motion_allowed or snapshot.active_goal:
            return GoalResponse.REJECT
        try:
            self._trajectory_targets(goal_request.trajectory)
        except (JointLimitsError, ValueError) as exc:
            self.get_logger().warning(f"Rejected motion goal: {exc}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        del goal_handle
        self._authority.pause("trajectory cancelled; explicit resume required")
        self._publish_state()
        return CancelResponse.ACCEPT

    def _trajectory_targets(self, trajectory):
        names = list(trajectory.joint_names)
        if not names or len(names) != len(set(names)):
            raise JointLimitsError("Trajectory joint names must be non-empty and unique")
        if len(trajectory.points) != 1:
            raise JointLimitsError("JetArm goals must contain exactly one synchronized target point")

        point = trajectory.points[0]
        if len(point.positions) != len(names):
            raise JointLimitsError("Trajectory positions must match joint names")
        duration = float(point.time_from_start.sec) + (point.time_from_start.nanosec / 1e9)
        if not math.isfinite(duration) or duration <= 0:
            raise JointLimitsError("Trajectory duration must be positive and finite")

        targets: dict[int, int] = {}
        for name, radians in zip(names, point.positions):
            joint = self._limits_by_name.get(name)
            if joint is None:
                raise JointLimitsError(f"Unknown joint name: {name}")
            targets[joint.servo_id] = joint.radians_to_pulse(radians)

        return targets, duration

    def _execute_goal(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            raw_targets, duration = self._trajectory_targets(goal_handle.request.trajectory)
            generation, targets = self._authority.begin_goal(raw_targets)
            self._publish_state()
            with self._execution_lock:
                completed = self._move_to_targets(targets, duration, generation)

            if completed:
                self._authority.finish_goal(generation)
                goal_handle.succeed()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Synchronized command completed"
            else:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Motion pre-empted by controller priority state"
        except Exception as exc:
            self._authority.pause(f"motion goal failed: {exc}")
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
        self._publish_state()
        return result

    def _publish_frame(
        self,
        positions: dict[int, int],
        duration: float,
        *,
        cap_duration: bool = True,
    ) -> None:
        validated = {
            servo_id: self._limits[servo_id].validate_pulse(pulse)
            for servo_id, pulse in positions.items()
        }
        message = ServosPosition(
            duration=float(
                min(duration, MAX_VENDOR_FRAME_SECONDS) if cap_duration else duration
            ),
            position=[
                ServoPosition(id=servo_id, position=pulse)
                for servo_id, pulse in sorted(validated.items())
            ],
        )
        self._servo_publisher.publish(message)
        self._last_positions.update(validated)

    def _move_to_targets(
        self,
        targets: dict[int, int],
        duration: float,
        generation: int | None,
        allow_shutdown: bool = False,
    ) -> bool:
        if not all(servo_id in self._last_positions for servo_id in targets):
            # The vendor controller performs the only safe first interpolation,
            # because this project currently has no verified position feedback.
            if generation is not None and not self._authority.goal_is_current(generation):
                return False
            self._publish_frame(targets, duration, cap_duration=False)
            time.sleep(duration)
            return allow_shutdown or (
                generation is not None and self._authority.goal_is_current(generation)
            )

        starts = {servo_id: self._last_positions[servo_id] for servo_id in targets}
        safe_duration = fastest_safe_duration(starts, targets, self._limits, duration)
        if safe_duration > duration:
            self.get_logger().info(
                f"Extended trajectory from {duration:.2f}s to {safe_duration:.2f}s "
                "to remain at the calibrated velocity limit"
            )
            duration = safe_duration
        steps = max(1, int(math.ceil(duration / FRAME_PERIOD_SECONDS)))
        frame_duration = duration / steps

        for step in range(1, steps + 1):
            if not allow_shutdown:
                if generation is None or not self._authority.goal_is_current(generation):
                    return False
            t = step / steps
            eased = t * t * (3.0 - 2.0 * t)
            frame = {
                servo_id: int(round(starts[servo_id] + (target - starts[servo_id]) * eased))
                for servo_id, target in targets.items()
            }
            self._publish_frame(frame, frame_duration)
            time.sleep(frame_duration)
        return True


def main() -> None:
    rclpy.init()
    node = JetArmControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
