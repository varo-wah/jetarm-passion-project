"""Blocking ROS client used by the web server and scanner processes."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Mapping

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from jetarm.control.control_node import ACTION_NAME, STATE_TOPIC
from jetarm.control.limits import load_joint_limits, validate_pulse_targets


class JetArmControlClient:
    """Serialize local callers while the controller serializes the whole robot."""

    def __init__(self, wait_seconds: float = 5.0) -> None:
        # Uvicorn owns SIGINT/SIGTERM in the dashboard process. A private ROS
        # context without rclpy signal handlers prevents Ctrl+C from invalidating
        # this client before FastAPI has run its safe-shutdown callbacks.
        self._context = Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._closed = False
        suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.node: Node = rclpy.create_node(
            f"jetarm_control_client_{suffix}",
            context=self._context,
        )
        self._limits = load_joint_limits()
        self._action = ActionClient(self.node, FollowJointTrajectory, ACTION_NAME)
        self._services = {
            name: self.node.create_client(Trigger, f"/jetarm_controller/{name}")
            for name in ("pause", "resume", "estop", "clear_estop", "safe_shutdown")
        }
        self._status_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._service_lock = threading.RLock()
        self._status = {
            "state": "BOOT_LOCKED",
            "paused": True,
            "estop_latched": False,
            "motion_allowed": False,
            "active_goal": False,
            "reason": "waiting for controller state",
        }
        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.node.create_subscription(String, STATE_TOPIC, self._state_callback, state_qos)

        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"jetarm-ros-client-{suffix}",
            daemon=True,
        )
        self._spin_thread.start()

        if not self._action.wait_for_server(timeout_sec=wait_seconds):
            raise RuntimeError(
                f"JetArm control action {ACTION_NAME} unavailable after {wait_seconds:.1f}s"
            )

    def _state_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        with self._status_lock:
            self._status = payload

    def status(self) -> dict[str, object]:
        with self._status_lock:
            return dict(self._status)

    def _wait_future(self, future, timeout: float):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for JetArm controller")
            time.sleep(0.01)
        return future.result()

    def command(self, positions: Mapping[int, float], duration: float = 1.0) -> bool:
        targets = validate_pulse_targets(positions, self._limits)
        if duration <= 0:
            raise ValueError("Motion duration must be positive")

        ordered = [self._limits[servo_id] for servo_id in sorted(targets)]
        seconds = int(duration)
        nanoseconds = int(round((duration - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds = 0
        trajectory = JointTrajectory(
            joint_names=[joint.name for joint in ordered],
            points=[
                JointTrajectoryPoint(
                    positions=[
                        joint.pulse_to_radians(targets[joint.servo_id])
                        for joint in ordered
                    ],
                    time_from_start=Duration(sec=seconds, nanosec=nanoseconds),
                )
            ],
        )
        goal = FollowJointTrajectory.Goal(trajectory=trajectory)

        with self._command_lock:
            goal_handle = self._wait_future(
                self._action.send_goal_async(goal),
                timeout=5.0,
            )
            if goal_handle is None or not goal_handle.accepted:
                return False
            result_response = self._wait_future(
                goal_handle.get_result_async(),
                timeout=max(duration + 5.0, 6.0),
            )
            return (
                result_response is not None
                and result_response.result.error_code
                == FollowJointTrajectory.Result.SUCCESSFUL
            )

    def call(self, operation: str, timeout: float = 3.0) -> bool:
        client = self._services.get(operation)
        if client is None:
            raise ValueError(f"Unknown controller operation: {operation}")
        if not client.wait_for_service(timeout_sec=timeout):
            return False
        # Safety services deliberately do not share the action lock: Pause and
        # E-stop must remain callable while another thread waits on a goal.
        with self._service_lock:
            response = self._wait_future(client.call_async(Trigger.Request()), timeout=timeout)
        return bool(response and response.success)

    def pause(self) -> bool:
        return self.call("pause")

    def resume(self) -> bool:
        return self.call("resume")

    def estop(self) -> bool:
        return self.call("estop")

    def clear_estop(self) -> bool:
        return self.call("clear_estop")

    def safe_shutdown(self) -> bool:
        return self.call("safe_shutdown", timeout=5.0)

    def close(self) -> None:
        """Stop the executor and release this client's private ROS context."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        if self._context.ok():
            rclpy.shutdown(context=self._context)
