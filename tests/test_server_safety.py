import json
import signal
import subprocess
import unittest
from unittest.mock import patch

from jetarm.hardware import Class_Execution as hardware
from jetarm.ui import server_app


class FakeProcess:
    def __init__(self, wait_results=None):
        self.wait_results = list(wait_results or [0])
        self.signals = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        result = self.wait_results.pop(0)
        if result == "timeout":
            raise subprocess.TimeoutExpired("scanner", timeout)
        return result


def response_payload(response):
    return json.loads(response.body.decode("utf-8"))


class ServerSafetyTests(unittest.TestCase):
    def setUp(self):
        hardware.clear_estop()
        hardware.resume_system()
        server_app._scanner_proc = None
        server_app._scanner_autocycle_enabled = False
        server_app._person_follow_stop.set()
        server_app._status["state"] = "IDLE"
        server_app._status["last_action"] = "--"
        server_app._status["last_error"] = "--"

    def tearDown(self):
        server_app._scanner_proc = None
        server_app._scanner_autocycle_enabled = False
        hardware.clear_estop()
        hardware.resume_system()

    def test_scanner_stop_escalates_to_kill(self):
        proc = FakeProcess(["timeout", "timeout", 0])
        server_app._scanner_proc = proc
        server_app._scanner_autocycle_enabled = True

        self.assertTrue(server_app._stop_scanner_process())

        self.assertEqual(proc.signals, [signal.SIGINT])
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertIsNone(server_app._scanner_proc)
        self.assertFalse(server_app._scanner_autocycle_enabled)

    def test_scanner_stop_pauses_controller_before_signaling_worker(self):
        events = []
        server_app._scanner_proc = FakeProcess([0])

        with patch.object(
            server_app,
            "stop_motion",
            side_effect=lambda: events.append("controller_pause") or True,
        ), patch.object(
            server_app._scanner_proc,
            "send_signal",
            side_effect=lambda value: events.append("worker_signal"),
        ):
            response = server_app.scanner_stop()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events, ["controller_pause", "worker_signal"])
        self.assertTrue(response_payload(response)["paused"])

    def test_estop_latches_parent_and_stops_scanner_process(self):
        proc = FakeProcess([0])
        server_app._scanner_proc = proc
        server_app._scanner_autocycle_enabled = True

        response = server_app.api_cmd({"type": "estop"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response_payload(response)["ok"])
        self.assertEqual(proc.signals, [signal.SIGINT])
        self.assertTrue(hardware.motion_safety_status()["estop_latched"])
        self.assertIsNone(server_app._scanner_proc)

    def test_manual_gripper_command_is_blocked_while_scanner_runs(self):
        server_app._scanner_proc = FakeProcess()

        with patch.object(server_app.gripper, "close_gripper") as close_gripper:
            response = server_app.api_cmd({"type": "close_gripper"})

        self.assertEqual(response.status_code, 409)
        self.assertIn("scanner", response_payload(response)["error"].lower())
        close_gripper.assert_not_called()

    def test_preview_configuration_blocks_manual_robot_motion(self):
        with patch.object(server_app, "SCANNER_ACTUATION_ENABLED", False), patch.object(
            server_app.gripper,
            "open_gripper",
        ) as open_gripper:
            response = server_app.api_cmd({"type": "open_gripper"})

        self.assertEqual(response.status_code, 409)
        self.assertIn("actuation disabled", response_payload(response)["error"].lower())
        open_gripper.assert_not_called()

    def test_clear_estop_keeps_system_paused_until_explicit_resume(self):
        hardware.estop_motion()

        response = server_app.api_cmd({"type": "clear_estop"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response_payload(response)["paused"])
        safety = hardware.motion_safety_status()
        self.assertFalse(safety["estop_latched"])
        self.assertTrue(safety["paused"])

    def test_scanner_launch_forces_preview_mode_by_default(self):
        with patch.object(server_app, "SCANNER_ACTUATION_ENABLED", False), patch.object(
            server_app.subprocess,
            "Popen",
            return_value=FakeProcess(),
        ) as popen:
            server_app._launch_scanner_process_unlocked()

        args, kwargs = popen.call_args
        self.assertEqual(args[0][-1], "jetarm.sorting.yolo_vision_scanner")
        self.assertEqual(kwargs["env"]["JETARM_ENABLE_ACTUATION"], "0")
        self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertTrue(kwargs["start_new_session"])

    def test_scanner_failure_is_reported_when_autocycle_is_disabled(self):
        proc = FakeProcess([7])
        server_app._scanner_proc = proc
        server_app._scanner_autocycle_enabled = False
        server_app._status["state"] = "SCANNER_RUNNING"

        with patch.object(server_app, "stop_motion") as stop_motion:
            server_app._scanner_autocycle_loop()

        self.assertEqual(server_app._status["state"], "SCANNER_ERROR")
        self.assertEqual(server_app._status["last_action"], "scanner_exited")
        self.assertEqual(server_app._status["last_error"], "YOLO scanner exited with code 7")
        self.assertIsNone(server_app._scanner_proc)
        stop_motion.assert_called_once_with()

    def test_shutdown_stops_camera_after_workers(self):
        with patch.object(server_app, "_request_person_follow_stop") as stop_follow, patch.object(
            server_app,
            "_stop_scanner_process",
        ) as stop_scanner, patch.object(server_app, "stop_camera") as stop_camera, patch.object(
            server_app,
            "shutdown_control_client",
        ) as close_client:
            server_app.on_shutdown()

        stop_follow.assert_called_once_with()
        stop_scanner.assert_called_once_with()
        stop_camera.assert_called_once_with()
        close_client.assert_called_once_with()

    def test_actuating_shutdown_pauses_then_requests_safe_pose(self):
        events = []
        with patch.object(server_app, "SCANNER_ACTUATION_ENABLED", True), patch.object(
            server_app,
            "_request_person_follow_stop",
            side_effect=lambda: events.append("follow_stop"),
        ), patch.object(
            server_app,
            "stop_motion",
            side_effect=lambda: events.append("controller_pause") or True,
        ), patch.object(
            server_app,
            "_stop_scanner_process",
            side_effect=lambda: events.append("scanner_stop") or True,
        ), patch.object(
            server_app,
            "safe_shutdown",
            side_effect=lambda: events.append("safe_shutdown") or True,
        ), patch.object(
            server_app,
            "stop_camera",
            side_effect=lambda: events.append("camera_stop"),
        ), patch.object(
            server_app,
            "shutdown_control_client",
            side_effect=lambda: events.append("client_close"),
        ):
            server_app.on_shutdown()

        self.assertEqual(
            events,
            [
                "follow_stop",
                "controller_pause",
                "scanner_stop",
                "safe_shutdown",
                "camera_stop",
                "client_close",
            ],
        )

    def test_shutdown_releases_remaining_subsystems_after_ros_failure(self):
        events = []
        with patch.object(
            server_app,
            "_request_person_follow_stop",
            side_effect=lambda: events.append("follow_stop"),
        ), patch.object(
            server_app,
            "stop_motion",
            side_effect=RuntimeError("ROS context invalid"),
        ), patch.object(
            server_app,
            "_stop_scanner_process",
            side_effect=lambda: events.append("scanner_stop"),
        ), patch.object(
            server_app,
            "stop_camera",
            side_effect=lambda: events.append("camera_stop"),
        ), patch.object(
            server_app,
            "shutdown_control_client",
            side_effect=lambda: events.append("client_close"),
        ):
            server_app.on_shutdown()

        self.assertEqual(events, ["follow_stop", "scanner_stop", "camera_stop", "client_close"])

    def test_actuating_scanner_is_rejected_when_ros_hardware_is_unavailable(self):
        with patch.object(server_app, "SCANNER_ACTUATION_REQUESTED", True), patch.object(
            server_app,
            "HARDWARE_AVAILABLE",
            False,
        ), patch.object(
            server_app,
            "HARDWARE_UNAVAILABLE_REASON",
            "No module named rclpy",
        ), patch.object(server_app, "_launch_scanner_process_unlocked") as launch:
            response = server_app.scanner_start()

        self.assertEqual(response.status_code, 503)
        self.assertIn("rclpy", response_payload(response)["error"])
        launch.assert_not_called()

    def test_status_exposes_scanner_and_motion_safety_modes(self):
        response = server_app.api_status()
        payload = response_payload(response)

        self.assertFalse(payload["scanner_actuation_enabled"])
        self.assertIn("hardware_available", payload)
        self.assertTrue(payload["motion_safety"]["motion_allowed"])


if __name__ == "__main__":
    unittest.main()
