import unittest
from unittest.mock import Mock, patch

import numpy as np

from jetarm.control.limits import JointLimitsError
from jetarm.sorting import yolo_vision_scanner as scanner


class ScannerSafetyTests(unittest.TestCase):
    def setUp(self):
        scanner._last_abort_event = None
        scanner._last_preflight_skip = None

    def test_actuation_environment_defaults_to_false(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(scanner.env_flag(scanner.ACTUATION_ENV, default=False))

    def test_preview_scan_does_not_move_robot(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detection = {
            "x": 1.0,
            "y": 10.0,
            "angle": 20.0,
            "color": "BLUE",
        }

        with patch.object(scanner.camera, "scan_position") as scan_position, patch.object(
            scanner,
            "take_snapshot",
            return_value=frame,
        ), patch.object(scanner, "detect_bricks_yolo", return_value=[detection]):
            bricks = scanner.scan_once(move_to_scan_pose=False)

        scan_position.assert_not_called()
        self.assertEqual(bricks[0]["frame_shape"], frame.shape)

    def test_preview_main_never_calls_pick_and_drop(self):
        target = {"x": 1.0, "y": 10.0, "angle": 20.0, "color": "RED"}

        with patch.object(scanner, "ENABLE_PICK_AND_DROP", False), patch.object(
            scanner,
            "scan_once",
            return_value=[target],
        ) as scan_once, patch.object(
            scanner,
            "pick_and_drop",
            Mock(side_effect=AssertionError("preview must not actuate")),
        ):
            scanner.main()

        scan_once.assert_called_once_with(move_to_scan_pose=False)

    def test_preflight_rejects_target_before_any_motion(self):
        target = {"x": -2.28, "y": 14.05, "angle": 0.0, "color": "NEUTRAL"}

        with patch.object(
            scanner.ik,
            "plan_to",
            side_effect=JointLimitsError("elbow_joint: outside calibrated range"),
        ), patch.object(scanner.ik, "move_to") as move_to:
            self.assertFalse(scanner.preflight_pick_and_drop(target))

        move_to.assert_not_called()

    def test_neutral_route_from_hardware_log_passes_preflight(self):
        target = {
            "x": 0.48,
            "y": 20.50,
            "angle": 138.1,
            "color": "NEUTRAL",
            "frame_shape": (480, 640, 3),
        }

        self.assertTrue(scanner.preflight_pick_and_drop(target))

    def test_motion_exception_becomes_controlled_failure(self):
        with patch.object(
            scanner.ik,
            "move_to",
            side_effect=JointLimitsError("elbow_joint: outside calibrated range"),
        ):
            self.assertFalse(scanner.move_wait(1.0, 10.0, 7.0, "approach"))

        self.assertEqual(scanner._last_abort_event["code"], "JOINT_LIMIT")
        self.assertEqual(scanner._last_abort_event["stage"], "motion")
        self.assertEqual(scanner._last_abort_event["object_state"], "not_gripped")

    def test_structured_event_is_forwarded_with_scanner_token(self):
        event = scanner.build_event(
            severity="abort",
            code="JOINT_LIMIT",
            stage_name="target_pickup",
            summary="Target rejected",
            detail="elbow pulse 112 below 120",
            action="Move the object and retry.",
        )
        response = Mock()
        response.raise_for_status.return_value = None

        with patch.dict(
            "os.environ",
            {scanner.SCANNER_EVENT_TOKEN_ENV: "scanner-secret"},
        ), patch.object(scanner.requests, "post", return_value=response) as post:
            self.assertTrue(scanner.report_event(event))

        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"], event)
        self.assertEqual(kwargs["headers"]["X-JetArm-Scanner-Token"], "scanner-secret")
        self.assertEqual(kwargs["timeout"], 0.75)

    def test_run_reports_the_exact_remembered_abort_before_reraising(self):
        error = JointLimitsError("elbow_joint pulse 112 is below minimum 120")
        remembered = scanner.remember_abort(
            error=error,
            stage_name="target_pickup",
            summary="Target pickup motion rejected",
            action="Move the object farther from the base.",
        )

        with patch.object(scanner, "main", side_effect=error), patch.object(
            scanner,
            "report_event",
        ) as report:
            with self.assertRaises(JointLimitsError):
                scanner.run()

        report.assert_called_once_with(remembered)


if __name__ == "__main__":
    unittest.main()
