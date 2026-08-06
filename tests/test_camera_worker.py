import subprocess
import unittest
from unittest.mock import patch

from jetarm.ui import camera_worker


class CameraWorkerTests(unittest.TestCase):
    def test_v4l2_defaults_are_parsed(self):
        output = """
                     brightness 0x00980900 (int) : min=-64 max=64 step=1 default=0 value=7
        white_balance_automatic 0x0098090c (bool) : default=1 value=0
                    pan_absolute 0x009a0908 (int) : min=-36000 max=36000 default=0 value=0
        """

        self.assertEqual(
            camera_worker._parse_v4l2_defaults(output),
            {
                "brightness": 0,
                "white_balance_automatic": 1,
                "pan_absolute": 0,
            },
        )

    @patch("jetarm.ui.camera_worker.shutil.which", return_value="/usr/bin/v4l2-ctl")
    @patch("jetarm.ui.camera_worker.subprocess.run")
    def test_image_controls_reset_to_driver_defaults(self, run, _which):
        run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "brightness 0x00980900 (int) : default=0 value=7\n"
                    "white_balance_automatic 0x0098090c (bool) : default=1 value=0\n"
                    "pan_absolute 0x009a0908 (int) : default=0 value=100\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        restored = camera_worker._reset_camera_controls(0)

        self.assertEqual(restored, 2)
        commands = [call.args[0] for call in run.call_args_list[1:]]
        self.assertIn(
            [
                "/usr/bin/v4l2-ctl",
                "--device",
                "/dev/video0",
                "--set-ctrl=brightness=0",
            ],
            commands,
        )
        self.assertIn(
            [
                "/usr/bin/v4l2-ctl",
                "--device",
                "/dev/video0",
                "--set-ctrl=white_balance_automatic=1",
            ],
            commands,
        )
        self.assertFalse(any("pan_absolute" in part for command in commands for part in command))

    @patch("jetarm.ui.camera_worker.shutil.which", return_value=None)
    def test_missing_v4l2_tool_reports_no_reset(self, _which):
        self.assertEqual(camera_worker._reset_camera_controls(0), 0)


if __name__ == "__main__":
    unittest.main()
