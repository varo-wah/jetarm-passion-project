import os
import unittest
from unittest.mock import patch

from jetarm.ui import camera_worker


class FakeCapture:
    def __init__(self):
        self.settings = []

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True


class CameraWorkerTests(unittest.TestCase):
    def test_low_light_defaults_are_applied(self):
        capture = FakeCapture()

        with patch.dict(os.environ, {}, clear=True):
            camera_worker._apply_camera_controls(capture)

        self.assertIn((camera_worker.cv2.CAP_PROP_BRIGHTNESS, 12), capture.settings)
        self.assertIn((camera_worker.cv2.CAP_PROP_GAIN, 20), capture.settings)
        self.assertIn((camera_worker.cv2.CAP_PROP_GAMMA, 140), capture.settings)

    def test_camera_controls_support_environment_override(self):
        capture = FakeCapture()

        with patch.dict(
            os.environ,
            {
                "JETARM_CAMERA_BRIGHTNESS": "7",
                "JETARM_CAMERA_GAIN": "11",
                "JETARM_CAMERA_GAMMA": "120",
            },
            clear=True,
        ):
            camera_worker._apply_camera_controls(capture)

        self.assertIn((camera_worker.cv2.CAP_PROP_BRIGHTNESS, 7), capture.settings)
        self.assertIn((camera_worker.cv2.CAP_PROP_GAIN, 11), capture.settings)
        self.assertIn((camera_worker.cv2.CAP_PROP_GAMMA, 120), capture.settings)


if __name__ == "__main__":
    unittest.main()
