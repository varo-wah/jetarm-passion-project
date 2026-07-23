import unittest
from unittest.mock import patch

import numpy as np

from jetarm.config.detection_roi import point_in_roi, roi_bounds_from_shape
from jetarm.vision import coordinatelogic
from jetarm.vision.wrist_safety import choose_safe_wrist_angle
from jetarm.vision.yolo_detector import choose_target, to_vision_scanner_format


class GeometryAndVisionTests(unittest.TestCase):
    def test_roi_bounds_and_membership_for_runtime_frame(self):
        shape = (480, 640, 3)

        self.assertEqual(roi_bounds_from_shape(shape), (128, 120, 560, 432))
        self.assertTrue(point_in_roi(shape, 128, 120))
        self.assertTrue(point_in_roi(shape, 560, 432))
        self.assertFalse(point_in_roi(shape, 127, 120))
        self.assertFalse(point_in_roi(shape, 560, 433))

    def test_pixel_to_robot_converts_calibration_millimeters_to_centimeters(self):
        homography = np.eye(3, dtype=np.float32)
        affine = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=np.float32,
        )

        with patch.object(coordinatelogic, "H_sheet", homography), patch.object(
            coordinatelogic,
            "A_robot",
            affine,
        ):
            self.assertEqual(coordinatelogic.pixel_to_robot(2.5, 3.5), (2.5, 3.5))

    def test_wrist_safety_overrides_all_roi_edges(self):
        shape = (480, 640, 3)
        rx0, ry0, rx1, ry1 = roi_bounds_from_shape(shape)
        common = {"angle": 37.0}
        cases = {
            "left": ({"center_x": rx0 + 5, "center_y": 250, "x1": rx0, "y1": 230, "x2": rx0 + 30, "y2": 270}, 90.0),
            "right": ({"center_x": rx1 - 5, "center_y": 250, "x1": rx1 - 30, "y1": 230, "x2": rx1, "y2": 270}, 90.0),
            "top": ({"center_x": 320, "center_y": ry0 + 5, "x1": 300, "y1": ry0, "x2": 340, "y2": ry0 + 30}, 0.0),
            "bottom": ({"center_x": 320, "center_y": ry1 - 5, "x1": 300, "y1": ry1 - 30, "x2": 340, "y2": ry1}, 0.0),
        }

        for expected_edge, (box, expected_angle) in cases.items():
            with self.subTest(edge=expected_edge):
                angle, edge = choose_safe_wrist_angle({**common, **box}, shape)
                self.assertEqual(edge, expected_edge)
                self.assertEqual(angle, expected_angle)

    def test_target_selection_filters_confidence_and_workspace(self):
        detections = [
            {"confidence": 0.99, "robot_x": 25.0, "robot_y": 15.0},
            {"confidence": 0.39, "robot_x": 0.0, "robot_y": 15.0},
            {"confidence": 0.72, "robot_x": 4.0, "robot_y": 17.0},
            {"confidence": 0.91, "robot_x": -3.0, "robot_y": 20.0},
        ]

        self.assertIs(choose_target(detections), detections[3])

    def test_scanner_adapter_preserves_geometry_metadata(self):
        detection = {
            "robot_x": 1.5,
            "robot_y": 12.0,
            "angle": 45.0,
            "color": "RED",
            "center_x": 300,
            "center_y": 200,
            "x1": 280,
            "y1": 180,
            "x2": 320,
            "y2": 220,
        }

        brick = to_vision_scanner_format([detection])[0]

        self.assertEqual(brick["x"], 1.5)
        self.assertEqual(brick["y"], 12.0)
        self.assertEqual(brick["angle"], 45.0)
        self.assertEqual(brick["center_x"], 300)
        self.assertEqual(brick["x2"], 320)


if __name__ == "__main__":
    unittest.main()
