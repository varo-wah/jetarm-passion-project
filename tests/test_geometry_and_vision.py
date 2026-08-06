import unittest
from unittest.mock import patch

import cv2
import numpy as np

from jetarm.config.detection_roi import point_in_roi, roi_bounds_from_shape
from jetarm.ui.display_appearance import enhance_display_frame
from jetarm.vision import coordinatelogic
from jetarm.vision.wrist_safety import choose_safe_wrist_angle
from jetarm.vision.yolo_detector import choose_target, to_vision_scanner_format


class GeometryAndVisionTests(unittest.TestCase):
    def test_green_center_outvotes_blue_tinted_box_background(self):
        frame = np.full((80, 80, 3), (85, 60, 50), dtype=np.uint8)
        cv2.rectangle(frame, (24, 20), (56, 60), (30, 180, 30), -1)

        self.assertEqual(coordinatelogic.detect_color(frame, 0, 0, 80, 80), "GREEN")

    def test_neutral_center_is_not_stolen_by_blue_tinted_background(self):
        frame = np.full((80, 80, 3), (85, 60, 50), dtype=np.uint8)
        cv2.rectangle(frame, (18, 16), (62, 64), (145, 145, 145), -1)

        self.assertEqual(coordinatelogic.detect_color(frame, 0, 0, 80, 80), "NEUTRAL")

    def test_primary_lego_colors_are_classified(self):
        colors = {
            "RED": (25, 25, 190),
            "GREEN": (25, 175, 30),
            "BLUE": (190, 55, 30),
        }

        for expected, bgr in colors.items():
            with self.subTest(color=expected):
                frame = np.full((40, 40, 3), bgr, dtype=np.uint8)
                self.assertEqual(coordinatelogic.detect_color(frame, 0, 0, 40, 40), expected)

    def test_natural_display_defaults_do_not_modify_pixels(self):
        frame = np.full((20, 20, 3), (70, 55, 45), dtype=np.uint8)
        original = frame.copy()

        enhanced = enhance_display_frame(frame)

        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(enhanced, original)

    def test_optional_display_gamma_brightens_without_modifying_source(self):
        frame = np.full((20, 20, 3), (70, 55, 45), dtype=np.uint8)
        original = frame.copy()

        with patch("jetarm.ui.display_appearance.SHADOW_GAMMA", 0.90):
            enhanced = enhance_display_frame(frame)
        source_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        np.testing.assert_array_equal(frame, original)
        self.assertGreater(
            float(enhanced_hsv[:, :, 2].mean()),
            float(source_hsv[:, :, 2].mean()),
        )

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
