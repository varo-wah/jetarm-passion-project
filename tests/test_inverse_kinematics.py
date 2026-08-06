import math
import unittest

from jetarm.control.limits import JointLimitsError
from jetarm.hardware.inverse_kinematics import (
    IKInputError,
    IKUnreachableError,
    JetArmKinematics,
)


class InverseKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.kinematics = JetArmKinematics()

    def test_scan_pose_matches_existing_calibrated_pulses(self):
        plan = self.kinematics.plan_wrist(0, 15, 23)

        self.assertEqual(
            plan.servo_targets,
            {1: 500, 2: 461, 3: 302, 4: 22},
        )
        self.assertAlmostEqual(sum(plan.angles.as_tuple()[1:]), 90.0)

    def test_table_tip_height_is_converted_before_solving(self):
        plan = self.kinematics.plan_table(0, 15, 13)

        self.assertAlmostEqual(plan.z_wrist_cm, 22.5)
        self.assertEqual(plan.z_table_cm, 13.0)
        self.assertEqual(
            plan.servo_targets,
            {1: 500, 2: 467, 3: 286, 4: 32},
        )

    def test_out_of_reach_target_is_rejected(self):
        with self.assertRaisesRegex(IKUnreachableError, "outside reachable range"):
            self.kinematics.plan_wrist(0, 0, 30.1)

    def test_shoulder_singularity_is_rejected(self):
        with self.assertRaisesRegex(IKUnreachableError, "shoulder singularity"):
            self.kinematics.solve_angles(0, 0, 0)

    def test_non_finite_target_is_rejected(self):
        with self.assertRaisesRegex(IKInputError, "x_cm must be a finite number"):
            self.kinematics.plan_wrist(math.nan, 15, 23)

    def test_base_angle_is_unwrapped_near_continuity_seam(self):
        angles = self.kinematics.solve_angles(
            -1,
            -0.01,
            23,
            previous_base_deg=179.0,
        )

        self.assertGreater(angles.base_deg, 180.0)
        self.assertLess(abs(angles.base_deg - 179.0), 2.0)

    def test_target_on_base_axis_retains_previous_yaw(self):
        angles = self.kinematics.solve_angles(
            0,
            0,
            23,
            previous_base_deg=90.0,
        )

        self.assertEqual(angles.base_deg, 90.0)

    def test_phase0_joint_limits_are_applied_during_planning(self):
        with self.assertRaisesRegex(
            JointLimitsError,
            r"wrist_pitch_joint: pulse 6 is outside calibrated range \[10, 700\]",
        ):
            self.kinematics.plan_table(-20, 10, 10)

    def test_elbow_branches_keep_the_same_tool_pitch_constraint(self):
        elbow_up = self.kinematics.solve_angles(0, 15, 23, elbow_up=True)
        elbow_down = self.kinematics.solve_angles(0, 15, 23, elbow_up=False)

        self.assertNotEqual(elbow_up.shoulder_deg, elbow_down.shoulder_deg)
        self.assertAlmostEqual(
            elbow_up.shoulder_deg + elbow_up.elbow_deg + elbow_up.wrist_pitch_deg,
            90.0,
        )
        self.assertAlmostEqual(
            elbow_down.shoulder_deg + elbow_down.elbow_deg + elbow_down.wrist_pitch_deg,
            90.0,
        )


if __name__ == "__main__":
    unittest.main()
