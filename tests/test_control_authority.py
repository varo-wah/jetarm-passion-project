import unittest

from jetarm.control.authority import ControlState, MotionAuthority
from jetarm.control.limits import (
    JointLimitsError,
    fastest_safe_duration,
    load_joint_limits,
    validate_pulse_targets,
)


class ControlAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.limits = load_joint_limits()
        self.authority = MotionAuthority(self.limits)

    def test_calibration_defines_six_distinct_servos_and_safe_poses(self):
        self.assertEqual(set(self.limits), {1, 2, 3, 4, 5, 10})
        self.assertEqual(len({joint.name for joint in self.limits.values()}), 6)
        for joint in self.limits.values():
            self.assertEqual(joint.validate_pulse(joint.safe_pose_pulse), joint.safe_pose_pulse)

    def test_per_joint_limits_reject_instead_of_clamp(self):
        with self.assertRaises(JointLimitsError):
            validate_pulse_targets({10: 99}, self.limits)
        with self.assertRaises(JointLimitsError):
            validate_pulse_targets({4: 701}, self.limits)
        with self.assertRaises(JointLimitsError):
            validate_pulse_targets({99: 500}, self.limits)

    def test_motion_duration_extends_to_fastest_calibrated_value(self):
        duration = fastest_safe_duration(
            {1: 500, 2: 490},
            {1: 8, 2: 424},
            self.limits,
            requested_duration=1.2,
        )

        self.assertAlmostEqual(duration, 1.23)

    def test_motion_duration_preserves_slower_operator_request(self):
        duration = fastest_safe_duration(
            {1: 500},
            {1: 8},
            self.limits,
            requested_duration=2.0,
        )

        self.assertEqual(duration, 2.0)

    def test_boot_is_locked_until_explicit_resume(self):
        snapshot = self.authority.snapshot()
        self.assertEqual(snapshot.state, ControlState.BOOT_LOCKED)
        self.assertFalse(snapshot.motion_allowed)
        self.assertTrue(self.authority.resume())
        self.assertTrue(self.authority.snapshot().motion_allowed)

    def test_pause_preempts_active_goal_and_requires_resume(self):
        self.authority.resume()
        generation, targets = self.authority.begin_goal({1: 500, 2: 461})
        self.assertEqual(targets, {1: 500, 2: 461})
        self.assertTrue(self.authority.goal_is_current(generation))

        self.assertTrue(self.authority.pause("scanner stop"))
        self.assertFalse(self.authority.goal_is_current(generation))
        self.assertEqual(self.authority.snapshot().state, ControlState.PAUSED)

    def test_estop_has_priority_and_clear_remains_paused(self):
        self.authority.resume()
        self.authority.estop()
        self.assertFalse(self.authority.resume())
        self.assertTrue(self.authority.snapshot().estop_latched)

        self.assertTrue(self.authority.clear_estop())
        snapshot = self.authority.snapshot()
        self.assertEqual(snapshot.state, ControlState.PAUSED)
        self.assertFalse(snapshot.motion_allowed)

    def test_concurrent_goal_is_rejected(self):
        self.authority.resume()
        self.authority.begin_goal({1: 500})
        with self.assertRaises(RuntimeError):
            self.authority.begin_goal({2: 461})


if __name__ == "__main__":
    unittest.main()
