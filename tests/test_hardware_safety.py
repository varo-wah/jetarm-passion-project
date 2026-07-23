import unittest

from jetarm.hardware import Class_Execution as hardware


class FakeArm:
    def __init__(self):
        self.commands = []

    def moveJetArm(self, servo_id, target_position, duration=1.0):
        self.commands.append(("single", servo_id, target_position, duration))
        return True

    def smoothMoveJetArmGroup(self, positions, duration=1.2, steps=24):
        self.commands.append(("smooth", dict(positions), duration, steps))
        return True


class HardwareSafetyTests(unittest.TestCase):
    def setUp(self):
        hardware.clear_estop()
        hardware.resume_system()
        self.arm = FakeArm()
        self.ik = hardware.JetArmIK()
        self.ik.Arm = self.arm
        self.gripper = hardware.JetArmGripper(self.ik)
        self.gripper.Arm = self.arm

    def tearDown(self):
        hardware.clear_estop()
        hardware.resume_system()

    def test_estop_blocks_wrist_ik_and_gripper_commands(self):
        hardware.estop_motion()

        self.assertFalse(self.ik.move_to_wrist(0, 15, 23))
        self.assertFalse(self.gripper.turn_wrist(90))
        self.assertFalse(self.gripper.close_gripper())
        self.assertFalse(self.gripper.open_gripper())
        self.assertEqual(self.arm.commands, [])

    def test_pause_blocks_table_referenced_motion(self):
        hardware.pause_system()

        self.assertFalse(self.ik.move_to(0, 15, 13))
        self.assertEqual(self.arm.commands, [])

    def test_clearing_estop_does_not_implicitly_resume_motion(self):
        hardware.estop_motion()
        hardware.clear_estop()

        self.assertFalse(hardware.motion_is_allowed())
        self.assertTrue(hardware.resume_system())
        self.assertTrue(hardware.motion_is_allowed())


if __name__ == "__main__":
    unittest.main()
