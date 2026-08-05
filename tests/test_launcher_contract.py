import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (PROJECT_ROOT / "scripts" / "run_jetarm.sh").read_text()


class LauncherContractTests(unittest.TestCase):
    def test_actuation_starts_only_the_low_level_servo_driver(self):
        self.assertIn(
            "setsid ros2 run ros_robot_controller ros_robot_controller &",
            LAUNCHER,
        )
        self.assertNotIn("ros2 launch sdk jetarm_sdk.launch.py", LAUNCHER)

    def test_cleanup_targets_managed_process_groups(self):
        self.assertIn('kill -TERM -- "-$leader_pid"', LAUNCHER)
        self.assertIn('kill -KILL -- "-$leader_pid"', LAUNCHER)
        self.assertIn('stop_process_group "$controller_pid"', LAUNCHER)
        self.assertIn('stop_process_group "$driver_pid"', LAUNCHER)


if __name__ == "__main__":
    unittest.main()
