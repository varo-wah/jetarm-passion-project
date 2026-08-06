import unittest
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (PROJECT_ROOT / "scripts" / "run_jetarm.sh").read_text()
PROJECT_COMMAND = PROJECT_ROOT / "jetarm"
PROJECT_COMMAND_TEXT = PROJECT_COMMAND.read_text()


class LauncherContractTests(unittest.TestCase):
    def test_project_command_delegates_to_the_safety_launcher(self):
        self.assertIn(
            'exec "$project_root/scripts/run_jetarm.sh" "$launch_mode"',
            PROJECT_COMMAND_TEXT,
        )
        self.assertIn('launch_mode="--preview"', PROJECT_COMMAND_TEXT)
        self.assertIn('launch_mode="--actuate"', PROJECT_COMMAND_TEXT)

    def test_project_command_help_documents_safe_preview_default(self):
        result = subprocess.run(
            [str(PROJECT_COMMAND), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("./jetarm website", result.stdout)
        self.assertIn("safe preview mode", result.stdout)

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

    def test_controller_readiness_does_not_spawn_short_lived_ros_clients(self):
        self.assertIn("controller_interfaces_ready", LAUNCHER)
        self.assertIn("/jetarm_controller/follow_joint_trajectory", LAUNCHER)
        self.assertNotIn("JetArmControlClient(wait_seconds", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
