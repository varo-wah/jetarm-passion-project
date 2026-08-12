import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = (PROJECT_ROOT / "src" / "jetarm" / "ui" / "static" / "index.html").read_text()


class DashboardAlertContractTests(unittest.TestCase):
    def test_dashboard_columns_stack_independently(self):
        self.assertIn('class="dashboard-grid"', DASHBOARD)
        self.assertEqual(DASHBOARD.count('class="column-stack"'), 2)
        self.assertLess(DASHBOARD.index("Vision Feed"), DASHBOARD.index("Manual Positioning"))
        self.assertLess(DASHBOARD.index("Operator Controls"), DASHBOARD.index("Active Programs"))
        self.assertNotIn('class="layout"', DASHBOARD)

    def test_dashboard_has_persistent_structured_alert_fields(self):
        for element_id in (
            "alertBanner",
            "alertSummary",
            "alertStage",
            "alertCode",
            "alertMotion",
            "alertObject",
            "alertDetail",
            "alertAction",
        ):
            self.assertIn(f'id="{element_id}"', DASHBOARD)

    def test_dashboard_supports_acknowledgement_and_diagnostic_copy(self):
        self.assertIn('id="btnAlertAck"', DASHBOARD)
        self.assertIn('id="btnAlertCopy"', DASHBOARD)
        self.assertIn('/api/alerts/acknowledge', DASHBOARD)
        self.assertIn('navigator.clipboard.writeText', DASHBOARD)

    def test_dashboard_renders_bounded_recent_alert_history(self):
        self.assertIn('id="alertHistory"', DASHBOARD)
        self.assertIn('history.slice(0, 5)', DASHBOARD)

    def test_dashboard_exposes_authoritative_motion_state(self):
        for element_id in (
            "motionStateBanner",
            "motionStateLabel",
            "motionStateHelp",
        ):
            self.assertIn(f'id="{element_id}"', DASHBOARD)
        self.assertIn("function applyMotionControls(status)", DASHBOARD)
        self.assertIn('safety.estop_latched === true', DASHBOARD)
        self.assertIn('safety.paused === true', DASHBOARD)
        self.assertIn('safety.motion_allowed === true', DASHBOARD)

    def test_dashboard_disables_motion_and_requires_explicit_estop_recovery(self):
        self.assertIn("for (const id of motionActionButtonIds)", DASHBOARD)
        self.assertRegex(DASHBOARD, r'setButtonAvailability\(\s*"btnResume"')
        self.assertRegex(DASHBOARD, r'setButtonAvailability\(\s*"btnClearEstop"')
        self.assertIn(
            'estopLatched ? "Clear the E-stop latch before resuming."',
            DASHBOARD,
        )
        self.assertIn(
            'online ? "No E-stop latch is active."',
            DASHBOARD,
        )

    def test_dashboard_fails_closed_when_status_is_unavailable(self):
        self.assertIn("applyMotionControls(null);", DASHBOARD)
        self.assertIn("CONTROLLER STATUS UNAVAILABLE", DASHBOARD)
        self.assertIn('id="btnClearEstop" disabled', DASHBOARD)


if __name__ == "__main__":
    unittest.main()
