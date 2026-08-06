import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = (PROJECT_ROOT / "src" / "jetarm" / "ui" / "static" / "index.html").read_text()


class DashboardAlertContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
