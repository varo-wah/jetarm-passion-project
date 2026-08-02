"""Compatibility facade for the centralized JetArm ROS controller."""

from __future__ import annotations

import time

try:
    from jetarm.control.client import JetArmControlClient
except ImportError as exc:
    JetArmControlClient = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


class CKMJetArm:
    """Preserve the legacy pulse API without publishing hardware topics here."""

    def __init__(self):
        if ROS_IMPORT_ERROR is not None or JetArmControlClient is None:
            raise RuntimeError(
                "JetArm ROS dependencies are unavailable. Start hardware with "
                "./scripts/run_jetarm.sh --actuate."
            ) from ROS_IMPORT_ERROR
        self.control = JetArmControlClient()

    def moveJetArm(self, servo_id, target_position, duration=1.0):
        ok = self.moveJetArmGroup({servo_id: target_position}, duration=duration)
        if ok:
            print(f"Command accepted for servo {servo_id} -> position {target_position}")
        return ok

    def moveJetArmGroup(self, positions, duration=1.0):
        return self.control.command(positions, duration=duration)

    def smoothMoveJetArmGroup(self, positions, duration=1.2, steps=24):
        del steps
        return self.control.command(positions, duration=duration)

    def safety_status(self):
        return self.control.status()

    def pause(self):
        return self.control.pause()

    def resume(self):
        return self.control.resume()

    def estop(self):
        return self.control.estop()

    def clear_estop(self):
        return self.control.clear_estop()

    def safe_shutdown(self):
        return self.control.safe_shutdown()

    def reset(self):
        return self.smoothMoveJetArmGroup(
            {1: 500, 2: 500, 3: 500, 4: 500, 10: 500}, duration=2.0
        )

    def front(self):
        if not self.smoothMoveJetArmGroup(
            {2: 330, 3: 250, 4: 400, 10: 100}, duration=1.0
        ):
            return False
        time.sleep(1.0)
        return self.moveJetArm(10, 700)

    def frontOpen(self):
        return self.smoothMoveJetArmGroup(
            {2: 330, 3: 250, 4: 400, 10: 100}, duration=1.0
        )

    def up(self):
        return self.smoothMoveJetArmGroup(
            {1: 500, 2: 500, 3: 500, 4: 500}, duration=1.2
        )
