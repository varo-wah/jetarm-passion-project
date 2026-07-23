"""Process-local motion safety state shared by every hardware command path."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SafetySnapshot:
    paused: bool
    estop_latched: bool

    @property
    def motion_allowed(self) -> bool:
        return not self.paused and not self.estop_latched

    def as_dict(self) -> dict[str, bool]:
        payload = asdict(self)
        payload["motion_allowed"] = self.motion_allowed
        return payload


class MotionSafetyState:
    """Thread-safe pause and E-stop latches for one Python process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paused = False
        self._estop_latched = False

    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return SafetySnapshot(
                paused=self._paused,
                estop_latched=self._estop_latched,
            )

    def motion_allowed(self) -> bool:
        return self.snapshot().motion_allowed

    def pause(self) -> bool:
        with self._lock:
            self._paused = True
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._estop_latched:
                return False
            self._paused = False
        return True

    def estop(self) -> bool:
        with self._lock:
            self._estop_latched = True
            self._paused = True
        return True

    def clear_estop(self) -> bool:
        """Clear only the E-stop latch; an explicit resume is still required."""
        with self._lock:
            self._estop_latched = False
        return True


MOTION_SAFETY = MotionSafetyState()
