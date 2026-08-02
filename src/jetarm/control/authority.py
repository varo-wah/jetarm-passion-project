"""Transport-independent priority and latching rules for JetArm motion."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from jetarm.control.limits import JointLimits, validate_pulse_targets


class ControlState(str, Enum):
    BOOT_LOCKED = "BOOT_LOCKED"
    PAUSED = "PAUSED"
    ACTIVE = "ACTIVE"
    ESTOP_LATCHED = "ESTOP_LATCHED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    FAULT = "FAULT"


@dataclass(frozen=True)
class AuthoritySnapshot:
    state: ControlState
    generation: int
    active_goal: bool
    reason: str

    @property
    def paused(self) -> bool:
        return self.state in {
            ControlState.BOOT_LOCKED,
            ControlState.PAUSED,
            ControlState.ESTOP_LATCHED,
            ControlState.SHUTTING_DOWN,
            ControlState.FAULT,
        }

    @property
    def estop_latched(self) -> bool:
        return self.state == ControlState.ESTOP_LATCHED

    @property
    def motion_allowed(self) -> bool:
        return self.state == ControlState.ACTIVE

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "generation": self.generation,
            "active_goal": self.active_goal,
            "reason": self.reason,
            "paused": self.paused,
            "estop_latched": self.estop_latched,
            "motion_allowed": self.motion_allowed,
        }


class MotionAuthority:
    """Single priority authority used by the ROS controller process."""

    def __init__(self, limits: Mapping[int, JointLimits]) -> None:
        self._limits = dict(limits)
        self._lock = threading.RLock()
        self._state = ControlState.BOOT_LOCKED
        self._generation = 0
        self._active_goal = False
        self._reason = "controller startup requires explicit resume"

    def snapshot(self) -> AuthoritySnapshot:
        with self._lock:
            return AuthoritySnapshot(
                state=self._state,
                generation=self._generation,
                active_goal=self._active_goal,
                reason=self._reason,
            )

    def resume(self) -> bool:
        with self._lock:
            if self._state in {
                ControlState.ESTOP_LATCHED,
                ControlState.SHUTTING_DOWN,
                ControlState.FAULT,
            }:
                return False
            self._state = ControlState.ACTIVE
            self._reason = "operator resume"
            return True

    def pause(self, reason: str = "operator pause") -> bool:
        with self._lock:
            if self._state == ControlState.ESTOP_LATCHED:
                return True
            if self._state in {ControlState.SHUTTING_DOWN, ControlState.FAULT}:
                return False
            self._generation += 1
            self._active_goal = False
            self._state = ControlState.PAUSED
            self._reason = reason
            return True

    def estop(self, reason: str = "software E-stop") -> bool:
        with self._lock:
            self._generation += 1
            self._active_goal = False
            self._state = ControlState.ESTOP_LATCHED
            self._reason = reason
            return True

    def clear_estop(self) -> bool:
        with self._lock:
            if self._state != ControlState.ESTOP_LATCHED:
                return False
            self._generation += 1
            self._active_goal = False
            self._state = ControlState.PAUSED
            self._reason = "E-stop cleared; explicit resume required"
            return True

    def begin_shutdown(self) -> bool:
        with self._lock:
            self._generation += 1
            self._active_goal = False
            self._state = ControlState.SHUTTING_DOWN
            self._reason = "safe shutdown requested"
            return True

    def fault(self, reason: str) -> None:
        with self._lock:
            self._generation += 1
            self._active_goal = False
            self._state = ControlState.FAULT
            self._reason = reason

    def begin_goal(self, targets: Mapping[int, float]) -> tuple[int, dict[int, int]]:
        validated = validate_pulse_targets(targets, self._limits)
        with self._lock:
            if self._state != ControlState.ACTIVE:
                raise RuntimeError(f"Motion blocked while controller is {self._state.value}")
            if self._active_goal:
                raise RuntimeError("Another motion goal is already active")
            self._generation += 1
            self._active_goal = True
            self._reason = "motion goal active"
            return self._generation, validated

    def goal_is_current(self, generation: int) -> bool:
        with self._lock:
            return (
                self._state == ControlState.ACTIVE
                and self._active_goal
                and self._generation == generation
            )

    def finish_goal(self, generation: int, reason: str = "motion goal complete") -> bool:
        with self._lock:
            if self._generation != generation:
                return False
            self._active_goal = False
            self._reason = reason
            return True
