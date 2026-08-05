"""Per-joint pulse calibration and command validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class JointLimitsError(ValueError):
    """Raised when calibration or a commanded target is invalid."""


@dataclass(frozen=True)
class JointLimits:
    name: str
    servo_id: int
    min_pulse: int
    max_pulse: int
    center_pulse: int
    radians_per_pulse: float
    max_velocity_pulses_per_second: float
    safe_pose_pulse: int

    def validate_pulse(self, pulse: float) -> int:
        if not math.isfinite(float(pulse)):
            raise JointLimitsError(f"{self.name}: pulse must be finite")
        rounded = int(round(float(pulse)))
        if not self.min_pulse <= rounded <= self.max_pulse:
            raise JointLimitsError(
                f"{self.name}: pulse {rounded} is outside calibrated range "
                f"[{self.min_pulse}, {self.max_pulse}]"
            )
        return rounded

    def pulse_to_radians(self, pulse: float) -> float:
        return (self.validate_pulse(pulse) - self.center_pulse) * self.radians_per_pulse

    def radians_to_pulse(self, radians: float) -> int:
        if not math.isfinite(float(radians)):
            raise JointLimitsError(f"{self.name}: position must be finite")
        pulse = self.center_pulse + (float(radians) / self.radians_per_pulse)
        return self.validate_pulse(pulse)


def default_limits_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "joint_limits.yaml"


def load_joint_limits(path: str | Path | None = None) -> dict[int, JointLimits]:
    """Load JSON-compatible YAML without requiring a YAML runtime dependency."""

    config_path = Path(path) if path is not None else default_limits_path()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointLimitsError(f"Unable to load joint calibration: {config_path}: {exc}") from exc

    joints = payload.get("joints")
    if not isinstance(joints, list) or not joints:
        raise JointLimitsError("Joint calibration must contain a non-empty joints list")

    limits: dict[int, JointLimits] = {}
    names: set[str] = set()
    for entry in joints:
        try:
            joint = JointLimits(
                name=str(entry["name"]),
                servo_id=int(entry["servo_id"]),
                min_pulse=int(entry["min_pulse"]),
                max_pulse=int(entry["max_pulse"]),
                center_pulse=int(entry["center_pulse"]),
                radians_per_pulse=float(entry["radians_per_pulse"]),
                max_velocity_pulses_per_second=float(entry["max_velocity_pulses_per_second"]),
                safe_pose_pulse=int(entry["safe_pose_pulse"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JointLimitsError(f"Malformed joint calibration entry: {entry!r}") from exc

        if joint.servo_id in limits or joint.name in names:
            raise JointLimitsError(f"Duplicate joint calibration: {joint.name}/{joint.servo_id}")
        if joint.min_pulse >= joint.max_pulse:
            raise JointLimitsError(f"{joint.name}: min_pulse must be below max_pulse")
        if joint.radians_per_pulse <= 0 or joint.max_velocity_pulses_per_second <= 0:
            raise JointLimitsError(f"{joint.name}: calibration scales must be positive")
        joint.validate_pulse(joint.center_pulse)
        joint.validate_pulse(joint.safe_pose_pulse)
        limits[joint.servo_id] = joint
        names.add(joint.name)

    return limits


def limits_by_name(limits: Mapping[int, JointLimits]) -> dict[str, JointLimits]:
    return {joint.name: joint for joint in limits.values()}


def validate_pulse_targets(
    targets: Mapping[int, float],
    limits: Mapping[int, JointLimits],
) -> dict[int, int]:
    if not targets:
        raise JointLimitsError("A motion command must contain at least one joint")

    validated: dict[int, int] = {}
    for raw_servo_id, pulse in targets.items():
        servo_id = int(raw_servo_id)
        if servo_id in validated:
            raise JointLimitsError(f"Duplicate servo ID: {servo_id}")
        joint = limits.get(servo_id)
        if joint is None:
            raise JointLimitsError(f"Unknown servo ID: {servo_id}")
        validated[servo_id] = joint.validate_pulse(pulse)
    return validated


def fastest_safe_duration(
    starts: Mapping[int, float],
    targets: Mapping[int, float],
    limits: Mapping[int, JointLimits],
    requested_duration: float,
) -> float:
    """Return the shortest duration that respects every calibrated velocity."""

    if not math.isfinite(float(requested_duration)) or requested_duration <= 0:
        raise JointLimitsError("Motion duration must be positive and finite")

    duration = float(requested_duration)
    for servo_id, target in validate_pulse_targets(targets, limits).items():
        if servo_id not in starts:
            continue
        start = limits[servo_id].validate_pulse(starts[servo_id])
        minimum = abs(target - start) / limits[servo_id].max_velocity_pulses_per_second
        duration = max(duration, minimum)
    return duration
