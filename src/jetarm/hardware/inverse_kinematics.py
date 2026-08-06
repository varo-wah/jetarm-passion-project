"""Pure inverse-kinematics planning for the JetArm Cartesian API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from jetarm.control.limits import JointLimits, load_joint_limits, validate_pulse_targets


BASE_SERVO_ID = 1
SHOULDER_SERVO_ID = 2
ELBOW_SERVO_ID = 3
WRIST_PITCH_SERVO_ID = 4


class IKError(ValueError):
    """Base exception for invalid or unsolvable Cartesian targets."""


class IKInputError(IKError):
    """Raised when an IK input is not finite or the model is malformed."""


class IKUnreachableError(IKError):
    """Raised when the wrist target lies outside the arm geometry."""


@dataclass(frozen=True, slots=True)
class IKConfig:
    link1_cm: float = 15.0
    link2_cm: float = 15.0
    degrees_per_pulse: float = 0.24
    base_zero_offset: float = 125.0
    arm_zero_offset: float = 125.0
    wrist_pitch_pulse_offset: int = 35
    tip_per_wrist_height: float = 0.8
    tip_height_bias_cm: float = -4.5
    sag_table: tuple[tuple[float, float], ...] = (
        (10.0, 0.0),
        (15.0, 0.5),
        (20.0, 1.1),
        (25.0, 1.7),
    )
    sag_max_cm: float = 5.0

    def __post_init__(self) -> None:
        positive_values = {
            "link1_cm": self.link1_cm,
            "link2_cm": self.link2_cm,
            "degrees_per_pulse": self.degrees_per_pulse,
            "tip_per_wrist_height": self.tip_per_wrist_height,
            "sag_max_cm": self.sag_max_cm,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise IKInputError(f"{name} must be finite and positive")
        if not self.sag_table:
            raise IKInputError("sag_table must contain at least one calibration point")
        previous_radius = -math.inf
        for radius, sag in self.sag_table:
            if not math.isfinite(radius) or not math.isfinite(sag):
                raise IKInputError("sag_table values must be finite")
            if radius <= previous_radius:
                raise IKInputError("sag_table radii must be strictly increasing")
            previous_radius = radius


@dataclass(frozen=True, slots=True)
class IKAngles:
    base_deg: float
    shoulder_deg: float
    elbow_deg: float
    wrist_pitch_deg: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.base_deg,
            self.shoulder_deg,
            self.elbow_deg,
            self.wrist_pitch_deg,
        )


@dataclass(frozen=True, slots=True)
class IKPlan:
    x_cm: float
    y_cm: float
    z_wrist_cm: float
    angles: IKAngles
    base_pulse: int
    shoulder_pulse: int
    elbow_pulse: int
    wrist_pitch_pulse: int
    z_table_cm: float | None = None

    @property
    def servo_targets(self) -> dict[int, int]:
        return {
            BASE_SERVO_ID: self.base_pulse,
            SHOULDER_SERVO_ID: self.shoulder_pulse,
            ELBOW_SERVO_ID: self.elbow_pulse,
            WRIST_PITCH_SERVO_ID: self.wrist_pitch_pulse,
        }


class JetArmKinematics:
    """Calculate and validate JetArm targets without commanding hardware."""

    def __init__(
        self,
        config: IKConfig | None = None,
        joint_limits: Mapping[int, JointLimits] | None = None,
    ) -> None:
        self.config = config or IKConfig()
        self.joint_limits = dict(joint_limits) if joint_limits is not None else load_joint_limits()

    @staticmethod
    def _finite(name: str, value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise IKInputError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise IKInputError(f"{name} must be a finite number")
        return result

    def base_to_pulse(self, angle_deg: float) -> int:
        angle = self._finite("base angle", angle_deg)
        return int(round(angle / self.config.degrees_per_pulse + self.config.base_zero_offset))

    def arm_to_pulse(self, angle_deg: float) -> int:
        angle = self._finite("arm angle", angle_deg)
        return int(round(angle / self.config.degrees_per_pulse + self.config.arm_zero_offset))

    def sag_cm(self, radius_cm: float) -> float:
        radius = self._finite("radius_cm", radius_cm)
        table = self.config.sag_table
        if radius <= table[0][0]:
            return max(0.0, min(self.config.sag_max_cm, table[0][1]))
        if radius >= table[-1][0]:
            return max(0.0, min(self.config.sag_max_cm, table[-1][1]))

        for (radius0, sag0), (radius1, sag1) in zip(table, table[1:]):
            if radius0 <= radius <= radius1:
                fraction = (radius - radius0) / (radius1 - radius0)
                interpolated = sag0 + fraction * (sag1 - sag0)
                return max(0.0, min(self.config.sag_max_cm, interpolated))

        raise IKInputError("radius did not intersect the configured sag table")

    def table_to_wrist_height(self, x_cm: float, y_cm: float, z_table_cm: float) -> float:
        x = self._finite("x_cm", x_cm)
        y = self._finite("y_cm", y_cm)
        z_table = self._finite("z_table_cm", z_table_cm)
        sag = self.sag_cm(math.hypot(x, y))
        return (
            z_table - self.config.tip_height_bias_cm + sag
        ) / self.config.tip_per_wrist_height

    @staticmethod
    def _unwrap_base_angle(base_deg: float, previous_base_deg: float) -> float:
        while previous_base_deg - base_deg > 180.0:
            base_deg += 360.0
        while base_deg - previous_base_deg > 180.0:
            base_deg -= 360.0
        return base_deg

    def solve_angles(
        self,
        x_cm: float,
        y_cm: float,
        z_wrist_cm: float,
        *,
        previous_base_deg: float = 0.0,
        elbow_up: bool = True,
    ) -> IKAngles:
        x = self._finite("x_cm", x_cm)
        y = self._finite("y_cm", y_cm)
        z_wrist = self._finite("z_wrist_cm", z_wrist_cm)
        previous_base = self._finite("previous_base_deg", previous_base_deg)

        radial = math.hypot(x, y)
        distance = math.hypot(radial, z_wrist)
        minimum_reach = abs(self.config.link1_cm - self.config.link2_cm)
        maximum_reach = self.config.link1_cm + self.config.link2_cm
        tolerance = 1e-9
        if radial <= tolerance:
            # Yaw is underdetermined on the base axis; retaining the previous
            # angle prevents an arbitrary rotation toward atan2(0, 0) == 0.
            base = previous_base
        else:
            base = self._unwrap_base_angle(
                math.degrees(math.atan2(y, x)),
                previous_base,
            )
        if distance <= tolerance:
            raise IKUnreachableError("Target is at the shoulder singularity")
        if distance < minimum_reach - tolerance or distance > maximum_reach + tolerance:
            raise IKUnreachableError(
                f"Target distance {distance:.3f} cm is outside reachable range "
                f"[{minimum_reach:.3f}, {maximum_reach:.3f}] cm"
            )

        shoulder_cos = (
            self.config.link1_cm**2
            + distance**2
            - self.config.link2_cm**2
        ) / (2.0 * self.config.link1_cm * distance)
        elbow_cos = (
            self.config.link1_cm**2
            + self.config.link2_cm**2
            - distance**2
        ) / (2.0 * self.config.link1_cm * self.config.link2_cm)
        shoulder_offset = math.degrees(math.acos(max(-1.0, min(1.0, shoulder_cos))))
        elbow_intersection = math.degrees(math.acos(max(-1.0, min(1.0, elbow_cos))))
        target_elevation = math.degrees(math.atan2(z_wrist, radial))

        if elbow_up:
            shoulder = target_elevation + shoulder_offset
            elbow = elbow_intersection - 90.0
        else:
            shoulder = target_elevation - shoulder_offset
            elbow = 270.0 - elbow_intersection
        wrist_pitch = 90.0 - (elbow + shoulder)

        return IKAngles(
            base_deg=base,
            shoulder_deg=shoulder,
            elbow_deg=elbow,
            wrist_pitch_deg=wrist_pitch,
        )

    def angles_to_pulses(self, angles: IKAngles) -> dict[int, int]:
        pulses = {
            BASE_SERVO_ID: self.base_to_pulse(angles.base_deg),
            SHOULDER_SERVO_ID: self.arm_to_pulse(angles.shoulder_deg),
            ELBOW_SERVO_ID: self.arm_to_pulse(angles.elbow_deg),
            WRIST_PITCH_SERVO_ID: (
                self.arm_to_pulse(angles.wrist_pitch_deg)
                + self.config.wrist_pitch_pulse_offset
            ),
        }
        return validate_pulse_targets(pulses, self.joint_limits)

    def plan_wrist(
        self,
        x_cm: float,
        y_cm: float,
        z_wrist_cm: float,
        *,
        previous_base_deg: float = 0.0,
        elbow_up: bool = True,
    ) -> IKPlan:
        angles = self.solve_angles(
            x_cm,
            y_cm,
            z_wrist_cm,
            previous_base_deg=previous_base_deg,
            elbow_up=elbow_up,
        )
        pulses = self.angles_to_pulses(angles)
        return IKPlan(
            x_cm=float(x_cm),
            y_cm=float(y_cm),
            z_wrist_cm=float(z_wrist_cm),
            angles=angles,
            base_pulse=pulses[BASE_SERVO_ID],
            shoulder_pulse=pulses[SHOULDER_SERVO_ID],
            elbow_pulse=pulses[ELBOW_SERVO_ID],
            wrist_pitch_pulse=pulses[WRIST_PITCH_SERVO_ID],
        )

    def plan_table(
        self,
        x_cm: float,
        y_cm: float,
        z_table_cm: float,
        *,
        previous_base_deg: float = 0.0,
        elbow_up: bool = True,
    ) -> IKPlan:
        z_wrist = self.table_to_wrist_height(x_cm, y_cm, z_table_cm)
        wrist_plan = self.plan_wrist(
            x_cm,
            y_cm,
            z_wrist,
            previous_base_deg=previous_base_deg,
            elbow_up=elbow_up,
        )
        return IKPlan(
            x_cm=wrist_plan.x_cm,
            y_cm=wrist_plan.y_cm,
            z_wrist_cm=wrist_plan.z_wrist_cm,
            z_table_cm=float(z_table_cm),
            angles=wrist_plan.angles,
            base_pulse=wrist_plan.base_pulse,
            shoulder_pulse=wrist_plan.shoulder_pulse,
            elbow_pulse=wrist_plan.elbow_pulse,
            wrist_pitch_pulse=wrist_plan.wrist_pitch_pulse,
        )
