import math
import time

from jetarm.control.limits import JointLimitsError, load_joint_limits, validate_pulse_targets
from jetarm.hardware.classCreation import CKMJetArm
from jetarm.hardware.safety_state import MOTION_SAFETY

class _UnavailableArm:
    def __init__(self, error: Exception):
        self.error = error

    def moveJetArm(self, servo_id, target_position, duration=1.0):
        raise RuntimeError("JetArm hardware is unavailable in this environment") from self.error

    def moveJetArmGroup(self, positions, duration=1.0):
        raise RuntimeError("JetArm hardware is unavailable in this environment") from self.error

    def smoothMoveJetArmGroup(self, positions, duration=1.2, steps=24):
        raise RuntimeError("JetArm hardware is unavailable in this environment") from self.error

    def safety_status(self):
        return MOTION_SAFETY.snapshot().as_dict()

    def pause(self):
        return MOTION_SAFETY.pause()

    def resume(self):
        return MOTION_SAFETY.resume()

    def estop(self):
        return MOTION_SAFETY.estop()

    def clear_estop(self):
        return MOTION_SAFETY.clear_estop()

    def safe_shutdown(self):
        return MOTION_SAFETY.pause()

    def close(self):
        return None


# Instantiate hardware once per process when ROS is available.
try:
    Arm = CKMJetArm()
except RuntimeError as exc:
    Arm = _UnavailableArm(exc)


def hardware_is_available() -> bool:
    return not isinstance(Arm, _UnavailableArm)


def hardware_unavailable_reason() -> str | None:
    if hardware_is_available():
        return None

    error = Arm.error
    root_cause = error
    while root_cause.__cause__ is not None:
        root_cause = root_cause.__cause__

    if root_cause is error:
        return str(error)
    return f"{error}: {root_cause}"


class JetArmIK:
    """
    move_to(x,y,z): Z is TABLE-REFERENCED TIP HEIGHT (cm above table) ✅
    move_to_wrist(x,y,z): Z is the old WRIST/J3 endpoint height used by your original IK ✅
    """
    def __init__(self):
        self.Arm = Arm
        self.L1 = 15.0
        self.L2 = 15.0

        self.DEG_PER_PULSE = 0.24
        self.BASE_ZERO_OFFSET = 125.0
        self.ANGLE_ZERO_OFFSET = 125.0

        self.ELBOW_UP = True
        self.last_base_angle = 0.0
        self._limits = load_joint_limits()

        # --- Your calibration (OPEN gripper, fixed wrist angle) ---
        # (0,15,20)->11 and (0,15,15)->7  => slope 0.8, bias -4.5
        # z_tip ≈ 0.8*z_wrist - 4.5 - sag(r)
        self.Z_TIP_PER_Z_WRIST = 0.8
        self.Z_TIP_BIAS_CM = -4.5

        # sag(r) at z_wrist=15: r=10->0.0, r=15->0.5, r=20->1.1, r=25->1.7
        self.SAG_TABLE = [(10.0, 0.0), (15.0, 0.5), (20.0, 1.1), (25.0, 1.7)]
        self.SAG_MAX_CM = 5.0

    def base_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))

    def arm_to_pulse(self, arm_deg):
        return int(round(arm_deg / self.DEG_PER_PULSE + self.ANGLE_ZERO_OFFSET))

    def _sag_cm(self, r_cm: float) -> float:
        t = self.SAG_TABLE
        if r_cm <= t[0][0]:
            return max(0.0, min(self.SAG_MAX_CM, t[0][1]))
        if r_cm >= t[-1][0]:
            return max(0.0, min(self.SAG_MAX_CM, t[-1][1]))

        for i in range(len(t) - 1):
            r0, s0 = t[i]
            r1, s1 = t[i + 1]
            if r0 <= r_cm <= r1:
                u = (r_cm - r0) / (r1 - r0)
                return max(0.0, min(self.SAG_MAX_CM, s0 + u * (s1 - s0)))

        return 0.0

    def _z_table_to_wrist(self, x: float, y: float, z_table_cm: float) -> float:
        # z_tip ≈ a*z_wrist + b - sag(r)  =>  z_wrist = (z_tip - b + sag)/a
        r = math.hypot(x, y)
        sag = self._sag_cm(r)
        a = self.Z_TIP_PER_Z_WRIST
        b = self.Z_TIP_BIAS_CM
        return (z_table_cm - b + sag) / a

    def calculate_angles(self, x, y, z_wrist, *, update_continuity=True):
        # 1) Base angle
        base_angle = math.degrees(math.atan2(y, x))

        # 2) Continuity seam-fix
        prev = getattr(self, "last_base_angle", 0.0)
        if prev - base_angle > 180.0:
            base_angle += 360.0
        elif base_angle - prev > 180.0:
            base_angle -= 360.0
        if update_continuity:
            self.last_base_angle = base_angle

        # 3) Planar IK (your method, with a safe reach guard)
        l = math.hypot(x, y)
        d = math.hypot(l, z_wrist)
        h = d / 2.0

        cos_arg = h / self.L1
        if cos_arg < -1.0 or cos_arg > 1.0:
            raise ValueError("Target out of reach for current IK geometry")

        theta = math.degrees(math.acos(cos_arg))
        phi = math.degrees(math.atan2(z_wrist, l))

        if self.ELBOW_UP:
            L1_angle = phi + theta
        else:
            L1_angle = phi - theta

        intersection = 180.0 - (2.0 * theta)
        if self.ELBOW_UP:
            L2_angle = intersection - 90.0
        else:
            L2_angle = 360.0 - (intersection + 90.0)

        L3_angle = 90.0 - (L2_angle + L1_angle)

        return base_angle, L1_angle, L2_angle, L3_angle

    def _pulse_targets(self, base_angle, L1_angle, L2_angle, L3_angle):
        return validate_pulse_targets(
            {
                1: self.base_to_pulse(base_angle),
                2: self.arm_to_pulse(L1_angle),
                3: self.arm_to_pulse(L2_angle),
                4: self.arm_to_pulse(L3_angle) + 35,
            },
            self._limits,
        )

    def plan_to_wrist(self, x, y, z_wrist):
        angles = self.calculate_angles(x, y, z_wrist, update_continuity=False)
        return self._pulse_targets(*angles)

    def plan_to(self, x, y, z_table):
        return self.plan_to_wrist(x, y, self._z_table_to_wrist(x, y, z_table))

    def _apply_pulses(self, base_angle, L1_angle, L2_angle, L3_angle, x, y, z_wrist):
        try:
            targets = self._pulse_targets(base_angle, L1_angle, L2_angle, L3_angle)
        except JointLimitsError as error:
            print(
                f"❌ Calibrated joint limit rejected x={x:.1f}, y={y:.1f}, "
                f"z_wrist={z_wrist:.1f} ({error})"
            )
            return False

        print(
            f"Smooth moving to: {targets[1]}, {targets[2]}, "
            f"{targets[3]}, {targets[4]}"
        )
        return self.Arm.smoothMoveJetArmGroup(
            targets,
            duration=1.2,
            steps=24,
        )

    def move_to_wrist(self, x, y, z_wrist):
        if not motion_is_allowed():
            print("🛑 Motion safety latch blocked IK movement")
            return False

        # Old behavior (raw IK wrist Z)
        try:
            base_angle, L1_angle, L2_angle, L3_angle = self.calculate_angles(x, y, z_wrist)
        except ValueError as e:
            print(f"❌ IK math failed for x={x:.1f}, y={y:.1f}, z_wrist={z_wrist:.1f} ({e})")
            return False
        return self._apply_pulses(base_angle, L1_angle, L2_angle, L3_angle, x, y, z_wrist)

    def move_to(self, x, y, z_table):
        if not motion_is_allowed():
            print("🛑 Motion safety latch blocked IK movement")
            return False

        # New default behavior: table-referenced tip height
        z_wrist = self._z_table_to_wrist(x, y, z_table)
        return self.move_to_wrist(x, y, z_wrist)


class JetArmGripper:
    def __init__(self, ik: JetArmIK):
        self.Arm = Arm
        self.ik = ik
        self.openGripperPulse = 150
        self.closeGripperPulse = 700
        self.BASE_ZERO_OFFSET = 125.0
        self.DEG_PER_PULSE = 0.24

    def wrist_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))

    def plan_wrist(self, angle, *, base_angle=None):
        if base_angle is None:
            base_angle = self.ik.last_base_angle
        final_angle = (angle - 90.0) + base_angle
        return validate_pulse_targets(
            {5: self.wrist_to_pulse(final_angle)},
            self.ik._limits,
        )

    def plan_gripper(self):
        open_target = validate_pulse_targets(
            {10: self.openGripperPulse},
            self.ik._limits,
        )[10]
        close_target = validate_pulse_targets(
            {10: self.closeGripperPulse},
            self.ik._limits,
        )[10]
        return {"open": open_target, "close": close_target}

    def turn_wrist(self, angle):
        if not motion_is_allowed():
            print("🛑 Motion safety latch blocked wrist movement")
            return False

        wrist_pulse = self.plan_wrist(angle)[5]
        return self.Arm.moveJetArm(5, wrist_pulse)

    def close_gripper(self):
        if not motion_is_allowed():
            print("🛑 Motion safety latch blocked gripper close")
            return False
        return self.Arm.moveJetArm(10, self.closeGripperPulse)

    def open_gripper(self):
        if not motion_is_allowed():
            print("🛑 Motion safety latch blocked gripper open")
            return False
        return self.Arm.moveJetArm(10, self.openGripperPulse)


class ComputerVision:
    def __init__(self, ik: JetArmIK, gripper: JetArmGripper):
        self.Arm = Arm
        self.ik = ik
        self.gripper = gripper

    def scan_position(self):
        # Preserve your previous behavior by using WRIST-Z here:
        if not self.ik.move_to_wrist(0, 15, 23):
            return False
        if not self.gripper.turn_wrist(90):
            return False
        return self.gripper.open_gripper()


class UserFriendlyMode:
    STEP_DELAY = 0.45
    POSE_DELAY = 0.7
    PERSON_FOLLOW_X_LIMIT_CM = 6.0
    PERSON_FOLLOW_SERVO4_FORWARD_PULSE = 400
    BUTTON_APPROACH_POSE = (0, 7.5, 13)
    BUTTON_PRESS_POSE = (0, 7.5, 12)
    BUTTON_PRESS_HOLD = 0.45
    HELLO_ARC_POINTS = (
        (0, 15.0, 23.0, 90, 0.35),
        (0, 17.0, 24.0, 70, 0.35),
        (0, 16.0, 22.0, 110, 0.35),
        (0, 13.5, 24.0, 70, 0.35),
        (0, 14.5, 22.0, 110, 0.35),
        (0, 15.0, 23.0, 90, 0.45),
    )

    def __init__(self, ik: JetArmIK, gripper: JetArmGripper, camera: ComputerVision):
        self.Arm = Arm
        self.ik = ik
        self.gripper = gripper
        self.camera = camera

    def _motion_allowed(self):
        if not motion_is_allowed():
            print("[USER FRIENDLY] Motion safety latch blocked movement")
            return False
        return True

    def _wait(self, seconds=None):
        time.sleep(self.STEP_DELAY if seconds is None else seconds)

    def _forward_scan_height_pose(self):
        ok = self.ik.move_to_wrist(0, 15, 23)
        if not ok:
            return False
        if not self.gripper.turn_wrist(90):
            return False
        if not self.gripper.open_gripper():
            return False
        self._wait(self.POSE_DELAY)
        return True

    def dummy_position(self):
        if not self._motion_allowed():
            return False
        if not self.Arm.smoothMoveJetArmGroup({1: 500, 2: 750, 3: 350, 4: 400}, duration=1.2):
            return False
        if not self.gripper.turn_wrist(90):
            return False
        return self.gripper.open_gripper()

    def idle_pose(self):
        if not self._motion_allowed():
            return False

        print("[USER FRIENDLY] Idle pose")
        return self._forward_scan_height_pose()

    def look_around(self):
        if not self._motion_allowed():
            return False

        print("[USER FRIENDLY] Look around")
        for x in (-4, 4, 0):
            if not self._motion_allowed():
                return False
            ok = self.ik.move_to_wrist(x, 15, 23)
            if not ok:
                return False
            if not self.gripper.turn_wrist(90):
                return False
            self._wait()
        return True

    def hello_wave(self):
        if not self._motion_allowed():
            return False

        print("[USER FRIENDLY] Hello wave")
        for x, y, z_wrist, wrist_angle, delay in self.HELLO_ARC_POINTS:
            if not self._motion_allowed():
                return False
            ok = self.ik.move_to_wrist(x, y, z_wrist)
            if not ok:
                return False
            if not self.gripper.turn_wrist(wrist_angle):
                return False
            self._wait(delay)
        return True

    def curious_idle(self):
        if not self._motion_allowed():
            return False

        print("[USER FRIENDLY] Curious idle")
        for x, y, z_wrist in ((-3, 15, 24), (3, 15, 24), (0, 14, 23), (0, 15, 23)):
            if not self._motion_allowed():
                return False
            ok = self.ik.move_to_wrist(x, y, z_wrist)
            if not ok:
                return False
            if not self.gripper.turn_wrist(90):
                return False
            self._wait()

        return self.gripper.open_gripper()

    def person_follow_pose(self, x_offset_cm):
        if not self._motion_allowed():
            return False

        x_offset_cm = max(
            -self.PERSON_FOLLOW_X_LIMIT_CM,
            min(self.PERSON_FOLLOW_X_LIMIT_CM, float(x_offset_cm)),
        )
        ok = self.ik.move_to_wrist(x_offset_cm, 15, 23)
        if not ok:
            return False

        if not self.Arm.moveJetArm(4, self.PERSON_FOLLOW_SERVO4_FORWARD_PULSE):
            return False
        if not self.gripper.turn_wrist(90):
            return False
        return self.gripper.open_gripper()

    def press_button(self):
        if not self._motion_allowed():
            return False

        print("[USER FRIENDLY] Press button")

        if not self.ik.move_to(*self.BUTTON_APPROACH_POSE):
            return False
        self._wait(self.POSE_DELAY)

        if not self.gripper.turn_wrist(90):
            return False
        if not self.gripper.close_gripper():
            return False
        self._wait(0.35)

        if not self.ik.move_to(*self.BUTTON_PRESS_POSE):
            self.gripper.open_gripper()
            return False
        self._wait(self.BUTTON_PRESS_HOLD)

        if not self.ik.move_to(*self.BUTTON_APPROACH_POSE):
            self.gripper.open_gripper()
            return False
        self._wait(0.35)

        return self.gripper.open_gripper()

    def idle_mode(self):
        return self.idle_pose()



ik = JetArmIK()
gripper = JetArmGripper(ik)
camera = ComputerVision(ik, gripper)
ufm = UserFriendlyMode(ik, gripper, camera)

# -----------------------------
# UI control latches
# -----------------------------
PAUSED = False
ESTOP_LATCHED = False


def _sync_legacy_flags() -> None:
    global PAUSED, ESTOP_LATCHED
    snapshot = motion_safety_status()
    PAUSED = bool(snapshot["paused"])
    ESTOP_LATCHED = bool(snapshot["estop_latched"])


def motion_is_allowed() -> bool:
    return bool(motion_safety_status()["motion_allowed"])


def motion_safety_status() -> dict[str, object]:
    if hardware_is_available():
        return Arm.safety_status()
    return MOTION_SAFETY.snapshot().as_dict()

def pause_system() -> bool:
    ok = Arm.pause()
    _sync_legacy_flags()
    print("⏸️ PAUSED")
    return ok

def resume_system() -> bool:
    ok = Arm.resume()
    _sync_legacy_flags()
    if not ok:
        print("❌ Cannot resume: E-STOP is latched")
        return False
    print("▶️ RESUMED")
    return True

def stop_motion() -> bool:
    # Soft stop = pause
    return pause_system()

def estop_motion() -> bool:
    ok = Arm.estop()
    _sync_legacy_flags()
    print("🛑 E-STOP LATCHED (software) — motion should halt at loop level")
    return ok

def clear_estop() -> bool:
    ok = Arm.clear_estop()
    _sync_legacy_flags()
    print("✅ E-STOP CLEARED; system remains paused until resume")
    return ok


def safe_shutdown() -> bool:
    """Cancel active work and request the controller's calibrated safe pose."""
    ok = Arm.safe_shutdown()
    _sync_legacy_flags()
    return ok


def shutdown_control_client() -> None:
    """Release the web process's ROS executor after its final safety request."""
    Arm.close()

if __name__ == "__main__":
    while True:
        try:
            code = input(">>> ")
            if code.lower() in ["exit", "quit"]:
                break
            exec(code)
        except Exception as e:
            print(f"❌ Error: {e}")
