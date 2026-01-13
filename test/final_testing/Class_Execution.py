import sys, os
import math

# Add parent directory of this file to the module search path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from classCreation import CKMJetArm
Arm = CKMJetArm()

# Optional ROS imports (kept safe if ROS isn't installed)
try:
    import rclpy
    from rclpy.node import Node
    from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition
except Exception:
    rclpy = None
    Node = object
    ServosPosition = None
    ServoPosition = None


class JetArmIK:
    def __init__(self):
        self.Arm = Arm
        self.L1 = 15.0
        self.L2 = 15.0

        self.DEG_PER_PULSE = 0.24
        self.BASE_ZERO_OFFSET = 125.0
        self.ANGLE_ZERO_OFFSET = 125.0

        self.ELBOW_UP = True
        self.last_base_angle = 0.0

        # -------------------------------
        # TABLE-Z CALIBRATION (YOUR DATA)
        # -------------------------------
        # Measured (gripper OPEN, fixed wrist angle):
        # (0,15,20) -> z_tip=11
        # (0,15,15) -> z_tip=7
        #
        # => z_tip ≈ 0.8*z_wrist + (-4.5) - sag(r)
        self.Z_TIP_PER_Z_WRIST = 0.8
        self.Z_TIP_BIAS_CM = -4.5

        # sag(r) derived at z_wrist=15:
        # r=10 => 0.0 ; r=15 => 0.5 ; r=20 => 1.1 ; r=25 => 1.7
        self.SAG_TABLE = [
            (10.0, 0.0),
            (15.0, 0.5),
            (20.0, 1.1),
            (25.0, 1.7),
        ]
        self.SAG_MAX_CM = 5.0  # clamp for safety

        # Optional: when gripper closes, your "lowest tip point" changes (~+2 cm).
        self.GRIPPER_CLOSE_LIFT_CM = 2.0

    def base_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))

    def arm_to_pulse(self, arm_deg):
        return int(round(arm_deg / self.DEG_PER_PULSE + self.ANGLE_ZERO_OFFSET))

    def calculate_angles(self, x, y, z):
        # 1) Base angle from atan2 in [-180, 180]
        base_angle = math.degrees(math.atan2(y, x))

        # 2) Seam-fix ONLY at the -180/+180 wrap
        prev = getattr(self, "last_base_angle", 0.0)
        if prev - base_angle > 180.0:
            base_angle += 360.0
        elif base_angle - prev > 180.0:
            base_angle -= 360.0

        # 3) Store for wrist alignment + continuity next call
        self.last_base_angle = base_angle

        # --- Existing planar IK math (kept, but with safer guards) ---
        l = math.hypot(x, y)
        d = math.hypot(l, z)

        # This IK assumes L1 == L2 and uses an isosceles construction (h=d/2).
        # Must have d <= 2*L1, otherwise acos() will fail.
        h = d / 2.0
        cos_arg = h / self.L1
        if cos_arg < -1.0 or cos_arg > 1.0:
            raise ValueError("Target out of reach for current IK geometry")

        theta = math.degrees(math.acos(cos_arg))
        phi = math.degrees(math.atan2(z, l))

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

    # -------------------------------
    # TABLE-Z HELPERS (NEW)
    # -------------------------------
    def sag_cm(self, r_cm: float) -> float:
        table = self.SAG_TABLE

        if r_cm <= table[0][0]:
            return max(0.0, min(self.SAG_MAX_CM, table[0][1]))
        if r_cm >= table[-1][0]:
            return max(0.0, min(self.SAG_MAX_CM, table[-1][1]))

        for i in range(len(table) - 1):
            r0, s0 = table[i]
            r1, s1 = table[i + 1]
            if r0 <= r_cm <= r1:
                t = (r_cm - r0) / (r1 - r0)
                sag = s0 + t * (s1 - s0)
                return max(0.0, min(self.SAG_MAX_CM, sag))

        return 0.0

    def z_table_to_wrist(self, x: float, y: float, z_table_cm: float) -> float:
        # z_tip ≈ a*z_wrist + b - sag(r)
        # => z_wrist = (z_tip - b + sag(r)) / a
        r = math.hypot(x, y)
        sag = self.sag_cm(r)
        a = self.Z_TIP_PER_Z_WRIST
        b = self.Z_TIP_BIAS_CM
        return (z_table_cm - b + sag) / a

    def move_to_table(self, x: float, y: float, z_table_cm: float) -> bool:
        z_wrist = self.z_table_to_wrist(x, y, z_table_cm)
        return self.move_to(x, y, z_wrist)

    def move_to(self, x, y, z):
        try:
            base_angle, L1_angle, L2_angle, L3_angle = self.calculate_angles(x, y, z)
        except ValueError as e:
            print(f"❌ IK math failed for x={x:.1f}, y={y:.1f}, z={z:.1f}  ({e})")
            return False

        base_pulse = self.base_to_pulse(base_angle)
        L1_pulse = self.arm_to_pulse(L1_angle)
        L2_pulse = self.arm_to_pulse(L2_angle)
        L3_pulse = self.arm_to_pulse(L3_angle) + 35

        pulses = {"base": base_pulse, "L1": L1_pulse, "L2": L2_pulse, "L3": L3_pulse}

        for name, p in pulses.items():
            if not isinstance(p, int):
                p = int(round(p))
                pulses[name] = p
            if p < 0 or p > 1000:
                print(f"❌ Joint limit: {name} pulse={p} (x={x:.1f}, y={y:.1f}, z={z:.1f})")
                return False

        print(f"Moving to: {pulses['base']}, {pulses['L1']}, {pulses['L2']}, {pulses['L3']}")
        self.Arm.moveJetArm(1, pulses["base"])
        self.Arm.moveJetArm(2, pulses["L1"])
        self.Arm.moveJetArm(3, pulses["L2"])
        self.Arm.moveJetArm(4, pulses["L3"])
        return True


class JetArmGripper:
    def __init__(self, ik: JetArmIK):
        self.Arm = Arm
        self.ik = ik

        self.openGripperPulse = 150
        self.closeGripperPulse = 850

        self.BASE_ZERO_OFFSET = 125.0
        self.DEG_PER_PULSE = 0.24

    def wrist_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))

    def turn_wrist(self, angle):
        base_angle = self.ik.last_base_angle
        final_angle = (angle - 90.0) + base_angle
        wrist_pulse = self.wrist_to_pulse(final_angle)
        self.Arm.moveJetArm(5, wrist_pulse)

    def close_gripper(self):
        self.Arm.moveJetArm(10, self.closeGripperPulse)

    def open_gripper(self):
        self.Arm.moveJetArm(10, self.openGripperPulse)


class ComputerVision:
    def __init__(self, ik: JetArmIK, gripper: JetArmGripper):
        self.Arm = Arm
        self.ik = ik
        self.gripper = gripper

    def scan_position(self):
        # Example: 15 cm above table at (0,15)
        self.ik.move_to_table(0, 15, 15)
        self.gripper.turn_wrist(90)
        self.gripper.open_gripper()


class UserFriendlyMode:
    def __init__(self, ik: JetArmIK, gripper: JetArmGripper, camera: ComputerVision):
        self.Arm = Arm
        self.ik = ik
        self.gripper = gripper
        self.camera = camera

    def dummy_position(self):
        self.Arm.moveJetArm(1, 500)
        self.Arm.moveJetArm(2, 750)
        self.Arm.moveJetArm(3, 350)
        self.Arm.moveJetArm(4, 400)
        self.gripper.turn_wrist(90)
        self.gripper.open_gripper()


ik = JetArmIK()
gripper = JetArmGripper(ik)
camera = ComputerVision(ik, gripper)
ufm = UserFriendlyMode(ik, gripper, camera)

if __name__ == "__main__":
    while True:
        try:
            code = input(">>> ")
            if code.lower() in ["exit", "quit"]:
                break
            exec(code)
        except Exception as e:
            print(f"❌ Error: {e}")
