import sys, os

# Add parent directory of this file to the module search path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from classCreation import CKMJetArm
Arm = CKMJetArm()
import math
import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition


class JetArmIK:
    def __init__(self):
        self.Arm = Arm
        self.L1 = 15.0
        self.L2 = 15.0
        self.DEG_PER_PULSE = 0.24
        self.BASE_ZERO_OFFSET = 125.0
        self.ANGLE_ZERO_OFFSET = 125.0
        self.ELBOW_UP = True

    def base_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))

    def arm_to_pulse(self, arm_deg):
        return int(round(arm_deg / self.DEG_PER_PULSE + self.ANGLE_ZERO_OFFSET))

    def calculate_angles(self, x, y, z):
        base_angle = math.degrees(math.atan2(y, x))
        l = math.hypot(x, y)
        d = math.hypot(l, z)
        h = d / 2
        theta = math.degrees(math.acos(h / self.L1))
        phi = math.degrees(math.atan2(z, l))

        if self.ELBOW_UP:
            L1_angle = phi + theta
        else:
            L1_angle = phi - theta

        intersection = 180 - (2 * theta)
        if self.ELBOW_UP:
            L2_angle = intersection - 90
        else:
            L2_angle = 360 - (intersection + 90)

        L3_angle = 90 - (L2_angle + L1_angle)
        
        self.last_base_angle = base_angle

        return base_angle, L1_angle, L2_angle, L3_angle

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

        # Servo safety limits (adjust max if your servos use a different range)
        pulses = {
            "base": base_pulse,
            "L1": L1_pulse,
            "L2": L2_pulse,
            "L3": L3_pulse
        }

        for name, p in pulses.items():
            if not isinstance(p, int):
                p_int = int(round(p))
                pulses[name] = p_int
                p = p_int

            # Your system seems to use ~0–1000 for positions. If yours differs, change these.
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
    def __init__(self):
        self.Arm = Arm
        self.openGripperPulse = 0
        self.closeGripperPulse = 1000
        self.BASE_ZERO_OFFSET = 125.0
        self.DEG_PER_PULSE = 0.24

    def wrist_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))
    def turn_wrist(self, angle):
        base_angle = ik.last_base_angle
        final_angle = (angle - 90) + base_angle
        wrist_pulse = self.wrist_to_pulse(final_angle)
        self.Arm.moveJetArm(5, wrist_pulse)
    def close_gripper(self):
        self.Arm.moveJetArm(10, self.closeGripperPulse)
    def open_gripper(self):
        self.Arm.moveJetArm(10, self.openGripperPulse)

class ComputerVision:
    def __init__(self, ik, gripper):
        self.Arm = Arm
        self.ik = ik
        self.gripper = gripper
    def scan_position(self): 
        self.ik.move_to(0, 15, 23)
        self.gripper.turn_wrist(90)
        self.gripper.open_gripper()

ik = JetArmIK()
gripper = JetArmGripper()
camera = ComputerVision(ik, gripper)

if __name__ == "__main__":
    while True:
        try:
            code = input(">>> ")
            if code.lower() in ['exit', 'quit']:
                break
            exec(code)
        except Exception as e:
            print(f"❌ Error: {e}")


