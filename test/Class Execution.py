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

        return base_angle, L1_angle, L2_angle, L3_angle

    def move_to(self, x, y, z):
        base_angle, L1_angle, L2_angle, L3_angle = self.calculate_angles(x, y, z)

        base_pulse = self.base_to_pulse(base_angle)
        L1_pulse = self.arm_to_pulse(L1_angle)
        L2_pulse = self.arm_to_pulse(L2_angle)
        L3_pulse = self.arm_to_pulse(L3_angle) + 35

        print(f"Moving to: {base_pulse}, {L1_pulse}, {L2_pulse}, {L3_pulse}")
        self.Arm.moveJetArm(1, base_pulse)
        self.Arm.moveJetArm(2, L1_pulse)
        self.Arm.moveJetArm(3, L2_pulse)
        self.Arm.moveJetArm(4, L3_pulse)
class JetArmGripper:
    def __init__(self):
        self.Arm = Arm
        self.openGripperPulse = 0
        self.closeGripperPulse = 1000

    def wrist_to_pulse(self, angle_deg):
        return int(round(angle_deg / self.DEG_PER_PULSE + self.BASE_ZERO_OFFSET))
    def turn_wrist(self, angle):
        wrist_pulse = self.wrist_to_pulse(angle)
        self.Arm.moveJetArm(5, wrist_pulse)
    def close_gripper(self):
        self.Arm.moveJetArm(10, self.closeGripperPulse)
    def open_gripper(self):
        self.Arm.moveJetArm(10, self.openGripperPulse)

class ComputerVision:
    def __init__(self, ik):
        self.Arm = Arm
        self.ik = ik
    def scan_position(self): 
        self.ik.move_to(0, 15, 23)

ik = JetArmIK()
gripper = JetArmGripper()
camera = ComputerVision(ik)

while True:
    try:
        code = input(">>> ")
        if code.lower() in ['exit', 'quit']:
            break
        exec(code)
    except Exception as e:
        print(f"❌ Error: {e}")


