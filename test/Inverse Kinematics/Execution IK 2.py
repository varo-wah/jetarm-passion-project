from classCreation import CKMJetArm
Arm1 = CKMJetArm()
import math

L1 = 15.0
L2 = 15.0
DEG_PER_PULSE = 0.24
BASE_ZERO_OFFSET = 125.0
ANGLE_ZERO_OFFSET = 125.0
GROUND_OFFSET = 10.0
HAND_OFFSET = 15.0
ELBOW_UP = True

x, y, z = map(float, input("Enter x y z: ").split())

# z = z + GROUND_OFFSET - HAND_OFFSET

base_angle = math.degrees(math.atan2(y, x))
l = math.hypot(x, y) # base distance
d = math.hypot(l, z) # hypotonous including with height with distance

def base_to_pulse(angle_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=BASE_ZERO_OFFSET):
    return int(round(((angle_deg) / deg_per_pulse) + zero_offset))

def arm_to_pulse(arm_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=ANGLE_ZERO_OFFSET):
    return int(round(((arm_deg) / deg_per_pulse) + zero_offset))

# L1 angle Section
h = d / 2

theta = math.degrees(math.acos(h / L1))
phi = math.degrees(math.atan2(z, l))

if ELBOW_UP:
    L1_angle = phi + theta
else:
    L1_angle = phi - theta
# L1 angle, refer to Easy Inverse Kinematics for Robot Arms by RoTechnic

# L2 angle section
Intersection = 180 - (2 * theta)

if ELBOW_UP:
    L2_angle = Intersection - 90
else:
    L2_angle = 360 - (Intersection + 90)

#L3 angle
L3_angle = 90 - (L2_angle + L1_angle)


print(f"Elbow (L2) angle: {L2_angle:.3f}°")
print(f"Shoulder (L1) angle: {L1_angle:.3f}°")
print(f"Shoulder (L3) angle: {L3_angle:.3f}°")
print(f"Base angle: {base_angle:.3f}°")

base_pulse = base_to_pulse(base_angle)
L2_pulse = arm_to_pulse(L2_angle)
L1_pulse = arm_to_pulse(L1_angle)
L3_pulse = arm_to_pulse(L3_angle) + 60
print(base_pulse, L1_pulse, L2_pulse, L3_pulse)

Arm1.moveJetArm(1, base_pulse)
Arm1.moveJetArm(2, L1_pulse)
Arm1.moveJetArm(3, L2_pulse)
Arm1.moveJetArm(4, 500)
Arm1.moveJetArm(4, 500)