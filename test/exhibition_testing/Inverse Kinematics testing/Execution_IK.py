from classCreation import CKMJetArm
import math
Arm1 = CKMJetArm()

L1 = 10.0
L2 = 15.0
DEG_PER_PULSE = 0.25 
BASE_ZERO_OFFSET = 125.0
ANGLE_ZERO_OFFSET = 125.0
ELBOW_UP = True 

def angle_cosine(a, b, c, *, degrees=True):
    """
    Angle opposite side c in a triangle with sides a, b, c.
    Returns degrees by default; set degrees=False for radians.
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Sides must be positive.")
    # prevent tiny floating-point overshoots
    x = (a*a + b*b - c*c) / (2*a*b)
    x = max(-1.0, min(1.0, x))      # clamp to [-1, 1]
    ang = math.acos(x)
    return math.degrees(ang) if degrees else ang

def base_to_pulse(angle_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=BASE_ZERO_OFFSET):
    return int(round(((angle_deg) / deg_per_pulse) + zero_offset))

def arm_to_pulse(arm_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=ANGLE_ZERO_OFFSET):
    return int(round(((arm_deg) / deg_per_pulse) + zero_offset))

x, y, z = map(float, input("Enter x y z: ").split())

base_angle = math.degrees(math.atan2(y, x))
l = math.hypot(x, y) # base distance
d = math.hypot(l, z) # hypothesis including with height with distance

# Elbow internal angle (at the joint between L1 and L2)
# B in radians; 0 < B < π
c_elbow = (L1*L1 + L2*L2 - d*d) / (2*L1*L2)
c_elbow = max(-1.0, min(1.0, c_elbow))
B_rad = math.acos(c_elbow)

# Helper terms for shoulder angle
k1 = L1 + L2 * math.cos(B_rad)
k2 = L2 * math.sin(B_rad)

# Choose configuration: + for elbow-up, - for elbow-down
sign = +1 if ELBOW_UP else -1

# Shoulder angle from the horizontal forward direction (degrees)
theta1 = math.degrees(math.atan2(z, l) + sign * math.atan2(k2, k1))
L1_angle = theta1  # this now points forward for reachable front targets


# B: internal elbow angle between L1 and L2
B = angle_cosine(L1, L2, d, degrees=True)
e = 180 - B                       # elbow bend: 0=straight, 90=front-right-angle
e_rel = e - 90 
e_pulse = arm_to_pulse(e)
L2_angle = e_rel

 
print(f"Elbow (L2) angle: {L2_angle:.3f}°")
print(f"Shoulder (L1) angle: {L1_angle:.3f}°")
print(f"Base angle: {base_angle:.3f}°")

base_pulse = base_to_pulse(base_angle)
L2_pulse = arm_to_pulse(L2_angle)
L1_pulse = arm_to_pulse(L1_angle)
print(base_pulse, L1_pulse, L2_pulse)

Arm1.moveJetArm(1, base_pulse)
Arm1.moveJetArm(2, L1_pulse)
Arm1.moveJetArm(3, L2_pulse)
Arm1.moveJetArm(4, 500)
Arm1.moveJetArm(4, 500)