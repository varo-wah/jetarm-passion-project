from classCreation import CKMJetArm
import math
Arm1 = CKMJetArm()

L1 = 10.0
L2 = 15.0
DEG_PER_PULSE = 0.25 
ZERO_OFFSET = -35.0 
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

def angle_to_pulse(angle_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=ZERO_OFFSET):
    return int(round((angle_deg - zero_offset) / deg_per_pulse))

def arm_to_pulse(arm_deg, deg_per_pulse=DEG_PER_PULSE, zero_offset=ZERO_OFFSET):
    return int(round((arm_deg - zero_offset) / deg_per_pulse))

x, y, z = map(float, input("Enter x y z: ").split())

base_angle = math.degrees(math.atan2(y, x))
l = math.hypot(x, y) # base distance
d = math.hypot(l, z) # hypothesis including with height with distance

if d > L1 + L2 or d < abs(L1 - L2):
    raise ValueError(f"Out of reach: d={d:.3f} (needs {abs(L1-L2):.3f} ≤ d ≤ {L1+L2:.3f})")

slope_degree = math.degrees(math.atan2(z, l))

# S: shoulder offset from the slope line (angle at vertex where sides L1 and d meet)
S = angle_cosine(L1, d, L2, degrees=True)
L1_angle = slope_degree + (S if ELBOW_UP else -S)

# B: internal elbow angle between L1 and L2
B = angle_cosine(L1, L2, d, degrees=True)
L2_angle = 180 - B  # servo = 0° when straight; increase as elbow bends
 
print(f"Elbow (L2) angle: {L2_angle:.3f}°")
print(f"Shoulder (L1) angle: {L1_angle:.3f}°")
print(f"Base angle: {base_angle:.3f}°")

base_pulse = angle_to_pulse(base_angle)
L1_pulse   = angle_to_pulse(L1_angle)
L2_pulse   = angle_to_pulse(L2_angle)
print(base_pulse, L1_pulse, L2_pulse)

Arm1.moveJetArm(1, base_pulse)
Arm1.moveJetArm(2, L1_pulse)
Arm1.moveJetArm(3, L2_pulse)
Arm1.moveJetArm(4, 500)