from classCreation import CKMJetArm
import math

Arm1 = CKMJetArm()
x, y, z = map(int, input("Enter x y z: ").split())

base_rad = math.atan2(y, x) 
base_angle = math.degrees(base_rad)
base_distance = math.sqrt(x**2+y**2)

print(base_distance)
angle = base_angle

print(angle)
pulse = int((angle + 30)*4)
print(pulse)


Arm1.moveJetArm(1, pulse)