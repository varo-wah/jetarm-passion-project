import rclpy
import time
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition

rclpy.init()
node = rclpy.create_node('jetarm_test')
pub = node.create_publisher(ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)

time.sleep(0.5)

def moveJetArm(servo_id, target_position): 
    pub.publish(ServosPosition(duration=1.0, position=[ServoPosition(id=servo_id, position=target_position)]))
    print(f"✅ Command sent to servo {servo_id} → position {target_position}")
    time.sleep(0.1)

def reset():
    for i in range(2, 5):
        moveJetArm(i, 500)
    moveJetArm(10, 500)
    moveJetArm(1, 500)
    time.sleep(3.0)
def front():
    moveJetArm(10, 10)
    moveJetArm(2, 330)
    moveJetArm(3, 250)
    moveJetArm(4, 400)
    time.sleep(1.0)
    moveJetArm(10, 700)
def frontopen():
    moveJetArm(2, 330)
    moveJetArm(3, 250)
    moveJetArm(4, 400)
    time.sleep(1.0)
    moveJetArm(10, 10)
def up():
    for i in range(2, 5):
        moveJetArm(i, 500)
    moveJetArm(1, 500)

while True: 
    try:
        code = input(">>> ")
        if code.lower() in ['exit', 'quit']:
            break
        exec(code)  # 🧨 Run the typed function call
    except Exception as e:
        print(f"❌ Error: {e}")
    

rclpy.shutdown()