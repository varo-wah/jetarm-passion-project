import rclpy
import time
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition

class CKMJetArm: 
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('jetarm_test')
        self.pub = self.node.create_publisher(ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)
        time.sleep(0.5)

    def moveJetArm(self, servo_id, target_position): 
        self.pub.publish(ServosPosition(duration=1.0, position=[ServoPosition(id=servo_id, position=target_position)]))
        print(f"✅ Command sent to servo {servo_id} → position {target_position}")
        time.sleep(0.1)
            
    def reset(self):
        for i in range(2, 5):
            self.moveJetArm(i, 500)
        self.moveJetArm(10, 500)
        self.moveJetArm(1, 500)
        time.sleep(3.0)

    def front(self):
        self.moveJetArm(10, 10)
        self.moveJetArm(2, 330)
        self.moveJetArm(3, 250)
        self.moveJetArm(4, 400)
        time.sleep(1.0)
        self.moveJetArm(10, 700)

    def frontOpen(self):
        self.moveJetArm(2, 330)
        self.moveJetArm(3, 250)
        self.moveJetArm(4, 400)
        time.sleep(1.0)
        self.moveJetArm(10, 10)

    def up(self):
        for i in range(2, 5):
            self.moveJetArm(i, 500)
        self.moveJetArm(1, 500)

