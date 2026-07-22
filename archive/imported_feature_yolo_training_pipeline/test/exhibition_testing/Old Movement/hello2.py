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

moveJetArm(1, 280)

rclpy.shutdown()
