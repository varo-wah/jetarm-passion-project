import rclpy
import time
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition

rclpy.init()
node = rclpy.create_node('jetarm_test')
pub = node.create_publisher(ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)
time.sleep(0.5)
for i in range(1, 8):
    msg = ServosPosition()
    msg.duration = 0.5
    msg.position = [ServoPosition(id=i, position=600)]
    pub.publish(msg)
    time.sleep(1)
    print(f"✅ Command sent to servo {i}")

rclpy.shutdown()
