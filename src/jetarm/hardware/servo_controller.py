import time

try:
    import rclpy
    from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition
except ImportError as exc:
    rclpy = None
    ServoPosition = None
    ServosPosition = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None

class CKMJetArm: 
    def __init__(self):
        if ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "JetArm ROS dependencies are unavailable. Source the JetArm ROS "
                "environment before commanding hardware."
            ) from ROS_IMPORT_ERROR

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('jetarm_test')
        self.pub = self.node.create_publisher(ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)
        self._last_positions = {}
        time.sleep(0.5)

    def moveJetArm(self, servo_id, target_position, duration=1.0): 
        self.moveJetArmGroup({servo_id: target_position}, duration=duration)
        print(f"✅ Command sent to servo {servo_id} → position {target_position}")
        time.sleep(0.1)

    def moveJetArmGroup(self, positions, duration=1.0):
        servo_positions = [
            ServoPosition(id=servo_id, position=int(round(position)))
            for servo_id, position in positions.items()
        ]
        self.pub.publish(ServosPosition(duration=float(duration), position=servo_positions))
        self._last_positions.update(
            {servo_id: int(round(position)) for servo_id, position in positions.items()}
        )

    def smoothMoveJetArmGroup(self, positions, duration=1.2, steps=24):
        targets = {servo_id: int(round(position)) for servo_id, position in positions.items()}
        if not all(servo_id in self._last_positions for servo_id in targets):
            self.moveJetArmGroup(targets, duration=duration)
            time.sleep(duration)
            return

        starts = {servo_id: self._last_positions[servo_id] for servo_id in targets}
        step_duration = max(duration / max(steps, 1), 0.02)

        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3.0 - 2.0 * t)
            frame = {
                servo_id: starts[servo_id] + (targets[servo_id] - starts[servo_id]) * eased
                for servo_id in targets
            }
            self.moveJetArmGroup(frame, duration=step_duration)
            time.sleep(step_duration)

        self._last_positions.update(targets)
            
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
