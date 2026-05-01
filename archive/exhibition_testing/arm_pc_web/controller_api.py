# controller_api.py
import servo_controller as sc
import action_group_controller as agc

def move_joint(servo_id: int, pulse: int, time_ms: int = 500):
    """Move a single servo to the requested pulse."""
    sc.setServoPulse(servo_id, pulse, time_ms)

def get_joint(servo_id: int):
    """Return current pulse of a servo."""
    return sc.getServoPulse(servo_id)

def run_action_group(name: str):
    agc.runAction(name)

def stop_action_group():
    agc.stop_action_group()