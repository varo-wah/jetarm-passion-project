#!/usr/bin/env bash

set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---preview}"

case "$mode" in
    --preview)
        actuation_enabled=0
        ;;
    --actuate)
        actuation_enabled=1
        ;;
    *)
        echo "Usage: $0 [--preview|--actuate]" >&2
        exit 2
        ;;
esac

cd "$project_root"

# Preserve any ROS paths already configured by the JetArm login environment.
# A direct `PYTHONPATH=src` assignment erases those paths and makes rclpy and
# ros_robot_controller_msgs unavailable.
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ "$actuation_enabled" == "1" ]]; then
    if ! python3 -c "import rclpy" >/dev/null 2>&1; then
        ros_distro="${ROS_DISTRO:-humble}"
        ros_setup="/opt/ros/$ros_distro/setup.bash"
        if [[ -f "$ros_setup" ]]; then
            # shellcheck disable=SC1090
            source "$ros_setup"
        fi
    fi

    if ! python3 -c "import rclpy; from ros_robot_controller_msgs.msg import ServosPosition; from control_msgs.action import FollowJointTrajectory" >/dev/null 2>&1; then
        for overlay_setup in \
            "$HOME/ros2_ws/install/setup.bash" \
            "$HOME/jetarm_ws/install/setup.bash" \
            "$HOME/jetarm_ros2_ws/install/setup.bash"
        do
            if [[ -f "$overlay_setup" ]]; then
                # shellcheck disable=SC1090
                source "$overlay_setup"
            fi
        done
    fi

    if ! python3 -c "import rclpy; from ros_robot_controller_msgs.msg import ServosPosition; from control_msgs.action import FollowJointTrajectory" >/dev/null 2>&1; then
        echo "ERROR: JetArm ROS hardware imports are unavailable." >&2
        echo "Open a JetArm ROS 2 shell, then rerun: ./scripts/run_jetarm.sh --actuate" >&2
        echo "Do not install rclpy with pip." >&2
        exit 3
    fi
fi

export JETARM_ENABLE_ACTUATION="$actuation_enabled"

controller_pid=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
        python3 -c "from jetarm.control.client import JetArmControlClient; JetArmControlClient().safe_shutdown()" >/dev/null 2>&1 || true
        kill -TERM "$controller_pid" 2>/dev/null || true
        wait "$controller_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if [[ "$actuation_enabled" == "1" ]]; then
    python3 -m jetarm.control.control_node &
    controller_pid=$!

    controller_ready=0
    for _ in {1..40}; do
        if ! kill -0 "$controller_pid" 2>/dev/null; then
            echo "ERROR: Central JetArm controller exited during startup." >&2
            exit 4
        fi
        if python3 -c "from jetarm.control.client import JetArmControlClient; JetArmControlClient(wait_seconds=0.25)" >/dev/null 2>&1; then
            controller_ready=1
            break
        fi
        sleep 0.25
    done

    if [[ "$controller_ready" != "1" ]]; then
        echo "ERROR: Central JetArm controller did not become ready." >&2
        exit 5
    fi
fi

if [[ "$actuation_enabled" == "1" ]]; then
    echo "JetArm dashboard starting in ACTUATION mode"
else
    echo "JetArm dashboard starting in PREVIEW mode"
fi
echo "Open http://0.0.0.0:8000"

if [[ "$actuation_enabled" == "1" ]]; then
    echo "Motion authority starts PAUSED; use Resume before commanding the arm."
fi

python3 -m uvicorn jetarm.ui.server_app:app \
    --host 0.0.0.0 \
    --port 8000
