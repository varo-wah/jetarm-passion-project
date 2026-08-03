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
sdk_pid=""

topic_endpoint_count() {
    local endpoint_label="$1"
    local topic_info
    topic_info="$(ros2 topic info -v /ros_robot_controller/bus_servo/set_position 2>/dev/null || true)"
    awk -v label="$endpoint_label" '$1 " " $2 == label ":" {print $3; found=1} END {if (!found) print 0}' <<<"$topic_info"
}

print_servo_topic_owners() {
    ros2 topic info -v /ros_robot_controller/bus_servo/set_position 2>/dev/null || true
}

cleanup() {
    trap - EXIT TERM
    if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
        python3 -c "from jetarm.control.client import JetArmControlClient; c = JetArmControlClient(); c.safe_shutdown(); c.close()" >/dev/null 2>&1 || true
        kill -TERM "$controller_pid" 2>/dev/null || true
        wait "$controller_pid" 2>/dev/null || true
    fi
    if [[ -n "$sdk_pid" ]] && kill -0 "$sdk_pid" 2>/dev/null; then
        kill -TERM "$sdk_pid" 2>/dev/null || true
        wait "$sdk_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT TERM

if [[ "$actuation_enabled" == "1" ]]; then
    if ! command -v setsid >/dev/null 2>&1; then
        echo "ERROR: setsid is required to preserve safe ROS shutdown ordering." >&2
        exit 11
    fi

    if systemctl is-active --quiet start_app_node.service 2>/dev/null; then
        echo "Stopping Hiwonder auto-start application to establish exclusive servo ownership..."
        sudo systemctl stop start_app_node.service
    fi

    # DDS discovery may briefly retain endpoints after systemd has stopped the
    # vendor application. Refuse to continue until every old command publisher
    # has disappeared; otherwise there is no centralized motion authority.
    for _ in {1..20}; do
        [[ "$(topic_endpoint_count "Publisher count")" == "0" ]] && break
        sleep 0.25
    done
    if [[ "$(topic_endpoint_count "Publisher count")" != "0" ]]; then
        echo "ERROR: Another node still publishes the JetArm servo-command topic:" >&2
        print_servo_topic_owners >&2
        echo "Stop the listed process before retrying; actuation remains disabled." >&2
        exit 6
    fi

    if command -v fuser >/dev/null 2>&1; then
        camera_pids=""
        for _ in {1..20}; do
            camera_pids="$(fuser /dev/video0 2>/dev/null || true)"
            [[ -z "$camera_pids" ]] && break
            sleep 0.25
        done
        if [[ -n "$camera_pids" ]]; then
            echo "ERROR: /dev/video0 is already owned by process(es): $camera_pids" >&2
            ps -fp $camera_pids >&2 || true
            echo "Stop the camera owner before retrying; the dashboard needs exclusive access." >&2
            exit 7
        fi
    fi

    # Keep driver and authority outside the terminal's foreground process group.
    # Ctrl+C must reach Uvicorn first so its safe-shutdown callback can still
    # contact both ROS processes before launcher cleanup terminates them.
    setsid ros2 launch sdk jetarm_sdk.launch.py &
    sdk_pid=$!

    sdk_ready=0
    for _ in {1..60}; do
        if ! kill -0 "$sdk_pid" 2>/dev/null; then
            echo "ERROR: JetArm SDK driver exited during startup." >&2
            exit 8
        fi
        if [[ "$(topic_endpoint_count "Subscription count")" -ge "1" ]]; then
            sdk_ready=1
            break
        fi
        sleep 0.25
    done

    if [[ "$sdk_ready" != "1" ]]; then
        echo "ERROR: JetArm SDK did not subscribe to the servo-command topic." >&2
        exit 9
    fi

    setsid python3 -m jetarm.control.control_node &
    controller_pid=$!

    controller_ready=0
    for _ in {1..40}; do
        if ! kill -0 "$controller_pid" 2>/dev/null; then
            echo "ERROR: Central JetArm controller exited during startup." >&2
            exit 4
        fi
        if python3 -c "from jetarm.control.client import JetArmControlClient; c = JetArmControlClient(wait_seconds=0.25); c.close()" >/dev/null 2>&1; then
            controller_ready=1
            break
        fi
        sleep 0.25
    done

    if [[ "$controller_ready" != "1" ]]; then
        echo "ERROR: Central JetArm controller did not become ready." >&2
        exit 5
    fi

    if [[ "$(topic_endpoint_count "Publisher count")" != "1" ]]; then
        echo "ERROR: Exclusive servo publisher verification failed after startup:" >&2
        print_servo_topic_owners >&2
        exit 10
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
