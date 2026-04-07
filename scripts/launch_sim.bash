#!/usr/bin/env bash
# Launch PX4 SITL + Micro XRCE-DDS Agent + ROS 2 bridge
# Reads config from config.env (copy config.env.example if you haven't)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Load user config
CONFIG_FILE="$REPO_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found. Copy config.env.example to config.env and edit it."
    echo "  cp $REPO_DIR/config.env.example $REPO_DIR/config.env"
    exit 1
fi
source "$CONFIG_FILE"

# Expand tilde in PX4_DIR
PX4_DIR="${PX4_DIR/#\~/$HOME}"

# Validate PX4 directory
if [[ ! -d "$PX4_DIR" ]]; then
    echo "ERROR: PX4_DIR=$PX4_DIR does not exist. Check your config.env."
    exit 1
fi

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Source colcon workspace if built
WORKSPACE_SETUP="$REPO_DIR/install/setup.bash"
if [[ -f "$WORKSPACE_SETUP" ]]; then
    source "$WORKSPACE_SETUP"
fi

echo "=== UAV Search Project ==="
echo "PX4_DIR:                   $PX4_DIR"
echo "PX4_SIM_MODEL:             ${PX4_SIM_MODEL:-gz_x500_mono_cam}"
echo "PX4_GZ_SIM_RENDER_ENGINE:  ${PX4_GZ_SIM_RENDER_ENGINE:-ogre2}"
echo "PX4_GZ_WORLD:              ${PX4_GZ_WORLD:-default}"
echo "=========================="

# Export env vars for PX4/Gazebo
export PX4_GZ_SIM_RENDER_ENGINE="${PX4_GZ_SIM_RENDER_ENGINE:-ogre2}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"

# Add our worlds/ directory so PX4/Gazebo can find custom world files
export GZ_SIM_RESOURCE_PATH="${REPO_DIR}/worlds:${GZ_SIM_RESOURCE_PATH:-}"

# World name is used in Gazebo topic paths
GZ_WORLD="${PX4_GZ_WORLD:-default}"

# Start Micro XRCE-DDS Agent in background
echo "Starting Micro XRCE-DDS Agent..."
MicroXRCEAgent udp4 -p 8888 &
XRCE_PID=$!

# Give agent a moment to bind
sleep 1

# Create a named pipe so we can send commands to the PX4 shell later
PX4_PIPE="/tmp/px4_pipe_$$"
mkfifo "$PX4_PIPE"
exec 3<>"$PX4_PIPE"   # hold the pipe open (read+write) so writes don't EOF PX4

# Start PX4 SITL (this also launches Gazebo)
echo "Starting PX4 SITL..."
cd "$PX4_DIR"
make px4_sitl "${PX4_SIM_MODEL:-gz_x500_mono_cam}" <&3 &
PX4_PID=$!

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $PX4_PID 2>/dev/null || true
    kill $XRCE_PID 2>/dev/null || true
    exec 3>&-                          # close the pipe fd
    rm -f "$PX4_PIPE"                  # remove the named pipe
    wait
}
trap cleanup EXIT INT TERM

# Wait for Gazebo to be up before starting bridge
echo "Waiting for Gazebo camera topic..."
TIMEOUT=120
ELAPSED=0
while ! gz topic -l 2>/dev/null | grep -q "camera"; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "WARNING: Timed out waiting for Gazebo camera topic. Starting bridge anyway."
        break
    fi
done

# Set camera render rate (defaults to 0 Hz)
echo "Setting camera render rate to ${CAMERA_RATE:-2} Hz..."
GZ_CAM_TOPIC="/world/${GZ_WORLD}/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

gz service -s "${GZ_CAM_TOPIC}/set_rate" \
    --reqtype gz.msgs.Double --reptype gz.msgs.Empty --req "data: ${CAMERA_RATE:-2}.0" --timeout 5000 || true

# Start ros_gz_bridge for camera (GZ -> ROS only)
echo "Starting ros_gz_bridge for camera..."
ros2 run ros_gz_bridge parameter_bridge \
    "${GZ_CAM_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image" \
    --ros-args -r "${GZ_CAM_TOPIC}:=/camera/image_raw" &
BRIDGE_PID=$!

# Wait for PX4 to finish booting before sending commands
TAKEOFF_DELAY="${TAKEOFF_DELAY:-30}"
echo ""
echo "Waiting ${TAKEOFF_DELAY}s for PX4 to initialize..."
sleep "$TAKEOFF_DELAY"

# Disable GCS/RC loss checks for SITL (no physical transmitter or GCS needed)
echo "Configuring PX4 for SITL (no GCS/RC required)..."
echo "param set COM_RCL_EXCEPT 4" > "$PX4_PIPE"
sleep 1
echo "param set NAV_RCL_ACT 0" > "$PX4_PIPE"
sleep 1
echo "param set NAV_DLL_ACT 0" > "$PX4_PIPE"
sleep 2

# Command the drone to take off and hover
echo "Sending takeoff command..."
echo "commander takeoff" > "$PX4_PIPE"

echo ""
echo "All processes running. Drone should be taking off."
echo "Verify with:"
echo "  ros2 topic list"
echo "  ros2 topic hz /camera/image_raw"
echo ""
echo "To send more PX4 commands:  echo 'commander land' > $PX4_PIPE"
echo "Press Ctrl+C to stop everything."

wait
