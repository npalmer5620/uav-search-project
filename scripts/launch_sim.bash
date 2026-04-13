#!/usr/bin/env bash
# Launch PX4 SITL + Micro XRCE-DDS Agent + ROS 2 bridge + Foxglove bridge
# Reads config from config.env (copy config.env.example if you haven't)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source_setup() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    set -u
}

prepare_prebuilt_px4_env() {
    local px4_gz_server_config="$PX4_DIR/share/gz/server.config"

    export PX4_GZ_MODELS="$PX4_DIR/share/gz/models"
    export PX4_GZ_WORLDS="$PX4_DIR/share/gz/worlds"
    export PX4_GZ_PLUGINS="$PX4_DIR/lib/gz/plugins"

    if [[ "${DISABLE_GST_CAMERA_SYSTEM:-0}" == "1" ]]; then
        local repo_server_config="$REPO_DIR/docker/gz-server-no-gst.config"
        if [[ -f "$repo_server_config" ]]; then
            px4_gz_server_config="$repo_server_config"
            echo "Using Gazebo server config without GstCameraSystem: $repo_server_config"
        else
            echo "WARNING: DISABLE_GST_CAMERA_SYSTEM=1 but $repo_server_config is missing"
        fi
    fi

    export PX4_GZ_SERVER_CONFIG="$px4_gz_server_config"
    export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}:${PX4_GZ_MODELS}:${PX4_GZ_WORLDS}"
    export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-}:${PX4_GZ_PLUGINS}"
    export GZ_SIM_SERVER_CONFIG_PATH="${PX4_GZ_SERVER_CONFIG}"

    # Match the upstream wrapper so Gazebo can find the DART engine without -dev packages.
    local gz_physics_engine_dir
    local unversioned
    local versioned
    gz_physics_engine_dir="$(find /usr/lib -maxdepth 3 -type d -name "engine-plugins" -path "*/gz-physics-7/*" 2>/dev/null | head -1)"
    if [[ -n "$gz_physics_engine_dir" && -d "$gz_physics_engine_dir" ]]; then
        unversioned="$gz_physics_engine_dir/libgz-physics-dartsim-plugin.so"
        if [[ ! -e "$unversioned" ]]; then
            versioned="$(find "$gz_physics_engine_dir" -maxdepth 1 -type f -name 'libgz-physics*-dartsim-plugin.so.*' | head -1)"
            if [[ -n "$versioned" ]]; then
                ln -sfn "$(basename "$versioned")" "$unversioned" 2>/dev/null || true
            fi
        fi
    fi
}

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

if [[ "${PX4_PREBUILT:-0}" == "1" && -n "${HEADLESS:-}" && "${PX4_GZ_SIM_RENDER_ENGINE:-ogre2}" == "ogre" ]]; then
    echo "HEADLESS prebuilt PX4 requires ogre2; overriding PX4_GZ_SIM_RENDER_ENGINE=ogre"
    PX4_GZ_SIM_RENDER_ENGINE=ogre2
fi

# Validate PX4 directory
if [[ ! -d "$PX4_DIR" ]]; then
    echo "ERROR: PX4_DIR=$PX4_DIR does not exist. Check your config.env."
    exit 1
fi

# Source ROS 2
source_setup /opt/ros/jazzy/setup.bash

# Source colcon workspace if built
WORKSPACE_SETUP="$REPO_DIR/install/setup.bash"
if [[ -f "$WORKSPACE_SETUP" ]]; then
    source_setup "$WORKSPACE_SETUP"
fi

echo "=== UAV Search Project ==="
echo "PX4_DIR:                   $PX4_DIR"
echo "PX4_SIM_MODEL:             ${PX4_SIM_MODEL:-gz_x500_mono_cam}"
echo "PX4_GZ_SIM_RENDER_ENGINE:  ${PX4_GZ_SIM_RENDER_ENGINE:-ogre2}"
echo "PX4_GZ_WORLD:              ${PX4_GZ_WORLD:-default}"
echo "PX4_PREBUILT:              ${PX4_PREBUILT:-0}"
echo "DISABLE_GST_CAMERA_SYSTEM: ${DISABLE_GST_CAMERA_SYSTEM:-0}"
echo "FOXGLOVE_ENABLED:          ${ENABLE_FOXGLOVE:-1}"
echo "=========================="

# Export env vars for PX4/Gazebo
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_mono_cam}"
export PX4_GZ_SIM_RENDER_ENGINE="${PX4_GZ_SIM_RENDER_ENGINE:-ogre2}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"
if [[ -n "${HEADLESS:-}" ]]; then
    export HEADLESS
fi

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

if [[ "${PX4_PREBUILT:-0}" == "1" ]]; then
    CUSTOM_WORLD_FILE="$REPO_DIR/worlds/${PX4_GZ_WORLD}.sdf"
    PX4_WORLD_FILE="$PX4_DIR/share/gz/worlds/${PX4_GZ_WORLD}.sdf"
    CUSTOM_CAMERA_MODEL_FILE=""
    PX4_CAMERA_MODEL_FILE=""

    if [[ -f "$CUSTOM_WORLD_FILE" ]]; then
        echo "Linking custom world into PX4 install: $CUSTOM_WORLD_FILE"
        ln -sfn "$CUSTOM_WORLD_FILE" "$PX4_WORLD_FILE"
    fi

    if [[ "${PX4_SIM_MODEL}" == "gz_x500_mono_cam" ]]; then
        CUSTOM_CAMERA_MODEL_FILE="$REPO_DIR/docker/models/mono_cam/model.sdf"
        PX4_CAMERA_MODEL_FILE="$PX4_DIR/share/gz/models/mono_cam/model.sdf"
    elif [[ "${PX4_SIM_MODEL}" == "gz_x500_depth" ]]; then
        CUSTOM_CAMERA_MODEL_FILE="$REPO_DIR/docker/models/OakD-Lite/model.sdf"
        PX4_CAMERA_MODEL_FILE="$PX4_DIR/share/gz/models/OakD-Lite/model.sdf"
    fi

    if [[ -n "$CUSTOM_CAMERA_MODEL_FILE" && -f "$CUSTOM_CAMERA_MODEL_FILE" ]]; then
        echo "Linking custom camera model into PX4 install: $CUSTOM_CAMERA_MODEL_FILE"
        ln -sfn "$CUSTOM_CAMERA_MODEL_FILE" "$PX4_CAMERA_MODEL_FILE"
    fi

    prepare_prebuilt_px4_env
    ./bin/px4 -s "$PX4_DIR/etc/init.d-posix/rcS" <&3 &
else
    make px4_sitl "${PX4_SIM_MODEL}" <&3 &
fi

PX4_PID=$!
BRIDGE_PID=""
FOXGLOVE_PID=""
POSE_BRIDGE_PID=""
CAMERA_LINK_TF_PID=""
CAMERA_OPTICAL_TF_PID=""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "${CAMERA_OPTICAL_TF_PID}" ]] && kill "${CAMERA_OPTICAL_TF_PID}" 2>/dev/null || true
    [[ -n "${CAMERA_LINK_TF_PID}" ]] && kill "${CAMERA_LINK_TF_PID}" 2>/dev/null || true
    [[ -n "${POSE_BRIDGE_PID}" ]] && kill "${POSE_BRIDGE_PID}" 2>/dev/null || true
    [[ -n "${FOXGLOVE_PID}" ]] && kill "${FOXGLOVE_PID}" 2>/dev/null || true
    [[ -n "${BRIDGE_PID}" ]] && kill "${BRIDGE_PID}" 2>/dev/null || true
    kill "${PX4_PID}" 2>/dev/null || true
    kill "${XRCE_PID}" 2>/dev/null || true
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
echo "Setting camera render rate to ${CAMERA_RATE:-30} Hz..."
MODEL_INSTANCE_NAME="${PX4_SIM_MODEL#gz_}_0"
GZ_SENSOR_PREFIX="/world/${GZ_WORLD}/model/${MODEL_INSTANCE_NAME}/link/camera_link/sensor"
GZ_CAM_TOPIC=""
GZ_CAM_INFO_TOPIC=""
GZ_DEPTH_TOPIC=""
GZ_DEPTH_INFO_TOPIC=""

if [[ "${PX4_SIM_MODEL}" == "gz_x500_depth" ]]; then
    GZ_CAM_TOPIC="${GZ_SENSOR_PREFIX}/IMX214/image"
    GZ_CAM_INFO_TOPIC="${GZ_SENSOR_PREFIX}/IMX214/camera_info"
    GZ_DEPTH_TOPIC="$(gz topic -l 2>/dev/null | grep "^${GZ_SENSOR_PREFIX}/StereoOV7251" | grep -v '/camera_info$' | grep -v '/points$' | head -1 || true)"
    GZ_DEPTH_INFO_TOPIC="$(gz topic -l 2>/dev/null | grep "^${GZ_SENSOR_PREFIX}/StereoOV7251" | grep '/camera_info$' | head -1 || true)"
else
    GZ_CAM_TOPIC="${GZ_SENSOR_PREFIX}/camera/image"
    GZ_CAM_INFO_TOPIC="${GZ_SENSOR_PREFIX}/camera/camera_info"
fi

gz service -s "${GZ_CAM_TOPIC}/set_rate" \
    --reqtype gz.msgs.Double --reptype gz.msgs.Empty --req "data: ${CAMERA_RATE:-30}.0" --timeout 5000 || true

if [[ -n "$GZ_DEPTH_TOPIC" ]]; then
    gz service -s "${GZ_DEPTH_TOPIC}/set_rate" \
        --reqtype gz.msgs.Double --reptype gz.msgs.Empty --req "data: ${CAMERA_RATE:-30}.0" --timeout 5000 || true
fi

# Start ros_gz_bridge for camera topics (GZ -> ROS only)
echo "Starting ros_gz_bridge for camera topics..."
BRIDGE_TOPICS=(
    "${GZ_CAM_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image"
    "${GZ_CAM_INFO_TOPIC}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
)
BRIDGE_REMAPS=(
    -r "${GZ_CAM_TOPIC}:=/camera/image_raw"
    -r "${GZ_CAM_INFO_TOPIC}:=/camera/camera_info"
)

if [[ -n "$GZ_DEPTH_TOPIC" ]]; then
    BRIDGE_TOPICS+=("${GZ_DEPTH_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image")
    BRIDGE_REMAPS+=(-r "${GZ_DEPTH_TOPIC}:=/camera/depth/image_raw")
fi

if [[ -n "$GZ_DEPTH_INFO_TOPIC" ]]; then
    BRIDGE_TOPICS+=("${GZ_DEPTH_INFO_TOPIC}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo")
    BRIDGE_REMAPS+=(-r "${GZ_DEPTH_INFO_TOPIC}:=/camera/depth/camera_info")
fi

ros2 run ros_gz_bridge parameter_bridge "${BRIDGE_TOPICS[@]}" --ros-args "${BRIDGE_REMAPS[@]}" &
BRIDGE_PID=$!

echo "Publishing static TF base_link -> camera_link..."
ros2 run tf2_ros static_transform_publisher \
    --x 0.12 --y 0.03 --z 0.242 \
    --roll 0.0 --pitch 0.0 --yaw 0.0 \
    --frame-id base_link --child-frame-id camera_link &
CAMERA_LINK_TF_PID=$!

echo "Publishing static TF camera_link -> camera_optical_frame..."
ros2 run tf2_ros static_transform_publisher \
    --x 0.0 --y 0.0 --z 0.0 \
    --roll -1.57079632679 --pitch 0.0 --yaw -1.57079632679 \
    --frame-id camera_link --child-frame-id camera_optical_frame &
CAMERA_OPTICAL_TF_PID=$!

if [[ "${ENABLE_FOXGLOVE:-1}" == "1" ]]; then
    FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
    echo "Starting Foxglove bridge on ws://localhost:${FOXGLOVE_PORT}..."
    ros2 run foxglove_bridge foxglove_bridge \
        --ros-args -p port:="${FOXGLOVE_PORT}" &
    FOXGLOVE_PID=$!
fi

if [[ "${ENABLE_POSE_BRIDGE:-1}" == "1" ]]; then
    if python3 -c "from px4_msgs.msg import VehicleOdometry" >/dev/null 2>&1; then
        echo "Starting PX4 pose bridge for Foxglove 3D..."
        python3 "$REPO_DIR/scripts/px4_pose_bridge.py" &
        POSE_BRIDGE_PID=$!
    else
        echo "WARNING: px4_msgs is unavailable; skipping Foxglove pose bridge"
    fi
fi

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

# Wait for health checks to pass after param changes. PX4 needs a few seconds
# to re-evaluate preflight checks once RC/GCS loss actions are disabled.
echo "Waiting for PX4 health checks to pass..."
sleep 10

# Command the drone to take off and hover
echo "Sending takeoff command..."
echo "commander takeoff" > "$PX4_PIPE"

echo ""
echo "All processes running. Drone should be taking off."
echo "Verify with:"
echo "  ros2 topic list"
echo "  ros2 topic hz /camera/image_raw"
if [[ "${ENABLE_FOXGLOVE:-1}" == "1" ]]; then
    echo "  open ws://localhost:${FOXGLOVE_PORT:-8765} in Foxglove Studio / web app"
    if [[ "${ENABLE_POSE_BRIDGE:-1}" == "1" ]]; then
        echo "  add /uav/drone_marker or /uav/pose in Foxglove 3D"
    fi
fi
echo ""
echo "To send more PX4 commands:  echo 'commander land' > $PX4_PIPE"
echo "Press Ctrl+C to stop everything."

wait
