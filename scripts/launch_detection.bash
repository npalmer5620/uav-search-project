#!/usr/bin/env bash
# Launch the YOLO detection node in a separate container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/config.env"

source_setup() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    set -u
}

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found. Copy config.env.example to config.env and edit it."
    exit 1
fi

source "$CONFIG_FILE"

source_setup /opt/ros/jazzy/setup.bash

if [[ -f "$REPO_DIR/install/setup.bash" ]]; then
    source_setup "$REPO_DIR/install/setup.bash"
fi

export PYTHONPATH="$REPO_DIR/src/uav_detection:${PYTHONPATH:-}"

MODEL_PATH="${DETECTION_MODEL_PATH:-${MODEL_PATH:-/root/.cache/yolo/yolo11n.onnx}}"
DETECTION_CONFIDENCE="${DETECTION_CONFIDENCE:-0.4}"
DETECTION_DEVICE="${DETECTION_DEVICE:-cpu}"
DETECTION_PUBLISH_VIZ="${DETECTION_PUBLISH_VIZ:-true}"
DETECTION_IMGSZ="${DETECTION_IMGSZ:-640}"
DETECTION_FRAME_SKIP="${DETECTION_FRAME_SKIP:-0}"
DETECTION_DEPTH_LOCALIZATION_ENABLED="${DETECTION_DEPTH_LOCALIZATION_ENABLED:-false}"
DETECTION_GROUND_PROJECTION_FALLBACK="${DETECTION_GROUND_PROJECTION_FALLBACK:-true}"
DETECTION_PREFER_GROUND_PROJECTION="${DETECTION_PREFER_GROUND_PROJECTION:-true}"
DETECTION_GROUND_PLANE_Z="${DETECTION_GROUND_PLANE_Z:-0.0}"
DETECTION_GROUND_PROJECTION_MAX_RANGE_M="${DETECTION_GROUND_PROJECTION_MAX_RANGE_M:-35.0}"
DETECTION_BBOX_GROUND_Y_FRACTION="${DETECTION_BBOX_GROUND_Y_FRACTION:-1.0}"

echo "=== UAV Detection ==="
echo "MODEL_PATH:           $MODEL_PATH"
echo "DETECTION_CONFIDENCE: $DETECTION_CONFIDENCE"
echo "DETECTION_DEVICE:     $DETECTION_DEVICE"
echo "DETECTION_PUBLISH_VIZ:$DETECTION_PUBLISH_VIZ"
echo "DETECTION_IMGSZ:      $DETECTION_IMGSZ"
echo "DETECTION_FRAME_SKIP: $DETECTION_FRAME_SKIP"
echo "DEPTH_LOCALIZATION:   $DETECTION_DEPTH_LOCALIZATION_ENABLED"
echo "GROUND_FALLBACK:      $DETECTION_GROUND_PROJECTION_FALLBACK"
echo "PREFER_GROUND:        $DETECTION_PREFER_GROUND_PROJECTION"
echo "GROUND_PLANE_Z:       $DETECTION_GROUND_PLANE_Z"
echo "GROUND_MAX_RANGE_M:   $DETECTION_GROUND_PROJECTION_MAX_RANGE_M"
echo "BBOX_GROUND_Y_FRAC:   $DETECTION_BBOX_GROUND_Y_FRACTION"
echo "====================="

if [[ ! -f "$MODEL_PATH" && "$MODEL_PATH" == *.onnx ]]; then
    echo "ONNX model missing at $MODEL_PATH. Exporting it now..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    (cd "$(dirname "$MODEL_PATH")" && python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=${DETECTION_IMGSZ}, simplify=False)")
fi

echo "Waiting for /camera/image_raw..."
until ros2 topic list 2>/dev/null | grep -qx '/camera/image_raw'; do
    sleep 2
done

echo "Starting detection node..."
exec python3 -m uav_detection.detection_node \
    --ros-args \
    -p model_path:="$MODEL_PATH" \
    -p confidence:="$DETECTION_CONFIDENCE" \
    -p device:="$DETECTION_DEVICE" \
    -p image_topic:=/camera/image_raw \
    -p publish_viz:="$DETECTION_PUBLISH_VIZ" \
    -p imgsz:="$DETECTION_IMGSZ" \
    -p frame_skip:="$DETECTION_FRAME_SKIP" \
    -p depth_localization_enabled:="$DETECTION_DEPTH_LOCALIZATION_ENABLED" \
    -p ground_projection_fallback:="$DETECTION_GROUND_PROJECTION_FALLBACK" \
    -p prefer_ground_projection:="$DETECTION_PREFER_GROUND_PROJECTION" \
    -p ground_plane_z:="$DETECTION_GROUND_PLANE_Z" \
    -p ground_projection_max_range_m:="$DETECTION_GROUND_PROJECTION_MAX_RANGE_M" \
    -p bbox_ground_y_fraction:="$DETECTION_BBOX_GROUND_Y_FRACTION"
