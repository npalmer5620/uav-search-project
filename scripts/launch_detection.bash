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

echo "=== UAV Detection ==="
echo "MODEL_PATH:           $MODEL_PATH"
echo "DETECTION_CONFIDENCE: $DETECTION_CONFIDENCE"
echo "DETECTION_DEVICE:     $DETECTION_DEVICE"
echo "DETECTION_PUBLISH_VIZ:$DETECTION_PUBLISH_VIZ"
echo "DETECTION_IMGSZ:      $DETECTION_IMGSZ"
echo "DETECTION_FRAME_SKIP: $DETECTION_FRAME_SKIP"
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
    -p frame_skip:="$DETECTION_FRAME_SKIP"
