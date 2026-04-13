#!/usr/bin/env bash
# Launch the mission controller (spiral search + detection response).
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

export PYTHONPATH="$REPO_DIR/src/uav_planning:${PYTHONPATH:-}"

# Configurable via config.env (or environment)
SPIRAL_MAX_RADIUS="${SPIRAL_MAX_RADIUS:-20.0}"
SPIRAL_SPACING="${SPIRAL_SPACING:-5.0}"
SPIRAL_ANGULAR_SPEED="${SPIRAL_ANGULAR_SPEED:-0.3}"
SPIRAL_ALTITUDE="${SPIRAL_ALTITUDE:--10.0}"
DETECTION_CONFIDENCE="${MISSION_DETECTION_CONFIDENCE:-0.6}"
INVESTIGATE_DURATION="${INVESTIGATE_DURATION:-10.0}"
INVESTIGATE_RADIUS="${INVESTIGATE_RADIUS:-3.0}"
TRACKING_MATCH_DISTANCE="${TRACKING_MATCH_DISTANCE:-2.5}"
TRACKING_DUPLICATE_DISTANCE="${TRACKING_DUPLICATE_DISTANCE:-1.5}"
TRACKING_TIMEOUT="${TRACKING_TIMEOUT:-1.0}"
TRACKING_HISTORY_SIZE="${TRACKING_HISTORY_SIZE:-5}"
TRACKING_MIN_HITS="${TRACKING_MIN_HITS:-3}"
TRACKING_MIN_AGE="${TRACKING_MIN_AGE:-0.5}"
TRACKING_MIN_CONFIDENCE="${TRACKING_MIN_CONFIDENCE:-0.65}"

echo "=== UAV Mission Controller ==="
echo "SPIRAL_MAX_RADIUS:    $SPIRAL_MAX_RADIUS"
echo "SPIRAL_ALTITUDE:      $SPIRAL_ALTITUDE"
echo "DETECTION_CONFIDENCE: $DETECTION_CONFIDENCE"
echo "INVESTIGATE_DURATION: $INVESTIGATE_DURATION"
echo "INVESTIGATE_RADIUS:   $INVESTIGATE_RADIUS"
echo "TRACKING_MATCH_DIST:  $TRACKING_MATCH_DISTANCE"
echo "TRACKING_DUP_DIST:    $TRACKING_DUPLICATE_DISTANCE"
echo "TRACKING_TIMEOUT:     $TRACKING_TIMEOUT"
echo "TRACKING_MIN_HITS:    $TRACKING_MIN_HITS"
echo "TRACKING_MIN_AGE:     $TRACKING_MIN_AGE"
echo "TRACKING_MIN_CONF:    $TRACKING_MIN_CONFIDENCE"
echo "=============================="

echo "Waiting for /fmu/out/vehicle_local_position_v1..."
until ros2 topic list 2>/dev/null | grep -qx '/fmu/out/vehicle_local_position_v1'; do
    sleep 2
done

echo "Starting mission controller..."
exec python3 -m uav_planning.mission_controller \
    --ros-args \
    -p spiral.max_radius:="$SPIRAL_MAX_RADIUS" \
    -p spiral.spacing:="$SPIRAL_SPACING" \
    -p spiral.angular_speed:="$SPIRAL_ANGULAR_SPEED" \
    -p spiral.altitude:="$SPIRAL_ALTITUDE" \
    -p detection_confidence_threshold:="$DETECTION_CONFIDENCE" \
    -p investigate_duration:="$INVESTIGATE_DURATION" \
    -p investigate_radius:="$INVESTIGATE_RADIUS" \
    -p tracking.match_distance:="$TRACKING_MATCH_DISTANCE" \
    -p tracking.duplicate_distance:="$TRACKING_DUPLICATE_DISTANCE" \
    -p tracking.track_timeout:="$TRACKING_TIMEOUT" \
    -p tracking.history_size:="$TRACKING_HISTORY_SIZE" \
    -p tracking.min_hits:="$TRACKING_MIN_HITS" \
    -p tracking.min_age:="$TRACKING_MIN_AGE" \
    -p tracking.min_confidence:="$TRACKING_MIN_CONFIDENCE"
