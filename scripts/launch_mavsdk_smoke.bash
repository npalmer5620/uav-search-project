#!/usr/bin/env bash
# Run the focused MAVSDK SITL smoke test from the planning container.
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

MAVSDK_SYSTEM_ADDRESS="${MAVSDK_SYSTEM_ADDRESS:-udpin://0.0.0.0:14540}"
SMOKE_ALTITUDE="${SMOKE_ALTITUDE:--10.0}"
SMOKE_GRID_WIDTH="${SMOKE_GRID_WIDTH:-20.0}"
SMOKE_GRID_HEIGHT="${SMOKE_GRID_HEIGHT:-20.0}"
SMOKE_GRID_SPACING="${SMOKE_GRID_SPACING:-10.0}"
SMOKE_GRID_SPEED="${SMOKE_GRID_SPEED:-2.0}"
SMOKE_SEARCH_WAYPOINTS="${SMOKE_SEARCH_WAYPOINTS:-4}"
SMOKE_GOTO_TIMEOUT="${SMOKE_GOTO_TIMEOUT:-75.0}"
SMOKE_HEALTH_TIMEOUT="${SMOKE_HEALTH_TIMEOUT:-150.0}"
SMOKE_LAND_TIMEOUT="${SMOKE_LAND_TIMEOUT:-90.0}"
SMOKE_HOLD_SECONDS="${SMOKE_HOLD_SECONDS:-2.0}"
SMOKE_VERIFY_HORIZONTAL_TOLERANCE="${SMOKE_VERIFY_HORIZONTAL_TOLERANCE:-1.5}"
SMOKE_VERIFY_VERTICAL_TOLERANCE="${SMOKE_VERIFY_VERTICAL_TOLERANCE:-1.0}"
SMOKE_VERIFY_MIN_PATH_RATIO="${SMOKE_VERIFY_MIN_PATH_RATIO:-0.55}"

echo "=== MAVSDK SITL Smoke Test ==="
echo "MAVSDK_ADDRESS:       $MAVSDK_SYSTEM_ADDRESS"
echo "SMOKE_ALTITUDE:       $SMOKE_ALTITUDE"
echo "SMOKE_GRID_WIDTH:     $SMOKE_GRID_WIDTH"
echo "SMOKE_GRID_HEIGHT:    $SMOKE_GRID_HEIGHT"
echo "SMOKE_GRID_SPACING:   $SMOKE_GRID_SPACING"
echo "SMOKE_GRID_SPEED:     $SMOKE_GRID_SPEED"
echo "SMOKE_WAYPOINTS:      $SMOKE_SEARCH_WAYPOINTS"
echo "SMOKE_GOTO_TIMEOUT:   $SMOKE_GOTO_TIMEOUT"
echo "SMOKE_HEALTH_TIMEOUT: $SMOKE_HEALTH_TIMEOUT"
echo "SMOKE_LAND_TIMEOUT:   $SMOKE_LAND_TIMEOUT"
echo "VERIFY_H_TOLERANCE:   $SMOKE_VERIFY_HORIZONTAL_TOLERANCE"
echo "VERIFY_V_TOLERANCE:   $SMOKE_VERIFY_VERTICAL_TOLERANCE"
echo "VERIFY_MIN_RATIO:     $SMOKE_VERIFY_MIN_PATH_RATIO"
echo "=============================="

exec python3 -m uav_planning.mavsdk_smoke_test \
    --system-address "$MAVSDK_SYSTEM_ADDRESS" \
    --altitude "$SMOKE_ALTITUDE" \
    --grid-width "$SMOKE_GRID_WIDTH" \
    --grid-height "$SMOKE_GRID_HEIGHT" \
    --grid-spacing "$SMOKE_GRID_SPACING" \
    --grid-speed "$SMOKE_GRID_SPEED" \
    --search-waypoints "$SMOKE_SEARCH_WAYPOINTS" \
    --goto-timeout "$SMOKE_GOTO_TIMEOUT" \
    --health-timeout "$SMOKE_HEALTH_TIMEOUT" \
    --land-timeout "$SMOKE_LAND_TIMEOUT" \
    --hold-seconds "$SMOKE_HOLD_SECONDS" \
    --verify-horizontal-tolerance "$SMOKE_VERIFY_HORIZONTAL_TOLERANCE" \
    --verify-vertical-tolerance "$SMOKE_VERIFY_VERTICAL_TOLERANCE" \
    --verify-min-path-ratio "$SMOKE_VERIFY_MIN_PATH_RATIO"
