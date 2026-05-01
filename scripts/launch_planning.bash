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

SEARCH_POLICY="${SEARCH_POLICY:-grid}"
MAVSDK_SYSTEM_ADDRESS="${MAVSDK_SYSTEM_ADDRESS:-udpin://0.0.0.0:14540}"
RL_POLICY_VERSION="${RL_POLICY_VERSION:-v2}"
RL_POLICY_KEY="$(printf '%s' "$RL_POLICY_VERSION" | tr '[:upper:]' '[:lower:]')"
RL_ALGORITHM="${RL_ALGORITHM:-dqn}"
if [[ -z "${RL_ARTIFACT_DIR:-}" ]]; then
    if [[ "$RL_POLICY_KEY" == "v1" ]]; then
        RL_ARTIFACT_DIR="artifacts/rl/search_policy"
    elif [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        RL_ARTIFACT_DIR="pyflyt_rl/artifacts/latest"
    else
        RL_ARTIFACT_DIR="artifacts/rl/search_policy_v2"
    fi
fi

# Configurable via config.env (or environment)
GRID_WIDTH="${GRID_WIDTH:-40.0}"
GRID_HEIGHT="${GRID_HEIGHT:-40.0}"
GRID_SPACING="${GRID_SPACING:-5.0}"
GRID_SPEED="${GRID_SPEED:-2.0}"
if [[ -z "${GRID_ALTITUDE:-}" ]]; then
    GRID_ALTITUDE="-4.0"
fi
GRID_ORIGIN_X="${GRID_ORIGIN_X:-0.0}"
GRID_ORIGIN_Y="${GRID_ORIGIN_Y:-0.0}"
DETECTION_CONFIDENCE="${MISSION_DETECTION_CONFIDENCE:-0.6}"
INVESTIGATE_DURATION="${INVESTIGATE_DURATION:-10.0}"
INVESTIGATE_RADIUS="${INVESTIGATE_RADIUS:-3.0}"
INVESTIGATE_SPIRAL_SPACING="${INVESTIGATE_SPIRAL_SPACING:-1.0}"
INVESTIGATE_SPIRAL_SPEED="${INVESTIGATE_SPIRAL_SPEED:-0.5}"
TRACKING_MATCH_DISTANCE="${TRACKING_MATCH_DISTANCE:-2.5}"
TRACKING_DUPLICATE_DISTANCE="${TRACKING_DUPLICATE_DISTANCE:-1.5}"
TRACKING_TIMEOUT="${TRACKING_TIMEOUT:-1.0}"
TRACKING_HISTORY_SIZE="${TRACKING_HISTORY_SIZE:-5}"
TRACKING_MIN_HITS="${TRACKING_MIN_HITS:-3}"
TRACKING_MIN_AGE="${TRACKING_MIN_AGE:-0.5}"
TRACKING_MIN_CONFIDENCE="${TRACKING_MIN_CONFIDENCE:-0.65}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"
TAKEOFF_TIMEOUT_SECONDS="${TAKEOFF_TIMEOUT_SECONDS:-60.0}"
HANDOFF_TIMEOUT_SECONDS="${HANDOFF_TIMEOUT_SECONDS:-60.0}"
MAVSDK_REQUIRE_FULL_HEALTH="${MAVSDK_REQUIRE_FULL_HEALTH:-false}"
RL_MODEL_PATH="${RL_MODEL_PATH:-$RL_ARTIFACT_DIR/model.zip}"
RL_VECNORMALIZE_PATH="${RL_VECNORMALIZE_PATH:-artifacts/rl/search_policy/vecnormalize.pkl}"
RL_PYFLYT_CONFIG_PATH="${RL_PYFLYT_CONFIG_PATH:-}"
if [[ -z "${RL_DECISION_PERIOD_S:-}" ]]; then
    if [[ "$RL_POLICY_KEY" == "v2" ]]; then
        RL_DECISION_PERIOD_S="2.0"
    elif [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        RL_DECISION_PERIOD_S="1.5"
    else
        RL_DECISION_PERIOD_S="0.5"
    fi
fi
RL_MAX_STEP_XY_M="${RL_MAX_STEP_XY_M:-4.0}"
if [[ -z "${RL_SCAN_YAW_STEP_DEG:-}" ]]; then
    if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        RL_SCAN_YAW_STEP_DEG="35.0"
    else
        RL_SCAN_YAW_STEP_DEG="25.0"
    fi
fi
RL_MAX_YAW_STEP_DEG="${RL_MAX_YAW_STEP_DEG:-25.0}"
RL_MAX_UNPRODUCTIVE_SCAN_STREAK="${RL_MAX_UNPRODUCTIVE_SCAN_STREAK:-2}"
RL_COVERAGE_GRID_SIDE="${RL_COVERAGE_GRID_SIDE:-16}"
if [[ -z "${RL_CELL_SIZE_M:-}" ]]; then
    if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        RL_CELL_SIZE_M="2.0"
    else
        RL_CELL_SIZE_M="4.0"
    fi
fi
RL_PATCH_SIDE="${RL_PATCH_SIDE:-11}"

case "$SEARCH_POLICY" in
    grid)
        export PYTHONPATH="$REPO_DIR/src/uav_planning:${PYTHONPATH:-}"
        PLANNING_MODULE="uav_planning.mission_controller"
        POLICY_LABEL="grid"
        EXTRA_ARGS=()
        ;;
    rl)
        export PYTHONPATH="$REPO_DIR/src/uav_planning:$REPO_DIR/src/uav_rl:${PYTHONPATH:-}"
        PLANNING_MODULE="uav_rl.rl_mission_controller"
        POLICY_LABEL="rl"
        EXTRA_ARGS=(
            -p rl.policy_version:="$RL_POLICY_VERSION"
            -p rl.algorithm:="$RL_ALGORITHM"
            -p rl.artifact_dir:="$RL_ARTIFACT_DIR"
            -p rl.model_path:="$RL_MODEL_PATH"
            -p rl.vecnormalize_path:="$RL_VECNORMALIZE_PATH"
            -p rl.decision_period_s:="$RL_DECISION_PERIOD_S"
            -p rl.max_step_xy_m:="$RL_MAX_STEP_XY_M"
            -p rl.scan_yaw_step_deg:="$RL_SCAN_YAW_STEP_DEG"
            -p rl.max_yaw_step_deg:="$RL_MAX_YAW_STEP_DEG"
            -p rl.max_unproductive_scan_streak:="$RL_MAX_UNPRODUCTIVE_SCAN_STREAK"
            -p rl.coverage_grid_side:="$RL_COVERAGE_GRID_SIDE"
            -p rl.cell_size_m:="$RL_CELL_SIZE_M"
            -p rl.patch_side:="$RL_PATCH_SIDE"
        )
        if [[ -n "$RL_PYFLYT_CONFIG_PATH" ]]; then
            EXTRA_ARGS+=(-p rl.pyflyt_config_path:="$RL_PYFLYT_CONFIG_PATH")
        fi
        ;;
    *)
        echo "ERROR: Unsupported SEARCH_POLICY='$SEARCH_POLICY' (expected 'grid' or 'rl')"
        exit 1
        ;;
esac

echo "=== UAV Mission Controller ==="
echo "SEARCH_POLICY:        $POLICY_LABEL"
echo "CONTROL:              mavsdk"
echo "MAVSDK_ADDRESS:       $MAVSDK_SYSTEM_ADDRESS"
echo "GRID_WIDTH:           $GRID_WIDTH"
echo "GRID_HEIGHT:          $GRID_HEIGHT"
echo "GRID_SPACING:         $GRID_SPACING"
echo "GRID_SPEED:           $GRID_SPEED"
echo "GRID_ALTITUDE:        $GRID_ALTITUDE"
echo "DETECTION_CONFIDENCE: $DETECTION_CONFIDENCE"
echo "INVESTIGATE_DURATION: $INVESTIGATE_DURATION"
echo "INVESTIGATE_RADIUS:   $INVESTIGATE_RADIUS"
echo "TRACKING_MATCH_DIST:  $TRACKING_MATCH_DISTANCE"
echo "TRACKING_DUP_DIST:    $TRACKING_DUPLICATE_DISTANCE"
echo "TRACKING_TIMEOUT:     $TRACKING_TIMEOUT"
echo "TRACKING_MIN_HITS:    $TRACKING_MIN_HITS"
echo "TRACKING_MIN_AGE:     $TRACKING_MIN_AGE"
echo "TRACKING_MIN_CONF:    $TRACKING_MIN_CONFIDENCE"
echo "USE_SIM_TIME:         $USE_SIM_TIME"
echo "TAKEOFF_TIMEOUT:      $TAKEOFF_TIMEOUT_SECONDS"
echo "HANDOFF_TIMEOUT:      $HANDOFF_TIMEOUT_SECONDS"
echo "MAVSDK_FULL_HEALTH:   $MAVSDK_REQUIRE_FULL_HEALTH"
if [[ "$POLICY_LABEL" == "rl" ]]; then
    echo "RL_POLICY_VERSION:    $RL_POLICY_VERSION"
    echo "RL_ALGORITHM:         $RL_ALGORITHM"
    echo "RL_ARTIFACT_DIR:      $RL_ARTIFACT_DIR"
    echo "RL_MODEL_PATH:        $RL_MODEL_PATH"
    echo "RL_VECNORMALIZE:      $RL_VECNORMALIZE_PATH"
    if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        echo "RL_PYFLYT_CONFIG:     ${RL_PYFLYT_CONFIG_PATH:-auto}"
    fi
    echo "RL_DECISION_PERIOD:   $RL_DECISION_PERIOD_S"
    echo "RL_MAX_STEP_XY:       $RL_MAX_STEP_XY_M"
    echo "RL_SCAN_YAW_STEP:     $RL_SCAN_YAW_STEP_DEG deg"
    echo "RL_MAX_YAW_STEP:      $RL_MAX_YAW_STEP_DEG deg"
    echo "RL_SCAN_STREAK_CAP:   $RL_MAX_UNPRODUCTIVE_SCAN_STREAK"
    echo "RL_COVERAGE_GRID:     $RL_COVERAGE_GRID_SIDE"
    echo "RL_CELL_SIZE_M:       $RL_CELL_SIZE_M"
    echo "RL_PATCH_SIDE:        $RL_PATCH_SIDE"
fi
echo "=============================="

echo "Starting mission controller..."
exec python3 -m "$PLANNING_MODULE" \
    --ros-args \
    -p grid.width:="$GRID_WIDTH" \
    -p grid.height:="$GRID_HEIGHT" \
    -p grid.spacing:="$GRID_SPACING" \
    -p grid.speed:="$GRID_SPEED" \
    -p grid.altitude:="$GRID_ALTITUDE" \
    -p grid.origin_x:="$GRID_ORIGIN_X" \
    -p grid.origin_y:="$GRID_ORIGIN_Y" \
    -p detection_confidence_threshold:="$DETECTION_CONFIDENCE" \
    -p investigate_duration:="$INVESTIGATE_DURATION" \
    -p investigate_radius:="$INVESTIGATE_RADIUS" \
    -p investigate_spiral_spacing:="$INVESTIGATE_SPIRAL_SPACING" \
    -p investigate_spiral_speed:="$INVESTIGATE_SPIRAL_SPEED" \
    -p tracking.match_distance:="$TRACKING_MATCH_DISTANCE" \
    -p tracking.duplicate_distance:="$TRACKING_DUPLICATE_DISTANCE" \
    -p tracking.track_timeout:="$TRACKING_TIMEOUT" \
    -p tracking.history_size:="$TRACKING_HISTORY_SIZE" \
    -p tracking.min_hits:="$TRACKING_MIN_HITS" \
    -p tracking.min_age:="$TRACKING_MIN_AGE" \
    -p tracking.min_confidence:="$TRACKING_MIN_CONFIDENCE" \
    -p mavsdk.system_address:="$MAVSDK_SYSTEM_ADDRESS" \
    -p mavsdk.require_full_health:="$MAVSDK_REQUIRE_FULL_HEALTH" \
    -p use_sim_time:="$USE_SIM_TIME" \
    -p takeoff_timeout_seconds:="$TAKEOFF_TIMEOUT_SECONDS" \
    -p handoff_timeout_seconds:="$HANDOFF_TIMEOUT_SECONDS" \
    "${EXTRA_ARGS[@]}"
