#!/usr/bin/env bash
# Launch the SITL evaluator that writes quantitative Gazebo run metrics.
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

float_param() {
    if [[ "$1" =~ ^-?[0-9]+$ ]]; then
        printf "%s.0" "$1"
    else
        printf "%s" "$1"
    fi
}

if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

source_setup /opt/ros/jazzy/setup.bash

if [[ -f "$REPO_DIR/install/setup.bash" ]]; then
    source_setup "$REPO_DIR/install/setup.bash"
fi

export PYTHONPATH="$REPO_DIR/src/uav_planning:${PYTHONPATH:-}"

SITL_EVAL_WORLD_PATH="${SITL_EVAL_WORLD_PATH:-worlds/search_area.sdf}"
SITL_EVAL_OUTPUT_DIR="${SITL_EVAL_OUTPUT_DIR:-artifacts/sitl_eval}"
SITL_EVAL_RUN_ID="${SITL_EVAL_RUN_ID:-}"
SITL_EVAL_POLICY_LABEL="${SITL_EVAL_POLICY_LABEL:-${SEARCH_POLICY:-unknown}}"
SITL_EVAL_MODEL_PATH="${SITL_EVAL_MODEL_PATH:-${RL_MODEL_PATH:-}}"
SITL_EVAL_TARGET_FILTER="${SITL_EVAL_TARGET_FILTER:-people}"
SITL_EVAL_SEARCH_BOUNDS_ONLY="${SITL_EVAL_SEARCH_BOUNDS_ONLY:-true}"
SITL_EVAL_MATCH_RADIUS_M="${SITL_EVAL_MATCH_RADIUS_M:-4.0}"
SITL_EVAL_REPORT_RADIUS_M="${SITL_EVAL_REPORT_RADIUS_M:-4.0}"
SITL_EVAL_SAMPLE_PERIOD_S="${SITL_EVAL_SAMPLE_PERIOD_S:-2.0}"
SITL_EVAL_SUMMARY_PERIOD_S="${SITL_EVAL_SUMMARY_PERIOD_S:-10.0}"
SITL_EVAL_DURATION_S="${SITL_EVAL_DURATION_S:-0.0}"
SITL_EVAL_STOP_ON_TERMINAL_PHASE="${SITL_EVAL_STOP_ON_TERMINAL_PHASE:-true}"
SITL_EVAL_COVERAGE_CELL_SIZE_M="${SITL_EVAL_COVERAGE_CELL_SIZE_M:-${RL_CELL_SIZE_M:-4.0}}"

GRID_WIDTH="${GRID_WIDTH:-40.0}"
GRID_HEIGHT="${GRID_HEIGHT:-40.0}"
GRID_ORIGIN_X="${GRID_ORIGIN_X:-0.0}"
GRID_ORIGIN_Y="${GRID_ORIGIN_Y:-0.0}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"

GRID_WIDTH="$(float_param "$GRID_WIDTH")"
GRID_HEIGHT="$(float_param "$GRID_HEIGHT")"
GRID_ORIGIN_X="$(float_param "$GRID_ORIGIN_X")"
GRID_ORIGIN_Y="$(float_param "$GRID_ORIGIN_Y")"
SITL_EVAL_MATCH_RADIUS_M="$(float_param "$SITL_EVAL_MATCH_RADIUS_M")"
SITL_EVAL_REPORT_RADIUS_M="$(float_param "$SITL_EVAL_REPORT_RADIUS_M")"
SITL_EVAL_SAMPLE_PERIOD_S="$(float_param "$SITL_EVAL_SAMPLE_PERIOD_S")"
SITL_EVAL_SUMMARY_PERIOD_S="$(float_param "$SITL_EVAL_SUMMARY_PERIOD_S")"
SITL_EVAL_DURATION_S="$(float_param "$SITL_EVAL_DURATION_S")"
SITL_EVAL_COVERAGE_CELL_SIZE_M="$(float_param "$SITL_EVAL_COVERAGE_CELL_SIZE_M")"

echo "=== SITL Evaluator ==="
echo "WORLD_PATH:           $SITL_EVAL_WORLD_PATH"
echo "OUTPUT_DIR:           $SITL_EVAL_OUTPUT_DIR"
echo "RUN_ID:               ${SITL_EVAL_RUN_ID:-auto}"
echo "POLICY_LABEL:         $SITL_EVAL_POLICY_LABEL"
echo "MODEL_PATH:           ${SITL_EVAL_MODEL_PATH:-none}"
echo "TARGET_FILTER:        $SITL_EVAL_TARGET_FILTER"
echo "SEARCH_BOUNDS_ONLY:   $SITL_EVAL_SEARCH_BOUNDS_ONLY"
echo "GRID:                 ${GRID_WIDTH}x${GRID_HEIGHT} origin=(${GRID_ORIGIN_X},${GRID_ORIGIN_Y})"
echo "MATCH_RADIUS:         $SITL_EVAL_MATCH_RADIUS_M"
echo "REPORT_RADIUS:        $SITL_EVAL_REPORT_RADIUS_M"
echo "DURATION:             $SITL_EVAL_DURATION_S"
echo "STOP_ON_DONE:         $SITL_EVAL_STOP_ON_TERMINAL_PHASE"
echo "======================"

PARAM_ARGS=(
    -p world_path:="$SITL_EVAL_WORLD_PATH" \
    -p output_dir:="$SITL_EVAL_OUTPUT_DIR" \
    -p policy_label:="$SITL_EVAL_POLICY_LABEL" \
    -p target_filter:="$SITL_EVAL_TARGET_FILTER" \
    -p search_bounds_only:="$SITL_EVAL_SEARCH_BOUNDS_ONLY" \
    -p grid.width:="$GRID_WIDTH" \
    -p grid.height:="$GRID_HEIGHT" \
    -p grid.origin_x:="$GRID_ORIGIN_X" \
    -p grid.origin_y:="$GRID_ORIGIN_Y" \
    -p coverage_cell_size_m:="$SITL_EVAL_COVERAGE_CELL_SIZE_M" \
    -p match_radius_m:="$SITL_EVAL_MATCH_RADIUS_M" \
    -p report_radius_m:="$SITL_EVAL_REPORT_RADIUS_M" \
    -p sample_period_s:="$SITL_EVAL_SAMPLE_PERIOD_S" \
    -p summary_period_s:="$SITL_EVAL_SUMMARY_PERIOD_S" \
    -p duration_s:="$SITL_EVAL_DURATION_S" \
    -p stop_on_terminal_phase:="$SITL_EVAL_STOP_ON_TERMINAL_PHASE" \
    -p use_sim_time:="$USE_SIM_TIME"
)

if [[ -n "$SITL_EVAL_RUN_ID" ]]; then
    PARAM_ARGS+=(-p run_id:="$SITL_EVAL_RUN_ID")
fi

if [[ -n "$SITL_EVAL_MODEL_PATH" ]]; then
    PARAM_ARGS+=(-p model_path:="$SITL_EVAL_MODEL_PATH")
fi

exec python3 -m uav_planning.sitl_evaluator --ros-args "${PARAM_ARGS[@]}"
