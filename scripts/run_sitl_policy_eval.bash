#!/usr/bin/env bash
# Run one quantitative Gazebo/SITL policy episode and validate that it flew.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

POLICY="${SITL_POLICY:-grid}"
RUN_ID="${SITL_RUN_ID:-}"
EVAL_DURATION_S="${SITL_EVAL_DURATION_S:-120.0}"
OUTPUT_DIR="${SITL_EVAL_OUTPUT_DIR:-artifacts/sitl_eval}"
GRID_ALTITUDE="${GRID_ALTITUDE:--4.0}"
SEARCH_TIMEOUT_S="${SITL_SEARCH_TIMEOUT_S:-220}"
MIN_ALTITUDE_M="${SITL_MIN_ALTITUDE_M:-2.0}"
MIN_SEARCH_PATH_M="${SITL_MIN_SEARCH_PATH_M:-5.0}"
RESTART_STACK="${SITL_RESTART_STACK:-true}"
REROUTE_MAVLINK="${SITL_REROUTE_MAVLINK:-true}"
MAVLINK_REMOTE_PORT="${MAVSDK_MAVLINK_REMOTE_PORT:-14540}"
MAVLINK_RATE="${MAVSDK_MAVLINK_RATE:-4000000}"
RL_POLICY_VERSION="${RL_POLICY_VERSION:-v2}"
RL_POLICY_KEY="$(printf '%s' "$RL_POLICY_VERSION" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${RL_ARTIFACT_DIR:-}" ]]; then
    if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        RL_ARTIFACT_DIR="pyflyt_rl/artifacts/latest"
    else
        RL_ARTIFACT_DIR="artifacts/rl/search_policy_v2_candidate_actions_50k"
    fi
fi
RL_MODEL_PATH="${RL_MODEL_PATH:-$RL_ARTIFACT_DIR/model.zip}"

usage() {
    cat <<'EOF'
Usage: scripts/run_sitl_policy_eval.bash [options]

Options:
  --policy grid|rl       Policy to evaluate (default: SITL_POLICY or grid)
  --run-id ID            Evaluator run id (default: auto)
  --duration SECONDS     SEARCH-duration budget for evaluator (default: 120.0)
  --output-dir DIR       Evaluator artifact root (default: artifacts/sitl_eval)
  --no-restart-stack     Do not recreate sim/detection/planning before this run
  -h, --help             Show this help

Environment knobs:
  RL_MODEL_PATH, RL_ARTIFACT_DIR, GRID_ALTITUDE, SITL_SEARCH_TIMEOUT_S,
  SITL_MIN_ALTITUDE_M, SITL_MIN_SEARCH_PATH_M, SITL_REROUTE_MAVLINK,
  RL_POLICY_VERSION, RL_PYFLYT_CONFIG_PATH
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --duration)
            EVAL_DURATION_S="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-restart-stack)
            RESTART_STACK=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$POLICY" != "grid" && "$POLICY" != "rl" ]]; then
    echo "ERROR: --policy must be 'grid' or 'rl'" >&2
    exit 2
fi

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${POLICY}"
fi

RUN_DIR="$REPO_DIR/$OUTPUT_DIR/$RUN_ID"

compose_env=(
    SEARCH_POLICY="$POLICY"
    GRID_ALTITUDE="$GRID_ALTITUDE"
    MAVSDK_REQUIRE_FULL_HEALTH=false
)

if [[ "$POLICY" == "rl" ]]; then
    if [[ -z "${RL_DECISION_PERIOD_S:-}" ]]; then
        if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
            RL_DECISION_PERIOD_S="1.5"
        else
            RL_DECISION_PERIOD_S="2.0"
        fi
    fi
    if [[ -z "${RL_SCAN_YAW_STEP_DEG:-}" ]]; then
        if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
            RL_SCAN_YAW_STEP_DEG="35.0"
        else
            RL_SCAN_YAW_STEP_DEG="25.0"
        fi
    fi
    if [[ -z "${RL_MAX_YAW_STEP_DEG:-}" ]]; then
        RL_MAX_YAW_STEP_DEG="25.0"
    fi
    if [[ -z "${RL_CELL_SIZE_M:-}" ]]; then
        if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
            RL_CELL_SIZE_M="2.0"
        else
            RL_CELL_SIZE_M="4.0"
        fi
    fi
    compose_env+=(
        RL_POLICY_VERSION="$RL_POLICY_VERSION"
        RL_ALGORITHM="${RL_ALGORITHM:-dqn}"
        RL_ARTIFACT_DIR="$RL_ARTIFACT_DIR"
        RL_MODEL_PATH="$RL_MODEL_PATH"
        RL_PYFLYT_CONFIG_PATH="${RL_PYFLYT_CONFIG_PATH:-}"
        RL_DECISION_PERIOD_S="$RL_DECISION_PERIOD_S"
        RL_SCAN_YAW_STEP_DEG="$RL_SCAN_YAW_STEP_DEG"
        RL_MAX_YAW_STEP_DEG="$RL_MAX_YAW_STEP_DEG"
        RL_MAX_UNPRODUCTIVE_SCAN_STREAK="${RL_MAX_UNPRODUCTIVE_SCAN_STREAK:-2}"
        RL_CELL_SIZE_M="$RL_CELL_SIZE_M"
        RL_PATCH_SIDE="${RL_PATCH_SIDE:-11}"
    )
fi

run_with_env() {
    env "${compose_env[@]}" "$@"
}

wait_for_log() {
    local service="$1"
    local pattern="$2"
    local timeout_s="$3"
    local start_s
    start_s="$(date +%s)"
    while true; do
        if docker compose logs --tail=260 "$service" 2>/dev/null | grep -Eq "$pattern"; then
            return 0
        fi
        if (( $(date +%s) - start_s >= timeout_s )); then
            return 1
        fi
        sleep 2
    done
}

reroute_mavlink_to_planning() {
    local planning_ip
    planning_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' uav-planning)"
    if [[ -z "$planning_ip" ]]; then
        echo "ERROR: could not determine uav-planning IP" >&2
        return 1
    fi
    local local_port=$((14542 + RANDOM % 1000))
    local command="mavlink start -x -u ${local_port} -r ${MAVLINK_RATE} -t ${planning_ip} -o ${MAVLINK_REMOTE_PORT}"
    echo "Rerouting PX4 MAVLink to planning (${planning_ip}:${MAVLINK_REMOTE_PORT}) from local port ${local_port}"
    docker compose exec -T sim bash -lc "pipe=(/tmp/px4_pipe_*); [[ -p \"\${pipe[0]}\" ]] && echo '$command' > \"\${pipe[0]}\""
}

write_metadata() {
    local status="$1"
    local reason="${2:-}"
    python3 - "$RUN_DIR" "$POLICY" "$RUN_ID" "$status" "$reason" "$EVAL_DURATION_S" "$GRID_ALTITUDE" "$RL_MODEL_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_path = run_dir / "summary.json"
summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text())

metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": sys.argv[2],
    "run_id": sys.argv[3],
    "status": sys.argv[4],
    "reason": sys.argv[5],
    "duration_s": float(sys.argv[6]),
    "grid_altitude_ned_m": float(sys.argv[7]),
    "rl_model_path": sys.argv[8] if sys.argv[2] == "rl" else "",
    "summary_path": str(summary_path),
    "mission_started": bool(summary.get("mission_started", False)),
    "phase": summary.get("phase"),
    "max_altitude_m": summary.get("max_altitude_m"),
    "search_path_length_m": summary.get("search_path_length_m"),
    "targets_detected": summary.get("targets_detected"),
    "targets_reported": summary.get("targets_reported"),
    "ground_truth_targets": summary.get("ground_truth_targets"),
    "mean_best_detection_error_m": summary.get("mean_best_detection_error_m"),
    "mean_best_report_error_m": summary.get("mean_best_report_error_m"),
    "visible_coverage_fraction": summary.get("visible_coverage_fraction"),
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
}

validate_summary() {
    python3 - "$RUN_DIR/summary.json" "$MIN_ALTITUDE_M" "$MIN_SEARCH_PATH_M" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
min_altitude_m = float(sys.argv[2])
min_search_path_m = float(sys.argv[3])
if not summary_path.exists():
    print("summary.json missing")
    sys.exit(1)
summary = json.loads(summary_path.read_text())
if not summary.get("mission_started", False):
    print("mission never reached SEARCH")
    sys.exit(1)
max_altitude = float(summary.get("max_altitude_m") or 0.0)
if max_altitude < min_altitude_m:
    print(f"max altitude {max_altitude:.2f}m < required {min_altitude_m:.2f}m")
    sys.exit(1)
search_path = float(summary.get("search_path_length_m") or 0.0)
if search_path < min_search_path_m:
    print(f"search path {search_path:.2f}m < required {min_search_path_m:.2f}m")
    sys.exit(1)
print(
    "valid: "
    f"targets_detected={summary.get('targets_detected')} "
    f"targets_reported={summary.get('targets_reported')} "
    f"visible_coverage={float(summary.get('visible_coverage_fraction') or 0.0):.2%} "
    f"search_path={search_path:.1f}m "
    f"max_altitude={max_altitude:.1f}m"
)
PY
}

echo "=== SITL Policy Episode ==="
echo "POLICY:        $POLICY"
echo "RUN_ID:        $RUN_ID"
echo "DURATION:      $EVAL_DURATION_S"
echo "OUTPUT_DIR:    $RUN_DIR"
echo "GRID_ALTITUDE: $GRID_ALTITUDE"
if [[ "$POLICY" == "rl" ]]; then
    echo "RL_POLICY_VERSION: $RL_POLICY_VERSION"
    echo "RL_MODEL_PATH: $RL_MODEL_PATH"
    if [[ "$RL_POLICY_KEY" == pyflyt* ]]; then
        echo "RL_PYFLYT_CONFIG: ${RL_PYFLYT_CONFIG_PATH:-auto}"
    fi
fi
echo "==========================="

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

if [[ "$RESTART_STACK" == "true" ]]; then
    docker compose stop evaluator planning detection sim >/dev/null 2>&1 || true
    run_with_env docker compose up -d --force-recreate sim detection planning
else
    docker compose stop evaluator >/dev/null 2>&1 || true
    run_with_env docker compose up -d --force-recreate planning
fi

if [[ "$REROUTE_MAVLINK" == "true" ]]; then
    if wait_for_log sim "All processes running|Starting PX4 MAVLink stream" 180; then
        reroute_mavlink_to_planning || true
    else
        echo "WARNING: timed out waiting for sim MAVLink readiness"
    fi
fi

eval_model_path=""
if [[ "$POLICY" == "rl" ]]; then
    eval_model_path="$RL_MODEL_PATH"
fi

env \
    "${compose_env[@]}" \
    SITL_EVAL_RUN_ID="$RUN_ID" \
    SITL_EVAL_OUTPUT_DIR="$OUTPUT_DIR" \
    SITL_EVAL_DURATION_S="$EVAL_DURATION_S" \
    SITL_EVAL_POLICY_LABEL="$POLICY" \
    SITL_EVAL_MODEL_PATH="$eval_model_path" \
    docker compose run --rm --no-deps evaluator &
evaluator_pid=$!

if wait_for_log planning "Phase: HANDOFF -> SEARCH|SEARCH rl_v2=|SEARCH rl_pyflyt=|SEARCH grid=" "$SEARCH_TIMEOUT_S"; then
    echo "Mission reached SEARCH."
else
    echo "ERROR: mission did not reach SEARCH within ${SEARCH_TIMEOUT_S}s" >&2
    wait "$evaluator_pid" || true
    write_metadata "invalid" "mission did not reach SEARCH"
    exit 1
fi

wait "$evaluator_pid"

validation_output=""
if validation_output="$(validate_summary 2>&1)"; then
    echo "$validation_output"
    write_metadata "valid" ""
else
    echo "ERROR: invalid episode: $validation_output" >&2
    write_metadata "invalid" "$validation_output"
    exit 1
fi

echo "Wrote metrics to $RUN_DIR"
