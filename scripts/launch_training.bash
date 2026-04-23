#!/usr/bin/env bash
# Launch PPO training for the task-level search environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source_setup() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    set -u
}

source_setup /opt/ros/jazzy/setup.bash

if [[ -f "$REPO_DIR/install/setup.bash" ]]; then
    source_setup "$REPO_DIR/install/setup.bash"
fi

export PYTHONPATH="$REPO_DIR/src/uav_rl:$REPO_DIR/src/uav_planning:${PYTHONPATH:-}"

TRAIN_CONFIG_PATH="${RL_TRAIN_CONFIG_PATH:-$REPO_DIR/src/uav_rl/config/search_policy.yaml}"
TRAIN_ARTIFACT_DIR="${RL_TRAIN_ARTIFACT_DIR:-}"
TRAIN_TOTAL_TIMESTEPS="${RL_TRAIN_TOTAL_TIMESTEPS:-}"

echo "=== UAV RL Training ==="
echo "CONFIG:   $TRAIN_CONFIG_PATH"
if [[ -n "$TRAIN_ARTIFACT_DIR" ]]; then
    echo "ARTIFACTS: $TRAIN_ARTIFACT_DIR"
fi
if [[ -n "$TRAIN_TOTAL_TIMESTEPS" ]]; then
    echo "TIMESTEPS: $TRAIN_TOTAL_TIMESTEPS"
fi
echo "======================="

ARGS=(--config "$TRAIN_CONFIG_PATH")
if [[ -n "$TRAIN_ARTIFACT_DIR" ]]; then
    ARGS+=(--artifact-dir "$TRAIN_ARTIFACT_DIR")
fi
if [[ -n "$TRAIN_TOTAL_TIMESTEPS" ]]; then
    ARGS+=(--total-timesteps "$TRAIN_TOTAL_TIMESTEPS")
fi

exec python3 -m uav_rl.train_search_policy "${ARGS[@]}"
