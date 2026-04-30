#!/usr/bin/env bash
# Reset the Gazebo drone model pose without recreating the Docker stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ "${1:-}" != "--inside" && ! -f "/.dockerenv" ]]; then
    cd "$REPO_DIR"
    exec docker compose exec -T sim bash /workspace/scripts/reset_gazebo_drone.bash --inside "$@"
fi

if [[ "${1:-}" == "--inside" ]]; then
    shift
fi

WORLD="${PX4_GZ_WORLD:-default}"
MODEL="${PX4_SIM_MODEL_NAME:-x500_depth_0}"
X="0.0"
Y="0.0"
Z="0.0"
YAW="0.0"
MODEL_ONLY_RESET=1

usage() {
    cat <<'EOF'
Usage: scripts/reset_gazebo_drone.bash [options]

Options:
  --world NAME       Gazebo world name (default: PX4_GZ_WORLD or default)
  --model NAME       Gazebo model name (default: x500_depth_0)
  --x METERS         Gazebo X position (default: 0.0)
  --y METERS         Gazebo Y position (default: 0.0)
  --z METERS         Gazebo Z position (default: 0.0)
  --yaw RADIANS      Yaw angle in radians (default: 0.0)
  --no-model-reset   Skip Gazebo model-only reset before setting pose
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --world)
            WORLD="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --x)
            X="$2"
            shift 2
            ;;
        --y)
            Y="$2"
            shift 2
            ;;
        --z)
            Z="$2"
            shift 2
            ;;
        --yaw)
            YAW="$2"
            shift 2
            ;;
        --no-model-reset)
            MODEL_ONLY_RESET=0
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

read -r QZ QW < <(
    python3 - "$YAW" <<'PY'
import math
import sys

yaw = float(sys.argv[1])
print(math.sin(yaw / 2.0), math.cos(yaw / 2.0))
PY
)

if [[ "$MODEL_ONLY_RESET" == "1" ]]; then
    gz service \
        -s "/world/${WORLD}/control" \
        --reqtype gz.msgs.WorldControl \
        --reptype gz.msgs.Boolean \
        --timeout 3000 \
        --req "reset { model_only: true }" >/dev/null
fi

gz service \
    -s "/world/${WORLD}/set_pose" \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 3000 \
    --req "name: \"${MODEL}\" position { x: ${X} y: ${Y} z: ${Z} } orientation { z: ${QZ} w: ${QW} }"

echo "Gazebo pose after reset:"
timeout 3 gz topic -e -n 1 -t "/world/${WORLD}/dynamic_pose/info" \
    | awk "/name: \"${MODEL}\"/{flag=1; c=0} flag{print; c++} c>14{exit}" || true
