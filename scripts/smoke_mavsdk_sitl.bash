#!/usr/bin/env bash
# Boot SITL and run the focused MAVSDK mission smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo "=== MAVSDK SITL smoke ==="
echo "This recreates sim/planning so PX4 resolves the current planning container IP."
echo "PX4_GZ_WORLD: ${PX4_GZ_WORLD:-default}"

docker compose stop planning detection sim >/dev/null 2>&1 || true

PLANNING_MODE=smoke \
PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}" \
docker compose up \
    --force-recreate \
    --abort-on-container-exit \
    --exit-code-from planning \
    sim planning
