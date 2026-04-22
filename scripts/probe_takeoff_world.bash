#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_DIR/config.env"
WORLD_NAME="${1:?usage: probe_takeoff_world.bash <world_name>}"
SUCCESS_Z="${SUCCESS_Z:--2.0}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-70}"
SIM_READY_TIMEOUT_SECONDS="${SIM_READY_TIMEOUT_SECONDS:-180}"
START_SERVICES="${START_SERVICES:-sim}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "config.env is missing at $CONFIG_FILE" >&2
    exit 1
fi

CONFIG_BACKUP="$(mktemp)"
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

restore_config() {
    cp "$CONFIG_BACKUP" "$CONFIG_FILE"
    rm -f "$CONFIG_BACKUP"
}

cleanup() {
    restore_config
    docker compose down --remove-orphans >/dev/null 2>&1 || true
}

trap cleanup EXIT

python3 - "$CONFIG_FILE" "$WORLD_NAME" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
world_name = sys.argv[2]
text = config_path.read_text()
line = f"PX4_GZ_WORLD={world_name}"

if re.search(r"^PX4_GZ_WORLD=.*$", text, flags=re.MULTILINE):
    text = re.sub(r"^PX4_GZ_WORLD=.*$", line, text, flags=re.MULTILINE)
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += line + "\n"

config_path.write_text(text)
PY

echo "=== Probing world: ${WORLD_NAME} ==="
docker compose down --remove-orphans >/dev/null 2>&1 || true
docker compose up -d ${START_SERVICES} >/dev/null

ready_deadline=$((SECONDS + SIM_READY_TIMEOUT_SECONDS))
while true; do
    sim_logs="$(docker compose logs sim --tail=250 2>&1 || true)"
    if grep -q "Ready for takeoff!" <<<"$sim_logs"; then
        break
    fi
    if grep -q "ERROR:" <<<"$sim_logs"; then
        echo "$sim_logs"
        echo "Sim failed before reaching ready state" >&2
        exit 1
    fi
    if (( SECONDS >= ready_deadline )); then
        echo "$sim_logs"
        echo "Timed out waiting for PX4 ready state" >&2
        exit 1
    fi
    sleep 2
done

probe_output="$(
    docker compose exec -T sim bash -lc "
        source /opt/ros/jazzy/setup.bash
        if [ -f /workspace/install/setup.bash ]; then
            source /workspace/install/setup.bash
        fi
        export PYTHONUNBUFFERED=1
        python3 - '$WORLD_NAME' '$SUCCESS_Z' '$MAX_WAIT_SECONDS' <<'PY'
import json
import socket
import sys
import time

import rclpy
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

world_name = sys.argv[1]
success_z = float(sys.argv[2])
max_wait_seconds = float(sys.argv[3])


class LocalPositionProbe(Node):
    def __init__(self):
        super().__init__('takeoff_probe')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.latest_z = None
        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._on_position,
            qos,
        )

    def _on_position(self, msg):
        self.latest_z = float(msg.z)


def send_command(sock, command):
    sock.sendall((command + '\n').encode('utf-8'))


rclpy.init()
node = LocalPositionProbe()

deadline = time.monotonic() + 20.0
while node.latest_z is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)

if node.latest_z is None:
    raise RuntimeError('No VehicleLocalPosition received')

sock = socket.create_connection(('127.0.0.1', 14600), timeout=5.0)
send_command(sock, 'commander disarm -f')
time.sleep(1.0)
send_command(sock, 'commander arm -f')
time.sleep(5.0)
send_command(sock, 'commander takeoff')

samples = []
start = time.monotonic()
next_sample = start
min_z = node.latest_z
success = False

while time.monotonic() - start < max_wait_seconds:
    rclpy.spin_once(node, timeout_sec=0.2)
    if node.latest_z is None:
        continue
    min_z = min(min_z, node.latest_z)
    now = time.monotonic()
    if now >= next_sample:
        samples.append({
            'elapsed_s': round(now - start, 1),
            'z': round(node.latest_z, 3),
        })
        next_sample += 5.0
    if min_z <= success_z:
        success = True
        break

send_command(sock, 'commander land')
time.sleep(2.0)
send_command(sock, 'commander disarm -f')
sock.close()

result = {
    'world': world_name,
    'success_z': success_z,
    'success': success,
    'min_z': round(min_z, 3),
    'samples': samples,
}
print(json.dumps(result))

node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
PY
    " 2>&1
)"

echo "$probe_output" | tail -n 1
echo "--- Sim log excerpt (${WORLD_NAME}) ---"
docker compose logs sim --tail=250 | rg "Ready for takeoff|Takeoff detected|Landing detected|Disarmed by landing|Armed by external command|WARN|ERROR|Preflight" || true
