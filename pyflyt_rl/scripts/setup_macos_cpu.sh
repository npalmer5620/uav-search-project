#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
  else
    PYTHON_BIN=python3
  fi
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required. Python 3.11 or 3.12 is recommended on macOS.")

print(f"Using Python {platform.python_version()} on {platform.platform()}")
PY

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "wheel>=0.42" "numpy>=1.26,<3"

# macOS PyPI wheels are CPU/MPS builds, not CUDA builds. Runtime code still
# forces device=cpu for Stable-Baselines3 and YOLO inference.
python -m pip install torch torchvision

# PyBullet's vendored zlib checks TARGET_OS_MAC and defines fdopen to NULL.
# The macOS 26 SDK then declares fdopen and the build fails. Defining fdopen as
# itself prevents that vendored macro without changing the function declaration.
export CFLAGS="${CFLAGS:-} -Dfdopen=fdopen"
python -m pip install -r requirements-macos-cpu.txt

python - <<'PY'
import gymnasium
import PyFlyt.gym_envs  # noqa: F401
import stable_baselines3
import torch
from ultralytics import YOLO  # noqa: F401

print("Sanity check passed")
print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
print(f"gymnasium={gymnasium.__version__}")
print(f"stable_baselines3={stable_baselines3.__version__}")
PY

echo "Activate with: source .venv/bin/activate"
