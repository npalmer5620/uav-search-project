# Lean PyFlyt RL Triage Search

This folder is a standalone UAV triage-search RL prototype. It intentionally
does not import ROS, PX4, Gazebo, colcon, Docker, or any package under the
repo-level `src/` tree.

The stack is:

- PyFlyt/PyBullet for lightweight quadrotor simulation
- Gymnasium for the RL environment API
- Stable-Baselines3 DQN for discrete search macro-actions
- YOLOv11n for optional CPU-only visual diagnostics in sim/eval

## Setup on macOS CPU

```bash
cd pyflyt_rl
bash scripts/setup_macos_cpu.sh
source .venv/bin/activate
```

The setup script installs `wheel` and `numpy` before `PyFlyt`, then installs
macOS PyTorch wheels from PyPI. Runtime code forces YOLO and SB3 inference to
`device="cpu"`.

## Train

```bash
python train.py --config configs/default.yaml --timesteps 50000
```

Artifacts are written to `artifacts/latest/` by default:

- `model.zip`
- `config.yaml`
- `eval_metrics.json`
- `training_summary.json`
- `monitor/`

Training defaults to analytic synthetic bounding boxes: known victim primitives
are projected into the camera image, then the boxes get jitter, dropout,
confidence noise, and localization noise. This keeps MacBook CPU training fast
while giving the policy detector-like instability without running YOLO.

To switch back to the simpler noisy world-point detector:

```bash
python train.py --config configs/default.yaml --timesteps 50000 --perception point
```

The same override is available for evaluation and simulation:

```bash
python eval.py --model artifacts/latest/model.zip --episodes 20 --perception bbox
python sim.py --model artifacts/latest/model.zip --render --perception bbox
```

## Evaluate

```bash
python eval.py --model artifacts/latest/model.zip --episodes 20
python eval.py --model artifacts/latest/model.zip --episodes 20 --yolo --yolo-episodes 1
```

The evaluator reports the learned policy plus simple random and lawnmower-style
baselines. The optional YOLO pass runs a short PyFlyt visual rollout and records
CPU YOLO detections separately from the fast metrics.

## Simulate and Run YOLO

```bash
python sim.py --model artifacts/latest/model.zip --render --yolo
```

`sim.py --render` uses PyFlyt camera frames in direct mode and writes rollout
frames, annotated video, and YOLO detection JSON under `artifacts/sim/`. Add
`--gui` only when you explicitly want the PyBullet GUI window.
The default YOLO model path is `../yolo11n.pt`, reusing the model at the repo
root without duplicating it.

## Environment

The Gymnasium environment is registered as:

```python
import gymnasium as gym
import uav_triage_rl  # registers UAVTriage/PyFlytSearch-v0

env = gym.make("UAVTriage/PyFlytSearch-v0")
```

The action space is `spaces.Discrete(10)`:

1. `frontier_best`
2. `frontier_second`
3. `frontier_third`
4. `investigate_best`
5. `investigate_offset`
6. `high_info_best`
7. `high_info_second`
8. `return_center`
9. `escape_stuck`
10. `hover_scan`

The observation is a flat `spaces.Box` for SB3 `MlpPolicy`, containing UAV
state, local belief-map patch, recent detection tracks, coverage, and progress.

## Try A PyFlyt Model In Gazebo/SITL

The Docker stack bind-mounts this repo at `/workspace`, so
`pyflyt_rl/artifacts/latest/model.zip` is already visible inside the Gazebo
containers. The ROS mission controller supports the PyFlyt observation/action
contract through `RL_POLICY_VERSION=pyflyt`. From the repo root:

```bash
SITL_POLICY=rl \
RL_POLICY_VERSION=pyflyt \
RL_ARTIFACT_DIR=pyflyt_rl/artifacts/latest \
RL_MODEL_PATH=pyflyt_rl/artifacts/latest/model.zip \
bash scripts/run_sitl_policy_eval.bash --policy rl --duration 120
```

The controller automatically reads `config.yaml` next to the model when present
and forces Stable-Baselines3 inference to CPU. Use `RL_PYFLYT_CONFIG_PATH` only
when the config is not next to the model.
