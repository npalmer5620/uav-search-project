# Docker workflow

This directory contains everything needed to run the UAV search sim inside a Docker container, replacing the legacy UTM VM setup.

## What's in the image

Built from `docker/Dockerfile`, single-stage, one image reused by the sim, detection, planning, and optional training services:

- `sim` — PX4 SITL, Gazebo, ROS bridge, Foxglove, pose/TF helpers
- `detection` — YOLO inference node subscribed to `/camera/image_raw` and fused with `/camera/depth/image_raw`
- `planning` — mission controller that drives takeoff, either grid or PPO search, and investigation
- `training` — task-level Gymnasium environment plus Stable-Baselines3 PPO training

- Ubuntu 24.04 (Noble) via `px4io/px4-sitl-gazebo`
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic (`gz-harmonic` from the OSRF apt repo)
- ROS<->Gazebo bridge (`ros-jazzy-ros-gz`, `ros-jazzy-ros-gz-bridge`, `ros-jazzy-ros-gz-image`, `ros-jazzy-ros-gz-sim`)
- Foxglove bridge (`ros-jazzy-foxglove-bridge`) for browser-based visualization on port 8765
- Micro XRCE-DDS Agent built from source at a pinned ref
- PX4 SITL + Gazebo preinstalled at `/opt/px4-gazebo`
- `ultralytics`, `opencv-python`, `numpy`, `gymnasium`, `stable-baselines3`, `tensorboard`, and `pyyaml` via pip
- `yolo11n.pt` pre-downloaded into `/root/.cache/yolo`

The repo itself is **not** baked into the image. `docker-compose.yml` bind-mounts the host repo to `/workspace` so code edits on the host are live inside both containers.

## Prerequisites

- Docker Desktop or OrbStack on macOS / Linux
- ~15 GB of free disk for the image + volumes
- First build takes a while, but PX4 is already prebuilt in the base image

## Usage

```bash
# Build the local uav-search-sim:latest image first on any new machine.
# This image is not published to Docker Hub, so skipping this can make Compose
# fail with "pull access denied for uav-search-sim".
docker compose build

# Start the full stack in the background
docker compose up -d

# Or build and start in one command
docker compose up -d --build

# Follow sim logs
docker compose logs -f sim

# Follow detection logs
docker compose logs -f detection

# Run PPO training
docker compose run --rm training

# Open a shell inside the running sim container
docker compose exec sim bash

# Shut everything down
docker compose down
```

The entrypoint seeds `config.env` from `config.env.docker` on first run if the user hasn't provided their own. Override by creating a plain `config.env` on the host before starting the container.

The detection container launches `scripts/launch_detection.bash`, which runs the detector directly from [`src/uav_detection`](/Users/nicknationwide/uav-search-project/src/uav_detection) using `PYTHONPATH`. That keeps iteration simple and avoids `ament_python` editable-install issues in the container runtime.

The planning container launches `scripts/launch_planning.bash`, which now honors `SEARCH_POLICY=grid|rl` plus `RL_MODEL_PATH`, `RL_VECNORMALIZE_PATH`, `RL_DECISION_PERIOD_S`, `RL_MAX_STEP_XY_M`, and `RL_COVERAGE_GRID_SIDE`.

The training container launches `scripts/launch_training.bash`, defaulting to [`src/uav_rl/config/search_policy.yaml`](/Users/nicknationwide/uav-search-project/src/uav_rl/config/search_policy.yaml) and writing artifacts into [`artifacts/rl/search_policy`](/Users/nicknationwide/uav-search-project/artifacts/rl/search_policy).

## Visualization (Foxglove)

The image includes `ros-jazzy-foxglove-bridge`, and `scripts/launch_sim.bash` starts it by default. To view camera feeds and detections from the host Mac:

1. Start the sim with `docker compose up -d`
2. On the host, open Foxglove Studio → `Open connection` → `Foxglove WebSocket` → `ws://localhost:8765`
3. Add an Image panel subscribed to `/camera/image_annotated`
4. Add a Raw Messages panel subscribed to `/detections` or `/detections_3d`
5. Add a 3D panel subscribed to `/uav/drone_marker` for the drone pose and `/detections_3d_markers` for depth-localized targets

No X11 or XQuartz needed.

## VS Code dev container

`.devcontainer/devcontainer.json` points at `docker-compose.yml` and service `sim`. Open the repo in VS Code → `Reopen in Container` → VS Code attaches to the running container with the ROS and Python extensions preinstalled.

## Rebuilding PX4

PX4 is baked into the image by the `px4io/px4-sitl-gazebo` base. To pick up a newer upstream PX4/Gazebo bundle, update the base tag in `docker/Dockerfile` and rebuild:

```bash
docker compose build --no-cache sim
```

## Volumes

`docker-compose.yml` uses named volumes for:

- `colcon-build`, `colcon-install`, `colcon-log` — `colcon build` outputs survive container recreation, and stay off the host filesystem
- `ultralytics-cache` — downloaded YOLO weights persist

Wipe everything with `docker compose down -v`.

## Troubleshooting

**`MicroXRCEAgent: command not found`** — MicroXRCEAgent is installed to `/usr/local/bin` by the cmake `make install` step. Verify it's on `PATH` inside the container; rerun `ldconfig` if the shared libs aren't found.

**PX4 can't find Gazebo models** — the sim uses `worlds/search_area.sdf` from the bind-mounted repo. `launch_sim.bash` sets `GZ_SIM_RESOURCE_PATH` to include `worlds/`. If you move worlds, update the script.

**Camera rate is lower than requested** — Docker launch now overrides PX4's `mono_cam` model with a repo-owned `640x480` / `visualize=false` variant from `docker/models/mono_cam/model.sdf`, and the default `CAMERA_RATE` is `30`. If Gazebo still runs slower than the requested rate, the remaining limit is renderer / host performance rather than the ROS bridge.

**Depth detections are missing for some targets** — the depth camera has a finite clip range and can still return `inf` for very distant objects or background-dominated boxes. Check `/camera/depth/image_raw` and `/detections_3d`; it is normal for `/detections` to contain more boxes than `/detections_3d`.

**Foxglove Studio can't connect** — check `docker compose logs -f sim`, confirm the bridge is running inside the container (`ros2 node list | grep foxglove`), that port 8765 is exposed in `docker-compose.yml`, and that nothing on the host is already bound to that port.

**Detector is not publishing** — check `docker compose logs -f detection`. A healthy startup shows the YOLO model loading, then repeated `target(s) detected` lines while `/camera/image_raw` is active.
