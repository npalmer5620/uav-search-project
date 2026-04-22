# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

UAV Search Project — quadcopter SITL simulation with real-time YOLOv11 object detection on monocular camera images.

**Stack:** ROS 2 Jazzy, Gazebo Harmonic, PX4 Autopilot (SITL), Micro XRCE-DDS Agent, YOLOv11 (Ultralytics)

**Environment:** The supported workflow is a single Docker image (`docker/Dockerfile`) bundling Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL + uXRCE-DDS Agent + ultralytics. Use `docker compose up -d && docker compose exec sim bash` to get a shell inside the sim container. The host repo is bind-mounted to `/workspace`, so edits on macOS are live inside the container. Colcon outputs (`build/`, `install/`, `log/`) live in named volumes, not on the host.

Legacy workflow (still supported, not the default): Ubuntu 24.04 ARM64 UTM VM on macOS Apple Silicon, with the repo shared via UTM directory passthrough at `/home/nicknationwide/utm/uav-search-project`. Prefer Docker for all new work.

## Architecture

PX4 SITL ↔ Micro XRCE-DDS Agent ↔ ROS 2 Jazzy. Gazebo Harmonic simulates the quadcopter with a monocular camera. Camera images are bridged to ROS 2 via `ros_gz_bridge` as `sensor_msgs/Image`. A YOLOv11 ROS 2 node subscribes to the image topic and publishes detections.

ROS 2 packages (in `src/`):
- `uav_bringup` — launch files and config (ros_gz_bridge config, etc.)
- `uav_description` — SDF models and Gazebo world files (planned)
- `uav_detection` — YOLOv11 inference node (planned)

## Per-developer Config

Developer-specific settings (PX4 path, Gazebo render engine) live in `config.env` (gitignored). Copy from `config.env.example`. The `scripts/launch_sim.bash` script reads this file.

## Build & Run

### Docker (default)

```bash
docker compose build                    # first build: slow (PX4 compile)
docker compose up -d
docker compose exec sim bash

# Inside the container:
colcon build --symlink-install
source install/setup.bash
bash scripts/launch_sim.bash
```

The entrypoint auto-sources ROS 2 Jazzy and the workspace install (if built), and seeds `config.env` from `config.env.docker` on first run. See `docker/README.md` for details (Foxglove viz, PX4 rebuild, volumes, troubleshooting).

### Native / VM (legacy)

```bash
# Build (from repo root — this IS the colcon workspace root)
colcon build --symlink-install
source install/setup.bash

# Build a single package
colcon build --symlink-install --packages-select uav_bringup

# Option A: all-in-one launch script (starts PX4 + XRCE agent + bridge)
bash scripts/launch_sim.bash

# Option B: manual (3 terminals)
# T1: source /opt/ros/jazzy/setup.bash && cd ~/PX4-Autopilot && make px4_sitl gz_x500_mono_cam
# T2: MicroXRCEAgent udp4 -p 8888
# T3: ros2 launch uav_bringup bridge.launch.py
```

## Conventions

- This repo IS the colcon workspace root (packages are in `src/`)
- ROS 2 packages use `ament_cmake` or `ament_python` build type
- Python detection code uses the `ultralytics` package for YOLOv11 inference
- Launch files use ROS 2 Python launch format (`.launch.py`)
- Camera bridge config is in `src/uav_bringup/config/bridge.yaml`
