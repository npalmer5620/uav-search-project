# UAV Search Project

Quadcopter SITL simulation with real-time YOLOv11 object detection on monocular camera imagery.

## Architecture

```
┌─────────────────┐      uXRCE-DDS       ┌──────────────────────┐
│  PX4 Autopilot  │◄────────────────────►│     ROS 2 Jazzy      │
│     (SITL)      │   Micro XRCE-DDS     │                      │
└─────────────────┘      Agent           │  ┌────────────────┐  │
                                         │  │ YOLO Detector  │  │
┌─────────────────┐   ros_gz_bridge      │  │  (YOLOv11)     │  │
│ Gazebo Harmonic │─────────────────────►│  └───────┬────────┘  │
│                 │  sensor_msgs/Image   │          │           │
│  Quadcopter +   │                      │  Detections &        │
│  Mono Camera    │                      │  Annotated Images    │
└─────────────────┘                      └──────────────────────┘
```

**Data flow:**

1. **Gazebo Harmonic** simulates the quadcopter and its monocular camera sensor
2. Camera images are bridged into ROS 2 as `sensor_msgs/Image` via `ros_gz_bridge`
3. A **YOLOv11** (Ultralytics) ROS 2 node subscribes to the image topic, runs inference, and publishes detections
4. **PX4 SITL** communicates with ROS 2 over the Micro XRCE-DDS agent for flight control and telemetry

## Tech Stack

| Component       | Version / Details              |
|-----------------|--------------------------------|
| ROS 2           | Jazzy Jalisco                  |
| Gazebo          | Harmonic                       |
| PX4 Autopilot   | SITL (latest main)            |
| DDS Bridge      | Micro XRCE-DDS Agent          |
| Object Detection| YOLOv11 (Ultralytics)          |
| OS              | Ubuntu 24.04                   |

## Quickstart (Docker &mdash; recommended)

The stack runs as three Compose services built from the same Docker image:

- `sim` — PX4 SITL, Gazebo, `ros_gz_bridge`, Foxglove, pose/TF publishers
- `detection` — YOLOv11 inference subscribed to `/camera/image_raw` with optional depth fusion from `/camera/depth/image_raw`
- `planning` — mission controller that arms, takes off, flies the search pattern, and investigates detections

**Prerequisites:** Docker Desktop or OrbStack on macOS / Linux.

```bash
# Build the image (first time is still substantial, but PX4 itself is prebuilt)
docker compose build

# Start the full stack in the background
docker compose up -d

# Follow sim logs
docker compose logs -f sim

# Follow detection logs
docker compose logs -f detection

# Open a shell inside the running sim container
docker compose exec sim bash
```

`scripts/launch_sim.bash` starts `foxglove_bridge` and a PX4 pose bridge by default, so visualization is browser-based: open **Foxglove Studio** on the host and connect to `ws://localhost:8765`. Useful topics now include `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/image_annotated`, `/detections`, `/detections_3d`, and `/detections_3d_markers`. In the 3D panel, add `/uav/drone_marker` for a visible drone pose marker and `/detections_3d_markers` for depth-localized targets.

`scripts/launch_detection.bash` runs the detector directly from the bind-mounted source tree in a separate container, so code edits under [`src/uav_detection`](/Users/nicknationwide/uav-search-project/src/uav_detection) are picked up on container restart without a manual `colcon build`.

See [`docker/README.md`](docker/README.md) for details (rebuilding, caching, VS Code dev container, troubleshooting).

## Legacy: native / VM install

The original workflow assumed Ubuntu 24.04 on bare metal or in a UTM VM. It still works but is no longer the default.

<details>
<summary>Click to expand native install instructions</summary>

**Prerequisites:**

- **Ubuntu 24.04**
- **ROS 2 Jazzy** &mdash; [installation guide](https://docs.ros.org/en/jazzy/Installation.html)
- **Gazebo Harmonic** &mdash; [installation guide](https://gazebosim.org/docs/harmonic/install)
- **PX4 Autopilot** &mdash; [source build for SITL](https://docs.px4.io/main/en/dev_setup/building_px4.html)
- **Micro XRCE-DDS Agent** &mdash; [setup guide](https://docs.px4.io/main/en/middleware/uxrce_dds.html)
- **Python 3** with `ultralytics`: `pip install ultralytics`

**Setup & run:**

```bash
# Build the workspace (repo root IS the colcon workspace)
colcon build --symlink-install
source install/setup.bash

# Per-machine config
cp config.env.example config.env   # edit PX4_DIR etc.

# Launch everything
./scripts/launch_sim.bash

# In a second terminal, verify
source /opt/ros/jazzy/setup.bash
ros2 topic hz /camera/image_raw
python3 scripts/view_camera.py      # optional OpenCV viewer (needs display)
```

See `config.env.example` for available settings (render engine, world, camera rate). The Docker workflow overrides PX4's `mono_cam` model at launch to use a lighter `640x480` camera with sensor visualization disabled.

</details>

## Project Structure

```
uav-search-project/
├── src/
│   ├── uav_description/     # URDF/SDF models, Gazebo world files
│   ├── uav_bringup/         # Launch files, config
│   └── uav_detection/       # YOLOv11 ROS 2 detection node
├── models/                  # YOLO model weights
├── worlds/                  # Gazebo world files
├── .gitignore
└── README.md
```

## License

TBD
