# UAV Search Project

Quadcopter SITL simulation with real-time YOLOv11 object detection on monocular camera imagery.

## Architecture

```
┌─────────────────┐       uXRCE-DDS        ┌─────────────────────┐
│  PX4 Autopilot  │◄──────────────────────►│      ROS 2 Jazzy     │
│     (SITL)      │    Micro XRCE-DDS      │                      │
└─────────────────┘       Agent             │  ┌────────────────┐  │
                                            │  │ YOLO Detector  │  │
┌─────────────────┐    ros_gz_bridge        │  │  (YOLOv11)     │  │
│ Gazebo Harmonic │────────────────────────►│  └───────┬────────┘  │
│                 │   sensor_msgs/Image     │          │           │
│  Quadcopter +   │                         │   Detections &       │
│  Mono Camera    │                         │   Annotated Images   │
└─────────────────┘                         └─────────────────────┘
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
| OS              | Ubuntu 24.04 ARM64 (UTM VM)   |
| Host            | macOS (Apple Silicon)          |

## Prerequisites

- **Ubuntu 24.04 ARM64** (running in UTM on macOS)
- **ROS 2 Jazzy** &mdash; [installation guide](https://docs.ros.org/en/jazzy/Installation.html)
- **Gazebo Harmonic** &mdash; [installation guide](https://gazebosim.org/docs/harmonic/install)
- **PX4 Autopilot** &mdash; [source build for SITL](https://docs.px4.io/main/en/dev_setup/building_px4.html)
- **Micro XRCE-DDS Agent** &mdash; [setup guide](https://docs.px4.io/main/en/middleware/uxrce_dds.html)
- **Python 3** with `ultralytics` package:
  ```bash
  pip install ultralytics
  ```

## Setup & Build

```bash
# Clone the workspace
mkdir -p ~/uav_ws/src
cd ~/uav_ws/src
git clone <this-repo> uav-search-project

# Build with colcon
cd ~/uav_ws
colcon build --symlink-install
source install/setup.bash
```

> Packages and launch files will be added here as the project develops.

## Usage

```bash
# Terminal 1 — Start PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam

# Terminal 2 — Micro XRCE-DDS Agent
MicroXRCEAgent udp4 -p 8888

# Terminal 3 — Launch ROS 2 nodes (bridge + YOLO detector)
ros2 launch uav_search bringup.launch.py
```

> Launch files and exact commands will be updated as packages are implemented.

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
