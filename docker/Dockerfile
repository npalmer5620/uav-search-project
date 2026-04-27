# syntax=docker/dockerfile:1.6
# UAV search project — all-in-one sim image.
#
# Built on top of the official prebuilt PX4 SITL + Gazebo image from px4io.
# That base already ships:
#   - Ubuntu 24.04 Noble
#   - PX4 SITL binary + all modules at /opt/px4-gazebo/bin/
#   - x500 / x500_mono_cam / x500_depth / ... Gazebo models at
#     /opt/px4-gazebo/share/gz/models/
#   - Gazebo Harmonic installed system-wide (gz-harmonic, gz-sim8, ...)
#   - A Docker-aware entrypoint that rewrites MAVLink/DDS targets to
#     host.docker.internal on Docker Desktop
#
# We layer on top:
#   - ROS 2 Jazzy (base) and the ros_gz / foxglove / vision_msgs packages
#   - Micro XRCE-DDS Agent (built from source; no apt package for Jazzy)
#   - ultralytics + opencv-python + numpy, with yolo11n.pt pre-fetched
#
# Skipping the PX4 source clone and compile cuts ~30-60 minutes off the
# first build. PX4 is not modified by this project, so there is no reason
# to rebuild it from source.
FROM px4io/px4-sitl-gazebo:latest

# The base image sets an entrypoint that execs px4. Clear it so our ROS/
# colcon workflow isn't hijacked; we set our own entrypoint at the bottom.
ENTRYPOINT []
CMD []

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PX4_DIR=/opt/px4-gazebo

# ---------------------------------------------------------------------------
# 1. Add the ROS 2 Jazzy apt source (the base only has Ubuntu + OSRF)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        locales \
        lsb-release \
        sudo \
        tzdata \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. ROS 2 Jazzy + ros-gz bridge + foxglove + build tools
# ---------------------------------------------------------------------------
# The ros-jazzy-ros-gz-* packages depend on gz-harmonic, which is already
# present in the base image via the OSRF repo, so dep resolution is fine.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        file \
        git \
        gstreamer1.0-gl \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        ninja-build \
        patchelf \
        python3-colcon-common-extensions \
        python3-pip \
        python3-venv \
        rsync \
        unzip \
        wget \
        zip \
        ros-jazzy-ros-base \
        ros-jazzy-ros-gz \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-ros-gz-image \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. MicroXRCEAgent — built from source (see docker/build-uxrce.sh)
# ---------------------------------------------------------------------------
# No apt package for Jazzy. The script handles the superbuild's quirks
# (missing `install` target, missing uagent install step, RPATH strip).
ARG UXRCE_REF=v3.0.0
COPY docker/build-uxrce.sh /tmp/build-uxrce.sh
RUN sed -i 's/\r$//' /tmp/build-uxrce.sh \
    && chmod +x /tmp/build-uxrce.sh \
    && /tmp/build-uxrce.sh "${UXRCE_REF}" \
    && rm /tmp/build-uxrce.sh

# ---------------------------------------------------------------------------
# 4. px4_msgs — built into a separate overlay so foxglove_bridge (and any
#    other node) can resolve PX4 message types without a colcon build at
#    runtime.  The overlay lives outside /workspace so the bind-mount doesn't
#    shadow it.
# ---------------------------------------------------------------------------
COPY src/px4_msgs /opt/ros/px4_msgs_ws/src/px4_msgs
SHELL ["/bin/bash", "-c"]
RUN . /opt/ros/jazzy/setup.bash \
    && cd /opt/ros/px4_msgs_ws \
    && colcon build --symlink-install --packages-select px4_msgs \
    && rm -rf log build/px4_msgs/CMakeFiles

# ---------------------------------------------------------------------------
# 5. Python / YOLO deps
# ---------------------------------------------------------------------------
# 1a) Reinstall debian-managed packages that pip can't uninstall (no RECORD).
RUN pip install --break-system-packages --no-cache-dir --ignore-installed \
        numpy scipy

# 1b) CPU-only PyTorch (separate so --index-url only applies to torch).
RUN pip install --break-system-packages --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) ultralytics + ONNX runtime. No --ignore-installed here so pip sees the
#    CPU torch from step 1b as satisfied and won't re-resolve it to CUDA.
RUN pip install --break-system-packages --no-cache-dir \
        gymnasium \
        pyyaml \
        stable-baselines3 \
        tensorboard \
        ultralytics-opencv-headless \
        onnx \
        onnxruntime \
    && mkdir -p /root/.cache/yolo \
    && cd /root/.cache/yolo \
    && python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, simplify=False)"

# 2b) Stable-Baselines3 imports matplotlib during logger setup. The Debian
#     matplotlib from apt is built against the system NumPy ABI, so overlay
#     a pip wheel that matches the pip-installed NumPy stack used above.
RUN pip install --break-system-packages --no-cache-dir --ignore-installed \
        "matplotlib>=3.10"

# ---------------------------------------------------------------------------
# 5b. Increase PX4 Gazebo world startup timeout (default 30s is too short for
#     first launch when Gazebo downloads assets / compiles shaders).
# ---------------------------------------------------------------------------
RUN sed -i 's/^ATTEMPTS=30$/ATTEMPTS=${PX4_GZ_WORLD_STARTUP_TIMEOUT:-120}/' \
        /opt/px4-gazebo/etc/init.d-posix/px4-rc.gzsim

# ---------------------------------------------------------------------------
# 6. Workspace + entrypoint
# ---------------------------------------------------------------------------
# /workspace is where the host repo gets bind-mounted at runtime. Nothing
# from the repo is COPY'd into the image.
WORKDIR /workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Exposed ports:
#   8888/udp — MicroXRCEAgent (PX4 <-> ROS 2 DDS bridge)
#   8765/tcp — foxglove_bridge WebSocket (browser viz from the Mac host)
EXPOSE 8888/udp 8765/tcp

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
