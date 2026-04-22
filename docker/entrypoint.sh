#!/usr/bin/env bash
# Docker entrypoint for the UAV search sim container.
# Sources ROS 2 + the colcon workspace (if built), makes sure config.env is in
# place, then execs whatever command was passed.

set -e

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Source pre-built px4_msgs overlay (built into the image)
if [[ -f /opt/ros/px4_msgs_ws/install/setup.bash ]]; then
    source /opt/ros/px4_msgs_ws/install/setup.bash
fi

# If the bind-mounted workspace has been built, source it too.
if [[ -f /workspace/install/setup.bash ]]; then
    source /workspace/install/setup.bash
fi

# Seed config.env from the committed container defaults if the user hasn't
# provided their own. launch_sim.bash expects config.env to exist.
if [[ ! -f /workspace/config.env && -f /workspace/config.env.docker ]]; then
    cp /workspace/config.env.docker /workspace/config.env
fi

cd /workspace
exec "$@"
