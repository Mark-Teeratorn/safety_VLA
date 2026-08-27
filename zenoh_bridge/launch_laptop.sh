#!/bin/bash
# Launch Zenoh Bridge — LAPTOP (ROS2 Humble)
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[Laptop] Starting zenoh-bridge-ros2dds → Orin at 192.168.1.20:7447"
zenoh-bridge-ros2dds --config "$SCRIPT_DIR/config/laptop_bridge.json5"
