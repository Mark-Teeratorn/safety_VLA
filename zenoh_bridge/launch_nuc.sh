#!/bin/bash
# Launch Zenoh Bridge — INTEL NUC
for D in jazzy humble iron foxy; do
    [ -f "/opt/ros/$D/setup.bash" ] && source "/opt/ros/$D/setup.bash" && break
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[NUC] Starting zenoh-bridge-ros2dds PEER → Orin at 192.168.1.20:7447"
zenoh-bridge-ros2dds --config "$SCRIPT_DIR/config/nuc_bridge.json5"
