#!/bin/bash
# Launch Zenoh Receiver — AGX ORIN
# NO ROS2 required — pure Python + eclipse-zenoh
#
# Install once:
#   pip install eclipse-zenoh opencv-python --break-system-packages
#
# Then run:
#   bash launch_orin.sh
#   bash launch_orin.sh --demo     # show live camera window

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[Orin] Starting Zenoh ROUTER receiver on port 7447..."
python3 "$SCRIPT_DIR/orin_receiver.py" "$@"
