#!/bin/bash
# ============================================================
# Install zenoh-bridge-ros2dds — Run on EACH machine
# Downloads the correct .deb from GitHub Releases
# No ROS2 node, no Python — standalone binary
# ============================================================
set -e

# ---- Auto-detect ROS2 distro ----
if [ -z "$ROS_DISTRO" ]; then
    for D in jazzy humble iron foxy; do
        if [ -f "/opt/ros/$D/setup.bash" ]; then
            source "/opt/ros/$D/setup.bash"
            break
        fi
    done
fi

if [ -z "$ROS_DISTRO" ]; then
    echo "[ERROR] Cannot detect ROS_DISTRO. Source your ROS2 setup first."
    exit 1
fi

ARCH=$(dpkg --print-architecture)   # amd64 or arm64
ZENOH_VER="1.0.0"
REPO="https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download"

echo "=============================="
echo " ROS_DISTRO : $ROS_DISTRO"
echo " ARCH       : $ARCH"
echo " Zenoh ver  : $ZENOH_VER"
echo "=============================="

DEB="${REPO}/${ZENOH_VER}/zenoh-bridge-ros2dds_${ZENOH_VER}-1_${ARCH}.deb"
TMP="/tmp/zenoh-bridge-ros2dds.deb"

echo "[INFO] Downloading $DEB ..."
wget -q --show-progress -O "$TMP" "$DEB"

echo "[INFO] Installing ..."
sudo dpkg -i "$TMP" || sudo apt-get install -f -y
rm -f "$TMP"

echo ""
echo "[OK] Installed:"
zenoh-bridge-ros2dds --version 2>&1 | head -2
