#!/usr/bin/env python3
"""
Zenoh ROS2 Bridge — AGX ORIN (Hub)
====================================
ROS2 Jazzy — Ubuntu 24.04

Two roles in one process:
  1. SUBSCRIBER: Receives sensor topics from Laptop via Zenoh,
                 deserializes CDR bytes, publishes to local ROS2
                 so the VLA node can subscribe normally.

  2. PUBLISHER:  Subscribes to VLA output topics on local ROS2,
                 serializes and forwards to Intel NUC via Zenoh.

The Orin runs as the Zenoh ROUTER — both Laptop and NUC connect to it.

Usage:
    source /opt/ros/jazzy/setup.bash
    pip install eclipse-zenoh
    python3 orin_bridge.py --nuc-ip 192.168.20.30

    # Or if NUC connects TO Orin (Orin is always router):
    python3 orin_bridge.py   # NUC connects to this machine's IP
"""

import argparse
import threading
import time

import zenoh
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String  # ← update to real VLA msg types when confirmed

# ---- Zenoh key expressions — must match laptop_publisher.py ----
KEY_KINEMATIC_STATE  = "ros2/laptop/localization/kinematic_state"
KEY_YOLOX_IMAGE      = "ros2/laptop/perception/yolox/image_compressed"
KEY_CAMERA_RAW       = "ros2/laptop/sensing/camera0/image_raw_compressed"

# ---- Zenoh keys for VLA output → NUC ----
KEY_VLA_DECEL        = "ros2/vla/deceleration_cmd"
KEY_VLA_BRAKE        = "ros2/vla/brake_cmd"
KEY_VLA_STATUS       = "ros2/vla/safety_status"


class OrinZenohBridge(Node):
    def __init__(self, zenoh_session: zenoh.Session):
        super().__init__("orin_zenoh_bridge")
        self.zs = zenoh_session

        # =========================================================
        # ROLE 1: Receive from Laptop → Publish to local ROS2
        # =========================================================

        # Local ROS2 publishers (these appear as topics on the Orin for VLA)
        self.pub_kinematic = self.create_publisher(
            Odometry, "/localization/kinematic_state", 10)
        self.pub_yolox = self.create_publisher(
            CompressedImage,
            "/perception/object_recognition/detection/tensorrt_yolox_node/out/image",
            5)
        self.pub_camera_raw = self.create_publisher(
            CompressedImage,
            "/sensing/camera/camera0/image_raw/compressed",
            5)

        # Zenoh subscribers — callback runs in Zenoh thread, we dispatch to ROS2
        self._sub_kinematic = self.zs.declare_subscriber(
            KEY_KINEMATIC_STATE, self._zenoh_cb_kinematic)
        self._sub_yolox = self.zs.declare_subscriber(
            KEY_YOLOX_IMAGE, self._zenoh_cb_yolox)
        self._sub_camera = self.zs.declare_subscriber(
            KEY_CAMERA_RAW, self._zenoh_cb_camera)

        # =========================================================
        # ROLE 2: Subscribe local VLA output → Send to NUC via Zenoh
        # =========================================================

        # Zenoh publishers toward NUC
        self.zpub_decel = self.zs.declare_publisher(
            KEY_VLA_DECEL, congestion_control=zenoh.CongestionControl.BLOCK)
        self.zpub_brake = self.zs.declare_publisher(
            KEY_VLA_BRAKE, congestion_control=zenoh.CongestionControl.BLOCK)
        self.zpub_status = self.zs.declare_publisher(
            KEY_VLA_STATUS, congestion_control=zenoh.CongestionControl.DROP)

        # ROS2 subscriptions to VLA output topics
        # ⚠ Update topic names and message types when VLA output is confirmed
        self.create_subscription(
            Float32, "/vla/deceleration_cmd", self._cb_vla_decel, 10)
        self.create_subscription(
            Float32, "/vla/brake_cmd", self._cb_vla_brake, 10)
        self.create_subscription(
            String, "/vla/safety_status", self._cb_vla_status, 10)

        # Stats
        self._rx = {"kinematic": 0, "yolox": 0, "camera": 0}
        self._tx = {"decel": 0, "brake": 0, "status": 0}
        self.create_timer(5.0, self._log_stats)
        self.get_logger().info("OrinZenohBridge ready — Router mode active")

    # ---- Zenoh → ROS2 callbacks ----

    def _zenoh_cb_kinematic(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), Odometry)
            self.pub_kinematic.publish(msg)
            self._rx["kinematic"] += 1
        except Exception as e:
            self.get_logger().warn(f"[kinematic] deserialize error: {e}")

    def _zenoh_cb_yolox(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), CompressedImage)
            self.pub_yolox.publish(msg)
            self._rx["yolox"] += 1
        except Exception as e:
            self.get_logger().warn(f"[yolox] deserialize error: {e}")

    def _zenoh_cb_camera(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), CompressedImage)
            self.pub_camera_raw.publish(msg)
            self._rx["camera"] += 1
        except Exception as e:
            self.get_logger().warn(f"[camera] deserialize error: {e}")

    # ---- ROS2 VLA output → Zenoh callbacks ----

    def _cb_vla_decel(self, msg: Float32):
        raw = serialize_message(msg)
        self.zpub_decel.put(bytes(raw))
        self._tx["decel"] += 1

    def _cb_vla_brake(self, msg: Float32):
        raw = serialize_message(msg)
        self.zpub_brake.put(bytes(raw))
        self._tx["brake"] += 1

    def _cb_vla_status(self, msg: String):
        raw = serialize_message(msg)
        self.zpub_status.put(bytes(raw))
        self._tx["status"] += 1

    def _log_stats(self):
        self.get_logger().info(
            f"[Zenoh RX from Laptop] kinematic={self._rx['kinematic']} "
            f"yolox={self._rx['yolox']} camera={self._rx['camera']}  |  "
            f"[Zenoh TX to NUC] decel={self._tx['decel']} "
            f"brake={self._tx['brake']} status={self._tx['status']}"
        )
        self._rx = {"kinematic": 0, "yolox": 0, "camera": 0}
        self._tx = {"decel": 0, "brake": 0, "status": 0}


def main():
    parser = argparse.ArgumentParser(description="Orin Zenoh Hub (Router)")
    parser.add_argument("--zenoh-port", default=7447, type=int,
                        help="Port for Zenoh router to listen on (default 7447)")
    args = parser.parse_args()

    # ---- Configure Zenoh as ROUTER ----
    # Orin is the central hub — Laptop and NUC both connect to this machine
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"router"')
    cfg.insert_json5("listen/endpoints",
                     f'["tcp/0.0.0.0:{args.zenoh_port}"]')

    print(f"[Zenoh] Starting ROUTER on 0.0.0.0:{args.zenoh_port} ...")
    zs = zenoh.open(cfg)
    print("[Zenoh] Router session opened. Waiting for Laptop and NUC connections...")

    rclpy.init(args=None)
    node = OrinZenohBridge(zenoh_session=zs)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        zs.close()
        print("[Zenoh] Session closed.")


if __name__ == "__main__":
    main()
