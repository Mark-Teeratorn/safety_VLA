#!/usr/bin/env python3
"""
Zenoh ROS2 Bridge — INTEL NUC (Subscriber)
============================================
Receives VLA deceleration/braking commands from AGX Orin via Zenoh,
deserializes CDR bytes, and republishes as local ROS2 topics.

The NUC's vehicle control node subscribes to these local topics normally.

Usage:
    source /opt/ros/<distro>/setup.bash
    pip install eclipse-zenoh
    python3 nuc_subscriber.py --orin-ip 192.168.1.20
"""

import argparse

import zenoh
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

from std_msgs.msg import Float32, String  # ← update to real VLA msg types when confirmed

# ---- Zenoh key expressions — must match orin_bridge.py ----
KEY_VLA_DECEL   = "ros2/vla/deceleration_cmd"
KEY_VLA_BRAKE   = "ros2/vla/brake_cmd"
KEY_VLA_STATUS  = "ros2/vla/safety_status"


class NucZenohSubscriber(Node):
    def __init__(self, zenoh_session: zenoh.Session):
        super().__init__("nuc_zenoh_subscriber")
        self.zs = zenoh_session

        # ---- Local ROS2 publishers ----
        # These topics are what the vehicle control node on NUC subscribes to
        self.pub_decel = self.create_publisher(
            Float32, "/vla/deceleration_cmd", 10)
        self.pub_brake = self.create_publisher(
            Float32, "/vla/brake_cmd", 10)
        self.pub_status = self.create_publisher(
            String, "/vla/safety_status", 10)

        # ---- Zenoh subscribers ----
        self._sub_decel = self.zs.declare_subscriber(
            KEY_VLA_DECEL, self._zenoh_cb_decel)
        self._sub_brake = self.zs.declare_subscriber(
            KEY_VLA_BRAKE, self._zenoh_cb_brake)
        self._sub_status = self.zs.declare_subscriber(
            KEY_VLA_STATUS, self._zenoh_cb_status)

        self._counts = {"decel": 0, "brake": 0, "status": 0}
        self.create_timer(5.0, self._log_stats)
        self.get_logger().info("NucZenohSubscriber ready — listening for VLA commands from Orin")

    def _zenoh_cb_decel(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), Float32)
            self.pub_decel.publish(msg)
            self._counts["decel"] += 1
        except Exception as e:
            self.get_logger().warn(f"[decel] deserialize error: {e}")

    def _zenoh_cb_brake(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), Float32)
            self.pub_brake.publish(msg)
            self._counts["brake"] += 1
        except Exception as e:
            self.get_logger().warn(f"[brake] deserialize error: {e}")

    def _zenoh_cb_status(self, sample: zenoh.Sample):
        try:
            msg = deserialize_message(bytes(sample.payload.to_bytes()), String)
            self.pub_status.publish(msg)
            self._counts["status"] += 1
        except Exception as e:
            self.get_logger().warn(f"[status] deserialize error: {e}")

    def _log_stats(self):
        self.get_logger().info(
            f"[Zenoh RX from Orin] decel={self._counts['decel']} "
            f"brake={self._counts['brake']} status={self._counts['status']}"
        )
        self._counts = {"decel": 0, "brake": 0, "status": 0}


def main():
    parser = argparse.ArgumentParser(description="NUC → receive VLA commands from Orin via Zenoh")
    parser.add_argument("--orin-ip", required=True,
                        help="AGX Orin Ethernet IP facing the NUC (e.g. 192.168.1.20)")
    parser.add_argument("--zenoh-port", default=7447, type=int)
    args = parser.parse_args()

    # ---- Configure Zenoh as PEER connecting to Orin ROUTER ----
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("connect/endpoints",
                     f'["tcp/{args.orin_ip}:{args.zenoh_port}"]')

    print(f"[Zenoh] Connecting to Orin router at tcp/{args.orin_ip}:{args.zenoh_port} ...")
    zs = zenoh.open(cfg)
    print("[Zenoh] Connected. Waiting for VLA commands...")

    rclpy.init(args=None)
    node = NucZenohSubscriber(zenoh_session=zs)

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
