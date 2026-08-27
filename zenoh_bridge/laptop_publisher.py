#!/usr/bin/env python3
"""
Zenoh Bridge — LAPTOP SIDE (Publisher)
=======================================
ROS2 Humble — Ubuntu 22.04

New architecture:
  - Camera is connected directly to Orin (RealSense USB)
  - Laptop only streams kinematic state over Zenoh

The Orin pairs each camera frame with the kinematic state having the
nearest wall-clock timestamp (cross-machine NTP sync assumed ~10ms error).

Install:
    pip install eclipse-zenoh --break-system-packages

Usage:
    source /opt/ros/humble/setup.bash
    python3 laptop_publisher.py --orin-ip 192.168.1.20
"""

import argparse
import json
import time

import zenoh
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry

# ---- Zenoh key expression — must match orin_receiver.py ----
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"


class LaptopKinematicPublisher(Node):
    def __init__(self, session: zenoh.Session):
        super().__init__("laptop_kinematic_publisher")
        self._zs = session

        # Zenoh publisher
        self._pub_kinematic = session.declare_publisher(
            KEY_KINEMATIC,
            congestion_control=zenoh.CongestionControl.DROP)

        # QoS: kinematic_state is published RELIABLE by Autoware's EKF node
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.create_subscription(
            Odometry,
            "/localization/kinematic_state",
            self._cb_kinematic,
            qos)

        self._count = 0
        self.create_timer(5.0, self._stats)
        self.get_logger().info("LaptopKinematicPublisher ready → Orin via Zenoh")

    def _cb_kinematic(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        # Include both ROS2 header stamp and wall-clock time for cross-machine sync
        payload = json.dumps({
            "wall_time": time.time(),                   # unix epoch seconds (float)
            "stamp": {
                "sec":     msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec,
            },
            "frame_id":    msg.header.frame_id,
            "position":    {"x": p.x, "y": p.y, "z": p.z},
            "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
            "linear_vel":  {"x": v.x, "y": v.y, "z": v.z},
            "angular_vel": {"x": w.x, "y": w.y, "z": w.z},
        }).encode()

        self._pub_kinematic.put(payload)
        self._count += 1

    def _stats(self):
        hz = self._count / 5.0
        self.get_logger().info(f"[Zenoh TX] kinematic={hz:.1f} Hz")
        self._count = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orin-ip", required=True,
                        help="Orin Ethernet IP (e.g. 192.168.1.20)")
    parser.add_argument("--zenoh-port", default=7447, type=int)
    args = parser.parse_args()

    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("connect/endpoints",
                     f'["tcp/{args.orin_ip}:{args.zenoh_port}"]')

    print(f"[Zenoh] Connecting to Orin at tcp/{args.orin_ip}:{args.zenoh_port} ...")
    zs = zenoh.open(cfg)
    print("[Zenoh] Connected — publishing kinematic state only (camera on Orin)")

    rclpy.init()
    node = LaptopKinematicPublisher(session=zs)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        zs.close()
        print("[Zenoh] Closed.")


if __name__ == "__main__":
    main()
