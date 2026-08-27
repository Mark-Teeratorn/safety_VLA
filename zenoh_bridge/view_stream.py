#!/usr/bin/env python3
"""
Zenoh Perception Stream Viewer — Laptop Client
==============================================
Subscribes to real-time YOLOX perception data and kinematic state
streamed from the AGX Orin over Zenoh (pure JSON metadata, NO images).

Usage on Laptop:
    python3 view_stream.py --orin-ip 192.168.1.20
"""

import argparse
import json
import time
import zenoh

KEY_PERCEPTION = "aimslab/orin/perception/objects"


def main():
    parser = argparse.ArgumentParser(description="Live Perception Data Viewer over Zenoh")
    parser.add_argument("--orin-ip", default="192.168.1.20", help="IP address of Orin Zenoh router")
    parser.add_argument("--port", default=7447, type=int, help="Zenoh router port (default 7447)")
    parser.add_argument("--topic", default=KEY_PERCEPTION, help="Zenoh perception topic")
    args = parser.parse_args()

    # Open Zenoh client connection to Orin
    cfg = zenoh.Config()
    endpoint = f"tcp/{args.orin_ip}:{args.port}"
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    print(f"[Viewer] Connecting to Orin Zenoh router at {endpoint}...")

    session = zenoh.open(cfg)
    print(f"[Viewer] Connected! Listening to real-time perception stream on: {args.topic}")
    print("=" * 70)

    count = 0

    def on_sample(sample: zenoh.Sample):
        nonlocal count
        try:
            payload = json.loads(bytes(sample.payload.to_bytes()).decode())
            ts = payload.get("timestamp", time.time())
            latency_ms = (time.time() - ts) * 1000
            inf_ms = payload.get("inference_ms", 0.0)
            detections = payload.get("detections", [])
            kinematic = payload.get("kinematic", {})

            count += 1
            print(f"\r[{count:04d}] Latency: {latency_ms:4.1f}ms | GPU Inf: {inf_ms:4.1f}ms | Objects: {len(detections)}", end="")

            # Print detailed breakdown every 10 frames
            if count % 15 == 0 or len(detections) > 0:
                det_str = ", ".join([f"{d['label']}({d['confidence']:.2f})" for d in detections]) if detections else "None"
                pos = kinematic.get("position", {}) if kinematic else {}
                pos_str = f"x:{pos.get('x',0):.2f}, y:{pos.get('y',0):.2f}" if pos else "N/A"
                print(f"\n   └── Detections: [{det_str}] | Kinematic Pos: [{pos_str}]")
        except Exception as e:
            print(f"\n[Viewer] Parse error: {e}")

    sub = session.declare_subscriber(args.topic, on_sample)
    print("[Viewer] Real-time stream active. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sub.undeclare()
        session.close()
        print("\n[Viewer] Stopped.")


if __name__ == "__main__":
    main()
