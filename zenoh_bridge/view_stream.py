#!/usr/bin/env python3
"""
Zenoh Stream Viewer — Laptop Client
==================================
Subscribes to live annotated camera video stream from AGX Orin over Zenoh
and displays it live on your laptop screen.

Usage on Laptop:
    python3 view_stream.py --orin-ip 192.168.1.20
"""

import argparse
import time
import cv2
import numpy as np
import zenoh


def main():
    parser = argparse.ArgumentParser(description="Live Video Stream Viewer over Zenoh")
    parser.add_argument("--orin-ip", default="192.168.1.20", help="IP address of Orin Zenoh router")
    parser.add_argument("--port", default=7447, type=int, help="Zenoh router port (default 7447)")
    parser.add_argument("--topic", default="aimslab/orin/perception/image_annotated", help="Zenoh image topic")
    args = parser.parse_args()

    # Open Zenoh client connection to Orin
    cfg = zenoh.Config()
    endpoint = f"tcp/{args.orin_ip}:{args.port}"
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    print(f"[Viewer] Connecting to Orin Zenoh router at {endpoint}...")

    session = zenoh.open(cfg)
    print(f"[Viewer] Connected! Subscribing to topic: {args.topic}")

    last_time = time.time()
    frame_count = 0

    has_gui = True

    def on_image_sample(sample: zenoh.Sample):
        nonlocal last_time, frame_count, has_gui
        try:
            raw_bytes = bytes(sample.payload.to_bytes())
            if len(raw_bytes) < 8:
                return

            # Header timestamp
            timestamp = np.frombuffer(raw_bytes[:8], dtype=np.float64)[0]
            jpeg_data = np.frombuffer(raw_bytes[8:], dtype=np.uint8)

            frame_bgr = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                latency_ms = (time.time() - timestamp) * 1000
                cv2.putText(frame_bgr, f"Network Latency: {latency_ms:.1f}ms",
                            (10, frame_bgr.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frame_count += 1

                if has_gui:
                    try:
                        cv2.imshow("Orin Real-Time YOLOX Stream", frame_bgr)
                    except Exception as gui_err:
                        print(f"[Viewer] GUI window not available on laptop ({gui_err}).")
                        print("[Viewer] Fallback: Saving live frame to 'latest_stream.jpg' continuously...")
                        has_gui = False
                        cv2.imwrite("latest_stream.jpg", frame_bgr)
                else:
                    cv2.imwrite("latest_stream.jpg", frame_bgr)
                    if frame_count % 30 == 0:
                        print(f"[Viewer] Stream active: Received {frame_count} frames | Latency: {latency_ms:.1f}ms | Saved to latest_stream.jpg")
        except Exception as e:
            print(f"[Viewer] Sample error: {e}")

    sub = session.declare_subscriber(args.topic, on_image_sample)
    print("[Viewer] Streaming video feed... Press Ctrl+C to stop.")

    try:
        while True:
            if has_gui:
                try:
                    if cv2.waitKey(1) == ord('q'):
                        break
                except Exception:
                    pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if has_gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        sub.undeclare()
        session.close()
        print("[Viewer] Stopped.")


if __name__ == "__main__":
    main()
