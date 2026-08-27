#!/usr/bin/env python3
"""
Zenoh Receiver — AGX ORIN (Pure Python, NO ROS2 required)
==========================================================
Ubuntu 24.04 — No ROS2 needed

Subscribes to Zenoh topics published by the Laptop:
  - JPEG image bytes  (raw, cv2-decodable directly)
  - JSON kinematic state

Exposes a simple API for the VLA to consume the latest data.

Install:
    pip install eclipse-zenoh opencv-python --break-system-packages

Usage:
    python3 orin_receiver.py                        # defaults to router on 0.0.0.0:7447
    python3 orin_receiver.py --demo                 # show live images in a window (for testing)
"""

import argparse
import json
import struct
import time
import threading

import cv2
import numpy as np
import zenoh

# ---- Zenoh key expressions — must match laptop_publisher.py ----
KEY_SYNCED    = "aimslab/laptop/camera/synced"
KEY_YOLOX     = "aimslab/laptop/camera/yolox"
KEY_CAMERA    = "aimslab/laptop/camera/raw"
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"


class OrinZenohReceiver:
    """
    Standalone Zenoh receiver for VLA input data.
    No ROS2, no rclpy — just eclipse-zenoh + opencv.

    Usage in VLA code:
        receiver = OrinZenohReceiver()
        receiver.start()
        ...
        raw_frame, yolox_frame = receiver.get_synced_frames() # Guaranteed 100% time-synchronized pair
        state = receiver.get_kinematic_state()               # dict or None
    """

    def __init__(self, zenoh_port: int = 7447):
        self._port = zenoh_port
        self._session = None

        self._lock = threading.Lock()
        self._latest_camera   = None   # numpy BGR
        self._latest_yolox    = None   # numpy BGR
        self._latest_kinematic = None  # dict

        self._counts = {"synced": 0, "camera": 0, "yolox": 0, "kinematic": 0}
        self._subs = []
        self._running = False
        self._stats_thread = None

    def start(self):
        """Open Zenoh session (router mode) and begin receiving."""
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"router"')
        cfg.insert_json5("listen/endpoints",
                         f'["tcp/0.0.0.0:{self._port}"]')

        print(f"[OrinReceiver] Starting Zenoh ROUTER on 0.0.0.0:{self._port} ...")
        self._session = zenoh.open(cfg)
        print("[OrinReceiver] Ready — waiting for Laptop connection...")

        self._subs = [
            self._session.declare_subscriber(KEY_SYNCED,    self._cb_synced),
            self._session.declare_subscriber(KEY_YOLOX,     self._cb_yolox),
            self._session.declare_subscriber(KEY_CAMERA,    self._cb_camera),
            self._session.declare_subscriber(KEY_KINEMATIC, self._cb_kinematic),
        ]

        self._running = True
        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True)
        self._stats_thread.start()

    def stop(self):
        self._running = False
        for s in self._subs:
            s.undeclare()
        if self._session:
            self._session.close()
        print("[OrinReceiver] Closed.")

    # ---- Public API for VLA ----

    def get_synced_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns (raw_camera_frame, yolox_frame) as BGR numpy arrays.
        Guaranteed to be matched to the exact same point in time.
        """
        with self._lock:
            cam = self._latest_camera.copy() if self._latest_camera is not None else None
            yolox = self._latest_yolox.copy() if self._latest_yolox is not None else None
            return cam, yolox

    def get_camera_frame(self) -> np.ndarray | None:
        """Returns latest raw camera frame as BGR numpy array, or None."""
        with self._lock:
            return self._latest_camera.copy() if self._latest_camera is not None else None

    def get_yolox_frame(self) -> np.ndarray | None:
        """Returns latest YOLOX detection frame as BGR numpy array, or None."""
        with self._lock:
            return self._latest_yolox.copy() if self._latest_yolox is not None else None

    def get_kinematic_state(self) -> dict | None:
        """Returns latest kinematic state as dict, or None.
        Keys: stamp, frame_id, position, orientation, linear_vel, angular_vel
        """
        with self._lock:
            return dict(self._latest_kinematic) if self._latest_kinematic else None

    # ---- Zenoh callbacks ----

    def _cb_synced(self, sample: zenoh.Sample):
        try:
            payload = bytes(sample.payload.to_bytes())
            if len(payload) < 8:
                return
            len_raw, len_yolox = struct.unpack("<II", payload[:8])
            if len(payload) < 8 + len_raw + len_yolox:
                return

            raw_bytes = payload[8 : 8 + len_raw]
            yolox_bytes = payload[8 + len_raw : 8 + len_raw + len_yolox]

            img_raw = self._decode_bytes(raw_bytes)
            img_yolox = self._decode_bytes(yolox_bytes)

            with self._lock:
                if img_raw is not None:
                    self._latest_camera = img_raw
                if img_yolox is not None:
                    self._latest_yolox = img_yolox
            self._counts["synced"] += 1
        except Exception as e:
            print(f"[OrinReceiver] synced packet decode error: {e}")

    def _cb_camera(self, sample: zenoh.Sample):
        img = self._decode_jpeg(sample)
        if img is not None:
            with self._lock:
                self._latest_camera = img
            self._counts["camera"] += 1

    def _cb_yolox(self, sample: zenoh.Sample):
        img = self._decode_jpeg(sample)
        if img is not None:
            with self._lock:
                self._latest_yolox = img
            self._counts["yolox"] += 1

    def _cb_kinematic(self, sample: zenoh.Sample):
        try:
            state = json.loads(bytes(sample.payload.to_bytes()).decode())
            with self._lock:
                self._latest_kinematic = state
            self._counts["kinematic"] += 1
        except Exception as e:
            print(f"[OrinReceiver] kinematic parse error: {e}")

    def _decode_bytes(self, buf_bytes: bytes) -> np.ndarray | None:
        try:
            buf = np.frombuffer(buf_bytes, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[OrinReceiver] JPEG decode error: {e}")
            return None

    def _decode_jpeg(self, sample: zenoh.Sample) -> np.ndarray | None:
        return self._decode_bytes(bytes(sample.payload.to_bytes()))

    def _stats_loop(self):
        while self._running:
            time.sleep(5.0)
            with self._lock:
                c = dict(self._counts)
                self._counts = {"synced": 0, "camera": 0, "yolox": 0, "kinematic": 0}
            print(f"[OrinReceiver] RX — synced={c['synced']/5.0:.1f} FPS | "
                  f"camera={c['camera']/5.0:.1f} FPS | "
                  f"yolox={c['yolox']/5.0:.1f} FPS | "
                  f"kinematic={c['kinematic']/5.0:.1f} Hz")


# ---- Standalone demo / test mode ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=7447, type=int)
    parser.add_argument("--demo", action="store_true",
                        help="Show live camera window (test mode)")
    args = parser.parse_args()

    receiver = OrinZenohReceiver(zenoh_port=args.port)
    receiver.start()

    print("[OrinReceiver] Press Ctrl+C to stop.")
    try:
        while True:
            if args.demo:
                frame, yolox = receiver.get_synced_frames()
                state = receiver.get_kinematic_state()

                if frame is not None:
                    cv2.imshow("Camera Raw", frame)
                if yolox is not None:
                    cv2.imshow("YOLOX", yolox)
                if state is not None:
                    p = state.get("position", {})
                    v = state.get("linear_vel", {})
                    print(f"  pos=({p.get('x',0):.2f}, {p.get('y',0):.2f}) "
                          f"vel=({v.get('x',0):.2f}, {v.get('y',0):.2f})",
                          end="\r")
                if cv2.waitKey(33) == ord("q"):
                    break
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        receiver.stop()


if __name__ == "__main__":
    main()
