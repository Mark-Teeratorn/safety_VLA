#!/usr/bin/env python3
"""
Zenoh Receiver + RealSense Camera — AGX ORIN (Pure Python, NO ROS2 required)
==============================================================================
Ubuntu 24.04 — No ROS2 needed

Architecture:
  - RealSense camera connected directly to Orin via USB
  - Kinematic state streamed from Laptop over Zenoh (with wall-clock timestamps)
  - Orin pairs each camera frame with the kinematic state nearest in wall time

Timestamp sync strategy:
  - Both machines are NTP-synced (chrony recommended, ~1-5ms accuracy on LAN)
  - Laptop stamps each kinematic message with time.time() (unix epoch, float)
  - Orin stamps each camera frame with time.time() at capture time
  - Orin holds a rolling buffer of kinematic states; for each frame it picks
    the entry with minimum |t_frame - t_kinematic|

Install:
    pip install eclipse-zenoh opencv-python pyrealsense2 --break-system-packages

Usage:
    python3 orin_receiver.py                        # router mode (default)
    python3 orin_receiver.py --demo                 # show live windows (test)
    python3 orin_receiver.py --max-sync-gap 0.2     # reject pairs >200ms apart
"""

import argparse
import collections
import json
import time
import threading

import cv2
import numpy as np
import zenoh

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False
    print("[OrinReceiver] WARNING: pyrealsense2 not installed — camera disabled")

# ---- Zenoh key expression — must match laptop_publisher.py ----
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"

# Max wall-clock gap (seconds) between frame and kinematic state to be considered "synced"
DEFAULT_MAX_SYNC_GAP = 0.3   # 300ms is generous; reduce if NTP is tight


class OrinReceiver:
    """
    Standalone receiver for the VLA.
    - Captures frames from RealSense (USB, directly on Orin)
    - Receives kinematic state from Laptop via Zenoh
    - Matches them by nearest wall-clock timestamp

    Usage in VLA code:
        receiver = OrinReceiver()
        receiver.start()

        # Blocks until a new synced pair is available (or timeout)
        result = receiver.get_synced()
        if result is not None:
            frame_bgr, kinematic_state, sync_gap_s = result
            # sync_gap_s: how far apart in wall-clock time they were (seconds)
    """

    def __init__(self,
                 zenoh_port: int = 7447,
                 max_sync_gap: float = DEFAULT_MAX_SYNC_GAP,
                 kinematic_buffer_size: int = 60,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):

        self._port = zenoh_port
        self._max_sync_gap = max_sync_gap
        self._width = width
        self._height = height
        self._fps = fps

        self._session = None
        self._lock = threading.Lock()

        # Rolling buffer: deque of (wall_time_float, state_dict)
        self._kinematic_buf: collections.deque = collections.deque(
            maxlen=kinematic_buffer_size)

        # Latest synced output (frame_bgr, kinematic_dict, gap_s)
        self._latest_synced: tuple | None = None

        self._counts = {"camera": 0, "kinematic": 0, "synced": 0, "dropped": 0}
        self._subs = []
        self._running = False

        # RealSense pipeline
        self._rs_pipeline = None
        self._camera_thread = None
        self._stats_thread = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        """Start Zenoh router, RealSense capture, and stats thread."""
        self._running = True

        # 1. Zenoh
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"router"')
        cfg.insert_json5("listen/endpoints",
                         f'["tcp/0.0.0.0:{self._port}"]')
        print(f"[OrinReceiver] Starting Zenoh ROUTER on 0.0.0.0:{self._port} ...")
        self._session = zenoh.open(cfg)
        self._subs = [
            self._session.declare_subscriber(KEY_KINEMATIC, self._cb_kinematic),
        ]
        print("[OrinReceiver] Zenoh ready — waiting for Laptop kinematic state...")

        # 2. RealSense
        if _HAS_REALSENSE:
            self._camera_thread = threading.Thread(
                target=self._realsense_loop, daemon=True, name="rs-capture")
            self._camera_thread.start()
        else:
            print("[OrinReceiver] RealSense disabled — no camera thread started")

        # 3. Stats
        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True, name="stats")
        self._stats_thread.start()

    def stop(self):
        self._running = False
        if self._rs_pipeline:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
        for s in self._subs:
            s.undeclare()
        if self._session:
            self._session.close()
        print("[OrinReceiver] Closed.")

    # -----------------------------------------------------------------------
    # Public API for VLA
    # -----------------------------------------------------------------------

    def get_synced(self, timeout: float = 1.0) -> tuple | None:
        """
        Returns (frame_bgr, kinematic_state, sync_gap_s) or None.

        frame_bgr        : numpy BGR array from RealSense
        kinematic_state  : dict with keys position, orientation, linear_vel, etc.
        sync_gap_s       : wall-clock gap between frame and kinematic state (seconds)

        Blocks up to `timeout` seconds waiting for a new synced pair.
        Returns None if timeout expires without a new pair.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_synced is not None:
                    result = self._latest_synced
                    self._latest_synced = None   # consume it
                    return result
            time.sleep(0.005)
        return None

    def get_latest_frame(self) -> np.ndarray | None:
        """Returns the most recent camera frame (BGR) regardless of kinematic sync."""
        with self._lock:
            if self._latest_synced is not None:
                return self._latest_synced[0].copy()
        return None

    def get_latest_kinematic(self) -> dict | None:
        """Returns the most recent kinematic state regardless of camera sync."""
        with self._lock:
            if self._kinematic_buf:
                return dict(self._kinematic_buf[-1][1])
        return None

    # -----------------------------------------------------------------------
    # RealSense capture loop
    # -----------------------------------------------------------------------

    def _realsense_loop(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self._width, self._height,
            rs.format.bgr8,
            self._fps)

        print(f"[RealSense] Starting pipeline {self._width}x{self._height} @ {self._fps}fps ...")
        try:
            pipeline.start(config)
            self._rs_pipeline = pipeline
            print("[RealSense] Pipeline started.")
        except Exception as e:
            print(f"[RealSense] Failed to start pipeline: {e}")
            return

        while self._running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Wall-clock time of this frame capture
            frame_wall_time = time.time()
            img = np.asanyarray(color_frame.get_data())   # BGR uint8

            self._counts["camera"] += 1

            # Match to nearest kinematic state
            synced = self._match_kinematic(img, frame_wall_time)
            if synced is not None:
                with self._lock:
                    self._latest_synced = synced
                self._counts["synced"] += 1
            else:
                self._counts["dropped"] += 1

        try:
            pipeline.stop()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Kinematic matching
    # -----------------------------------------------------------------------

    def _match_kinematic(self, frame_bgr: np.ndarray, frame_time: float) -> tuple | None:
        """Find the kinematic state nearest in wall time to frame_time."""
        with self._lock:
            if not self._kinematic_buf:
                return None
            buf_snapshot = list(self._kinematic_buf)

        # Find entry with minimum |t_frame - t_kinematic|
        best_t, best_state = min(buf_snapshot, key=lambda x: abs(x[0] - frame_time))
        gap = abs(frame_time - best_t)

        if gap > self._max_sync_gap:
            return None   # Too far apart — kinematic state stale or not yet arrived

        return (frame_bgr, best_state, gap)

    # -----------------------------------------------------------------------
    # Zenoh callbacks
    # -----------------------------------------------------------------------

    def _cb_kinematic(self, sample: zenoh.Sample):
        try:
            state = json.loads(bytes(sample.payload.to_bytes()).decode())
            wall_time = state.get("wall_time", time.time())  # fallback if missing
            with self._lock:
                self._kinematic_buf.append((wall_time, state))
            self._counts["kinematic"] += 1
        except Exception as e:
            print(f"[OrinReceiver] kinematic parse error: {e}")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def _stats_loop(self):
        while self._running:
            time.sleep(5.0)
            with self._lock:
                c = dict(self._counts)
                self._counts = {"camera": 0, "kinematic": 0, "synced": 0, "dropped": 0}
            print(
                f"[OrinReceiver] camera={c['camera']/5.0:.1f} FPS | "
                f"kinematic={c['kinematic']/5.0:.1f} Hz | "
                f"synced={c['synced']/5.0:.1f} FPS | "
                f"dropped={c['dropped']/5.0:.1f} FPS")


# ---------------------------------------------------------------------------
# Standalone demo / test mode
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=7447, type=int,
                        help="Zenoh router port (default 7447)")
    parser.add_argument("--max-sync-gap", default=DEFAULT_MAX_SYNC_GAP, type=float,
                        help="Max wall-clock gap in seconds for a valid sync pair (default 0.3)")
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--demo", action="store_true",
                        help="Show live camera + kinematic window")
    args = parser.parse_args()

    receiver = OrinReceiver(
        zenoh_port=args.port,
        max_sync_gap=args.max_sync_gap,
        width=args.width,
        height=args.height,
        fps=args.fps)
    receiver.start()

    print("[OrinReceiver] Press Ctrl+C to stop.")
    try:
        while True:
            if args.demo:
                result = receiver.get_synced(timeout=0.1)
                if result is not None:
                    frame, state, gap = result
                    p = state.get("position", {})
                    v = state.get("linear_vel", {})

                    # Overlay sync gap on frame
                    overlay = frame.copy()
                    cv2.putText(overlay,
                                f"sync_gap={gap*1000:.0f}ms  "
                                f"pos=({p.get('x',0):.2f},{p.get('y',0):.2f})  "
                                f"vel=({v.get('x',0):.2f},{v.get('y',0):.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    cv2.imshow("RealSense + Kinematic (synced)", overlay)

                if cv2.waitKey(1) == ord("q"):
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
