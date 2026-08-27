#!/usr/bin/env python3
"""
Zenoh Image Interface — Lightweight Image Streaming over Zenoh
================================================================
Transmits and receives JPEG-compressed camera frames over Zenoh without ROS 2.

Usage:
    from image_interface import ZenohImagePublisher, ZenohImageSubscriber

    # Publisher (e.g. Laptop or Camera node):
    pub = ZenohImagePublisher(session, topic="aimslab/camera/image_raw")
    pub.publish(frame_bgr, quality=80)

    # Subscriber (e.g. Orin perception node):
    sub = ZenohImageSubscriber(session, topic="aimslab/camera/image_raw", callback=my_frame_handler)
"""

import time
import cv2
import numpy as np
import zenoh


class ZenohImagePublisher:
    """Publishes OpenCV BGR images over Zenoh as compressed JPEG bytes."""

    def __init__(self, session: zenoh.Session, topic: str = "aimslab/camera/image_raw"):
        self.session = session
        self.publisher = session.declare_publisher(topic)

    def publish(self, frame_bgr: np.ndarray, quality: int = 80, frame_id: int = 0):
        """Compress BGR numpy frame to JPEG and transmit payload over Zenoh."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode('.jpg', frame_bgr, encode_params)
        if not success:
            return False

        # Header structure: wall_timestamp (float64) + frame_id (int32) + jpeg_bytes
        timestamp = time.time()
        header = np.array([timestamp], dtype=np.float64).tobytes()
        payload = header + encoded_img.tobytes()

        self.publisher.put(payload)
        return True


class ZenohImageSubscriber:
    """Subscribes to Zenoh image topic and decodes JPEG bytes into OpenCV BGR numpy arrays."""

    def __init__(self, session: zenoh.Session, topic: str = "aimslab/camera/image_raw", callback=None):
        self.session = session
        self.callback = callback
        self.subscriber = session.declare_subscriber(topic, self._on_sample)
        self.latest_frame = None
        self.latest_timestamp = None

    def _on_sample(self, sample: zenoh.Sample):
        try:
            raw_bytes = bytes(sample.payload.to_bytes())
            if len(raw_bytes) < 8:
                return

            # Unpack timestamp (first 8 bytes float64)
            timestamp = np.frombuffer(raw_bytes[:8], dtype=np.float64)[0]
            jpeg_bytes = np.frombuffer(raw_bytes[8:], dtype=np.uint8)

            # Decode JPEG to BGR numpy array
            frame_bgr = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return

            self.latest_frame = frame_bgr
            self.latest_timestamp = timestamp

            if self.callback is not None:
                self.callback(frame_bgr, timestamp)
        except Exception as e:
            print(f"[ZenohImageSubscriber] Decode error: {e}")

    def get_latest(self):
        """Get the latest decoded (frame_bgr, timestamp) or (None, None)."""
        return self.latest_frame, self.latest_timestamp
