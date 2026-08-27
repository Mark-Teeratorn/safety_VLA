#!/usr/bin/env python3
"""
Main Autonomous Safety System - Real Camera Integration Pipeline
================================================================
Implements the exact multi-threaded Autonomous Driving Safety Pipeline as main_simulation.py,
replacing CARLA simulation with live RealSense HD (1280x720) or USB camera capture.

Pipeline Architecture:
  1. Fast Reflex Perception (Level 1): YOLOX/YOLO Fast Perception (30-60 FPS)
  2. Cognitive Safety Reasoner (Level 2): Cosmos-Reason2-2B (Async Thread Worker)
  3. Real-Time HUD Visual Overlay & Zenoh Control Telemetry Command Streaming
"""

import sys
import time
import threading
import queue
import argparse
import cv2
import numpy as np
import zenoh

from config import SystemConfig
from fast_perception import FastPerceptionLayer
from cognitive_reasoner import CosmosCognitiveReasoner

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False


class RealCameraBridge:
    """Interface for RealSense HD or USB camera streams."""

    def __init__(self, camera_type: str = "realsense", width: int = 1280, height: int = 720):
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.rs_pipeline = None
        self.cap = None
        self.connected = False

    def connect((self) -> bool:
        if self.camera_type == "realsense":
            if not _HAS_REALSENSE:
                print("[RealCameraBridge] pyrealsense2 not installed. Falling back to USB camera...")
                self.camera_type = "usb"
            else:
                try:
                    print(f"[RealCameraBridge] Initializing RealSense Camera Pipeline ({self.width}x{self.height} HD)...")
                    self.rs_pipeline = rs.pipeline()
                    cfg = rs.config()
                    cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
                    self.rs_pipeline.start(cfg)
                    self.connected = True
                    return True
                except Exception as e:
                    print(f"[RealCameraBridge] RealSense initialization failed ({e}). Falling back to USB camera...")
                    self.camera_type = "usb"

        if self.camera_type == "usb":
            print(f"[RealCameraBridge] Initializing USB Camera ({self.width}x{self.height} HD)...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.cap.isOpened():
                self.connected = True
                return True
            else:
                print("[RealCameraBridge] Unable to open USB camera!")
                return False
        return False

    def get_frame(self) -> np.ndarray:
        if not self.connected:
            return None

        if self.camera_type == "realsense" and self.rs_pipeline:
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        return None

    def destroy(self):
        if self.rs_pipeline:
            self.rs_pipeline.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


class SafeDrivingRealSystem:
    """Exact SafeDrivingSystem pipeline from main_simulation.py adapted for real camera hardware."""

    def __init__(self, camera_type: str = "realsense", cruise_speed: float = 3.0, config: SystemConfig = SystemConfig()):
        self.config = config
        self.cruise_speed = cruise_speed
        self.perception = FastPerceptionLayer(
            model_path=config.perception.yolo_model_path,
            conf_thresh=config.perception.conf_threshold
        )
        self.reasoner = CosmosCognitiveReasoner(model_path=config.reasoner.model_path)
        self.camera_bridge = RealCameraBridge(camera_type=camera_type, width=1280, height=720)
        
        self.roi_queue = queue.Queue(maxsize=2)
        self.latest_reasoning = None
        self.running = False
        
        # Zenoh Telemetry Publisher
        try:
            cfg_z = zenoh.Config()
            cfg_z.insert_json5("mode", '"router"')
            cfg_z.insert_json5("listen/endpoints", '["tcp/0.0.0.0:7447"]')
            self.zenoh_session = zenoh.open(cfg_z)
            self.pub_control = self.zenoh_session.declare_publisher("aimslab/orin/control_cmd")
            print("[Zenoh] Control publisher online on aimslab/orin/control_cmd.")
        except Exception as e:
            print(f"[Zenoh] Zenoh initialization info ({e}). Continuing standalone...")
            self.zenoh_session = None

    def _cognitive_reasoning_worker(self):
        """Asynchronous Level 2 Cognitive Reasoner thread worker (matching main_simulation.py)."""
        print("[System] Cognitive Reasoning Worker active.")
        while self.running:
            try:
                target_info, frame = self.roi_queue.get(timeout=0.2)
                reasoning_output = self.reasoner.reason_on_roi(target_info, frame)
                self.latest_reasoning = reasoning_output
                self.roi_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def run(self, demo: bool = True, max_frames: int = 100000):
        self.running = True
        reasoning_thread = threading.Thread(target=self._cognitive_reasoning_worker, daemon=True)
        reasoning_thread.start()

        connected = self.camera_bridge.connect()
        print("=================================================================")
        print("REAL-CAMERA JETSON AI ADVISORY & PERCEPTION RUNNING (HD 1280x720)")
        print("=================================================================")

        win_name = "Autonomous Safety Pipeline - Real Camera (HD)"
        if demo:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)

        fps_timer = time.time()
        frame_count = 0

        try:
            while self.running and (max_frames is None or frame_count < max_frames):
                frame_start = time.time()
                frame = self.camera_bridge.get_frame() if connected else None
                if frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]

                # 1. Fast Reflex Perception (Level 1)
                targets, annotated_frame = self.perception.detect_and_filter(frame)

                # 2. Enqueue Primary Candidate to Cognitive Reasoner Queue
                if targets and self.roi_queue.empty():
                    primary_target = sorted(targets, key=lambda x: (x.get('in_path', False), x.get('area_ratio', 0.0)), reverse=True)[0]
                    self.roi_queue.put((primary_target, frame))

                # 3. Compute Adaptive Control & Risk Assessment
                risk_level = "SAFE"
                target_speed = self.cruise_speed
                emergency_brake = False

                if targets:
                    primary = sorted(targets, key=lambda x: (x.get('in_path', False), x.get('area_ratio', 0.0)), reverse=True)[0]
                    area_ratio = primary.get("area_ratio", 0.0)
                    in_path = primary.get("in_path", False)
                    cls_name = primary.get("class", "object")

                    if in_path and area_ratio > 0.08:
                        risk_level = "CRITICAL"
                        target_speed = 0.0
                        emergency_brake = True
                        cot_msg = f"[CoT Reasoning]: 1. Visual Grounding: [{cls_name}] directly blocking lane. 2. Threat: High proximity. 3. Risk: CRITICAL. 4. Control: EMERGENCY BRAKE (0.0 m/s)!"
                    elif in_path or area_ratio > 0.03:
                        risk_level = "WARNING"
                        target_speed = max(1.0, self.cruise_speed * 0.5)
                        cot_msg = f"[CoT Reasoning]: 1. Visual Grounding: [{cls_name}] in path. 2. Threat: Medium distance. 3. Risk: WARNING. 4. Control: SLOW DOWN ({target_speed:.1f} m/s)."
                    else:
                        risk_level = "SAFE"
                        target_speed = self.cruise_speed
                        cot_msg = f"[CoT Reasoning]: 1. Visual Grounding: [{cls_name}] safe zone. 2. Threat: Low. 3. Risk: SAFE. 4. Control: CRUISING ({self.cruise_speed:.1f} m/s)."
                else:
                    cot_msg = "[CoT Reasoning]: 1. Visual Scene: Trajectory clear. 2. Threat: Zero targets. 3. Risk: SAFE. 4. Control: CRUISING (3.0 m/s)."

                # Use background LLM reasoning text if available
                if self.latest_reasoning:
                    reasoning_msg = self.latest_reasoning.get("reasoning", cot_msg)
                else:
                    reasoning_msg = cot_msg

                # 4. Stream Zenoh Telemetry Command Packet
                if hasattr(self, 'pub_control') and self.pub_control:
                    try:
                        import json
                        cmd_packet = {
                            "timestamp": time.time(),
                            "risk_level": risk_level,
                            "target_speed": target_speed,
                            "emergency_brake": emergency_brake,
                            "reasoning": reasoning_msg
                        }
                        self.pub_control.put(json.dumps(cmd_packet).encode())
                    except Exception:
                        pass

                # 5. Draw HUD Banner & Overlay
                cv2.rectangle(annotated_frame, (0, 0), (w, 70), (15, 20, 28), -1)
                cv2.line(annotated_frame, (0, 70), (w, 70), (60, 80, 110), 2)

                r_color = (0, 255, 0) if risk_level == "SAFE" else ((0, 165, 255) if risk_level == "WARNING" else (0, 0, 255))
                cv2.rectangle(annotated_frame, (10, 10), (140, 60), r_color, -1)
                cv2.putText(annotated_frame, risk_level, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.putText(annotated_frame, "AUTONOMOUS SAFETY BRAIN (REAL CAMERA HD)", (155, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                cv2.putText(annotated_frame, f"DECISION: {reasoning_msg}", (155, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 255), 1)

                frame_count += 1
                loop_latency_ms = (time.time() - frame_start) * 1000
                fps = frame_count / max(0.001, (time.time() - fps_timer))

                cv2.putText(annotated_frame, f"{fps:.1f} FPS ({loop_latency_ms:.1f}ms)", (w - 190, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

                if demo:
                    cv2.imshow(win_name, annotated_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

                if frame_count % 30 == 0:
                    print(f"[RealCamera Frame {frame_count:04d}] FPS: {fps:4.1f} | Latency: {loop_latency_ms:4.1f}ms | Risk: {risk_level} | Target Speed: {target_speed:.1f} m/s")

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.camera_bridge.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Safety Pipeline with Real Camera")
    parser.add_argument("--camera", choices=["realsense", "usb"], default="realsense", help="Camera type (realsense or usb)")
    parser.add_argument("--speed", type=float, default=3.0, help="Target cruising speed (m/s)")
    parser.add_argument("--demo", action="store_true", help="Launch live HD GUI window")
    args = parser.parse_args()

    system = SafeDrivingRealSystem(camera_type=args.camera, cruise_speed=args.speed)
    system.run(demo=args.demo)
