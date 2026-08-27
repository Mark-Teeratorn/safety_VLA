#!/usr/bin/env python3
"""
VLA Cosmos Simulation & Reasoning Engine — Laptop / NUC
=========================================================
Consumes real-time perception detections (YOLOX) and kinematic state over Zenoh.
Runs Vision-Language-Action (VLA) safety reasoning and outputs control commands.

Subscribes:
  - aimslab/orin/perception/objects (YOLOX detections + Orin timestamps)
  - aimslab/laptop/localization/kinematic_state (Vehicle pose & velocity)

Publishes:
  - aimslab/orin/control_cmd (VLA target speed, steering, emergency brake trigger)

Usage:
    python3 zenoh_bridge/vla_cosmos.py --orin-ip 192.168.1.20
    python3 zenoh_bridge/vla_cosmos.py --orin-ip 192.168.1.20 --dashboard
"""

import argparse
import json
import time
import threading
import numpy as np
import cv2
import zenoh

# ---- Zenoh Topics ----
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"
KEY_PERCEPTION = "aimslab/orin/perception/objects"
KEY_CONTROL = "aimslab/orin/control_cmd"


class VLACosmosReasoningEngine:
    """Vision-Language-Action (VLA) Safety & Simulation Reasoner."""

    def __init__(self, target_speed_mps: float = 3.0):
        self.target_speed = target_speed_mps  # Standard cruising speed in m/s (~10 km/h)
        self.current_speed = 0.0
        self.risk_level = "SAFE"  # SAFE, WARNING, CRITICAL
        self.active_obstacles = []
        self.last_decision_reason = "System initialized. Path clear."

    def evaluate_scene(self, detections: list, kinematic_data: dict = None):
        """
        VLA Scene Assessment:
        - Evaluates detected objects (PEDESTRIAN, CAR, BUS, TRUCK, BICYCLE, MOTORCYCLE).
        - Estimates obstacle proximity based on bounding box size and position in frame.
        - Outputs target velocity and emergency brake flags.
        """
        if kinematic_data:
            self.current_speed = kinematic_data.get("twist", {}).get("linear", {}).get("x", 0.0)

        if not detections:
            self.risk_level = "SAFE"
            self.last_decision_reason = "No obstacles detected in field of view. Cruising."
            return {"target_speed": self.target_speed, "steering_angle": 0.0, "emergency_brake": False, "reason": self.last_decision_reason}

        high_priority_threats = []
        max_box_area = 0.0

        for det in detections:
            label = det.get("label", "UNKNOWN")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            
            x1, y1, x2, y2 = bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height
            
            # Normalize area relative to 640x480 frame (307,200 total pixels)
            norm_area = area / 307200.0
            max_box_area = max(max_box_area, norm_area)

            # Pedestrians and Cyclists have highest vulnerability weight
            weight = 2.0 if label in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"] else 1.0
            risk_score = norm_area * weight * conf

            if risk_score > 0.05:  # Threshold for notable proximity
                high_priority_threats.append({
                    "label": label,
                    "confidence": conf,
                    "norm_area": norm_area,
                    "risk_score": risk_score
                })

        # Sort threats by risk score descending
        high_priority_threats.sort(key=lambda t: t["risk_score"], reverse=True)
        self.active_obstacles = high_priority_threats

        # Reasoning Logic:
        if not high_priority_threats:
            self.risk_level = "SAFE"
            self.last_decision_reason = "Obstacles distant. Maintaining cruise speed."
            return {"target_speed": self.target_speed, "steering_angle": 0.0, "emergency_brake": False, "reason": self.last_decision_reason}

        top_threat = high_priority_threats[0]
        top_label = top_threat["label"]
        top_score = top_threat["risk_score"]

        if top_score > 0.25:
            # Immediate Critical Threat -> Emergency Stop
            self.risk_level = "CRITICAL"
            self.last_decision_reason = f"CRITICAL HAZARD: Near {top_label} (Risk {top_score:.2f})! Emergency Brake Activated."
            return {"target_speed": 0.0, "steering_angle": 0.0, "emergency_brake": True, "reason": self.last_decision_reason}

        elif top_score > 0.10:
            # Moderate Warning -> Slow Down
            reduced_speed = max(0.5, self.target_speed * 0.4)
            self.risk_level = "WARNING"
            self.last_decision_reason = f"WARNING: Approaching {top_label} (Risk {top_score:.2f}). Decelerating to {reduced_speed:.1f} m/s."
            return {"target_speed": reduced_speed, "steering_angle": 0.0, "emergency_brake": False, "reason": self.last_decision_reason}

        else:
            self.risk_level = "SAFE"
            self.last_decision_reason = f"NOTICE: Tracked {top_label} at safe distance."
            return {"target_speed": self.target_speed, "steering_angle": 0.0, "emergency_brake": False, "reason": self.last_decision_reason}


class VLAZenohSimulator:
    """Zenoh Data Bridge & Interactive Simulation Manager."""

    def __init__(self, orin_ip: str, zenoh_port: int = 7447):
        self.orin_ip = orin_ip
        self.zenoh_port = zenoh_port
        self.reasoner = VLACosmosReasoningEngine()
        self.latest_perception = None
        self.latest_kinematic = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        print(f"[VLA Cosmos] Connecting to Zenoh router at {self.orin_ip}:{self.zenoh_port}...")
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"client"')
        cfg.insert_json5("connect/endpoints", f'["tcp/{self.orin_ip}:{self.zenoh_port}"]')

        self.session = zenoh.open(cfg)
        print("[VLA Cosmos] Connected to Orin Perception Stream successfully.")

        # Declare Subscribers & Publishers
        self.sub_perception = self.session.declare_subscriber(KEY_PERCEPTION, self._cb_perception)
        self.sub_kinematic = self.session.declare_subscriber(KEY_KINEMATIC, self._cb_kinematic)
        self.pub_control = self.session.declare_publisher(KEY_CONTROL)
        self.running = True

    def _cb_perception(self, sample: zenoh.Sample):
        try:
            payload = json.loads(sample.payload.to_bytes().decode())
            with self.lock:
                self.latest_perception = payload

            # Run VLA Reasoning on incoming frame
            detections = payload.get("detections", [])
            kinematic = payload.get("kinematic", None)
            cmd = self.reasoner.evaluate_scene(detections, kinematic)

            # Publish Control Command back to Vehicle Controller
            control_payload = json.dumps({
                "timestamp": time.time(),
                "command": cmd
            }).encode()
            self.pub_control.put(control_payload)

        except Exception as e:
            print(f"[VLA Cosmos] Perception callback error: {e}")

    def _cb_kinematic(self, sample: zenoh.Sample):
        try:
            payload = json.loads(sample.payload.to_bytes().decode())
            with self.lock:
                self.latest_kinematic = payload
        except Exception as e:
            print(f"[VLA Cosmos] Kinematic callback error: {e}")

    def render_dashboard(self):
        """Render GUI Dashboard showing real-time VLA simulation telemetry."""
        canvas = np.zeros((500, 800, 3), dtype=np.uint8)
        
        with self.lock:
            perception = self.latest_perception
            kinematic = self.latest_kinematic

        # Color palette
        bg_color = (20, 24, 30)
        panel_color = (35, 42, 54)
        border_color = (60, 75, 95)
        text_color = (240, 245, 255)
        
        canvas[:] = bg_color

        # Header Title
        cv2.rectangle(canvas, (10, 10), (790, 60), panel_color, -1)
        cv2.rectangle(canvas, (10, 10), (790, 60), border_color, 1)
        cv2.putText(canvas, "VLA COSMOS SAFETY REASONING DASHBOARD", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)

        # Risk Status Box
        risk = self.reasoner.risk_level
        risk_color = (0, 255, 0) if risk == "SAFE" else ((0, 165, 255) if risk == "WARNING" else (0, 0, 255))
        
        cv2.rectangle(canvas, (10, 75), (390, 175), panel_color, -1)
        cv2.rectangle(canvas, (10, 75), (390, 175), border_color, 1)
        cv2.putText(canvas, "SAFETY RISK LEVEL:", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        cv2.putText(canvas, risk, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)

        # VLA Reason Box
        cv2.rectangle(canvas, (400, 75), (790, 175), panel_color, -1)
        cv2.rectangle(canvas, (400, 75), (790, 175), border_color, 1)
        cv2.putText(canvas, "VLA DECISION REASONING:", (415, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        
        # Word wrap reasoning text
        reason_text = self.reasoner.last_decision_reason
        words = reason_text.split(" ")
        line1, line2 = "", ""
        for word in words:
            if len(line1 + " " + word) < 35:
                line1 += " " + word
            else:
                line2 += " " + word
        cv2.putText(canvas, line1.strip(), (415, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)
        if line2:
            cv2.putText(canvas, line2.strip(), (415, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)

        # Detections Table
        cv2.rectangle(canvas, (10, 190), (790, 480), panel_color, -1)
        cv2.rectangle(canvas, (10, 190), (790, 480), border_color, 1)
        cv2.putText(canvas, "ACTIVE OBJECT PERCEPTION STREAM (YOLOX AGX ORIN)", (25, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        headers = ["LABEL", "CONFIDENCE", "BBOX [x1, y1, x2, y2]", "LATENCY"]
        cv2.putText(canvas, f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<30} {headers[3]}", (25, 250),
                    cv2.FONT_HERSHEY_MONOSPACE, 0.45, (150, 175, 200), 1)
        cv2.line(canvas, (25, 260), (775, 260), border_color, 1)

        if perception:
            dets = perception.get("detections", [])
            inf_ms = perception.get("inference_ms", 0.0)
            y_offset = 285
            for i, d in enumerate(dets[:7]):  # Display top 7
                lbl = d["label"]
                conf = f"{d['confidence']:.2f}"
                box_str = str([int(v) for v in d["bbox"]])
                lat_str = f"{inf_ms:.1f}ms"
                line_str = f"{lbl:<15} {conf:<12} {box_str:<30} {lat_str}"
                cv2.putText(canvas, line_str, (25, y_offset), cv2.FONT_HERSHEY_MONOSPACE, 0.45, (0, 255, 180), 1)
                y_offset += 25
        else:
            cv2.putText(canvas, "Waiting for perception stream from Orin...", (25, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 140, 160), 1)

        return canvas

    def stop(self):
        self.running = False
        if hasattr(self, 'session'):
            self.session.close()
        print("[VLA Cosmos] Simulator stopped.")


def main():
    parser = argparse.ArgumentParser(description="VLA Cosmos Simulation & Safety Reasoner")
    parser.add_argument("--orin-ip", default="192.168.1.20", help="IP address of AGX Orin Zenoh router")
    parser.add_argument("--port", type=int, default=7447, help="Zenoh port (default: 7447)")
    parser.add_argument("--dashboard", action="store_true", help="Launch live OpenCV simulation dashboard")
    args = parser.parse_args()

    simulator = VLAZenohSimulator(args.orin_ip, args.port)
    simulator.start()

    print("[VLA Cosmos] Engine active. Press Ctrl+C to exit.")
    try:
        while True:
            if args.dashboard:
                dashboard_img = simulator.render_dashboard()
                cv2.imshow("VLA Cosmos Simulation Dashboard", dashboard_img)
                if cv2.waitKey(30) == ord('q'):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if args.dashboard:
            cv2.destroyAllWindows()
        simulator.stop()


if __name__ == "__main__":
    main()
