import sys
import time
import threading
import queue
import cv2
import numpy as np
from config import SystemConfig
from fast_perception import FastPerceptionLayer
from cognitive_reasoner import CosmosCognitiveReasoner
from safety_controller import SafetyDecisionController
from carla_sensor_bridge import CarlaBridge

class SafeDrivingSystem:
    def __init__(self, config: SystemConfig = SystemConfig()):
        self.config = config
        self.perception = FastPerceptionLayer(
            model_path=config.perception.yolo_model_path,
            conf_thresh=config.perception.conf_threshold
        )
        self.reasoner = CosmosCognitiveReasoner(model_path=config.reasoner.model_path)
        self.controller = SafetyDecisionController()
        self.carla_bridge = CarlaBridge(
            host=config.carla.host,
            port=config.carla.port,
            width=config.carla.image_width,
            height=config.carla.image_height
        )
        self.roi_queue = queue.Queue(maxsize=2)
        self.latest_reasoning = None
        self.running = False

    def _cognitive_reasoning_worker(self):
        print('[System] Cognitive Reasoning Worker active.')
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

    def run(self, max_frames: int = 100000):
        self.running = True
        reasoning_thread = threading.Thread(target=self._cognitive_reasoning_worker, daemon=True)
        reasoning_thread.start()
        
        connected = self.carla_bridge.connect()
        print('=================================================================')
        print(f'JETSON AI ADVISORY & PERCEPTION RUNNING (Connected to {self.config.carla.host})')
        print('=================================================================')
        
        fps_timer = time.time()
        frame_count = 0
        
        try:
            while self.running and (max_frames is None or frame_count < max_frames):
                frame_start = time.time()
                frame = self.carla_bridge.latest_frame if connected else self.carla_bridge.generate_mock_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                targets, annotated_frame = self.perception.detect_and_filter(frame)
                
                if targets and self.roi_queue.empty():
                    primary_target = sorted(targets, key=lambda x: (x['in_path'], x['area_ratio']), reverse=True)[0]
                    self.roi_queue.put((primary_target, frame))
                    
                control_signal = self.controller.compute_control(targets, self.latest_reasoning)
                
                reasoning_msg = self.latest_reasoning.get('reasoning', '') if self.latest_reasoning else ''
                self.carla_bridge.apply_control(
                    throttle=control_signal.throttle,
                    brake=control_signal.brake,
                    steer=control_signal.steer,
                    recommended_speed=control_signal.recommended_speed,
                    risk_level=control_signal.risk_level,
                    action_advisory=control_signal.action_advisory,
                    status_message=control_signal.status_message,
                    reasoning=reasoning_msg,
                    targets=targets
                )
                
                frame_count += 1
                loop_latency_ms = (time.time() - frame_start) * 1000
                
                if frame_count % 15 == 0:
                    fps = frame_count / (time.time() - fps_timer)
                    print(f'[Jetson Frame {frame_count:04d}] FPS: {fps:4.1f} | Latency: {loop_latency_ms:4.1f}ms | Targets: {len(targets)} | RecSpeed: {control_signal.recommended_speed:4.1f} km/h | Risk: {control_signal.risk_level} | {control_signal.action_advisory}')
                          
                time.sleep(max(0.0, 0.033 - (time.time() - frame_start)))
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.carla_bridge.destroy()

if __name__ == '__main__':
    max_f = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    system = SafeDrivingSystem()
    system.run(max_frames=max_f)
