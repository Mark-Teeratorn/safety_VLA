#!/usr/bin/env python3
"""
FastVLM Live OpenCV Desktop GUI Window with Real-Time Inference Latency Benchmarking.
Displays live camera stream, VLA visual safety decision HUD, and exact GPU inference latency (ms) & FPS.
"""

import time
import os
import cv2
import torch
import numpy as np
from PIL import Image

# FastVLM Native HuggingFace imports
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = "apple/FastVLM-0.5B"

class FastVLMLocalGUI:
    def __init__(self):
        print("[FastVLM Local GUI] Loading Apple FastVLM natively via Hugging Face...")
        t0 = time.time()
        
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
            
        print(f"[FastVLM Local GUI] Model loaded successfully in {time.time() - t0:.2f}s!")
        self.risk_level = "SAFE ✅"
        self.latency_ms = 0.0
        self.fps = 0.0
        self.reason_text = "Initializing FastVLM..."

    def infer(self, frame_bgr: np.ndarray) -> str:
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "You are an autonomous driving vision system. Assess road safety ahead: SAFE, WARNING, or CRITICAL."}
            ]}
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt")
        inputs = {k: v.to("cuda") if not torch.is_floating_point(v) else v.to("cuda", torch.float16) for k, v in inputs.items()}

        t_start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False
            )
        self.latency_ms = (time.time() - t_start) * 1000
        self.fps = 1000.0 / max(self.latency_ms, 1.0)

        # Decode only the newly generated tokens
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"[FastVLM GPU {self.latency_ms:.1f}ms] Output: {response_text}")

        resp_upper = response_text.upper()
        if "CRITICAL" in resp_upper or "HAZARD" in resp_upper or "STOP" in resp_upper:
            self.risk_level = "CRITICAL 🚨"
        elif "WARNING" in resp_upper or "CAUTION" in resp_upper or "OBSTACLE" in resp_upper:
            self.risk_level = "WARNING ⚠️"
        else:
            self.risk_level = "SAFE ✅"

        self.reason_text = response_text
        return response_text

    def render_hud(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        # Top HUD Banner Background (made slightly taller for bigger text)
        cv2.rectangle(img, (0, 0), (w, 80), (15, 20, 28), -1)
        cv2.line(img, (0, 80), (w, 80), (60, 80, 110), 2)

        # Risk Status Pill
        r_color = (0, 255, 0) if "SAFE" in self.risk_level else ((0, 165, 255) if "WARNING" in self.risk_level else (0, 0, 255))
        cv2.rectangle(img, (10, 10), (170, 70), r_color, -1)
        cv2.putText(img, self.risk_level.split()[0], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # FastVLM Reasoning Output (Larger Font)
        display_text = f"FastVLM: {self.reason_text}"
        if len(display_text) > 75:
            display_text = display_text[:72] + "..."
            
        cv2.putText(img, display_text, (190, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 245, 255), 2)

        # Exact GPU Inference Latency & FPS
        lat_text = f"⚡ {self.latency_ms:.1f}ms ({self.fps:.1f} FPS)"
        cv2.putText(img, lat_text, (w - 280, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        return img

def main():
    engine = FastVLMLocalGUI()

    # Try RealSense First
    use_rs = False
    rs_pipeline = None
    try:
        import pyrealsense2 as rs
        print("[FastVLM GUI] Connecting to RealSense Camera...")
        rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        rs_pipeline.start(cfg)
        use_rs = True
        print("[FastVLM GUI] RealSense Pipeline Started successfully!")
    except Exception as e:
        print(f"[FastVLM GUI] RealSense failed: {e}. Trying USB/Synthetic...")

    cap = None
    if not use_rs:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[FastVLM GUI] Connected to USB Camera /dev/video0...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            print("[FastVLM GUI] No physical camera found. Using Synthetic HD Road Camera Stream...")
            cap = None

    win_name = "Apple FastVLM-0.5B Real-Time GPU Benchmark (HD 1280x720)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    frame_count = 0
    try:
        while True:
            frame_count += 1
            frame_bgr = None
            
            if use_rs:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
            elif cap and cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
            else:
                # Generate synthetic HD road scene
                frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame_bgr[:360, :] = [180, 140, 100]
                frame_bgr[360:, :] = [60, 60, 65]
                cv2.line(frame_bgr, (640, 360), (200, 720), (255, 255, 255), 4)
                cv2.line(frame_bgr, (640, 360), (1080, 720), (255, 255, 255), 4)
                if (frame_count // 30) % 2 == 1:
                    cv2.rectangle(frame_bgr, (580, 480), (700, 580), (30, 30, 200), -1)
                    cv2.putText(frame_bgr, "ROAD OBSTACLE", (560, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                time.sleep(0.03)

            # Run inference every 5 frames for continuous smooth display
            if frame_count % 5 == 1 and frame_bgr is not None:
                engine.infer(frame_bgr)

            if frame_bgr is not None:
                frame_hud = engine.render_hud(frame_bgr)
                cv2.imshow(win_name, frame_hud)
            
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if use_rs and rs_pipeline:
            rs_pipeline.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
