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

# FastVLM / LLaVA imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    _HAS_FASTVLM = True
except ImportError:
    _HAS_FASTVLM = False

MODEL_PATH = "apple/FastVLM-0.5B"

class FastVLMLocalGUI:
    def __init__(self):
        print("[FastVLM Local GUI] Loading Apple FastVLM model onto AGX Orin GPU...")
        t0 = time.time()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            MODEL_PATH, None, "FastVLM-0.5B", device_map="cuda"
        )
        if self.image_processor is not None and not hasattr(self.image_processor, 'image_mean'):
            self.image_processor.image_mean = [0.48145466, 0.4578275, 0.40821073]
        print(f"[FastVLM Local GUI] Model loaded successfully in {time.time() - t0:.2f}s!")
        self.risk_level = "SAFE ✅"
        self.latency_ms = 0.0
        self.fps = 0.0
        self.reason_text = "Initializing FastVLM..."

    def infer(self, frame_bgr: np.ndarray) -> str:
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        prompt = f"{DEFAULT_IMAGE_TOKEN}\nYou are an autonomous driving vision system. Assess road safety ahead: SAFE, WARNING, or CRITICAL."
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        image_tensor = process_images([pil_img], self.image_processor, self.model.config)[0].unsqueeze(0).half().cuda()

        t_start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=48,
                do_sample=False
            )
        self.latency_ms = (time.time() - t_start) * 1000
        self.fps = 1000.0 / max(self.latency_ms, 1.0)

        response_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
        # Top HUD Banner Background
        cv2.rectangle(img, (0, 0), (w, 60), (15, 20, 28), -1)
        cv2.line(img, (0, 60), (w, 60), (60, 80, 110), 2)

        # Risk Status Pill
        r_color = (0, 255, 0) if "SAFE" in self.risk_level else ((0, 165, 255) if "WARNING" in self.risk_level else (0, 0, 255))
        cv2.rectangle(img, (10, 10), (150, 50), r_color, -1)
        cv2.putText(img, self.risk_level, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # FastVLM Reasoning Output
        cv2.putText(img, f"FastVLM: {self.reason_text[:65]}", (165, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 255), 1)

        # Exact GPU Inference Latency & FPS
        lat_text = f"⚡ {self.latency_ms:.1f}ms ({self.fps:.1f} FPS)"
        cv2.putText(img, lat_text, (w - 220, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        return img

def main():
    if not _HAS_FASTVLM:
        print("ERROR: FastVLM package not found. Activate fastvlm_env first.")
        return

    engine = FastVLMLocalGUI()

    # OpenCV Camera / Synthetic Stream
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[FastVLM GUI] Connected to USB Camera /dev/video0...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        print("[FastVLM GUI] No physical camera found. Using Synthetic HD Road Camera Stream...")
        cap = None

    win_name = "Apple FastVLM-7B Real-Time GPU Benchmark (HD 1280x720)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    frame_count = 0
    try:
        while True:
            frame_count += 1
            if cap and cap.isOpened():
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
            if frame_count % 5 == 1:
                engine.infer(frame_bgr)

            frame_hud = engine.render_hud(frame_bgr)
            cv2.imshow(win_name, frame_hud)
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
