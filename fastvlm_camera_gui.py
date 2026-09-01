#!/usr/bin/env python3
"""
FastVLM Live OpenCV Desktop GUI Window with Real-Time Inference Latency Benchmarking.
Displays live camera stream, VLA visual safety decision HUD, and exact GPU inference latency (ms) & FPS.
"""

import time
import os
# Suppress Qt font logging warnings
os.environ["QT_LOGGING_RULES"] = "*=false;qt.qpa.font=false"

import cv2
import torch
import numpy as np
from PIL import Image

# FastVLM via ml-fastvlm llava package
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "apple/FastVLM-0.5B"

class FastVLMLocalGUI:
    def __init__(self):
        print("[FastVLM Local GUI] Loading Apple FastVLM-0.5B onto AGX Orin GPU...")
        t0 = time.time()

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            MODEL_PATH, None, "FastVLM-0.5B", device_map="cuda"
        )
        
        # --- Diagnostics only. No weight mutation until we know which case we're in. ---
        try:
            lm_head_w = self.model.get_output_embeddings().weight
            embed_w = self.model.get_input_embeddings().weight
            is_tied = torch.equal(lm_head_w, embed_w)
            cfg_tied = getattr(self.model.config, "tie_word_embeddings", None)

            print(f"[DEBUG FastVLM] tie_word_embeddings config: {cfg_tied}")
            print(f"[DEBUG FastVLM] lm_head == embed_tokens (tied): {is_tied}")
            print(f"[DEBUG FastVLM] lm_head stats  - mean: {lm_head_w.mean().item():.5f}, std: {lm_head_w.std().item():.5f}")
            print(f"[DEBUG FastVLM] embed_tokens stats - mean: {embed_w.mean().item():.5f}, std: {embed_w.std().item():.5f}")

            # Only treat as a bug if config says tied but weights disagree.
            if cfg_tied is True and not is_tied:
                print("[DEBUG FastVLM] MISMATCH: config says tied, weights are not. Fixing.")
                self.model.set_output_embeddings(self.model.get_input_embeddings())
            elif cfg_tied is False:
                print("[DEBUG FastVLM] Config says untied — leaving lm_head alone (likely intentional).")
        except Exception as e:
            print(f"[DEBUG FastVLM] Weight tie check warning: {e}")

        # --- Check the vision-language projector actually has trained (non-random) weights ---
        try:
            get_model_fn = getattr(self.model, "get_model", None)
            base_model = get_model_fn() if callable(get_model_fn) else self.model
            projector = getattr(base_model, "mm_projector", None)
            if projector is not None:
                for name, p in projector.named_parameters():
                    print(f"[DEBUG FastVLM] mm_projector.{name} - mean: {p.mean().item():.5f}, std: {p.std().item():.5f}")
            else:
                print("[DEBUG FastVLM] WARNING: no mm_projector found on model — image features may never reach the LLM.")
        except Exception as e:
            print(f"[DEBUG FastVLM] Projector check warning: {e}")

        # Retrieve native FastViTHD image_processor directly from vision tower
        if self.image_processor is None and hasattr(self.model, "get_vision_tower"):
            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            self.image_processor = getattr(vision_tower, "image_processor", None)

        # Fallback image_mean if missing
        if self.image_processor is not None and getattr(self.image_processor, 'image_mean', None) is None:
            self.image_processor.image_mean = [0.485, 0.456, 0.406]

        print(f"[FastVLM Local GUI] Model loaded successfully in {time.time() - t0:.2f}s!")
        self.risk_level = "SAFE"
        self.latency_ms = 0.0
        self.fps = 0.0
        self.reason_text = "Initializing FastVLM..."

    def infer(self, frame_bgr: np.ndarray) -> str:
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        from llava.conversation import conv_templates
        template_name = "qwen_2" if "qwen_2" in conv_templates else ("qwen_1_5" if "qwen_1_5" in conv_templates else "v1")
        conv = conv_templates[template_name].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nWhat are you seeing?")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        if self.image_processor is not None:
            image_tensor = process_images([pil_img], self.image_processor, self.model.config)
        else:
            import torchvision.transforms as T
            _tf = T.Compose([
                T.Resize((1024, 1024), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = _tf(pil_img).unsqueeze(0)

        model_dtype = next(self.model.parameters()).dtype
        if isinstance(image_tensor, list):
            image_tensor = [img.to(device="cuda", dtype=model_dtype) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(device="cuda", dtype=model_dtype)

        # GPU Memory Watch Telemetry
        if not hasattr(self, "_frame_n"):
            self._frame_n = 0
        self._frame_n += 1
        alloc = torch.cuda.memory_allocated() / 1e6
        reserv = torch.cuda.memory_reserved() / 1e6
        print(f"[DEBUG FastVLM] Frame {self._frame_n} — GPU Alloc: {alloc:.1f}MB | Reserved: {reserv:.1f}MB")

        t_start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[pil_img.size],
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
            )
        self.latency_ms = (time.time() - t_start) * 1000
        self.fps = 1000.0 / max(self.latency_ms, 1.0)

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not response_text:
            response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        print(f"[FastVLM {self.latency_ms:.0f}ms] → {response_text}")

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
