#!/usr/bin/env python3
"""
FastVLM Live Interactive Web UI with Real-Time Inference Latency Benchmarking.
Displays model response, visual safety decision, and exact GPU inference time (ms) & FPS.
"""

import time
import torch
import gradio as gr
from PIL import Image
import numpy as np

# FastVLM / LLaVA imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    _HAS_FASTVLM = True
except ImportError:
    _HAS_FASTVLM = False

MODEL_PATH = "apple/FastVLM-7B"
tokenizer, model, image_processor = None, None, None

def init_fastvlm():
    global tokenizer, model, image_processor
    if tokenizer is None and _HAS_FASTVLM:
        print("[FastVLM WebUI] Loading Apple FastVLM model into GPU VRAM...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            MODEL_PATH, None, "FastVLM-7B", device_map="cuda"
        )
        print("[FastVLM WebUI] Model loaded successfully!")

def analyze_image(input_image, prompt_text):
    if input_image is None:
        return "Please upload or capture an image.", "0 ms (0 FPS)", "N/A"
    
    init_fastvlm()
    if model is None:
        return "FastVLM package not found. Ensure fastvlm_env is activated.", "0 ms", "ERROR"

    pil_img = Image.fromarray(input_image).convert("RGB")
    
    formatted_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    input_ids = tokenizer_image_token(formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image_tensor = process_images([pil_img], image_processor, model.config)[0].unsqueeze(0).half().cuda()

    # Measure exact GPU inference latency
    t_start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=64,
            do_sample=False
        )
    t_latency_ms = (time.time() - t_start) * 1000
    fps = 1000.0 / max(t_latency_ms, 1.0)

    # Decode response
    response_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Determine risk level
    risk = "SAFE"
    resp_upper = response_text.upper()
    if "CRITICAL" in resp_upper or "HAZARD" in resp_upper or "STOP" in resp_upper:
        risk = "CRITICAL 🚨"
    elif "WARNING" in resp_upper or "CAUTION" in resp_upper or "OBSTACLE" in resp_upper:
        risk = "WARNING ⚠️"
    else:
        risk = "SAFE ✅"

    latency_str = f"⚡ {t_latency_ms:.1f} ms  ({fps:.1f} FPS)"
    return response_text, latency_str, risk

# Build Gradio UI
with gr.Blocks(title="Apple FastVLM Real-Time Safety Benchmarking UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Apple FastVLM-7B Real-Time Safety & Inference Latency Benchmark")
    gr.Markdown("Upload or capture a road camera image to run **Apple FastVLM** on the Jetson AGX Orin GPU and view real-time latency.")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="numpy", label="Camera Input / Road Image Source", sources=["upload", "webcam"])
            prompt_input = gr.Textbox(
                value="You are an autonomous driving vision system. Assess road safety ahead and output one rating: SAFE, WARNING, or CRITICAL.",
                label="VLM Prompt Instruction",
                lines=2
            )
            btn_run = gr.Button("🔍 Run FastVLM GPU Inference", variant="primary")

        with gr.Column(scale=1):
            latency_output = gr.Textbox(label="⚡ GPU Inference Time & FPS", interactive=False)
            risk_output = gr.Textbox(label="🛡️ VLA Safety Decision", interactive=False)
            response_output = gr.Textbox(label="🧠 FastVLM Reasoning Output", lines=5, interactive=False)

    btn_run.click(
        fn=analyze_image,
        inputs=[img_input, prompt_input],
        outputs=[response_output, latency_output, risk_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)
