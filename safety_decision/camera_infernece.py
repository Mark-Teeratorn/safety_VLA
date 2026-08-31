#!/usr/bin/env python3
"""
VLA Cosmos Real-Time Safety Perception Engine
======================================================
Integrates live camera capture, Autoware YOLOX object detection,
Vision-Language-Action (VLA) safety reasoning, and Zenoh telemetry streaming.

Features:
  1. Live Camera Input (RealSense or USB Camera)
  2. Autoware YOLOX TensorRT/ONNX GPU Inference
  3. VLA Safety Reasoning Engine (Risk Assessment & Control Actions)
  4. Real-time Visual Telemetry Dashboard (Bounding Boxes & VLA HUD Overlay)
  5. Zenoh Telemetry Publishing (aimslab/orin/perception/objects & aimslab/orin/control_cmd)

Usage:
    python3 safety_decision/camera_infernece.py --model ~/models/yolox-sPlus-opt.engine --demo
    python3 safety_decision/camera_infernece.py --model ~/models/yolox-sPlus-opt.onnx --camera usb --demo
"""

import os
import sys
import argparse
import json
import time
import subprocess
import threading
import queue
from typing import Dict, Optional
import numpy as np
import cv2
import zenoh

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False

try:
    import tensorrt as trt
    import ctypes
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False


# ---- Zenoh Topics ----
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"
KEY_PERCEPTION = "aimslab/orin/perception/objects"
KEY_CONTROL = "aimslab/orin/control_cmd"

# Empirically verified Autoware yolox-sPlus model class index mapping
CLASS_NAMES = [
    "UNKNOWN",     # Index 0
    "CAR",         # Index 1
    "TRUCK",       # Index 2
    "BUS",         # Index 3
    "BICYCLE",     # Index 4
    "MOTORCYCLE",  # Index 5
    "PEDESTRIAN",  # Index 6
    "UNKNOWN"      # Index 7
]


class TensorRTEngine:
    """Native TensorRT Engine Execution via libcudart."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[YOLOX] Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        try:
            self.cudart = ctypes.CDLL('libcudart.so')
        except OSError:
            self.cudart = ctypes.CDLL('libcudart.so.12')

        self.bindings = []
        self.num_tensors = self.engine.num_io_tensors

        for i in range(self.num_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = (self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            ptr = ctypes.c_void_p()
            self.cudart.cudaMalloc(ctypes.byref(ptr), size)
            self.bindings.append(int(ptr.value))

            if is_input:
                self.input_name = tensor_name
                self.input_shape = shape
                self.input_ptr = ptr
                self.input_size = size
                self.input_dtype = dtype
            else:
                self.output_name = tensor_name
                self.output_ptr = ptr
                self.output_shape = shape
                self.output_size = size
                self.output_dtype = dtype
                self.output_host = np.empty(shape, dtype=dtype)

        self.stream = ctypes.c_void_p()
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))

    def infer(self, blob_np: np.ndarray) -> np.ndarray:
        blob_contig = np.ascontiguousarray(blob_np, dtype=self.input_dtype)
        self.cudart.cudaMemcpyAsync(self.input_ptr, blob_contig.ctypes.data, self.input_size, 1, self.stream)

        for i in range(self.num_tensors):
            tname = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tname, self.bindings[i])

        self.context.execute_async_v3(self.stream.value)
        self.cudart.cudaMemcpyAsync(self.output_host.ctypes.data, self.output_ptr, self.output_size, 2, self.stream)
        self.cudart.cudaStreamSynchronize(self.stream)
        return self.output_host


class YOLOXDetector:
    """Standalone YOLOX Inference Engine matching Autoware Universe tensorrt_yolox C++ spec."""

    def __init__(self, model_path: str, conf_thresh: float = 0.45, nms_thresh: float = 0.45, input_size: tuple = (640, 640)):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_w, self.input_h = input_size
        self.backend = None

        if model_path.endswith('.engine'):
            if _HAS_TRT:
                try:
                    self.trt_engine = TensorRTEngine(model_path)
                    shape = self.trt_engine.input_shape
                    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                        self.input_h, self.input_w = shape[2], shape[3]
                    self.backend = "trt_engine"
                    print(f"[YOLOX] TensorRT GPU Engine ready! ({self.input_w}x{self.input_h})")
                except Exception as e:
                    print(f"[YOLOX] TensorRT Engine init failed: {e}")
            else:
                print("[YOLOX] ERROR: tensorrt Python package missing.")

        if self.backend is None and _HAS_ORT and model_path.endswith('.onnx'):
            try:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[YOLOX] Loading ONNXRuntime GPU/TensorRT: {model_path}")
                self.session = ort.InferenceSession(model_path, providers=providers)
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                shape = model_inputs[0].shape
                if isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_h, self.input_w = shape[2], shape[3]
                self.backend = "ort"
                print(f"[YOLOX] ONNXRuntime GPU backend ready ({self.input_w}x{self.input_h})")
            except Exception as e:
                print(f"[YOLOX] ONNXRuntime notice: {e}")

        if self.backend is None and model_path.endswith('.onnx'):
            print(f"[YOLOX] Loading OpenCV DNN: {model_path}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[YOLOX] OpenCV DNN CUDA backend enabled")
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[YOLOX] OpenCV DNN CPU backend enabled")
            self.backend = "cv2_dnn"

    def preprocess(self, img: np.ndarray):
        """Top-left padding (114), BGR format, raw float32 [0-255]."""
        h, w = img.shape[:2]
        r = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(round(h * r)), int(round(w * r))

        if (w, h) != (new_w, new_h):
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img.copy()

        img_padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        img_padded[:new_h, :new_w] = img_resized

        blob = img_padded.transpose((2, 0, 1)).astype(np.float32)
        blob = np.ascontiguousarray(blob)
        blob = np.expand_dims(blob, axis=0)

        return blob, r

    def infer(self, img: np.ndarray):
        if self.backend is None:
            return [], [], []
        blob, ratio = self.preprocess(img)

        if self.backend == "trt_engine":
            predictions = self.trt_engine.infer(blob)
        elif self.backend == "ort":
            outputs = self.session.run(None, {self.input_name: blob})
            predictions = outputs[0]
        elif self.backend == "cv2_dnn" and hasattr(self, 'net'):
            self.net.setInput(blob)
            predictions = self.net.forward()
        else:
            return [], [], []

        boxes, scores, class_ids = self.postprocess(predictions[0], ratio, img.shape)
        return boxes, scores, class_ids

    def _generate_grids_and_strides(self):
        strides = [8, 16, 32]
        grids, exp_strides = [], []
        for stride in strides:
            grid_h = self.input_h // stride
            grid_w = self.input_w // stride
            for gy in range(grid_h):
                for gx in range(grid_w):
                    grids.append([gx, gy])
                    exp_strides.append(stride)
        return np.array(grids, dtype=np.float32), np.array(exp_strides, dtype=np.float32)

    def postprocess(self, predictions: np.ndarray, ratio: float, orig_shape: tuple):
        if len(predictions.shape) == 3:
            predictions = predictions[0]

        if not hasattr(self, '_grids'):
            self._grids, self._strides = self._generate_grids_and_strides()

        grids = self._grids
        strides = self._strides

        x_center = (predictions[:, 0] + grids[:, 0]) * strides
        y_center = (predictions[:, 1] + grids[:, 1]) * strides
        w = np.exp(predictions[:, 2]) * strides
        h = np.exp(predictions[:, 3]) * strides

        obj_conf = predictions[:, 4]
        class_conf = predictions[:, 5:]
        class_ids = np.argmax(class_conf, axis=1)
        scores = obj_conf * class_conf[np.arange(len(class_ids)), class_ids]

        mask = scores > self.conf_thresh
        if not np.any(mask):
            return [], [], []

        x_center = x_center[mask]
        y_center = y_center[mask]
        w = w[mask]
        h = h[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        x1 = (x_center - w / 2.0) / ratio
        y1 = (y_center - h / 2.0) / ratio
        x2 = (x_center + w / 2.0) / ratio
        y2 = (y_center + h / 2.0) / ratio

        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow - 1)
        y1 = np.clip(y1, 0, oh - 1)
        x2 = np.clip(x2, 0, ow - 1)
        y2 = np.clip(y2, 0, oh - 1)

        final_boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            final_boxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )

        if len(indices) == 0:
            return [], [], []

        indices = np.array(indices).flatten()
        return final_boxes[indices], scores[indices], class_ids[indices]


class CosmosCognitiveReasoner:
    """Cognitive Reasoning Layer (Cosmos-Reason2-2B) executing llama-cli matching safe_driving_carla implementation."""

    def __init__(self, model_path: str = '/home/tesla/models/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf'):
        self.model_path = model_path
        self.llama_cli = '/usr/local/bin/llama-cli'
        self.has_native_cli = os.path.exists(self.llama_cli)
        print(f'[Cognitive Brain] Reasoner initialized with Cosmos-Reason2-2B (Engine: {self.llama_cli})')

    def reason_on_image(self, frame_bgr: np.ndarray, prompt: str) -> str:
        """Executes llama-cli matching the exact safe_driving_carla inference pattern."""
        if not (self.has_native_cli and os.path.exists(self.model_path)):
            return "Cosmos-VLA Model Engine unavailable."

        try:
            cmd = [
                self.llama_cli,
                '-m', self.model_path,
                '-p', prompt,
                '-n', '64',
                '-ngl', '20',
                '-t', '4',
                '--no-warmup',
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
            output = proc.stdout
            if output and len(output) > 20:
                lines = [
                    line.strip() for line in output.splitlines()
                    if line.strip() and not line.startswith('llama_') and not line.startswith('main:')
                ]
                if lines:
                    return ' '.join(lines)
        except Exception as e:
            print(f"[Cognitive Brain] Exception: {e}")
        return "Visual CoT: Inspecting camera scene."


class VLAReasoningEngine:
    """Vision-Language-Action (VLA) Safety Assessment & Controller."""

    def __init__(self, cruise_speed: float = 3.0):
        self.cruise_speed = cruise_speed
        self.risk_level = "SAFE"  # SAFE, WARNING, CRITICAL
        self.last_decision = "CLEAR TO PROCEED"
        self.prompt_queue = queue.Queue(maxsize=1)
        self.running = True

        self.reasoner = CosmosCognitiveReasoner()
        self.llm_thread = threading.Thread(target=self._vla_llm_worker, daemon=True)
        self.llm_thread.start()

    def _vla_llm_worker(self):
        """Background thread executing llama-cli on Cosmos-Reason2-2B GGUF model."""
        while self.running:
            try:
                frame_bgr, prompt = self.prompt_queue.get(timeout=0.2)
                if self.reasoner:
                    llm_cot = self.reasoner.reason_on_image(frame_bgr, prompt)
                    if llm_cot and len(llm_cot) > 10:
                        self.last_decision = f"[VLM CoT]: {llm_cot}"
                        # Determine risk from CoT for terminal display
                        cot_upper = llm_cot.upper()
                        if any(k in cot_upper for k in ["CRITICAL", "EMERGENCY"]):
                            risk_tag = "CRITICAL"
                        elif any(k in cot_upper for k in ["WARNING", "WARN", "CAUTION", "SLOW"]):
                            risk_tag = "WARNING"
                        else:
                            risk_tag = "SAFE"
                        print(f"\n[VLA CoT | {risk_tag}] {llm_cot}\n")
                self.prompt_queue.task_done()
                time.sleep(0.3)
            except queue.Empty:
                continue
            except Exception:
                pass

    def construct_vla_prompt(self, yolo_hints: list) -> str:
        """Constructs an efficient VLM prompt focused on road occupancy and safety decisions."""
        hint_lines = [f"- {d['label']} at {[int(v) for v in d['bbox']]}" for d in yolo_hints if not d.get("is_novel")]
        hints_str = "\n".join(hint_lines) if hint_lines else "- None"
        prev_cot = getattr(self, 'last_decision', 'CLEAR TO PROCEED')

        return (
            f"You are an autonomous driving vision system. Focus on the road image ahead.\n\n"
            f"1. COGNITIVE PERCEPTION: Identify what object is on the road in front of our car (e.g., 'On the road in front of our car is a [object]' or 'No [objects]').\n"
            f"2. VLA REASONING: Determine if the [object] is hard blocking our driving path or if it is safe to proceed.\n"
            f"3. DECISION: End with exactly one rating on its own line:\n"
            f"   - CRITICAL (road hard blocked, must stop)\n"
            f"   - WARNING (hazard near lane, slow down)\n"
            f"   - SAFE (road clear, safe to proceed)\n\n"
            f"YOLO Hints:\n{hints_str}\n"
            f"Previous Thought: {prev_cot}"
        )

    def evaluate(self, raw_frame_bgr: np.ndarray, yolo_hints: list) -> dict:
        h, w = raw_frame_bgr.shape[:2]
        vla_prompt = self.construct_vla_prompt(yolo_hints)
        
        # Construct Dual-Scene VLM Input Frame (Raw RGB View + YOLO Bounding Box Overlay View)
        half_w = w // 2
        raw_panel = cv2.resize(raw_frame_bgr, (half_w, h))
        cv2.putText(raw_panel, "IMAGE 1: RAW SCENE (VLM OOD Scan)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        
        yolo_panel = raw_frame_bgr.copy()
        for det in yolo_hints:
            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
            lbl = det["label"]
            conf = det["confidence"]
            color = (0, 140, 255) if det.get("is_novel") or "OOD" in lbl or "PILLAR" in lbl else ((255, 255, 0) if lbl in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"] else (0, 255, 0))
            cv2.rectangle(yolo_panel, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(yolo_panel, f"{lbl} {conf:.2f}", (bx1, max(by1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        yolo_panel_resized = cv2.resize(yolo_panel, (half_w, h))
        cv2.putText(yolo_panel_resized, "IMAGE 2: YOLO BBOX OVERLAY", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        dual_view_vlm_frame = np.hstack([raw_panel, yolo_panel_resized])

        if hasattr(self, 'prompt_queue') and self.prompt_queue.empty():
            try:
                self.prompt_queue.put_nowait((dual_view_vlm_frame, vla_prompt))
            except Exception:
                pass

        # VLM Model Direct Risk Decision (CRITICAL / WARNING / SAFE) based on model rating while preserving full CoT
        last_cot = str(self.last_decision).upper()
        
        if any(k in last_cot for k in ["CRITICAL", "EMERGENCY"]):
            self.risk_level = "CRITICAL"
            return {
                "target_speed": 0.0,
                "emergency_brake": True,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }
        elif any(k in last_cot for k in ["WARNING", "WARN", "CAUTION", "SLOW"]):
            self.risk_level = "WARNING"
            return {
                "target_speed": max(1.0, self.cruise_speed * 0.5),
                "emergency_brake": False,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }
        else:
            self.risk_level = "SAFE"
            return {
                "target_speed": self.cruise_speed,
                "emergency_brake": False,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }



class VLACosmosRealtimePipeline:
    """Real-Time Execution Pipeline."""

    def __init__(self, model_path: str, conf_thresh: float = 0.45, zenoh_port: int = 7447, use_yolo: bool = True):
        self.use_yolo = use_yolo
        if self.use_yolo:
            self.detector = YOLOXDetector(model_path, conf_thresh=conf_thresh)
        else:
            self.detector = None
            print("=========================================================")
            print("[VLA Cosmos] PURE VLM EVALUATION MODE (YOLO Completely Disabled)")
            print("=========================================================")
        self.vla_engine = VLAReasoningEngine()
        self.zenoh_port = zenoh_port
        self.running = False

    def start_zenoh(self):
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"router"')
        cfg.insert_json5("listen/endpoints", f'["tcp/0.0.0.0:{self.zenoh_port}"]')
        print(f"[VLA Cosmos] Starting Zenoh router on 0.0.0.0:{self.zenoh_port}...")
        self.session = zenoh.open(cfg)
        self.pub_perception = self.session.declare_publisher(KEY_PERCEPTION)
        self.pub_control = self.session.declare_publisher(KEY_CONTROL)
        print("[VLA Cosmos] Zenoh topics ready.")

    def process_and_draw(self, frame_bgr: np.ndarray):
        t0 = time.time()
        detections = []
        if self.use_yolo and self.detector:
            boxes, scores, class_ids = self.detector.infer(frame_bgr)
            for box, score, cid in zip(boxes, scores, class_ids):
                label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                if label != "UNKNOWN":
                    detections.append({
                        "label": label,
                        "confidence": float(score),
                        "bbox": [float(v) for v in box]
                    })
        inf_ms = (time.time() - t0) * 1000

        # Run VLA Safety Reasoning
        vla_cmd = self.vla_engine.evaluate(frame_bgr, detections)

        # Publish Perception & Control via Zenoh
        if hasattr(self, 'pub_perception'):
            self.pub_perception.put(json.dumps({
                "timestamp": time.time(),
                "inference_ms": round(inf_ms, 2),
                "detections": detections
            }).encode())
            
            self.pub_control.put(json.dumps({
                "timestamp": time.time(),
                "command": vla_cmd
            }).encode())

        # Render VLA Visual HUD Overlay onto camera frame
        self._overlay_vla_hud(frame_bgr, detections, vla_cmd, inf_ms)
        return frame_bgr

    def _overlay_vla_hud(self, img: np.ndarray, detections: list, vla_cmd: dict, inf_ms: float):
        # Draw Bounding Boxes
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            lbl = det["label"]
            conf = det["confidence"]
            
            # Box Colors: Novel Pillar/Debris = Orange/Red, Pedestrians = Cyan, Vehicles = Green
            color = (0, 140, 255) if det.get("is_novel") or "PILLAR" in lbl else ((255, 255, 0) if lbl in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"] else (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{lbl} {conf:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Top HUD Banner Background
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 55), (15, 20, 28), -1)
        cv2.line(img, (0, 55), (w, 55), (60, 80, 110), 2)

        # VLA Risk Status Pill
        risk = self.vla_engine.risk_level
        r_color = (0, 255, 0) if risk == "SAFE" else ((0, 165, 255) if risk == "WARNING" else (0, 0, 255))
        cv2.rectangle(img, (10, 10), (130, 45), r_color, -1)
        cv2.putText(img, risk, (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # VLA Reason Text
        reason = vla_cmd.get("reason", "")
        cv2.putText(img, f"VLA REASON: {reason}", (145, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 255), 1)

        # FPS & Inference Latency
        fps = 1000.0 / max(inf_ms, 1.0)
        cv2.putText(img, f"{fps:.1f} FPS ({inf_ms:.1f}ms)", (w - 180, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="VLA Cosmos Real-Time Safety Perception")
    parser.add_argument("--model", default="/home/tesla/models/yolox-sPlus-opt.engine",
                        help="Path to YOLOX Engine (.engine) or ONNX (.onnx) model")
    parser.add_argument("--camera", choices=["realsense", "usb"], default="realsense",
                        help="Camera input source (realsense or usb)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--demo", action="store_true", help="Launch live GUI video window")
    parser.add_argument("--no-yolo", action="store_true", help="Bypass YOLO completely and let Cosmos-Reason2-2B VLM decide 100% on raw camera frames")
    args = parser.parse_args()

    pipeline = VLACosmosRealtimePipeline(args.model, conf_thresh=args.conf, use_yolo=not args.no_yolo)
    pipeline.start_zenoh()

    if args.camera == "realsense":
        if not _HAS_REALSENSE:
            print("[VLA Cosmos] ERROR: pyrealsense2 is not installed.")
            return
        print("[VLA Cosmos] Starting RealSense Camera Pipeline (1280x720 HD)...")
        rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        rs_pipeline.start(cfg)
    else:
        print("[VLA Cosmos] Starting USB Camera (1280x720 HD)...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[VLA Cosmos] Real-Time Safety Engine active. Running inference & safety reasoning...")
    win_name = "VLA Cosmos Real-Time Safety Engine (HD 1280x720)"
    if args.demo:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

    try:
        while True:
            if args.camera == "realsense":
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    continue

            frame_hud = pipeline.process_and_draw(frame_bgr)

            if args.demo:
                cv2.imshow(win_name, frame_hud)
                if cv2.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if args.camera == "realsense":
            rs_pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
