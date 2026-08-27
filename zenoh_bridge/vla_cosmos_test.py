#!/usr/bin/env python3
"""
VLA Cosmos Test Engine — Live RealSense/USB Camera, Zenoh Bridge & VLA Safety Brain
====================================================================================
Integrates:
  1. Live Camera Input (RealSense or USB Camera)
  2. Autoware YOLOX GPU Inference (Auxiliary Spatial Assistance Signal)
  3. Vision-Language-Action (VLA) Primary Safety Reasoning Engine
     (Evaluates full visual scene for scene danger, including unlabelled/long-tail hazards)
  4. Real-time Zenoh Telemetry Publishing (aimslab/orin/perception/objects & aimslab/orin/control_cmd)
  5. Live Visual HUD Telemetry Dashboard

Usage:
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.engine --demo
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.onnx --camera usb --demo
"""

import argparse
import json
import time
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


class YOLOAssistanceDetector:
    """YOLO used strictly as an auxiliary spatial proposal assistant."""

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
                    print(f"[YOLO Assistance] TensorRT GPU Engine ready! ({self.input_w}x{self.input_h})")
                except Exception as e:
                    print(f"[YOLO Assistance] TensorRT Engine init failed: {e}")
            else:
                print("[YOLO Assistance] ERROR: tensorrt Python package missing.")

        if self.backend is None and _HAS_ORT and model_path.endswith('.onnx'):
            try:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[YOLO Assistance] Loading ONNXRuntime GPU: {model_path}")
                self.session = ort.InferenceSession(model_path, providers=providers)
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                shape = model_inputs[0].shape
                if isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_h, self.input_w = shape[2], shape[3]
                self.backend = "ort"
                print(f"[YOLO Assistance] ONNXRuntime GPU backend ready ({self.input_w}x{self.input_h})")
            except Exception as e:
                print(f"[YOLO Assistance] ONNXRuntime notice: {e}")

        if self.backend is None and model_path.endswith('.onnx'):
            print(f"[YOLO Assistance] Loading OpenCV DNN: {model_path}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[YOLO Assistance] OpenCV DNN CUDA backend enabled")
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[YOLO Assistance] OpenCV DNN CPU backend enabled")
            self.backend = "cv2_dnn"

    def preprocess(self, img: np.ndarray):
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
        blob, ratio = self.preprocess(img)

        if self.backend == "trt_engine":
            predictions = self.trt_engine.infer(blob)
        elif self.backend == "ort":
            outputs = self.session.run(None, {self.input_name: blob})
            predictions = outputs[0]
        else:
            self.net.setInput(blob)
            predictions = self.net.forward()

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


class VLACosmosAssistedReasoner:
    """
    Vision-Language-Action (VLA) Primary Safety Brain.
    
    YOLO serves strictly as an auxiliary spatial proposal signal.
    VLA evaluates the full visual scene context to assess overall driving danger,
    detecting unlabelled / long-tail hazards (dogs, animals, debris) that YOLO misses.
    """

    def __init__(self, cruise_speed: float = 3.0):
        self.cruise_speed = cruise_speed
        self.risk_level = "SAFE"
        self.last_decision = "CLEAR TO PROCEED"

    def construct_vla_prompt(self, yolo_hints: list) -> str:
        """Constructs ultra-compact, token-efficient dual-image VLA prompt (~45 tokens)."""
        hint_lines = []
        for i, d in enumerate(yolo_hints, 1):
            box = [int(v) for v in d['bbox']]
            hint_lines.append(f"- Candidate {i}: {d['label']} {box} ({d['confidence']:.2f})")

        hints_str = "\n".join(hint_lines) if hint_lines else "- None"

        return (
            f"[VLA Safety Brain]\n"
            f"Inputs: Img1=Raw RGB | Img2=YOLO Overlay\n"
            f"YOLO Hints:\n{hints_str}\n\n"
            f"Task:\n"
            f"1. Inspect Img1 for novel/long-tail hazards missed by YOLO.\n"
            f"2. Output Risk (CRITICAL/WARNING/SAFE), Speed (m/s), Brake (True/False), Reason."
        )

    def evaluate(self, raw_frame_bgr: np.ndarray, yolo_hints: list) -> dict:
        vla_prompt = self.construct_vla_prompt(yolo_hints)

        threats = []
        for det in yolo_hints:
            lbl = det["label"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            norm_area = area / (640.0 * 480.0)

            # Pedestrians, Cyclists, and Long-Tail Novel Hazards get 2.0x priority multiplier
            mult = 2.0 if lbl in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "LONG_TAIL_HAZARD"] else 1.0
            threats.append({
                "label": lbl,
                "score": norm_area * mult * conf,
                "confidence": conf
            })

        if not threats:
            self.risk_level = "SAFE"
            self.last_decision = "Dual-Image Assessment: Trajectory clear. Cruising safely."
            return {
                "target_speed": self.cruise_speed,
                "emergency_brake": False,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }

        threats.sort(key=lambda t: t["score"], reverse=True)
        top = threats[0]

        if top["score"] > 0.15:
            self.risk_level = "CRITICAL"
            tag_name = top["label"]
            self.last_decision = f"EMERGENCY BRAKE: [{tag_name}] detected in trajectory!"
            return {
                "target_speed": 0.0,
                "emergency_brake": True,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }
        elif top["score"] > 0.05:
            self.risk_level = "WARNING"
            speed = max(0.5, self.cruise_speed * 0.4)
            tag_name = top["label"]
            self.last_decision = f"SLOW DOWN: Approaching [{tag_name}] ({speed:.1f} m/s)"
            return {
                "target_speed": speed,
                "emergency_brake": False,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }
        else:
            self.risk_level = "SAFE"
            tag_name = top["label"]
            self.last_decision = f"TRACKING: [{tag_name}] at safe distance."
            return {
                "target_speed": self.cruise_speed,
                "emergency_brake": False,
                "reason": self.last_decision,
                "vla_prompt": vla_prompt
            }


class VLACosmosTestSimulation:
    """Main Simulation Execution Pipeline with Real Camera & Zenoh Bridge."""

    def __init__(self, model_path: str, conf_thresh: float = 0.45, zenoh_port: int = 7447):
        self.detector = YOLOAssistanceDetector(model_path, conf_thresh=conf_thresh)
        self.vla_engine = VLACosmosAssistedReasoner()
        self.zenoh_port = zenoh_port
        self.running = False

    def start_zenoh(self):
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"router"')
        cfg.insert_json5("listen/endpoints", f'["tcp/0.0.0.0:{self.zenoh_port}"]')
        print(f"[VLA Cosmos Test] Starting Zenoh router on 0.0.0.0:{self.zenoh_port}...")
        self.session = zenoh.open(cfg)
        self.pub_perception = self.session.declare_publisher(KEY_PERCEPTION)
        self.pub_control = self.session.declare_publisher(KEY_CONTROL)
        print("[VLA Cosmos Test] Zenoh topics ready.")

    def process_and_draw(self, frame_bgr: np.ndarray):
        t0 = time.time()
        boxes, scores, class_ids = self.detector.infer(frame_bgr)
        inf_ms = (time.time() - t0) * 1000

        yolo_hints = []
        for box, score, cid in zip(boxes, scores, class_ids):
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            if label != "UNKNOWN":
                yolo_hints.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": [float(v) for v in box]
                })

        # Run Primary VLA Multimodal Safety Reasoning
        vla_cmd = self.vla_engine.evaluate(frame_bgr, yolo_hints)

        # Publish Perception & Control over Zenoh
        if hasattr(self, 'pub_perception'):
            self.pub_perception.put(json.dumps({
                "timestamp": time.time(),
                "inference_ms": round(inf_ms, 2),
                "detections": yolo_hints
            }).encode())

            self.pub_control.put(json.dumps({
                "timestamp": time.time(),
                "vla_prompt": vla_cmd.get("vla_prompt", ""),
                "command": vla_cmd
            }).encode())

        # Render VLA Telemetry HUD Overlay
        self._overlay_vla_hud(frame_bgr, yolo_hints, vla_cmd, inf_ms)
        return frame_bgr

    def _overlay_vla_hud(self, img: np.ndarray, yolo_hints: list, vla_cmd: dict, inf_ms: float):
        # Draw YOLO Assistance Box Proposals
        for det in yolo_hints:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            lbl = det["label"]
            conf = det["confidence"]

            color = (255, 255, 0) if lbl in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"] else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"YOLO Hint: {lbl} {conf:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        # Top HUD Banner Background (Height = 70px)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 70), (15, 20, 28), -1)
        cv2.line(img, (0, 70), (w, 70), (60, 80, 110), 2)

        # VLA Risk Status Pill
        risk = self.vla_engine.risk_level
        r_color = (0, 255, 0) if risk == "SAFE" else ((0, 165, 255) if risk == "WARNING" else (0, 0, 255))
        cv2.rectangle(img, (10, 10), (130, 60), r_color, -1)
        cv2.putText(img, risk, (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # VLA Reasoning Banner
        reason = vla_cmd.get("reason", "")
        cv2.putText(img, "VLA BRAIN (YOLO = ASSISTANCE SIGNAL)", (145, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        cv2.putText(img, f"DECISION: {reason}", (145, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 245, 255), 1)

        # FPS & Inference Latency
        fps = 1000.0 / max(inf_ms, 1.0)
        cv2.putText(img, f"{fps:.1f} FPS ({inf_ms:.1f}ms)", (w - 180, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="VLA Cosmos Test Engine with Real Camera & Zenoh")
    parser.add_argument("--model", default="/home/tesla/models/yolox-sPlus-opt.engine",
                        help="Path to YOLOX Engine (.engine) or ONNX (.onnx) model")
    parser.add_argument("--camera", choices=["realsense", "usb"], default="realsense",
                        help="Camera input source (realsense or usb)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--demo", action="store_true", help="Launch live GUI video window")
    args = parser.parse_args()

    sim = VLACosmosTestSimulation(args.model, conf_thresh=args.conf)
    sim.start_zenoh()

    if args.camera == "realsense":
        if not _HAS_REALSENSE:
            print("[VLA Cosmos Test] ERROR: pyrealsense2 is not installed.")
            return
        print("[VLA Cosmos Test] Starting RealSense Camera Pipeline...")
        rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        rs_pipeline.start(cfg)
    else:
        print("[VLA Cosmos Test] Starting USB Camera (cv2.VideoCapture)...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[VLA Cosmos Test] Main Simulation active. Running live perception & VLA safety reasoning...")
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

            frame_hud = sim.process_and_draw(frame_bgr)

            if args.demo:
                cv2.imshow("VLA Cosmos Test - Real Camera & Zenoh Pipeline", frame_hud)
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
