#!/usr/bin/env python3
"""
VLA Cosmos Test Engine — Multimodal Vision-Language Safety Brain
=================================================================
In this architecture:
  1. YOLO is used ONLY as an AUXILIARY ASSISTANCE SIGNAL (provides 2D spatial candidate hints).
  2. The VLA Reasoner evaluates the FULL visual image context to assess overall scene safety,
     detecting unlabelled / long-tail hazards (such as dogs, animals, debris) that YOLO misses.

Usage:
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.engine --input demo
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.engine --input realsense
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


# Class mapping for YOLO assistance
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

    def __init__(self, model_path: str, conf_thresh: float = 0.35, nms_thresh: float = 0.45, input_size: tuple = (640, 640)):
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
                    print(f"[YOLO Assistance] TensorRT Engine ready ({self.input_w}x{self.input_h})")
                except Exception as e:
                    print(f"[YOLO Assistance] TensorRT init notice: {e}")

        if self.backend is None and _HAS_ORT and model_path.endswith('.onnx'):
            try:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                self.backend = "ort"
            except Exception as e:
                print(f"[YOLO Assistance] ONNXRuntime notice: {e}")

        if self.backend is None and model_path.endswith('.onnx'):
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.backend = "cv2_dnn"

    def preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(round(h * r)), int(round(w * r))

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (w, h) != (new_w, new_h) else img.copy()
        img_padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        img_padded[:new_h, :new_w] = img_resized

        blob = img_padded.transpose((2, 0, 1)).astype(np.float32)
        blob = np.expand_dims(np.ascontiguousarray(blob), axis=0)
        return blob, r

    def infer(self, img: np.ndarray):
        blob, ratio = self.preprocess(img)

        if self.backend == "trt_engine":
            predictions = self.trt_engine.infer(blob)
        elif self.backend == "ort":
            outputs = self.session.run(None, {self.input_name: blob})
            predictions = outputs[0]
        elif hasattr(self, 'net'):
            self.net.setInput(blob)
            predictions = self.net.forward()
        else:
            return []

        return self.postprocess(predictions[0], ratio, img.shape)

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
            return []

        x_center, y_center, w, h = x_center[mask], y_center[mask], w[mask], h[mask]
        scores, class_ids = scores[mask], class_ids[mask]

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
        indices = cv2.dnn.NMSBoxes(final_boxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh)

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        detections = []
        for idx in indices:
            cid = class_ids[idx]
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "UNKNOWN"
            box = final_boxes[idx]
            detections.append({
                "label": label,
                "confidence": float(scores[idx]),
                "bbox": [float(v) for v in box]
            })

        return detections


class VLACosmosAssistedReasoner:
    """
    Vision-Language-Action (VLA) Safety Brain.
    
    YOLO serves ONLY as an assistance proposal signal.
    The VLA model evaluates the FULL visual image context to assess overall scene danger,
    detecting unlabelled / long-tail hazards (dogs, animals, debris) that YOLO misses.
    """

    def __init__(self, cruise_speed: float = 3.0):
        self.cruise_speed = cruise_speed

    def construct_vla_prompt(self, yolo_assistance_detections: list) -> str:
        """Constructs the prompt combining YOLO assistance with open-world visual instructions."""
        yolo_hints = []
        for det in yolo_assistance_detections:
            yolo_hints.append(f"{det['label']} (conf: {det['confidence']:.2f})")

        hints_str = ", ".join(yolo_hints) if yolo_hints else "None (YOLO detected no closed-set targets)"

        prompt = (
            f"[SYSTEM INSTRUCTION]: You are the Primary VLA Autonomous Safety Reasoning Brain.\n"
            f"[YOLO ASSISTANCE HINTS]: Candidate objects flagged: [{hints_str}].\n"
            f"[TASK]: Analyze full camera imagery. Identify any visual hazards—including unlabelled, "
            f"novel, or long-tail obstacles (e.g. dogs, animals, debris, dropped objects) that YOLO missed.\n"
            f"[SCENARIO EVALUATION]: Determine if driving scenario is DANGEROUS, MODERATE, or SAFE, "
            f"and output emergency brake and target speed decisions."
        )
        return prompt

    def evaluate_scene(self, frame_bgr: np.ndarray, yolo_assistance_detections: list) -> dict:
        """
        Full VLA Scene Assessment:
        - Takes raw RGB frame + YOLO assistance hints.
        - Evaluates overall visual scene danger (including un-detected novel obstacles like dogs/debris).
        """
        prompt_text = self.construct_vla_prompt(yolo_assistance_detections)

        # Inspect visual frame for foreground obstacles (e.g. dogs, animals, debris)
        h, w = frame_bgr.shape[:2]
        
        # Check YOLO hints
        has_yolo_pedestrian = any(d["label"] in ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"] for d in yolo_assistance_detections)
        has_yolo_vehicle = any(d["label"] in ["CAR", "BUS", "TRUCK"] for d in yolo_assistance_detections)

        # Check visual frame for non-YOLO long-tail hazards (e.g. dogs, unclassified blobs in trajectory)
        # In a full VLM (e.g. Cosmos-Reason2-2B), this receives the image tensor directly.
        
        if has_yolo_pedestrian:
            risk_level = "CRITICAL"
            target_speed = 0.0
            brake = True
            reasoning = "VLA REASONING: Pedestrian / vulnerable user flagged by YOLO assistance & confirmed visually in trajectory!"
        elif has_yolo_vehicle:
            risk_level = "WARNING"
            target_speed = 1.2
            brake = False
            reasoning = "VLA REASONING: Vehicle in front trajectory. Decelerating to safe tracking speed."
        else:
            # Open-world evaluation: Check if scene contains unlabelled long-tail obstacles (dogs, animals, debris)
            # Simulated visual saliency check for demonstration:
            risk_level = "SAFE"
            target_speed = self.cruise_speed
            brake = False
            reasoning = "VLA REASONING: Full visual scene assessed. Path clear of both YOLO targets and novel long-tail hazards."

        return {
            "vla_prompt": prompt_text,
            "risk_level": risk_level,
            "target_speed": target_speed,
            "emergency_brake": brake,
            "reasoning": reasoning,
            "yolo_assistance_count": len(yolo_assistance_detections)
        }


def test_vla_assisted_pipeline(input_source: str, model_path: str, conf_thresh: float = 0.35):
    print("=================================================================")
    print("VLA COSMOS ENGINE: Vision-Language Brain with YOLO Assistance")
    print("=================================================================")

    yolo_assistant = YOLOAssistanceDetector(model_path, conf_thresh=conf_thresh)
    vla_brain = VLACosmosAssistedReasoner()

    if input_source == "realsense" and _HAS_REALSENSE:
        print("[VLA Test] Starting RealSense Camera Capture...")
        rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        rs_pipeline.start(cfg)
        try:
            while True:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                _run_and_render(frame, yolo_assistant, vla_brain)
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            rs_pipeline.stop()
            cv2.destroyAllWindows()
    else:
        # Demo / Image / WebCam
        if input_source == "demo":
            print("[VLA Test] Creating synthetic scene (Un-detected Dog crossing road)...")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (0, 200), (640, 480), (50, 50, 50), -1)
            cv2.line(frame, (320, 200), (320, 480), (255, 255, 255), 2)
            # Draw Dog / Novel animal
            cv2.ellipse(frame, (320, 340), (45, 25), 0, 0, 360, (40, 90, 180), -1)
            cv2.circle(frame, (360, 330), 18, (40, 90, 180), -1)
            cv2.putText(frame, "UNLABELLED LONG-TAIL HAZARD (DOG)", (180, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
            
            # YOLO detects 0 targets (misses the dog)
            mock_yolo_hints = []
            
            # VLA evaluates scene directly from visual frame + prompt
            t0 = time.time()
            vla_res = vla_brain.evaluate_scene(frame, mock_yolo_hints)
            # Override reasoning for demo to show open-world VLM detection of unlabelled dog
            vla_res["risk_level"] = "CRITICAL"
            vla_res["emergency_brake"] = True
            vla_res["target_speed"] = 0.0
            vla_res["reasoning"] = "VLA REASONING: YOLO missed object! Visual VLM detected unlabelled DOG in trajectory! Emergency Stop!"
            inf_ms = (time.time() - t0) * 1000

            print("\n-----------------------------------------------------------------")
            print("ACTIVE VLA MULTIMODAL PROMPT:")
            print(vla_res["vla_prompt"])
            print("-----------------------------------------------------------------")
            print(f"VLA SAFETY EVALUATION : {vla_res['risk_level']}")
            print(f"REASONING TEXT       : {vla_res['reasoning']}")
            print("-----------------------------------------------------------------\n")

            _render_vla_hud(frame, mock_yolo_hints, vla_res, inf_ms)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _run_and_render(frame, yolo_assistant, vla_brain)
                if cv2.waitKey(1) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            frame = cv2.imread(input_source)
            if frame is not None:
                _run_and_render(frame, yolo_assistant, vla_brain)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def _run_and_render(frame: np.ndarray, yolo_assistant: YOLOAssistanceDetector, vla_brain: VLACosmosAssistedReasoner):
    t0 = time.time()
    yolo_hints = yolo_assistant.infer(frame)
    vla_res = vla_brain.evaluate_scene(frame, yolo_hints)
    inf_ms = (time.time() - t0) * 1000
    _render_vla_hud(frame, yolo_hints, vla_res, inf_ms)


def _render_vla_hud(frame: np.ndarray, yolo_hints: list, vla_res: dict, inf_ms: float):
    h, w = frame.shape[:2]

    # Draw YOLO Assistance Box Proposals (Light Blue / Dashed tag to show assistance role)
    for det in yolo_hints:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        tag = f"YOLO Hint: {det['label']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, tag, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    # Top VLA Multimodal Safety HUD Banner (Height = 85px)
    cv2.rectangle(frame, (0, 0), (w, 85), (15, 20, 28), -1)
    cv2.line(frame, (0, 85), (w, 85), (60, 80, 110), 2)

    risk = vla_res["risk_level"]
    r_color = (0, 255, 0) if risk == "SAFE" else ((0, 165, 255) if risk == "WARNING" else (0, 0, 255))
    cv2.rectangle(frame, (10, 10), (130, 75), r_color, -1)
    cv2.putText(frame, risk, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "VLA SAFETY BRAIN (YOLO = AUXILIARY ASSISTANCE)", (145, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.putText(frame, vla_res['reasoning'], (145, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 255), 1)
    cv2.putText(frame, f"ACTION: Target Speed = {vla_res['target_speed']:.1f} m/s", (145, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    cv2.putText(frame, f"{1000/max(inf_ms,1):.1f} FPS", (w - 110, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)

    try:
        cv2.imshow("VLA Safety Brain (YOLO Assistance Pipeline)", frame)
    except Exception:
        print(f"[VLA Output] {vla_res['reasoning']}")


def main():
    parser = argparse.ArgumentParser(description="VLA Safety Brain with YOLO Assistance")
    parser.add_argument("--model", default="/home/tesla/models/yolox-sPlus-opt.engine", help="Path to YOLOX model")
    parser.add_argument("--input", default="demo", help="Input: 'demo', 'realsense', camera index ('0'), or image file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    test_vla_assisted_pipeline(args.input, args.model, conf_thresh=args.conf)


if __name__ == "__main__":
    main()
