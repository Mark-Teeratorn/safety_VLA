#!/usr/bin/env python3
"""
VLA Cosmos Open-Vocabulary & Distance Estimation Test Script
=============================================================
Integrates YOLO spatial grounding (proximity/distance estimation) with open-vocabulary
VLA safety reasoning that handles BOTH known target classes (Pedestrians, Vehicles)
AND novel/unknown obstacles (dogs, animals, strollers, debris, unclassified obstacles).

Usage:
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.engine --demo
    python3 zenoh_bridge/vla_cosmos_test.py --model ~/models/yolox-sPlus-opt.onnx --demo
"""

import argparse
import json
import time
import math
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


# ---- Class Mapping including Novel/Unknown Obstacle Handling ----
CLASS_NAMES = [
    "NOVEL_OBSTACLE", # Index 0 (Unclassified / Unknown obstacle, e.g. dog/debris)
    "CAR",            # Index 1
    "TRUCK",          # Index 2
    "BUS",            # Index 3
    "BICYCLE",        # Index 4
    "MOTORCYCLE",     # Index 5
    "PEDESTRIAN",     # Index 6
    "NOVEL_OBSTACLE"  # Index 7 (Unclassified / Trailer / Other)
]

# Standard Real-World Height Estimates (meters) for Distance Calculation:
# Distance (m) ≈ (focal_length * real_height) / bbox_height_pixels
REAL_HEIGHT_MAP = {
    "PEDESTRIAN": 1.70,
    "CAR": 1.50,
    "TRUCK": 2.50,
    "BUS": 3.20,
    "BICYCLE": 1.20,
    "MOTORCYCLE": 1.30,
    "NOVEL_OBSTACLE": 0.80  # Default assumed height for unknown obstacles (dog, box, etc.)
}


class TensorRTEngine:
    """Native TensorRT Engine Execution via libcudart."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[YOLOX Test] Loading TensorRT Engine: {engine_path}")
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


class OpenVocabYOLOXDetector:
    """YOLOX Detector providing distance estimation and novel obstacle tracking."""

    def __init__(self, model_path: str, conf_thresh: float = 0.35, nms_thresh: float = 0.45, input_size: tuple = (640, 640)):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_w, self.input_h = input_size
        self.focal_length = 500.0  # Approximate camera focal length in pixels for 640x480 resolution
        self.backend = None

        if model_path.endswith('.engine'):
            if _HAS_TRT:
                try:
                    self.trt_engine = TensorRTEngine(model_path)
                    shape = self.trt_engine.input_shape
                    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                        self.input_h, self.input_w = shape[2], shape[3]
                    self.backend = "trt_engine"
                    print(f"[YOLOX Test] TensorRT GPU Engine ready! ({self.input_w}x{self.input_h})")
                except Exception as e:
                    print(f"[YOLOX Test] TensorRT Engine init failed: {e}")

        if self.backend is None and _HAS_ORT and model_path.endswith('.onnx'):
            try:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                self.backend = "ort"
                print("[YOLOX Test] ONNXRuntime GPU backend ready")
            except Exception as e:
                print(f"[YOLOX Test] ONNXRuntime notice: {e}")

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

    def estimate_distance(self, label: str, bbox_h_px: float) -> float:
        """Estimate distance (meters) based on pixel height and object real-world height."""
        real_h = REAL_HEIGHT_MAP.get(label, 0.80)
        bbox_h_px = max(bbox_h_px, 1.0)
        distance = (self.focal_length * real_h) / bbox_h_px
        return round(float(distance), 2)

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
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "NOVEL_OBSTACLE"
            box = final_boxes[idx]
            bbox_h = max(1.0, box[3] - box[1])
            dist_m = self.estimate_distance(label, bbox_h)
            is_novel = (label == "NOVEL_OBSTACLE")

            detections.append({
                "label": label,
                "confidence": float(scores[idx]),
                "bbox": [float(v) for v in box],
                "distance_m": dist_m,
                "is_novel": is_novel
            })

        return detections


class OpenVocabVLAReasoner:
    """Vision-Language-Action Reasoner evaluating Known & Novel/Unknown Obstacles."""

    def __init__(self, cruise_speed: float = 3.0):
        self.cruise_speed = cruise_speed
        self.risk_level = "SAFE"
        self.reasoning = "Path clear."

    def evaluate(self, detections: list) -> dict:
        if not detections:
            self.risk_level = "SAFE"
            self.reasoning = "No obstacles in trajectory. Path clear."
            return {"target_speed": self.cruise_speed, "emergency_brake": False, "reason": self.reasoning, "risk_level": "SAFE"}

        # Sort all detected obstacles (Known + Novel) by distance (closest first)
        sorted_dets = sorted(detections, key=lambda d: d["distance_m"])
        closest = sorted_dets[0]
        label = closest["label"]
        dist = closest["distance_m"]
        is_novel = closest["is_novel"]

        display_name = "NOVEL UNKNOWN OBSTACLE (Dog/Debris)" if is_novel else label

        if dist < 2.0:
            # Dangerously Close (< 2m) -> Immediate Emergency Stop
            self.risk_level = "CRITICAL"
            self.reasoning = f"CRITICAL HAZARD: {display_name} at {dist:.1f}m! Emergency Brake!"
            return {"target_speed": 0.0, "emergency_brake": True, "reason": self.reasoning, "risk_level": "CRITICAL"}

        elif dist < 5.0:
            # Medium Proximity (2m - 5m) -> Slow Down & Yield
            slow_speed = max(0.5, self.cruise_speed * (dist / 5.0))
            self.risk_level = "WARNING"
            self.reasoning = f"WARNING: {display_name} detected at {dist:.1f}m. Decelerating to {slow_speed:.1f} m/s."
            return {"target_speed": slow_speed, "emergency_brake": False, "reason": self.reasoning, "risk_level": "WARNING"}

        else:
            # Far Proximity (> 5m) -> Safe Tracking
            self.risk_level = "SAFE"
            self.reasoning = f"TRACKING: {display_name} detected at safe distance ({dist:.1f}m)."
            return {"target_speed": self.cruise_speed, "emergency_brake": False, "reason": self.reasoning, "risk_level": "SAFE"}


def test_vla_on_image(image_path_or_cam: str, model_path: str, conf_thresh: float = 0.35):
    print("=================================================================")
    print("VLA COSMOS TEST ENGINE: Open-Vocab & Distance Grounding Assessment")
    print("=================================================================")

    detector = OpenVocabYOLOXDetector(model_path, conf_thresh=conf_thresh)
    reasoner = OpenVocabVLAReasoner()

    if image_path_or_cam == "realsense" and _HAS_REALSENSE:
        print("[VLA Test] Initializing RealSense Camera...")
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
                _process_and_display(frame, detector, reasoner)
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            rs_pipeline.stop()
            cv2.destroyAllWindows()
    else:
        # Load sample image file or web camera
        if image_path_or_cam.isdigit():
            cap = cv2.VideoCapture(int(image_path_or_cam))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _process_and_display(frame, detector, reasoner)
                if cv2.waitKey(1) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            frame = cv2.imread(image_path_or_cam)
            if frame is None:
                print(f"[VLA Test] Creating synthetic test image for validation...")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Draw simulated pedestrian & novel obstacle
                cv2.rectangle(frame, (250, 150), (350, 400), (200, 200, 200), -1)  # Pedestrian
                cv2.putText(frame, "SIMULATED TARGET", (230, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            _process_and_display(frame, detector, reasoner)
            print("[VLA Test] Press any key on window or Ctrl+C to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _process_and_display(frame: np.ndarray, detector: OpenVocabYOLOXDetector, reasoner: OpenVocabVLAReasoner):
    t0 = time.time()
    detections = detector.infer(frame)
    inf_ms = (time.time() - t0) * 1000

    vla_result = reasoner.evaluate(detections)

    # Render Telemetry HUD
    h, w = frame.shape[:2]

    # Draw Bounding Boxes with Distance in Meters
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]
        dist = det["distance_m"]
        is_novel = det["is_novel"]

        color = (0, 0, 255) if is_novel else ((255, 255, 0) if label == "PEDESTRIAN" else (0, 255, 0))
        tag = f"{label} {dist:.1f}m ({conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, max(y1 - 25, 0)), (x1 + len(tag) * 9, max(y1, 25)), color, -1)
        cv2.putText(frame, tag, (x1 + 4, max(y1 - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # Top HUD Banner
    cv2.rectangle(frame, (0, 0), (w, 70), (15, 20, 28), -1)
    cv2.line(frame, (0, 70), (w, 70), (60, 80, 110), 2)

    risk = vla_result["risk_level"]
    r_color = (0, 255, 0) if risk == "SAFE" else ((0, 165, 255) if risk == "WARNING" else (0, 0, 255))
    cv2.rectangle(frame, (10, 10), (130, 60), r_color, -1)
    cv2.putText(frame, risk, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"VLA REASONING: {vla_result['reason']}", (145, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 245, 255), 1)
    cv2.putText(frame, f"ACTION: Target Speed = {vla_result['target_speed']:.1f} m/s", (145, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    cv2.putText(frame, f"{1000/max(inf_ms,1):.1f} FPS ({inf_ms:.1f}ms)", (w - 170, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)

    try:
        cv2.imshow("VLA Cosmos Open-Vocab & Distance Grounding Test", frame)
    except Exception:
        print(f"[VLA Test Output] {vla_result['reason']} | Distance Output: {[d['distance_m'] for d in detections]}m")


def main():
    parser = argparse.ArgumentParser(description="VLA Cosmos Open-Vocab & Distance Test")
    parser.add_argument("--model", default="/home/tesla/models/yolox-sPlus-opt.engine", help="Path to YOLOX model")
    parser.add_argument("--input", default="realsense", help="Input image file, camera index (0), or 'realsense'")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    test_vla_on_image(args.input, args.model, conf_thresh=args.conf)


if __name__ == "__main__":
    main()
