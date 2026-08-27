#!/usr/bin/env python3
"""
Standalone YOLOX Perception & Zenoh Bridge — AGX ORIN (Pure Python, NO ROS2)
==============================================================================
Runs YOLOX object detection on AGX Orin using ONNXRuntime-GPU / TensorRT.
No ROS 2 installation required.

Pipeline:
  1. Capture camera frames directly via RealSense (or Zenoh stream)
  2. Perform YOLOX inference using ~/models/yolox-sPlus-opt.onnx (or .engine)
  3. Pair detections with kinematic state received from Laptop via Zenoh
  4. Publish detection results / control commands over Zenoh to NUC

Dependencies on Orin:
    pip install eclipse-zenoh opencv-python onnxruntime-gpu pyrealsense2 numpy

Usage:
    python3 orin_yolox_standalone.py --model ~/models/yolox-sPlus-opt.onnx
    python3 orin_yolox_standalone.py --model ~/models/yolox-sPlus-opt.onnx --demo
"""

import argparse
import json
import time
import threading
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

# ---- Zenoh Topics ----
KEY_KINEMATIC = "aimslab/laptop/localization/kinematic_state"
KEY_PERCEPTION = "aimslab/orin/perception/objects"
KEY_CONTROL = "aimslab/orin/control_cmd"

# Autoware / COCO class labels for YOLOX
CLASS_NAMES = [
    "car", "truck", "bus", "pedestrian", "cyclist",
    "motorcycle", "trailer", "obstacle"
]
try:
    import tensorrt as trt
    import ctypes
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False


class TensorRTEngine:
    """Native TensorRT Engine Execution via libcudart."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[YOLOX] Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Load CUDA runtime library
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

        # Create a dedicated CUDA stream for asynchronous execution
        self.stream = ctypes.c_void_p()
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))

    def infer(self, blob_np: np.ndarray) -> np.ndarray:
        # Host to Device (async)
        blob_contig = np.ascontiguousarray(blob_np, dtype=self.input_dtype)
        self.cudart.cudaMemcpyAsync(self.input_ptr, blob_contig.ctypes.data, self.input_size, 1, self.stream)

        # Set addresses for inputs and outputs
        for i in range(self.num_tensors):
            tname = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tname, self.bindings[i])

        # Execute using non-default stream
        self.context.execute_async_v3(self.stream.value)

        # Device to Host (async) and synchronize stream
        self.cudart.cudaMemcpyAsync(self.output_host.ctypes.data, self.output_ptr, self.output_size, 2, self.stream)
        self.cudart.cudaStreamSynchronize(self.stream)
        return self.output_host


class YOLOXDetector:
    """Standalone YOLOX Inference Engine using TensorRT Engine / ONNXRuntime GPU / OpenCV DNN."""

    def __init__(self, model_path: str, conf_thresh: float = 0.3, nms_thresh: float = 0.45, input_size: tuple = (640, 640)):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_w, self.input_h = input_size
        self.backend = None

        # 1. Native TensorRT .engine file
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
                print("[YOLOX] ERROR: tensorrt Python package is missing for .engine file.")

        # 2. Try ONNXRuntime with TensorRT / CUDA
        if self.backend is None and _HAS_ORT and model_path.endswith('.onnx'):
            try:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[YOLOX] Loading model with ONNXRuntime GPU/TensorRT: {model_path}")
                self.session = ort.InferenceSession(model_path, providers=providers)
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                shape = model_inputs[0].shape
                if isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_h, self.input_w = shape[2], shape[3]
                self.backend = "ort"
                print(f"[YOLOX] ONNXRuntime GPU backend ready ({self.input_w}x{self.input_h})")
            except Exception as e:
                print(f"[YOLOX] ONNXRuntime init notice: {e}")

        # 3. Try OpenCV DNN
        if self.backend is None and model_path.endswith('.onnx'):
            print(f"[YOLOX] Loading model with OpenCV DNN: {model_path}")
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
        """Autoware tensorrt_yolox Preprocessing: Centered Letterbox + RGB float32."""
        h, w = img.shape[:2]
        r = min(self.input_h / h, self.input_w / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (self.input_w - new_unpad[0]) / 2.0
        dh = (self.input_h - new_unpad[1]) / 2.0

        if (w, h) != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img.copy()

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        if self.backend in ["ort", "trt_engine"]:
            blob = img_padded.transpose((2, 0, 1))[::-1]  # BGR to RGB (Autoware standard)
            blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0  # Normalize to [0.0, 1.0]
            blob = np.expand_dims(blob, axis=0)
        else:
            blob = cv2.dnn.blobFromImage(img_padded, scalefactor=1.0 / 255.0, size=(self.input_w, self.input_h), swapRB=True)

        return blob, r, (dw, dh)

    def infer(self, img: np.ndarray):
        """Run object detection on BGR image."""
        blob, ratio, (dw, dh) = self.preprocess(img)

        if self.backend == "trt_engine":
            predictions = self.trt_engine.infer(blob)
        elif self.backend == "ort":
            outputs = self.session.run(None, {self.input_name: blob})
            predictions = outputs[0]
        else:
            self.net.setInput(blob)
            predictions = self.net.forward()

        boxes, scores, class_ids = self.postprocess(predictions[0], ratio, dw, dh, img.shape)
        return boxes, scores, class_ids

    def postprocess(self, predictions: np.ndarray, ratio: float, dw: float, dh: float, orig_shape: tuple):
        """Decode YOLOX predictions and apply NMS."""
        if len(predictions.shape) == 3:
            predictions = predictions[0]

        # Extract confidence scores
        obj_conf = predictions[:, 4]
        class_conf = predictions[:, 5:]
        class_ids = np.argmax(class_conf, axis=1)
        scores = obj_conf * class_conf[np.arange(len(class_ids)), class_ids]

        mask = scores > self.conf_thresh
        if not np.any(mask):
            return [], [], []

        boxes = predictions[mask, :4]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert [center_x, center_y, w, h] to [x1, y1, x2, y2] centered unpadded coordinates
        x1 = (boxes[:, 0] - boxes[:, 2] / 2.0 - dw) / ratio
        y1 = (boxes[:, 1] - boxes[:, 3] / 2.0 - dh) / ratio
        x2 = (boxes[:, 0] + boxes[:, 2] / 2.0 - dw) / ratio
        y2 = (boxes[:, 1] + boxes[:, 3] / 2.0 - dh) / ratio

        # Clip to original image boundaries
        h, w = orig_shape[:2]
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        final_boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            final_boxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )

        if len(indices) == 0:
            return [], [], []

        indices = np.array(indices).flatten()
        return final_boxes[indices], scores[indices], class_ids[indices]


class StandaloneOrinPipeline:
    """Complete Standalone Orin Pipeline: Camera + YOLOX + Zenoh."""

    def __init__(self, model_path: str, zenoh_port: int = 7447):
        self.detector = YOLOXDetector(model_path)
        self.zenoh_port = zenoh_port

        self.latest_kinematic = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        # Open Zenoh session
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"router"')
        cfg.insert_json5("listen/endpoints", f'["tcp/0.0.0.0:{self.zenoh_port}"]')
        print(f"[OrinPipeline] Starting Zenoh router on 0.0.0.0:{self.zenoh_port}...")
        self.session = zenoh.open(cfg)

        # Subscribe to kinematic state from laptop
        self.sub_kinematic = self.session.declare_subscriber(KEY_KINEMATIC, self._cb_kinematic)
        # Publisher for perception output (lightweight JSON detections + kinematics)
        self.pub_perception = self.session.declare_publisher(KEY_PERCEPTION)
        print("[OrinPipeline] Zenoh router active. Ready for processing.")

    def _cb_kinematic(self, sample: zenoh.Sample):
        try:
            data = json.loads(sample.payload.to_bytes().decode())
            with self.lock:
                self.latest_kinematic = data
        except Exception as e:
            print(f"[OrinPipeline] Kinematic error: {e}")

    def process_frame(self, frame_bgr: np.ndarray):
        """Run local YOLOX detection on AGX Orin GPU and publish JSON detections + kinematics over Zenoh."""
        t0 = time.time()
        boxes, scores, class_ids = self.detector.infer(frame_bgr)
        inference_time_ms = (time.time() - t0) * 1000

        detections = []
        for box, score, cid in zip(boxes, scores, class_ids):
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            detections.append({
                "label": label,
                "confidence": float(score),
                "bbox": [float(v) for v in box]  # [x1, y1, x2, y2]
            })

        with self.lock:
            kinematic = self.latest_kinematic

        # Pack payload for downstream VLA / NUC controller
        payload = json.dumps({
            "timestamp": time.time(),
            "inference_ms": round(inference_time_ms, 2),
            "detections": detections,
            "kinematic": kinematic
        }).encode()

        self.pub_perception.put(payload)
        return detections, inference_time_ms

    def stop(self):
        self.running = False
        if hasattr(self, 'session'):
            self.session.close()
        print("[OrinPipeline] Stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/tesla/models/yolox-sPlus-opt.engine",
                        help="Path to YOLOX Engine (.engine) or ONNX (.onnx) model file")
    parser.add_argument("--demo", action="store_true", help="Display live detection window")
    args = parser.parse_args()

    pipeline = StandaloneOrinPipeline(args.model)
    pipeline.start()

    print("[OrinPipeline] Starting RealSense camera capture...")
    if not _HAS_REALSENSE:
        print("[OrinPipeline] ERROR: pyrealsense2 not installed.")
        return

    rs_pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_pipeline.start(config)

    try:
        while True:
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            detections, inf_time = pipeline.process_frame(frame_bgr)

            if args.demo:
                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"{det['label']} {det['confidence']:.2f}",
                                (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame_bgr, f"FPS: {1000/max(inf_time,1):.1f} ({inf_time:.1f}ms)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                try:
                    cv2.imshow("Orin Standalone YOLOX Perception", frame_bgr)
                    if cv2.waitKey(1) == ord('q'):
                        break
                except Exception as e:
                    print(f"[OrinPipeline] Headless display mode (no GUI window): {e}")
                    args.demo = False  # Disable GUI mode to prevent loop errors
    except KeyboardInterrupt:
        pass
    finally:
        rs_pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pipeline.stop()


if __name__ == "__main__":
    main()
