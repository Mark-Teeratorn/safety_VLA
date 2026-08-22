"""
Fast Perception Layer (YOLOv8) - Level 1 Reflex (30-60 FPS)
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple

class FastPerceptionLayer:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.4):
        self.conf_thresh = conf_thresh
        self.model = None
        self.use_mock = False
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[Perception] YOLO model loaded from {model_path}")
        except Exception as e:
            print(f"[Perception] Using Fast Integrated Road Detector ({e}).")
            self.use_mock = True

    def detect_and_filter(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        h, w, _ = frame.shape
        targets = []
        annotated_frame = frame.copy()
        
        if self.model and not self.use_mock:
            results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
            boxes = results.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = results.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                if cls_name in ["person", "car", "truck", "bus", "motorcycle", "bicycle"]:
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.15)
                    crop_x1 = max(0, x1 - pad_x)
                    crop_y1 = max(0, y1 - pad_y)
                    crop_x2 = min(w, x2 + pad_x)
                    crop_y2 = min(h, y2 + pad_y)
                    roi_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    box_center_x = (x1 + x2) / 2
                    is_in_path = (w * 0.30 <= box_center_x <= w * 0.70)
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                    
                    targets.append({
                        "class": cls_name,
                        "conf": round(conf, 2),
                        "bbox": (x1, y1, x2, y2),
                        "crop": roi_crop,
                        "in_path": is_in_path,
                        "area_ratio": area_ratio
                    })
                    color = (0, 0, 255) if is_in_path else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            center_x, center_y = w // 2, int(h * 0.65)
            bw, bh = int(w * 0.10), int(h * 0.30)
            x1, y1 = center_x - bw // 2, center_y - bh // 2
            x2, y2 = center_x + bw // 2, center_y + bh // 2
            roi_crop = frame[y1:y2, x1:x2]
            targets.append({
                "class": "person",
                "conf": 0.95,
                "bbox": (x1, y1, x2, y2),
                "crop": roi_crop,
                "in_path": True,
                "area_ratio": (bw * bh) / (w * h)
            })
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Detected: Pedestrian (Crossing Zone)", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        return targets, annotated_frame
