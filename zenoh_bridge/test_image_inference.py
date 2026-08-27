#!/usr/bin/env python3
"""
Standalone YOLOX Image Interface & Tester — AGX ORIN / Laptop
==============================================================
Runs YOLOX object detection on single images, image folders, or saved frames.
Saves annotated images with bounding boxes, labels, and confidence scores.

Usage:
    # Test a single image:
    python3 test_image_inference.py --image sample.jpg --model ~/models/yolox-sPlus-opt.engine --output result.jpg

    # Test an entire directory of images:
    python3 test_image_inference.py --dir ./test_images/ --model ~/models/yolox-sPlus-opt.engine --output-dir ./results/
"""

import argparse
import os
import time
import cv2
import numpy as np

# Import YOLOXDetector from standalone module
from orin_yolox_standalone import YOLOXDetector, CLASS_NAMES


def process_single_image(detector: YOLOXDetector, img_path: str, output_path: str = None):
    if not os.path.exists(img_path):
        print(f"[ERROR] Image path does not exist: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return

    t0 = time.time()
    boxes, scores, class_ids = detector.infer(img)
    inf_time_ms = (time.time() - t0) * 1000

    print(f"\n--- Detection Results for: {os.path.basename(img_path)} ---")
    print(f"Inference Time: {inf_time_ms:.2f} ms ({1000/max(inf_time_ms,1):.1f} FPS)")
    print(f"Total Objects Detected: {len(boxes)}")

    annotated = img.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        print(f"  - [{label}] confidence: {score:.2f} | box: [{x1}, {y1}, {x2}, {y2}]")

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label header box
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw summary banner
    banner = f"FPS: {1000/max(inf_time_ms,1):.1f} ({inf_time_ms:.1f}ms) | Detections: {len(boxes)}"
    cv2.putText(annotated, banner, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"[Saved] Annotated result saved to: {output_path}")

    return annotated, boxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser(description="YOLOX Standalone Image Inference Tester")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory containing input images")
    parser.add_argument("--model", type=str, default="/home/tesla/models/yolox-sPlus-opt.engine",
                        help="Path to .engine or .onnx model file")
    parser.add_argument("--output", type=str, default="output_result.jpg", help="Output path for single image result")
    parser.add_argument("--output-dir", type=str, default="output_results/", help="Output directory for folder results")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default 0.3)")
    args = parser.parse_args()

    detector = YOLOXDetector(args.model, conf_thresh=args.conf)

    if args.image:
        process_single_image(detector, args.image, args.output)
    elif args.dir:
        os.makedirs(args.output_dir, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(args.dir) if f.lower().endswith(exts)]
        print(f"[ImageInterface] Processing {len(files)} images in directory: {args.dir}")

        for fname in sorted(files):
            in_path = os.path.join(args.dir, fname)
            out_path = os.path.join(args.output_dir, f"det_{fname}")
            process_single_image(detector, in_path, out_path)
    else:
        print("[ImageInterface] Please specify --image <path> or --dir <folder_path>")


if __name__ == "__main__":
    main()
