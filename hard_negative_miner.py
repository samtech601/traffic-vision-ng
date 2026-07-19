"""
hard_negative_miner.py
─────────────────────────────────────────────────────────────────────────────
Scans one or more videos with your trained model and collects every
HIGH-CONFIDENCE detection of the classes you're currently confusing
(default: truck vs bus vs van), saving:

  1. A tight crop of each detection      -> crops/<label>/frame123_id45.jpg
  2. The full frame with box drawn       -> context/<label>/frame123_id45.jpg
  3. A manifest.csv you can sort/filter  -> hard_negatives/manifest.csv

Why this matters: the model is often right MOST of the time on a class —
what actually fixes confident-wrong predictions is finding the specific
cases it gets wrong and feeding those back into training with correct
labels (hard-negative mining), not just adding more generic data.

WORKFLOW:
  1. Run this script on your training/deployment videos.
  2. Open hard_negatives/manifest.csv in Excel/Sheets, sorted by confidence
     (descending) — these are the model's most confident predictions, so
     any WRONG ones here are exactly what's confusing it.
  3. Open crops/truck/, crops/bus/, and crops/van/ folders, eyeball each crop next to
     its context frame (context/<label>/ has the same filename with the box
     drawn on the full scene, for reference).
  4. For every crop that's actually mislabeled, drag it into a "relabel"
     pile. Upload those + the context frame to Roboflow with the CORRECT
     class, then retrain.

USAGE:
    python hard_negative_miner.py --video project.mp4 --model best1.pt
    python hard_negative_miner.py --video clip1.mp4 clip2.mp4 --model best1.pt --classes truck bus van --conf 0.6 --every 15
"""

import argparse
import csv
import os
from datetime import datetime

import cv2
from ultralytics import YOLO


def mine_hard_negatives(video_paths, model_path, watch_classes,
                        min_conf, sample_every, output_dir,
                        detect_w=640, detect_h=480):
    model = YOLO(model_path)
    print("Model classes:", model.names)

    crops_dir   = os.path.join(output_dir, "crops")
    context_dir = os.path.join(output_dir, "context")
    os.makedirs(crops_dir,   exist_ok=True)
    os.makedirs(context_dir, exist_ok=True)

    for cls in watch_classes:
        os.makedirs(os.path.join(crops_dir,   cls), exist_ok=True)
        os.makedirs(os.path.join(context_dir, cls), exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    write_header   = not os.path.exists(manifest_path)
    manifest_file  = open(manifest_path, "a", newline="")
    manifest       = csv.writer(manifest_file)
    if write_header:
        manifest.writerow(["video", "frame_number", "label", "confidence",
                           "x1", "y1", "x2", "y2", "crop_path", "context_path",
                           "saved_at"])

    total_saved = 0

    for video_path in video_paths:
        print(f"\nScanning {video_path} ...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Could not open {video_path}, skipping.")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_id = 0
        saved_this_video = 0

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Sample every N frames — no need to scan every single frame,
            # and it avoids saving near-duplicate crops of the same vehicle.
            if frame_id % sample_every != 0:
                continue

            frame = cv2.resize(raw_frame, (detect_w, detect_h))

            results = model.predict(frame, conf=0.25, verbose=False)
            boxes   = results[0].boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                label  = model.names.get(cls_id, str(cls_id))
                conf   = float(box.conf[0])

                if label not in watch_classes or conf < min_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad = 6
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                fname = f"{video_name}_frame{frame_id}_conf{int(conf*100)}.jpg"
                crop_path    = os.path.join(crops_dir, label, fname)
                context_path = os.path.join(context_dir, label, fname)

                cv2.imwrite(crop_path, crop)

                context_frame = frame.copy()
                cv2.rectangle(context_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(context_frame, f"{label} {conf:.0%}", (x1, max(0, y1 - 8)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(context_path, context_frame)

                manifest.writerow([video_name, frame_id, label, f"{conf:.3f}",
                                   x1, y1, x2, y2, crop_path, context_path,
                                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

                total_saved      += 1
                saved_this_video += 1

        cap.release()
        print(f"  Saved {saved_this_video} high-confidence crops from this video.")

    manifest_file.close()
    print(f"\nDone. {total_saved} crops saved to '{output_dir}/'.")
    print(f"Manifest: {manifest_path}")
    print("\nNext step: open the manifest, sort by confidence, and check the")
    print("crops/context images for each watched class — any that are")
    print("actually mislabeled are your hard negatives for retraining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine high-confidence detections for manual review/relabeling.")
    parser.add_argument("--video", nargs="+", required=True, help="One or more video file paths")
    parser.add_argument("--model", default="best1.pt", help="Path to your trained .pt model")
    parser.add_argument("--classes", nargs="+", default=["truck", "bus", "van"],
                        help="Which class labels to watch (must match model.names)")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="Minimum confidence to save a detection (default 0.6 — 'confidently wrong' cases)")
    parser.add_argument("--every", type=int, default=15,
                        help="Sample every Nth frame (default 15 — avoids near-duplicate crops)")
    parser.add_argument("--output", default="hard_negatives",
                        help="Output folder for crops/context/manifest")
    args = parser.parse_args()

    mine_hard_negatives(
        video_paths=args.video,
        model_path=args.model,
        watch_classes=args.classes,
        min_conf=args.conf,
        sample_every=args.every,
        output_dir=args.output,
    )
