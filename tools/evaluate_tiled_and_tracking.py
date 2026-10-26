#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 21 — Selective Tiled Inference and Rigorous Tracking Evaluation Suite
================================================================================
Compares detection, tracking, small-object recall, and resource efficiency across:
  A. YOLOv8n Full-Frame @ 640 (Baseline)
  B. YOLOv8n Full-Frame @ 800 (High-Resolution Baseline)
  C. Selective Tiled Inference (640x540 tiles, 20% overlap, Corridor ROI, Cross-Tile NMS)
  D. Selective Tiled Inference + ByteTrack Integration

Evaluation Integrity Rules:
- ZERO surrogate sequential IDs: Missing or unassociated track IDs are never fabricated.
- Official motmetrics accumulator: Computes MOTA, MOTP, IDF1, and CLEAR ID Switches.
- Granular recall breakdown: Distinguishes small pedestrians (H < 60px) from standard pedestrians (H >= 60px).
- Duplicate detection measurement: Quantifies redundant boxes overlapping ground truth.
- Resource Profiling: Measures FPS, ms/frame latency breakdown, and peak RSS memory.
- Identical test frames: 120 frames evaluated identically across MOT17-02 and MOT20-01.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import motmetrics as mm
import psutil
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiled_inference import SelectiveTiledDetector, SimpleDetections


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_sequence_mode(
    video_path: Path,
    gt_parsed_path: Path,
    mode: str,  # 'A', 'B', 'C', 'D'
    model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.25,
    max_frames: int = 120,
    iou_eval_thresh: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates a video sequence under one of the four experimental configurations.
    """
    with open(gt_parsed_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    model = YOLO(model_name)
    model.predictor = None

    # Configure mode-specific detector / tracker
    imgsz = 640
    tiled_detector = None
    byte_tracker = None

    if mode == "A":
        # Full-frame @ 640
        imgsz = 640
    elif mode == "B":
        # Full-frame @ 800
        imgsz = 800
    elif mode == "C":
        # Selective tiled inference (pure detection)
        imgsz = 640
        tiled_detector = SelectiveTiledDetector(
            model=model,
            tile_size=(640, 540),
            overlap=(0.20, 0.20),
            roi_rel=(0.0, 0.15, 1.0, 0.65),
            conf_thresh=conf_thresh,
            iou_nms_thresh=0.50,
            enabled=True
        )
    elif mode == "D":
        # Selective tiled inference + ByteTrack
        imgsz = 640
        tiled_detector = SelectiveTiledDetector(
            model=model,
            tile_size=(640, 540),
            overlap=(0.20, 0.20),
            roi_rel=(0.0, 0.15, 1.0, 0.65),
            conf_thresh=conf_thresh,
            iou_nms_thresh=0.50,
            enabled=True
        )
        tracker_args = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.08,
            new_track_thresh=0.20,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True
        )
        byte_tracker = BYTETracker(tracker_args)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # motmetrics accumulator for tracking evaluation
    acc = mm.MOTAccumulator(auto_id=True)

    # Detection counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_duplicates = 0
    small_tp = 0
    small_fn = 0
    large_tp = 0
    large_fn = 0
    iou_accum = []
    latencies = []

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    frame_idx = 0
    t_start = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        str_fidx = str(frame_idx)
        cur_gt = gt_data.get(str_fidx, [])
        # Active pedestrians in ground truth (conf >= 0.25)
        valid_gt = [g for g in cur_gt if g.get("conf", 1.0) >= 0.25]

        t_f0 = time.time()
        pred_boxes: List[List[float]] = []
        pred_ids: List[int] = []  # Strictly integer >= 1 or empty

        if mode == "A" or mode == "B":
            res = model.predict(
                frame,
                imgsz=imgsz,
                conf=conf_thresh,
                classes=[0],
                verbose=False
            )
            if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                coords = res[0].boxes.xyxy.cpu().numpy()
                pred_boxes = [list(c) for c in coords]
                # Detection mode: No tracking IDs
                pred_ids = [-1] * len(pred_boxes)

        elif mode == "C":
            boxes, scores, classes, meta = tiled_detector.detect(frame, imgsz=imgsz)
            pred_boxes = [list(b) for b in boxes]
            pred_ids = [-1] * len(pred_boxes)

        elif mode == "D":
            tracks, meta = tiled_detector.detect_and_track(frame, byte_tracker, imgsz=imgsz)
            if len(tracks) > 0:
                # tracks rows: [x1, y1, x2, y2, track_id, score, cls, idx]
                pred_boxes = [list(t[:4]) for t in tracks]
                pred_ids = [int(t[4]) for t in tracks]

        t_f_elapsed = time.time() - t_f0
        latencies.append(t_f_elapsed * 1000.0)

        # 1. Detection Evaluation (Bipartite matching)
        matched_gt = set()
        matched_pred = set()
        gt_matched_counts = [0] * len(valid_gt)

        # Match predictions to GT greedily
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_item in enumerate(valid_gt):
                iou = compute_box_iou(p_box, g_item["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_eval_thresh and best_g_idx >= 0:
                if best_g_idx not in matched_gt:
                    matched_gt.add(best_g_idx)
                    matched_pred.add(p_idx)
                    total_tp += 1
                    iou_accum.append(best_iou)

                    # Classify small vs large pedestrian
                    gh = valid_gt[best_g_idx]["bbox"][3] - valid_gt[best_g_idx]["bbox"][1]
                    if gh < 60.0:
                        small_tp += 1
                    else:
                        large_tp += 1
                else:
                    # Redundant prediction covering already-matched GT box: duplicate!
                    total_duplicates += 1
                    total_fp += 1
                gt_matched_counts[best_g_idx] += 1
            else:
                total_fp += 1

        # False negatives (unmatched ground truth)
        for g_idx, g_item in enumerate(valid_gt):
            if g_idx not in matched_gt:
                total_fn += 1
                gh = g_item["bbox"][3] - g_item["bbox"][1]
                if gh < 60.0:
                    small_fn += 1
                else:
                    large_fn += 1

        # 2. Tracking Evaluation (motmetrics accumulator)
        # Only tracks with genuine IDs (tid > 0) are evaluated for identity consistency
        valid_pred_tids = []
        valid_pred_boxes_for_track = []
        for p_box, p_id in zip(pred_boxes, pred_ids):
            if p_id > 0:
                valid_pred_tids.append(p_id)
                valid_pred_boxes_for_track.append(p_box)

        gt_ids = [int(g["track_id"]) for g in valid_gt]
        gt_boxes = [g["bbox"] for g in valid_gt]

        if len(gt_boxes) > 0 and len(valid_pred_boxes_for_track) > 0:
            # iou_matrix computes (1 - IoU). Threshold at 0.50 (max_iou=0.50 in motmetrics means dist <= 0.50)
            dists = mm.distances.iou_matrix(gt_boxes, valid_pred_boxes_for_track, max_iou=0.50)
        elif len(gt_boxes) > 0:
            dists = np.empty((len(gt_boxes), 0), dtype=np.float32)
        else:
            dists = np.empty((0, len(valid_pred_boxes_for_track)), dtype=np.float32)

        acc.update(gt_ids, valid_pred_tids, dists)

    cap.release()
    total_time = time.time() - t_start
    overall_fps = frame_idx / max(total_time, 0.001)
    mean_lat_ms = float(np.mean(latencies)) if latencies else 0.0
    mem_after = process.memory_info().rss / (1024 * 1024)

    # Compute MOT metrics
    mh = mm.metrics.create()
    mot_summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "num_switches", "mostly_tracked", "mostly_lost"],
        name=video_path.stem
    )

    mota = float(mot_summary["mota"].iloc[0]) if not np.isnan(mot_summary["mota"].iloc[0]) else 0.0
    motp = float(mot_summary["motp"].iloc[0]) if not np.isnan(mot_summary["motp"].iloc[0]) else 0.0
    idf1 = float(mot_summary["idf1"].iloc[0]) if not np.isnan(mot_summary["idf1"].iloc[0]) else 0.0
    id_switches = int(mot_summary["num_switches"].iloc[0])

    prec = total_tp / max(1, (total_tp + total_fp))
    rec = total_tp / max(1, (total_tp + total_fn))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
    mean_iou = float(np.mean(iou_accum)) if iou_accum else 0.0

    small_rec = small_tp / max(1, (small_tp + small_fn))
    large_rec = large_tp / max(1, (large_tp + large_fn))

    return {
        "mode": mode,
        "sequence": video_path.stem,
        "frames_evaluated": frame_idx,
        "eval_fps": round(overall_fps, 1),
        "mean_latency_ms": round(mean_lat_ms, 1),
        "memory_rss_mb": round(mem_after, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "detection_metrics": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "mean_iou": round(mean_iou, 3),
            "duplicate_detections": total_duplicates,
            "small_pedestrian_recall_pct": round(small_rec * 100.0, 2),
            "small_pedestrian_gt_count": small_tp + small_fn,
            "large_pedestrian_recall_pct": round(large_rec * 100.0, 2),
            "large_pedestrian_gt_count": large_tp + large_fn
        },
        "tracking_metrics": {
            "mota_pct": round(mota * 100.0, 2),
            "motp_pct": round((1.0 - motp) * 100.0, 2),  # motmetrics MOTP is avg dist (1-IoU)
            "idf1_pct": round(idf1 * 100.0, 2),
            "id_switches": id_switches
        }
    }


def run_phase21_benchmark_suite(
    mot_dir: Path,
    output_json: Path,
    max_frames: int = 120
) -> Dict[str, Any]:
    """
    Runs the full 4-mode benchmark suite (A, B, C, D) across MOT17-02 and MOT20-01.
    """
    sequences = [
        ("MOT17-02", mot_dir / "MOT17-02" / "mot17-02.mp4", mot_dir / "MOT17-02" / "gt_parsed.json"),
        ("MOT20-01", mot_dir / "MOT20-01" / "mot20-01.mp4", mot_dir / "MOT20-01" / "gt_parsed.json")
    ]

    modes = [
        ("A", "YOLOv8n Full-Frame @ 640"),
        ("B", "YOLOv8n Full-Frame @ 800"),
        ("C", "Selective Tiled Inference (640x540 tiles, Corridor ROI)"),
        ("D", "Selective Tiled Inference + ByteTrack")
    ]

    suite_results = []

    print("=" * 75)
    print(" AEGIS Phase 21 — Selective Tiled Inference Benchmark Suite")
    print(f" Sequences: MOT17-02 (Outdoor), MOT20-01 (Dense Crowd) | Frames: {max_frames}")
    print("=" * 75)

    for seq_name, vid_path, gt_path in sequences:
        if not vid_path.exists() or not gt_path.exists():
            print(f"[!] Warning: Missing sequence files for {seq_name}")
            continue

        print(f"\n>>> Running Sequence: {seq_name}")
        for mode_code, mode_desc in modes:
            print(f"  Executing Mode {mode_code}: {mode_desc}...")
            res = evaluate_sequence_mode(
                video_path=vid_path,
                gt_parsed_path=gt_path,
                mode=mode_code,
                max_frames=max_frames
            )
            res["description"] = mode_desc
            suite_results.append(res)

            dm = res["detection_metrics"]
            tm = res["tracking_metrics"]
            print(f"    [Mode {mode_code}] P={dm['precision_pct']}%, R={dm['recall_pct']}%, F1={dm['f1_score_pct']}% | Small R={dm['small_pedestrian_recall_pct']}% | Dups={dm['duplicate_detections']}")
            print(f"            IDSW={tm['id_switches']}, IDF1={tm['idf1_pct']}%, MOTA={tm['mota_pct']}% | Latency={res['mean_latency_ms']}ms, FPS={res['eval_fps']}, RSS={res['memory_rss_mb']}MB")

    suite_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_frames_per_sequence": max_frames,
        "modes_evaluated": dict(modes),
        "results": suite_results
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)
    print(f"\n[✓] Saved complete Phase 21 benchmark results to: {output_json}")

    return suite_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 21 Benchmark Suite")
    parser.add_argument("--mot-dir", type=str, default="data/mot", help="Path to MOT directory")
    parser.add_argument("--max-frames", type=int, default=120, help="Number of frames per sequence")
    parser.add_argument("--output", type=str, default="outputs/reports/phase21_tiled_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    run_phase21_benchmark_suite(
        mot_dir=PROJECT_ROOT / args.mot_dir,
        output_json=PROJECT_ROOT / args.output,
        max_frames=args.max_frames
    )
