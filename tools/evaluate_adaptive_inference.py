#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 22 — Adaptive Crowd Inference Evaluation Suite
==========================================================
Benchmarks and rigorously compares 4 inference policies on identical frames
(120 frames each) of MOT17-02 and MOT20-01:

  Policy 1: Full-Frame @ 640 Baseline (Single full-frame forward pass + ByteTrack)
  Policy 2: Phase 21 Static 4-Tile Inference (Full frame + 4 upper corridor tiles on EVERY frame)
  Policy 3: Spatial ROI Optimization (Full frame + 2 targeted corridor tiles on EVERY frame)
  Policy 4: Temporal Adaptive Cadence (Full frame every frame + 2 corridor tiles on keyframes K=3)

Evaluation Integrity Protocol:
- ZERO surrogate sequential IDs: Untracked detections are assigned id=-1.
- Official motmetrics accumulator: Computes MOTA, MOTP, IDF1, and CLEAR ID Switches.
- Granular recall breakdown: Distinguishes small pedestrians (H < 60px) from standard pedestrians (H >= 60px).
- Duplicate detection measurement: Redundant predictions covering already-matched GT boxes.
- Resource Profiling: Measures effective FPS, latency breakdown, and peak process RSS memory.
- Identical test frames: Frames 1..120 evaluated identically across all configurations.
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


def evaluate_adaptive_sequence(
    video_path: Path,
    gt_parsed_path: Path,
    policy_name: str,
    policy_cfg: Dict[str, Any],
    model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.25,
    max_frames: int = 120,
    iou_eval_thresh: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates a single video sequence under the specified adaptive crowd inference policy.
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

    # Common ByteTrack tracker setup
    tracker_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.08,
        new_track_thresh=0.20,
        track_buffer=30,
        match_thresh=0.80,
        fuse_score=True
    )
    byte_tracker = BYTETracker(tracker_args)

    # Policy configuration
    mode_type = policy_cfg["type"]
    imgsz = policy_cfg.get("imgsz", 640)
    keyframe_interval = policy_cfg.get("keyframe_interval", 1)

    tiled_detector: Optional[SelectiveTiledDetector] = None
    if mode_type in ("static_4tile", "corridor_2tile", "adaptive_cadence"):
        tiled_detector = SelectiveTiledDetector(
            model=model,
            tile_size=policy_cfg.get("tile_size", (640, 540)),
            overlap=policy_cfg.get("overlap", (0.20, 0.20)),
            roi_rel=policy_cfg.get("roi_rel", (0.0, 0.15, 1.0, 0.65)),
            conf_thresh=conf_thresh,
            iou_nms_thresh=0.50,
            enabled=True
        )

    # motmetrics accumulator for tracking evaluation
    acc = mm.MOTAccumulator(auto_id=True)

    # Metrics counters
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
    tiles_total_processed = 0

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    peak_mem = mem_before

    frame_idx = 0
    t_start = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        str_fidx = str(frame_idx)
        cur_gt = gt_data.get(str_fidx, [])
        valid_gt = [g for g in cur_gt if g.get("conf", 1.0) >= 0.25]

        t_f0 = time.time()
        pred_boxes: List[List[float]] = []
        pred_ids: List[int] = []

        if mode_type == "full_frame":
            # Policy 1: Full-Frame 640 Baseline
            res = model.predict(frame, imgsz=imgsz, conf=conf_thresh, classes=[0], verbose=False)
            if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                b_xyxy = res[0].boxes.xyxy.cpu().numpy()
                b_conf = res[0].boxes.conf.cpu().numpy()
                b_cls = res[0].boxes.cls.cpu().numpy()
                det_container = SimpleDetections(b_xyxy, b_conf, b_cls)
            else:
                det_container = SimpleDetections([], [], [])
            tracks = byte_tracker.update(det_container)
            if tracks is not None and len(tracks) > 0:
                pred_boxes = [list(t[:4]) for t in tracks]
                pred_ids = [int(t[4]) for t in tracks]

        elif mode_type in ("static_4tile", "corridor_2tile"):
            # Policy 2 & 3: Sliced inference on every frame
            tracks, meta = tiled_detector.detect_and_track(frame, byte_tracker, imgsz=imgsz)
            tiles_total_processed += meta.get("tiles_processed", 0)
            if len(tracks) > 0:
                pred_boxes = [list(t[:4]) for t in tracks]
                pred_ids = [int(t[4]) for t in tracks]

        elif mode_type == "adaptive_cadence":
            # Policy 4: Full frame on every frame; Sliced tiles only on keyframes
            is_keyframe = (frame_idx % keyframe_interval == 1) or (keyframe_interval == 1)
            if is_keyframe:
                tracks, meta = tiled_detector.detect_and_track(frame, byte_tracker, imgsz=imgsz)
                tiles_total_processed += meta.get("tiles_processed", 0)
            else:
                # Fast full-frame pass
                res = model.predict(frame, imgsz=imgsz, conf=conf_thresh, classes=[0], verbose=False)
                if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                    b_xyxy = res[0].boxes.xyxy.cpu().numpy()
                    b_conf = res[0].boxes.conf.cpu().numpy()
                    b_cls = res[0].boxes.cls.cpu().numpy()
                    det_container = SimpleDetections(b_xyxy, b_conf, b_cls)
                else:
                    det_container = SimpleDetections([], [], [])
                tracks = byte_tracker.update(det_container)

            if tracks is not None and len(tracks) > 0:
                pred_boxes = [list(t[:4]) for t in tracks]
                pred_ids = [int(t[4]) for t in tracks]

        t_f_elapsed = time.time() - t_f0
        latencies.append(t_f_elapsed * 1000.0)

        # Track peak memory
        rss_cur = process.memory_info().rss / (1024 * 1024)
        if rss_cur > peak_mem:
            peak_mem = rss_cur

        # 1. Detection Evaluation (Bipartite matching)
        matched_gt = set()
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
                    total_tp += 1
                    iou_accum.append(best_iou)

                    gh = valid_gt[best_g_idx]["bbox"][3] - valid_gt[best_g_idx]["bbox"][1]
                    if gh < 60.0:
                        small_tp += 1
                    else:
                        large_tp += 1
                else:
                    total_duplicates += 1
                    total_fp += 1
            else:
                total_fp += 1

        for g_idx, g_item in enumerate(valid_gt):
            if g_idx not in matched_gt:
                total_fn += 1
                gh = g_item["bbox"][3] - g_item["bbox"][1]
                if gh < 60.0:
                    small_fn += 1
                else:
                    large_fn += 1

        # 2. Tracking Evaluation (motmetrics accumulator)
        valid_pred_tids = []
        valid_pred_boxes_for_track = []
        for p_box, p_id in zip(pred_boxes, pred_ids):
            if p_id > 0:
                valid_pred_tids.append(p_id)
                valid_pred_boxes_for_track.append(p_box)

        gt_ids = [int(g["track_id"]) for g in valid_gt]
        gt_boxes = [g["bbox"] for g in valid_gt]

        if len(gt_boxes) > 0 and len(valid_pred_boxes_for_track) > 0:
            # motmetrics.distances.iou_matrix expects [x, y, w, h] format
            gt_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in gt_boxes]
            pred_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in valid_pred_boxes_for_track]
            dists = mm.distances.iou_matrix(gt_boxes_xywh, pred_boxes_xywh, max_iou=0.50)
        elif len(gt_boxes) > 0:
            dists = np.empty((len(gt_boxes), 0), dtype=np.float32)
        else:
            dists = np.empty((0, len(valid_pred_boxes_for_track)), dtype=np.float32)

        acc.update(gt_ids, valid_pred_tids, dists)

    cap.release()
    total_time = time.time() - t_start
    overall_fps = frame_idx / max(total_time, 0.001)
    mean_lat_ms = float(np.mean(latencies)) if latencies else 0.0

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
    mt = int(mot_summary["mostly_tracked"].iloc[0])
    ml = int(mot_summary["mostly_lost"].iloc[0])

    prec = total_tp / max(1, (total_tp + total_fp))
    rec = total_tp / max(1, (total_tp + total_fn))
    f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
    mean_iou = float(np.mean(iou_accum)) if iou_accum else 0.0

    small_rec = small_tp / max(1, (small_tp + small_fn))
    large_rec = large_tp / max(1, (large_tp + large_fn))

    return {
        "policy": policy_name,
        "type": mode_type,
        "sequence": video_path.stem,
        "frames_evaluated": frame_idx,
        "fps": round(overall_fps, 2),
        "mean_latency_ms": round(mean_lat_ms, 1),
        "peak_rss_mb": round(peak_mem, 1),
        "tiles_processed_total": tiles_total_processed,
        "detection_metrics": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "duplicates": total_duplicates,
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "mean_iou_pct": round(mean_iou * 100.0, 2),
            "small_pedestrians": {
                "tp": small_tp,
                "fn": small_fn,
                "recall_pct": round(small_rec * 100.0, 2)
            },
            "standard_pedestrians": {
                "tp": large_tp,
                "fn": large_fn,
                "recall_pct": round(large_rec * 100.0, 2)
            }
        },
        "tracking_metrics": {
            "mota_pct": round(mota * 100.0, 2),
            "motp_pct": round(motp * 100.0, 2),
            "idf1_pct": round(idf1 * 100.0, 2),
            "id_switches": id_switches,
            "mostly_tracked": mt,
            "mostly_lost": ml
        }
    }


def run_adaptive_inference_benchmark(
    mot_dir: Path,
    output_json: Optional[Path] = None,
    max_frames: int = 120
) -> Dict[str, Any]:
    """
    Runs the full 4-policy benchmark across MOT17-02 and MOT20-01.
    """
    sequences = [
        {
            "name": "MOT17-02",
            "video": mot_dir / "MOT17-02" / "mot17-02.mp4",
            "gt": mot_dir / "MOT17-02" / "gt_parsed.json",
            "corridor_roi": (0.20, 0.35, 0.65, 0.60)
        },
        {
            "name": "MOT20-01",
            "video": mot_dir / "MOT20-01" / "mot20-01.mp4",
            "gt": mot_dir / "MOT20-01" / "gt_parsed.json",
            "corridor_roi": (0.10, 0.15, 0.90, 0.65)
        }
    ]

    policies = {
        "Policy 1 (Baseline Full-Frame @ 640)": {
            "type": "full_frame",
            "imgsz": 640
        },
        "Policy 2 (Phase 21 Static 4-Tile Grid)": {
            "type": "static_4tile",
            "imgsz": 640,
            "tile_size": (640, 540),
            "overlap": (0.20, 0.20),
            "roi_rel": (0.0, 0.15, 1.0, 0.65)
        },
        "Policy 3 (Spatial ROI Optimization: 2-Corridor Tiles)": {
            "type": "corridor_2tile",
            "imgsz": 640,
            "tile_size": (640, 540),
            "overlap": (0.15, 0.15),
            "roi_rel": None  # Will be set to sequence-specific corridor
        },
        "Policy 4 (Temporal Adaptive Cadence: Keyframes K=3)": {
            "type": "adaptive_cadence",
            "imgsz": 640,
            "tile_size": (640, 540),
            "overlap": (0.15, 0.15),
            "keyframe_interval": 3,
            "roi_rel": None  # Will be set to sequence-specific corridor
        }
    }

    all_results = []
    print("=" * 80)
    print(" AEGIS Adaptive Crowd Inference Benchmark — Phase 22")
    print("=" * 80)

    for seq in sequences:
        print(f"\n>>> SEQUENCE: {seq['name']} (120 frames)")
        for pol_name, pol_cfg in policies.items():
            cfg_copy = dict(pol_cfg)
            if cfg_copy["type"] in ("corridor_2tile", "adaptive_cadence"):
                cfg_copy["roi_rel"] = seq["corridor_roi"]

            print(f"  Running: {pol_name}...")
            res = evaluate_adaptive_sequence(
                video_path=seq["video"],
                gt_parsed_path=seq["gt"],
                policy_name=pol_name,
                policy_cfg=cfg_copy,
                max_frames=max_frames
            )
            all_results.append(res)

            dm = res["detection_metrics"]
            tm = res["tracking_metrics"]
            print(f"    FPS: {res['fps']} | Latency: {res['mean_latency_ms']}ms | Peak RAM: {res['peak_rss_mb']}MB")
            print(f"    Det: Recall={dm['recall_pct']}%, Small Recall={dm['small_pedestrians']['recall_pct']}%, Dups={dm['duplicates']}")
            print(f"    Track: MOTA={tm['mota_pct']}%, IDF1={tm['idf1_pct']}%, ID Switches={tm['id_switches']}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "Phase 22 Adaptive Crowd Inference",
        "frames_per_sequence": max_frames,
        "results": all_results
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[✓] Saved adaptive inference benchmark report to: {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS Adaptive Inference Benchmark")
    parser.add_argument("--mot-dir", type=str, default="data/mot", help="Path to MOT directory")
    parser.add_argument("--max-frames", type=int, default=120, help="Frames to evaluate per sequence")
    parser.add_argument("--output", type=str, default="outputs/reports/phase23_adaptive_benchmark.json", help="Output path")
    args = parser.parse_args()

    run_adaptive_inference_benchmark(
        mot_dir=PROJECT_ROOT / args.mot_dir,
        output_json=PROJECT_ROOT / args.output,
        max_frames=args.max_frames
    )
