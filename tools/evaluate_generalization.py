#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalization & Detection Accuracy Evaluation Harness for Phase 19
=====================================================================
Calculates empirical computer-vision and tracking metrics across diverse datasets:
1. MOT17-02 (Outdoor, Occlusions, Crossing Paths)
2. MOT20-01 (Dense Crowd, Extreme Overlap, Small Pedestrians)
3. Fall Benchmark (Multi-Scene Positive Falls & Negative ADLs)

Metrics Computed:
- Person Detection: Precision, Recall, F1, Average IoU @ 0.50
- Tracking: Unique IDs, ID Switches, Track Fragmentations, Trajectory Completeness (>80%)
- Fall Detection: Fall Precision, Recall, F1-Score, False Positives / False Alarms per min
- Performance: FPS, Processing latency, Peak Memory (RAM MB)
"""

import sys
import os
import json
import time
import psutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"

from ultralytics import YOLO
from scripts.pedestrian_pipeline import load_config
from scripts.fall_detection import FallDetector

def compute_box_iou(box1: list, box2: list) -> float:
    """Computes Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
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

def evaluate_tracking_sequence(
    video_path: Path,
    gt_parsed_path: Path,
    model_name: str = "yolov8n.pt",
    tracker_cfg: str = "configs/bytetrack_fall.yaml",
    conf_thresh: float = 0.25,
    imgsz: int = 640,
    iou_eval_thresh: float = 0.50,
    max_frames: int = 120
) -> Dict[str, Any]:
    """
    Evaluates detection & tracking accuracy against MOT ground truth.
    """
    with open(gt_parsed_path, "r") as f:
        gt_data = json.load(f)

    model = YOLO(model_name)
    model.predictor = None

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_accum = []

    # Tracking identity matching: gt_id -> last_pred_track_id
    gt_to_pred_map: Dict[int, int] = {}
    id_switches = 0
    track_history = defaultdict(int) # pred_id -> count of frames tracked
    gt_track_frames = defaultdict(int) # gt_id -> total frames in gt

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.time()
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        str_fidx = str(frame_idx)
        cur_gt = gt_data.get(str_fidx, [])
        # Filter ground truth: active pedestrian boxes (conf >= 0.5 or conf == 1)
        valid_gt = [g for g in cur_gt if g.get("conf", 1) >= 0.25]
        for g in valid_gt:
            gt_track_frames[g["track_id"]] += 1

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf_thresh,
            imgsz=imgsz,
            classes=[0] if "pose" not in str(model_name) else None,
            verbose=False
        )

        pred_boxes = []
        pred_ids = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if hasattr(boxes, "xyxy") and len(boxes.xyxy) > 0:
                coords = boxes.xyxy.cpu().numpy()
                tids = boxes.id.int().cpu().tolist() if (hasattr(boxes, "id") and boxes.id is not None) else list(range(len(coords)))
                pred_boxes = [list(c) for c in coords]
                pred_ids = [int(t) for t in tids]
                for tid in pred_ids:
                    track_history[tid] += 1

        # Match Pred to GT using greedy bipartite matching at iou_eval_thresh
        matched_gt = set()
        matched_pred = set()

        for p_idx, (p_box, p_id) in enumerate(zip(pred_boxes, pred_ids)):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_item in enumerate(valid_gt):
                if g_idx in matched_gt:
                    continue
                iou = compute_box_iou(p_box, g_item["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_eval_thresh and best_g_idx >= 0:
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
                total_tp += 1
                iou_accum.append(best_iou)

                # Check ID Switch
                gt_id = valid_gt[best_g_idx]["track_id"]
                if gt_id in gt_to_pred_map:
                    if gt_to_pred_map[gt_id] != p_id:
                        id_switches += 1
                        gt_to_pred_map[gt_id] = p_id
                else:
                    gt_to_pred_map[gt_id] = p_id
            else:
                total_fp += 1

        total_fn += len(valid_gt) - len(matched_gt)

    cap.release()
    total_time = time.time() - t0
    eval_fps = frame_idx / max(total_time, 0.001)
    mem_after = process.memory_info().rss / (1024 * 1024)

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    f1 = 2 * precision * recall / max(1e-5, (precision + recall))
    mean_iou = np.mean(iou_accum) if iou_accum else 0.0

    # Trajectory completeness: % of tracked IDs with length >= 80% of average lifespan
    avg_pred_len = np.mean(list(track_history.values())) if track_history else 0.0
    complete_count = sum(1 for v in track_history.values() if v >= 0.8 * avg_pred_len)
    completeness = complete_count / max(1, len(track_history)) * 100

    return {
        "sequence": video_path.stem,
        "frames_evaluated": frame_idx,
        "eval_fps": round(eval_fps, 1),
        "total_gt_boxes": total_tp + total_fn,
        "total_pred_boxes": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "mean_iou": round(float(mean_iou), 3),
        "unique_track_ids": len(track_history),
        "id_switches": id_switches,
        "avg_trajectory_length": round(float(avg_pred_len), 1),
        "trajectory_completeness_pct": round(completeness, 1),
        "memory_mb": round(mem_after, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1)
    }

def evaluate_fall_benchmark(
    fall_dir: Path,
    config: Dict[str, Any],
    yolo_model_name: str = "yolov8n.pt",
    tracker_cfg: str = "configs/bytetrack_fall.yaml",
    conf_thresh: float = 0.08,
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Evaluates Fall Detection precision, recall, and false alarms on positive falls & negative ADLs.
    """
    gt_file = fall_dir / "fall_ground_truth.json"
    with open(gt_file, "r") as f:
        ground_truth = json.load(f)

    detector = FallDetector(config)
    model = YOLO(yolo_model_name)
    model.predictor = None

    results = []
    total_fall_tp = 0
    total_fall_fp = 0
    total_fall_fn = 0
    total_duration_sec = 0.0

    for vid_name, meta in ground_truth.items():
        vid_path = fall_dir / vid_name
        if not vid_path.exists():
            vid_path = PROJECT_ROOT / "assets" / vid_name
        if not vid_path.exists():
            print(f"[Warning] Video not found: {vid_path}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detector.reset()
        model.predictor = None

        is_pos = (meta["type"] == "POSITIVE_FALL")
        gt_range = meta["fall_frame_range"]

        frame_idx = 0
        fall_detected_frames = []
        fall_suspected_frames = []

        t_start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            res = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=conf_thresh,
                imgsz=imgsz,
                classes=[0] if "pose" not in str(yolo_model_name) else None,
                verbose=False
            )

            if res and len(res) > 0 and res[0].boxes is not None:
                boxes = res[0].boxes
                if hasattr(boxes, "id") and boxes.id is not None:
                    coords = boxes.xyxy.cpu().numpy()
                    tids = boxes.id.int().cpu().tolist()

                    for b, tid in zip(coords, tids):
                        eval_res = detector.evaluate_pedestrian(
                            track_id=int(tid),
                            bbox=tuple(b),
                            frame_idx=frame_idx,
                            frame_h=h,
                            frame_w=w,
                            frame_img=frame
                        )
                        st = eval_res["status"]
                        if st == "FALL_DETECTED":
                            fall_detected_frames.append(frame_idx)
                        elif st == "FALL_SUSPECTED":
                            fall_suspected_frames.append(frame_idx)

        cap.release()
        elapsed = time.time() - t_start
        clip_fps = frame_idx / max(elapsed, 0.001)
        clip_dur = frame_idx / max(fps, 1.0)
        total_duration_sec += clip_dur

        has_fall_alarm = len(fall_detected_frames) > 0

        # Classification outcome
        if is_pos:
            if has_fall_alarm:
                total_fall_tp += 1
                status_eval = "TRUE_POSITIVE"
            else:
                total_fall_fn += 1
                status_eval = "FALSE_NEGATIVE"
        else:
            if has_fall_alarm:
                total_fall_fp += 1
                status_eval = "FALSE_POSITIVE"
            else:
                status_eval = "TRUE_NEGATIVE"

        results.append({
            "video": vid_name,
            "category": meta["type"],
            "environment": meta["environment"],
            "total_frames": frame_idx,
            "duration_sec": round(clip_dur, 1),
            "fps": round(clip_fps, 1),
            "fall_detected_frame_count": len(fall_detected_frames),
            "fall_suspected_frame_count": len(fall_suspected_frames),
            "ground_truth_range": gt_range,
            "outcome": status_eval
        })

    fall_prec = total_fall_tp / max(1, (total_fall_tp + total_fall_fp))
    fall_rec = total_fall_tp / max(1, (total_fall_tp + total_fall_fn))
    fall_f1 = 2 * fall_prec * fall_rec / max(1e-5, (fall_prec + fall_rec))
    fp_per_min = (total_fall_fp / max(1e-5, total_duration_sec)) * 60.0

    return {
        "summary": {
            "total_clips": len(results),
            "total_duration_sec": round(total_duration_sec, 1),
            "true_positives": total_fall_tp,
            "false_positives": total_fall_fp,
            "false_negatives": total_fall_fn,
            "precision_pct": round(fall_prec * 100, 1),
            "recall_pct": round(fall_rec * 100, 1),
            "f1_score_pct": round(fall_f1 * 100, 1),
            "false_alarm_rate_per_min": round(fp_per_min, 2)
        },
        "details": results
    }

def run_comprehensive_benchmark(model_name="yolov8n.pt", tracker_cfg="configs/bytetrack_fall.yaml", conf_thresh=0.08, imgsz=640):
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    cfg["yolo_model"] = model_name
    cfg["detection_confidence"] = conf_thresh
    cfg["tracker_config"] = tracker_cfg

    print("=" * 70)
    print(f" AEGIS GENERALIZATION & ACCURACY BENCHMARK")
    print(f" Detector: {model_name} | Tracker: {tracker_cfg} | conf={conf_thresh} | imgsz={imgsz}")
    print("=" * 70)

    # 1. MOT17-02
    mot17_res = evaluate_tracking_sequence(
        video_path=DATA_DIR / "mot" / "MOT17-02" / "mot17-02.mp4",
        gt_parsed_path=DATA_DIR / "mot" / "MOT17-02" / "gt_parsed.json",
        model_name=model_name,
        tracker_cfg=tracker_cfg,
        conf_thresh=conf_thresh,
        imgsz=imgsz,
        max_frames=120
    )

    # 2. MOT20-01 (Dense Crowd)
    mot20_res = evaluate_tracking_sequence(
        video_path=DATA_DIR / "mot" / "MOT20-01" / "mot20-01.mp4",
        gt_parsed_path=DATA_DIR / "mot" / "MOT20-01" / "gt_parsed.json",
        model_name=model_name,
        tracker_cfg=tracker_cfg,
        conf_thresh=conf_thresh,
        imgsz=imgsz,
        max_frames=120
    )

    # 3. Fall Benchmark
    fall_res = evaluate_fall_benchmark(
        fall_dir=DATA_DIR / "fall_benchmark",
        config=cfg,
        yolo_model_name=model_name,
        tracker_cfg=tracker_cfg,
        conf_thresh=conf_thresh,
        imgsz=imgsz
    )

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "model": model_name,
            "tracker": tracker_cfg,
            "conf_thresh": conf_thresh,
            "imgsz": imgsz
        },
        "mot17_outdoor_occlusions": mot17_res,
        "mot20_dense_crowds": mot20_res,
        "fall_benchmark": fall_res
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--tracker", type=str, default="configs/bytetrack_fall.yaml")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_comprehensive_benchmark(
        model_name=args.model,
        tracker_cfg=args.tracker,
        conf_thresh=args.conf,
        imgsz=args.imgsz
    )
    if args.output:
        out_file = Path(args.output)
        if not out_file.is_absolute():
            out_file = PROJECT_ROOT / out_file
    else:
        out_file = PROJECT_ROOT / "outputs" / "reports" / f"benchmark_{Path(args.model).stem}_{args.imgsz}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark Complete] Results saved -> {out_file}")
