#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Temporal Fall Detection Evaluation Harness
=================================================
Implements rigorous frame-level and event-level temporal evaluation for fall detection.

Replaces coarse clip-level binary classification with:
1. Frame-level Confusion Matrix: TP, FP, FN, TN, Precision, Recall, F1
2. Event-level Metrics: Event Recall, Missed Events, Detection Delay (latency in seconds)
3. Continuous Alert Clustering: Merges consecutive alarm frames into discrete alarm events
4. Coverage Metric: Percentage of annotated fall duration during which system maintained alert
5. False Alarms per Minute on Negative ADL clips
6. Explicit semantic separation: FALL_SUSPECTED vs FALL_DETECTED
7. Train/Dev vs Test split separation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fall_detection import FallDetector
from scripts.pedestrian_pipeline import load_config


def cluster_alarm_frames(alarm_frames: List[int], max_gap: int = 5) -> List[Tuple[int, int]]:
    """
    Clusters a sorted list of alarm frame numbers into discrete contiguous event windows [start, end].
    Tolerates transient drops up to max_gap frames.
    """
    if not alarm_frames:
        return []

    sorted_frames = sorted(set(alarm_frames))
    clusters = []
    start = sorted_frames[0]
    prev = sorted_frames[0]

    for f in sorted_frames[1:]:
        if f - prev <= max_gap + 1:
            prev = f
        else:
            clusters.append((start, prev))
            start = f
            prev = f
    clusters.append((start, prev))
    return clusters


def evaluate_video_temporal_fall(
    video_path: Path,
    meta: Dict[str, Any],
    fall_detector: FallDetector,
    yolo_model: YOLO,
    tracker_cfg: str = "configs/bytetrack_fall.yaml",
    conf_thresh: float = 0.08,
    imgsz: int = 640,
    enable_persistence: bool = True,
    detection_window_sec: float = 3.0
) -> Dict[str, Any]:
    """
    Evaluates a single video with rigorous temporal metrics.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fall_detector.reset()
    if hasattr(yolo_model, "predictor") and yolo_model.predictor is not None:
        yolo_model.predictor.trackers = None
        yolo_model.predictor = None

    is_positive = (meta.get("type") == "POSITIVE_FALL")
    gt_range = meta.get("fall_frame_range")  # [start_frame, end_frame] or None

    detected_frames = []
    suspected_frames = []
    coasting_frames_list = []

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        res = yolo_model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf_thresh,
            imgsz=imgsz,
            classes=[0],
            verbose=False
        )

        has_detected_in_frame = False
        has_suspected_in_frame = False
        tids = []

        if res and len(res) > 0 and res[0].boxes is not None:
            boxes = res[0].boxes
            if hasattr(boxes, "id") and boxes.id is not None:
                coords = boxes.xyxy.cpu().numpy()
                tids = boxes.id.int().cpu().tolist()

                for b, tid in zip(coords, tids):
                    eval_res = fall_detector.evaluate_pedestrian(
                        track_id=int(tid),
                        bbox=tuple(b),
                        frame_idx=frame_idx,
                        frame_h=h,
                        frame_w=w,
                        frame_img=frame
                    )
                    st = eval_res["status"]
                    if st == "FALL_DETECTED" or st == "FALL_CONFIRMED":
                        has_detected_in_frame = True
                    elif st == "FALL_SUSPECTED":
                        has_suspected_in_frame = True

        # Phase 22 Persistent Fall Tracking Hook
        if enable_persistence and hasattr(fall_detector, "process_frame_persistence"):
            # Extract raw low-confidence person candidate boxes for spatial re-association
            raw_candidates = []
            res_raw = yolo_model.predict(frame, conf=0.04, classes=[0], verbose=False)
            if res_raw and len(res_raw) > 0 and res_raw[0].boxes is not None and len(res_raw[0].boxes) > 0:
                raw_candidates = [list(c) for c in res_raw[0].boxes.xyxy.cpu().numpy()]

            persist_outputs = fall_detector.process_frame_persistence(
                frame_idx=frame_idx,
                observed_track_ids=tids,
                raw_candidate_boxes=raw_candidates,
                frame_img=frame,
                frame_h=h,
                frame_w=w
            )
            for p_out in persist_outputs:
                st = p_out["status"]
                if st == "FALL_DETECTED" or st == "FALL_CONFIRMED":
                    has_detected_in_frame = True
                elif st == "FALL_SUSPECTED":
                    has_suspected_in_frame = True
                if p_out.get("tracking_state") == "LOST_COASTING":
                    coasting_frames_list.append(frame_idx)

        if has_detected_in_frame:
            detected_frames.append(frame_idx)
        if has_suspected_in_frame:
            suspected_frames.append(frame_idx)

    cap.release()
    elapsed = time.time() - t0
    clip_fps = frame_idx / max(elapsed, 0.001)
    clip_dur_sec = frame_idx / max(fps, 1.0)

    # 1. Frame-Level Metrics (Evaluating definitive FALL_DETECTED)
    tp_frames = 0
    fp_frames = 0
    fn_frames = 0
    tn_frames = 0

    detected_set = set(detected_frames)

    for f in range(1, frame_idx + 1):
        in_gt_fall = False
        if is_positive and gt_range is not None:
            if gt_range[0] <= f <= gt_range[1]:
                in_gt_fall = True

        alert_active = (f in detected_set)

        if in_gt_fall and alert_active:
            tp_frames += 1
        elif not in_gt_fall and alert_active:
            fp_frames += 1
        elif in_gt_fall and not alert_active:
            fn_frames += 1
        else:
            tn_frames += 1

    precision_frame = tp_frames / max(1, (tp_frames + fp_frames))
    recall_frame = tp_frames / max(1, (tp_frames + fn_frames))
    f1_frame = (2 * precision_frame * recall_frame) / max(1e-5, (precision_frame + recall_frame))

    # 2. Event-Level Metrics & Clustering
    alarm_events = cluster_alarm_frames(detected_frames, max_gap=int(fps * 0.25))
    suspected_events = cluster_alarm_frames(suspected_frames, max_gap=int(fps * 0.25))

    event_tp = 0
    event_fp = 0
    event_fn = 0
    detection_delay_sec: Optional[float] = None
    event_coverage_pct: float = 0.0

    if is_positive and gt_range is not None:
        gt_start, gt_end = gt_range
        gt_len = max(1, gt_end - gt_start + 1)
        max_allowed_frame = gt_start + int(detection_window_sec * fps)

        # Check if any alert event matches the ground truth fall
        matched = False
        first_alarm = None

        for a_start, a_end in alarm_events:
            # Overlaps with GT fall or triggers within detection window
            if (a_start <= gt_end and a_end >= gt_start) or (gt_start <= a_start <= max_allowed_frame):
                matched = True
                if first_alarm is None or a_start < first_alarm:
                    first_alarm = a_start

        if matched and first_alarm is not None:
            event_tp = 1
            delay_frames = max(0, first_alarm - gt_start)
            detection_delay_sec = round(delay_frames / fps, 3)

            # Measure coverage: overlap frames / gt_len
            overlapping_frames = sum(1 for f in detected_frames if gt_start <= f <= gt_end)
            event_coverage_pct = round((overlapping_frames / gt_len) * 100.0, 1)
        else:
            event_fn = 1
            detection_delay_sec = None
            event_coverage_pct = 0.0

        # Any alarm event that is completely outside [gt_start - tolerance, gt_end + tolerance] is an event FP
        tolerance = int(fps * 1.0)
        for a_start, a_end in alarm_events:
            if a_end < gt_start - tolerance or a_start > gt_end + tolerance:
                event_fp += 1

        tail_threshold = gt_end - max(3, int(0.20 * gt_len))
        remains_monitored = any(f >= tail_threshold for f in detected_frames)
        fall_in_range = [f for f in detected_frames if gt_start <= f <= gt_end]
        if fall_in_range:
            gaps = [fall_in_range[i] - fall_in_range[i - 1] - 1 for i in range(1, len(fall_in_range))]
            max_gap_frames = max(gaps) if gaps else 0
        else:
            max_gap_frames = gt_len
    else:
        # Negative clip (ADL): any alarm event is a false alarm
        event_fp = len(alarm_events)
        remains_monitored = False
        max_gap_frames = 0

    return {
        "video": video_path.name,
        "type": meta.get("type"),
        "role": meta.get("role", "test"),
        "environment": meta.get("environment", ""),
        "total_frames": frame_idx,
        "fps": round(fps, 1),
        "duration_sec": round(clip_dur_sec, 2),
        "eval_fps": round(clip_fps, 1),
        "gt_range": gt_range,
        "detected_frames_count": len(detected_frames),
        "suspected_frames_count": len(suspected_frames),
        "coasting_frames_count": len(coasting_frames_list),
        "alarm_events": alarm_events,
        "frame_metrics": {
            "tp": tp_frames,
            "fp": fp_frames,
            "fn": fn_frames,
            "tn": tn_frames,
            "precision": round(precision_frame * 100.0, 2),
            "recall": round(recall_frame * 100.0, 2),
            "f1": round(f1_frame * 100.0, 2)
        },
        "event_metrics": {
            "event_tp": event_tp,
            "event_fp": event_fp,
            "event_fn": event_fn,
            "detection_delay_sec": detection_delay_sec,
            "event_coverage_pct": event_coverage_pct,
            "remains_monitored_after_impact": remains_monitored,
            "max_track_loss_gap_sec": round(max_gap_frames / fps, 2)
        }
    }


def run_temporal_fall_benchmark(
    fall_dir: Path,
    config_path: Path,
    yolo_model_name: str = "yolov8n.pt",
    tracker_cfg: str = "configs/bytetrack_fall.yaml",
    conf_thresh: float = 0.08,
    imgsz: int = 640,
    enable_persistence: bool = True,
    output_json: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes the full temporal fall detection benchmark across all positive and negative clips.
    """
    gt_path = fall_dir / "fall_ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Label roles: Dev/Regression vs Final Test
    # fall.mp4 and le2i_coffee_fall.mp4 are used for regression & dev calibration;
    # indoor_fall.mp4 and both ADL clips serve as unseen evaluation test.
    roles = {
        "fall.mp4": "regression_dev",
        "le2i_coffee_fall.mp4": "dev",
        "indoor_fall.mp4": "test",
        "indoor_normal_adl.mp4": "test",
        "le2i_coffee_adl.mp4": "test"
    }

    config = load_config(config_path)
    # Prefer live console settings from config when caller left defaults
    if imgsz == 640 and config.get("inference_imgsz"):
        imgsz = int(config.get("inference_imgsz", imgsz))
    if tracker_cfg == "configs/bytetrack_fall.yaml" and config.get("tracker_config"):
        tracker_cfg = str(config.get("tracker_config"))
    if conf_thresh == 0.08 and config.get("detection_confidence") is not None:
        conf_thresh = float(config.get("detection_confidence", conf_thresh))

    fall_detector = FallDetector(config)
    yolo_model = YOLO(yolo_model_name)

    results = []
    total_adl_dur_sec = 0.0
    total_adl_fps = 0

    mode_label = "Phase 22 Persistent Fall Tracking" if enable_persistence else "Phase 21 Baseline (No Persistence)"
    print("=" * 75)
    print(f" AEGIS Temporal Fall Detection Benchmark — {mode_label}")
    print(f" imgsz={imgsz} conf={conf_thresh} tracker={tracker_cfg}")
    print("=" * 75)

    for vid_name, meta in ground_truth.items():
        meta["role"] = roles.get(vid_name, "test")
        v_path = fall_dir / vid_name
        if not v_path.exists():
            v_path = PROJECT_ROOT / "assets" / vid_name
        if not v_path.exists():
            print(f"[!] Warning: Video file not found: {vid_name}")
            continue

        print(f"\nEvaluating: {vid_name} ({meta['type']}, Role: {meta['role']})...")
        res = evaluate_video_temporal_fall(
            video_path=v_path,
            meta=meta,
            fall_detector=fall_detector,
            yolo_model=yolo_model,
            tracker_cfg=tracker_cfg,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
            enable_persistence=enable_persistence
        )
        results.append(res)

        fm = res["frame_metrics"]
        em = res["event_metrics"]
        print(f"  -> Frame TP={fm['tp']}, FP={fm['fp']}, FN={fm['fn']}, TN={fm['tn']} | P={fm['precision']}%, R={fm['recall']}%, F1={fm['f1']}%")
        print(f"  -> Event TP={em['event_tp']}, FP={em['event_fp']}, FN={em['event_fn']} | Delay={em['detection_delay_sec']}s, Coverage={em['event_coverage_pct']}%")

        if meta["type"] == "NEGATIVE_ADL":
            total_adl_dur_sec += res["duration_sec"]
            total_adl_fps += em["event_fp"]

    # Compute overall benchmark statistics
    tot_frame_tp = sum(r["frame_metrics"]["tp"] for r in results)
    tot_frame_fp = sum(r["frame_metrics"]["fp"] for r in results)
    tot_frame_fn = sum(r["frame_metrics"]["fn"] for r in results)
    tot_frame_tn = sum(r["frame_metrics"]["tn"] for r in results)

    tot_event_tp = sum(r["event_metrics"]["event_tp"] for r in results if r["type"] == "POSITIVE_FALL")
    tot_event_fn = sum(r["event_metrics"]["event_fn"] for r in results if r["type"] == "POSITIVE_FALL")
    tot_event_fp = sum(r["event_metrics"]["event_fp"] for r in results)

    pos_results = [r for r in results if r["type"] == "POSITIVE_FALL"]
    delays = [r["event_metrics"]["detection_delay_sec"] for r in pos_results if r["event_metrics"]["detection_delay_sec"] is not None]
    coverages = [r["event_metrics"]["event_coverage_pct"] for r in pos_results if r["event_metrics"]["event_tp"] == 1]

    mean_delay = float(np.mean(delays)) if delays else 0.0
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0

    frame_prec = tot_frame_tp / max(1, (tot_frame_tp + tot_frame_fp))
    frame_rec = tot_frame_tp / max(1, (tot_frame_tp + tot_frame_fn))
    frame_f1 = (2 * frame_prec * frame_rec) / max(1e-5, (frame_prec + frame_rec))

    event_recall = tot_event_tp / max(1, (tot_event_tp + tot_event_fn))
    adl_min = max(0.01, total_adl_dur_sec / 60.0)
    fa_per_min = total_adl_fps / adl_min

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_clips": len(results),
        "frame_level_metrics": {
            "tp": tot_frame_tp,
            "fp": tot_frame_fp,
            "fn": tot_frame_fn,
            "tn": tot_frame_tn,
            "precision_pct": round(frame_prec * 100.0, 2),
            "recall_pct": round(frame_rec * 100.0, 2),
            "f1_score_pct": round(frame_f1 * 100.0, 2)
        },
        "event_level_metrics": {
            "total_fall_events": tot_event_tp + tot_event_fn,
            "detected_events": tot_event_tp,
            "missed_events": tot_event_fn,
            "event_recall_pct": round(event_recall * 100.0, 2),
            "mean_detection_delay_sec": round(mean_delay, 2),
            "mean_event_coverage_pct": round(mean_coverage, 2),
            "total_adl_duration_min": round(adl_min, 2),
            "adl_false_alarms_count": total_adl_fps,
            "adl_false_alarms_per_min": round(fa_per_min, 2)
        },
        "clip_details": results
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)
        print(f"\n[✓] Saved temporal fall benchmark report to: {output_json}")

    print("\n" + "=" * 70)
    print(" TEMPORAL FALL BENCHMARK RESULTS")
    print(f" Frame Level: Precision={benchmark_summary['frame_level_metrics']['precision_pct']}%, Recall={benchmark_summary['frame_level_metrics']['recall_pct']}%, F1={benchmark_summary['frame_level_metrics']['f1_score_pct']}%")
    print(f" Event Level: Recall={benchmark_summary['event_level_metrics']['event_recall_pct']}% ({tot_event_tp}/{tot_event_tp + tot_event_fn}), Missed={tot_event_fn}")
    print(f" Mean Latency: {mean_delay:.2f}s | Mean Event Coverage: {mean_coverage:.1f}%")
    print(f" ADL False Alarms: {total_adl_fps} ({fa_per_min:.2f} FA/min over {adl_min:.2f} min)")
    print("=" * 70)

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Temporal Fall Evaluation")
    parser.add_argument("--fall-dir", type=str, default="data/fall_benchmark", help="Directory with fall clips")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-persistence", action="store_true", help="Disable Phase 22 persistence (run Phase 21 baseline mode)")
    parser.add_argument("--output", type=str, default="outputs/reports/phase23_persistent_fall_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_temporal_fall_benchmark(
        fall_dir=PROJECT_ROOT / args.fall_dir,
        config_path=PROJECT_ROOT / args.config,
        enable_persistence=(not args.no_persistence),
        output_json=PROJECT_ROOT / args.output
    )
