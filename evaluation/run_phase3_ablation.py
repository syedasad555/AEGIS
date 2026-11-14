#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 3 — Ablation Study on Development Videos
===================================================
Evaluates the isolated and combined contributions of each Phase 3 improvement:
1. Phase 2 Baseline (all Phase 3 flags disabled)
2. + Component 1: Multi-Horizon Direction Estimation (6f, 12f, 24-32f)
3. + Component 2: Trajectory Curvature & Avoidance Recognition
4. + Component 3: False Wrong-Direction Suppression on Stationary Jitter
5. + Component 4: Lifespan Spatial Disambiguation & Adaptive Stopping
6. Full Phase 3 Architecture (All components combined)

Extracts video tracking once (cached in-memory), then benchmarks each ablation configuration.
Strictly preserved: held-out test videos (5 videos, 13 events) are excluded.
"""

import csv
import json
import yaml
import time
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

EVAL_DIR = PROJECT_ROOT / "evaluation"
ucf_dir = PROJECT_ROOT / "UCF_CrowdsDataset"
ann_file = EVAL_DIR / "ucf_annotations.csv"

with open(ann_file, "r", encoding="utf-8") as f:
    events = list(csv.DictReader(f))

# Grouped Video Folds: 12 development videos
FOLDS = {
    'Fold 1': ['2018-6_70.mov', '620-72_l.mov', '9019-13_l.mov', '3687-18_70.mov'],
    'Fold 2': ['2082-3_70.mov', '637-147_l.mov', '913-35_l.mov', '821-10_l.mov'],
    'Fold 3': ['341-46_l.mov', '7399-1_70.mov', '9-19_l.mov', '913-37_l.mov']
}
DEV_VIDEOS = [v for fv in FOLDS.values() for v in fv]
dev_events = [e for e in events if e['video_name'] in DEV_VIDEOS]

with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

model_path = PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"


def calc_metrics(preds):
    classes = ["NORMAL", "WRONG_DIRECTION", "PROLONGED_STOPPING"]
    c_to_i = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((3, 3), dtype=int)
    for (v, tid), (gt, p) in preds.items():
        cm[c_to_i[gt], c_to_i[p]] += 1
        
    correct = sum(cm[i, i] for i in range(3))
    total = max(1, cm.sum())
    acc = correct / total
    
    f1s = []
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-6) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_class[c] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "prec": float(round(prec, 4)),
            "rec": float(round(rec, 4)),
            "f1": float(round(f1, 4))
        }
    macro_f1 = sum(f1s) / len(f1s)
    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def precompute_tracks(max_frames=120):
    print("Precomputing detections and tracks for 12 development videos...")
    cached_tracks = {}
    by_video = defaultdict(list)
    for e in dev_events:
        by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"]))

    for (vname, exp_dir), track_list in by_video.items():
        vpath = ucf_dir / vname
        if not vpath.exists():
            continue
        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        f_idx = 0
        frames_data = []

        while f_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            f_idx += 1
            res = model.track(
                source=frame,
                persist=True,
                tracker=str(PROJECT_ROOT / "configs" / "bytetrack_crowd.yaml"),
                imgsz=800,
                conf=0.15,
                iou=0.55,
                classes=[0],
                device="cpu",
                verbose=False
            )
            frame_dets = []
            if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                tids = res[0].boxes.id.cpu().numpy().astype(int)
                confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else [None] * len(tids)
                for bbox, tid, conf in zip(boxes, tids, confs):
                    frame_dets.append((int(tid), tuple(bbox), float(conf) if conf is not None else None))
            frames_data.append((f_idx, frame_dets))
        cap.release()
        cached_tracks[(vname, exp_dir)] = (fps, track_list, frames_data)
        print(f"  Processed {vname} ({len(frames_data)} frames, {len(track_list)} events)")

    return cached_tracks


def evaluate_with_cache(cached_tracks, cfg_overrides):
    all_preds = {}
    total_time_ms = 0.0
    total_frames = 0

    for (vname, exp_dir), (fps, track_list, frames_data) in cached_tracks.items():
        c = dict(base_cfg)
        c['expected_evacuation_direction'] = exp_dir
        c['stopping_detection']['threshold_sec'] = 3.0
        c['stopping_detection']['min_frames'] = 75
        c['direction_min_displacement_px'] = 9.0
        c['direction_fast_displacement_px'] = 9.0
        c['direction_detection']['wrong_way_sustain_frames'] = 3
        c['stopping_detection']['long_horizon_verification'] = True
        c['stopping_detection']['max_cumulative_disp_px'] = 20.0
        
        # Apply ablation overrides
        for k, v in cfg_overrides.items():
            if isinstance(v, dict) and k in c:
                c[k].update(v)
            else:
                c[k] = v

        extractor = TrajectoryFeatureExtractor(c)
        engine = EvacuationAnomalyEngine(c)

        t_counts = defaultdict(Counter)
        t_total = defaultdict(int)

        t_start = time.perf_counter()
        for f_idx, frame_dets in frames_data:
            total_frames += 1
            for tid, bbox, conf in frame_dets:
                feat = extractor.update_pedestrian(
                    tid, bbox, f_idx, fps=fps, conf=conf, is_detection=True
                )
                feat = engine.evaluate_pedestrian(feat)
                atype = feat.get('anomaly_type')
                t_counts[tid][atype] += 1
                t_total[tid] += 1
        t_end = time.perf_counter()
        total_time_ms += (t_end - t_start) * 1000.0

        for tid, gt_label in track_list:
            cnt = t_counts[tid]
            N = max(1, t_total[tid])
            p_stop = cnt['PROLONGED_STOPPING']
            p_wrong = cnt['WRONG_DIRECTION']
            
            is_stop_sig = (p_stop >= 4) or (p_stop / N >= 0.10)
            is_wrong_sig = (p_wrong >= 3) or (p_wrong / N >= 0.08)
            
            if not is_stop_sig and not is_wrong_sig:
                pred = 'NORMAL'
            elif is_stop_sig and is_wrong_sig:
                pred = 'PROLONGED_STOPPING' if p_stop >= p_wrong else 'WRONG_DIRECTION'
            elif is_stop_sig:
                pred = 'PROLONGED_STOPPING'
            else:
                pred = 'WRONG_DIRECTION'
                
            all_preds[(vname, tid)] = (gt_label, pred)

    overall = calc_metrics(all_preds)
    avg_latency_ms = total_time_ms / max(1, total_frames)
    fps_achieved = 1000.0 / max(1e-3, avg_latency_ms)

    return {
        "overall": overall,
        "avg_pipeline_eval_latency_ms": round(avg_latency_ms, 3),
        "fps_achieved": round(fps_achieved, 1),
    }


def main():
    print("==================================================================")
    print("AEGIS Phase 3 — Ablation Study on 12 Development Videos (36 Events)")
    print("==================================================================")

    cached_tracks = precompute_tracks(max_frames=120)

    ablations = [
        (
            "Phase 2 Baseline",
            {
                "enable_multi_horizon": False,
                "enable_curvature_suppression": False,
                "enable_stationary_jitter_suppression": False,
                "enable_adaptive_stopping_ratio": False,
            }
        ),
        (
            "+ Component 1: Multi-Horizon Direction",
            {
                "enable_multi_horizon": True,
                "enable_curvature_suppression": False,
                "enable_stationary_jitter_suppression": False,
                "enable_adaptive_stopping_ratio": False,
            }
        ),
        (
            "+ Component 2: Curvature & Avoidance Recognition",
            {
                "enable_multi_horizon": False,
                "enable_curvature_suppression": True,
                "enable_stationary_jitter_suppression": False,
                "enable_adaptive_stopping_ratio": False,
            }
        ),
        (
            "+ Component 3: False Wrong-Direction Suppression on Jitter",
            {
                "enable_multi_horizon": False,
                "enable_curvature_suppression": False,
                "enable_stationary_jitter_suppression": True,
                "enable_adaptive_stopping_ratio": False,
            }
        ),
        (
            "+ Component 4: Adaptive Stopping & Spatial Disambiguation",
            {
                "enable_multi_horizon": False,
                "enable_curvature_suppression": False,
                "enable_stationary_jitter_suppression": False,
                "enable_adaptive_stopping_ratio": True,
            }
        ),
        (
            "Phase 3 Architecture (Full Combined)",
            {
                "enable_multi_horizon": True,
                "enable_curvature_suppression": True,
                "enable_stationary_jitter_suppression": True,
                "enable_adaptive_stopping_ratio": True,
            }
        ),
    ]

    results = {}
    for name, overrides in ablations:
        print(f"\nEvaluating: {name}...")
        res = evaluate_with_cache(cached_tracks, overrides)
        ov = res["overall"]
        results[name] = res
        correct = ov['per_class']['NORMAL']['TP'] + ov['per_class']['WRONG_DIRECTION']['TP'] + ov['per_class']['PROLONGED_STOPPING']['TP']
        print(f"  -> Accuracy: {ov['accuracy']:.4f} ({correct}/36) | Macro F1: {ov['macro_f1']:.4f}")
        print(f"     Normal Rec: {ov['per_class']['NORMAL']['rec']:.4f} | Wrong-Dir Rec: {ov['per_class']['WRONG_DIRECTION']['rec']:.4f} | Stopping Rec: {ov['per_class']['PROLONGED_STOPPING']['rec']:.4f}")
        print(f"     CM: {ov['confusion_matrix']}")

    out_file = EVAL_DIR / "phase3_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation study complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
