#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 2 — 3-Fold Grouped Cross-Validation on Development Videos
=====================================================================
Evaluates the Phase 2 generalisation improvements across the 12 development videos
(36 ground-truth events) grouped into 3 distinct video folds.
Held-out test set is strictly excluded and preserved.
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

# Grouped Video Folds: 12 development videos, exactly 4 videos per fold
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
        per_class[c] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "prec": prec, "rec": rec, "f1": f1}
    macro_f1 = sum(f1s) / len(f1s)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(), "per_class": per_class}


def run_cv(exp_name, stop_verification=True, max_disp=20.0, exit_regions_cfg=None):
    all_preds = {}
    fold_results = {}
    total_time_ms = 0.0
    total_frames = 0
    
    for fname, fvids in FOLDS.items():
        fold_events = [e for e in dev_events if e['video_name'] in fvids]
        by_video = defaultdict(list)
        for e in fold_events:
            by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"]))
            
        fold_preds = {}
        for (vname, exp_dir), track_list in by_video.items():
            vpath = ucf_dir / vname
            if not vpath.exists():
                continue
                
            c = dict(base_cfg)
            c['expected_evacuation_direction'] = exp_dir
            c['stopping_detection']['threshold_sec'] = 3.0
            c['stopping_detection']['min_frames'] = 75
            c['direction_min_displacement_px'] = 9.0
            c['direction_fast_displacement_px'] = 9.0
            c['direction_detection']['wrong_way_sustain_frames'] = 3
            c['stopping_detection']['long_horizon_verification'] = stop_verification
            c['stopping_detection']['max_cumulative_disp_px'] = max_disp
            if exit_regions_cfg:
                c['exit_regions'] = exit_regions_cfg
                
            extractor = TrajectoryFeatureExtractor(c)
            engine = EvacuationAnomalyEngine(c)
            model = YOLO(str(model_path))
            
            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            f_idx = 0
            t_counts = defaultdict(Counter)
            t_total = defaultdict(int)
            
            while f_idx < 120:
                ret, frame = cap.read()
                if not ret:
                    break
                f_idx += 1
                total_frames += 1
                
                t_start = time.perf_counter()
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
                
                if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                    tids = res[0].boxes.id.cpu().numpy().astype(int)
                    confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else [None] * len(tids)
                    for bbox, tid, conf in zip(boxes, tids, confs):
                        feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps, conf=float(conf) if conf is not None else None, is_detection=True)
                        feat = engine.evaluate_pedestrian(feat)
                        atype = feat.get('anomaly_type')
                        t_counts[tid][atype] += 1
                        t_total[tid] += 1
                t_end = time.perf_counter()
                total_time_ms += (t_end - t_start) * 1000.0

            cap.release()
            
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
                    
                fold_preds[(vname, tid)] = (gt_label, pred)
                
        all_preds.update(fold_preds)
        fold_results[fname] = calc_metrics(fold_preds)
        
    overall = calc_metrics(all_preds)
    avg_latency_ms = total_time_ms / max(1, total_frames)
    fps_achieved = 1000.0 / max(1e-3, avg_latency_ms)

    print(f"\n=======================================================")
    print(f"[{exp_name}] Overall Dev Accuracy: {overall['accuracy']:.4f} | Macro F1: {overall['macro_f1']:.4f}")
    print(f"Latency: {avg_latency_ms:.2f} ms/frame ({fps_achieved:.1f} FPS)")
    print(f"=======================================================")
    for fname, fr in fold_results.items():
        print(f"  {fname}: Acc={fr['accuracy']:.4f}, MacroF1={fr['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(overall['confusion_matrix']))
    
    preds_serializable = {f"{k[0]}_id_{k[1]}": list(v) for k, v in all_preds.items()}
    return {
        "name": exp_name,
        "overall": overall,
        "folds": fold_results,
        "preds": preds_serializable,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps_achieved": round(fps_achieved, 1),
    }


if __name__ == "__main__":
    print("Running Phase 2 3-Fold Grouped Cross-Validation on Dev Videos (12 videos, 36 events)...")

    # Load baseline results
    with open(EVAL_DIR / "generalisation_cv_comparison.json", "r") as f:
        prior_data = json.load(f)
    baseline_res = prior_data.get("improved")  # Phase 1 improved architecture is our baseline

    # Run Phase 2 Combined
    phase2_res = run_cv("Phase 2 Improved Architecture", stop_verification=True, max_disp=20.0)

    comparison_out = {
        "phase1_baseline": baseline_res,
        "phase2_improved": phase2_res,
    }

    with open(EVAL_DIR / "phase2_cv_comparison.json", "w") as f:
        json.dump(comparison_out, f, indent=2)

    print("\nSaved evaluation/phase2_cv_comparison.json successfully!")

