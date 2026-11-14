#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 2 — Final Independent Evaluation on 5 Held-Out Test Videos
=====================================================================
Strictly evaluates the frozen Phase 2 architecture on the 5 held-out test videos:
- 688-10_l.mov
- 861-13_70.mov
- 879-38_l.mov
- 906-17_l.mov
- 913-36_l.mov

Ground truth annotations and thresholds are completely untouched.
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

# The 5 held-out videos strictly
TEST_VIDEOS = {'688-10_l.mov', '861-13_70.mov', '879-38_l.mov', '906-17_l.mov', '913-36_l.mov'}

with open(ann_file, "r", encoding="utf-8") as f:
    all_events = list(csv.DictReader(f))

test_events = [e for e in all_events if e["video_name"] in TEST_VIDEOS]

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
            "prec": float(prec), "rec": float(rec), "f1": float(f1)
        }
    macro_f1 = sum(f1s) / len(f1s)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(), "per_class": per_class}


def evaluate_phase2_held_out(max_frames=120):
    by_video = defaultdict(list)
    for e in test_events:
        by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"], e["notes"]))

    preds = {}
    event_details = []
    video_accuracies = {}
    total_time_ms = 0.0
    total_frames = 0

    print(f"Evaluating {len(by_video)} held-out video scenarios (13 events)...")

    for (vname, exp_dir), track_list in by_video.items():
        vpath = ucf_dir / vname
        if not vpath.exists():
            print(f"Warning: {vpath} not found!")
            continue

        c = dict(base_cfg)
        c['expected_evacuation_direction'] = exp_dir
        c['stopping_detection']['threshold_sec'] = 3.0
        c['stopping_detection']['min_frames'] = 75
        c['direction_min_displacement_px'] = 9.0
        c['direction_fast_displacement_px'] = 9.0
        c['direction_detection']['wrong_way_sustain_frames'] = 3
        c['stopping_detection']['long_horizon_verification'] = True
        c['stopping_detection']['max_cumulative_disp_px'] = 20.0

        # Phase 2 active components
        c['enable_impulse_filter'] = True
        c['enable_confidence_ema'] = True
        c['enable_multi_horizon'] = True
        if 'exit_regions' not in c or not isinstance(c['exit_regions'], dict):
            c['exit_regions'] = {}
        c['exit_regions']['enable_approach_check'] = True

        extractor = TrajectoryFeatureExtractor(c)
        engine = EvacuationAnomalyEngine(c)
        model = YOLO(str(model_path))

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        f_idx = 0
        t_counts = defaultdict(Counter)
        t_total = defaultdict(int)

        while f_idx < max_frames:
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

        v_correct = 0
        for tid, gt_label, notes in track_list:
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

            is_corr = (gt_label == pred)
            if is_corr:
                v_correct += 1
            preds[(vname, tid)] = (gt_label, pred)
            event_details.append({
                "video_name": vname,
                "track_id": tid,
                "ground_truth": gt_label,
                "predicted": pred,
                "correct": is_corr,
                "counts": dict(cnt),
                "total_frames": N,
                "notes": notes
            })
        video_accuracies[vname] = {
            "correct": v_correct,
            "total": len(track_list),
            "accuracy": v_correct / len(track_list) if len(track_list) > 0 else 0.0
        }

    metrics = calc_metrics(preds)
    avg_latency = total_time_ms / max(1, total_frames)
    metrics["avg_latency_ms"] = round(avg_latency, 2)
    metrics["fps"] = round(1000.0 / max(1e-3, avg_latency), 1)
    return metrics, event_details, video_accuracies


def main():
    print("======================================================================")
    print("AEGIS Phase 2 — Final Independent Evaluation on 5 Held-Out Videos")
    print("======================================================================")

    # Load baseline and Phase 1 improved from final_held_out_comparison.json
    with open(EVAL_DIR / "final_held_out_comparison.json", "r") as f:
        prior_eval = json.load(f)

    baseline_metrics = prior_eval["baseline"]["metrics"]
    baseline_events = prior_eval["baseline"]["events"]
    phase1_metrics = prior_eval["improved"]["metrics"]
    phase1_events = prior_eval["improved"]["events"]

    # Run Phase 2
    phase2_metrics, phase2_events, phase2_video_acc = evaluate_phase2_held_out(max_frames=120)

    print("\n----------------------------------------------------------------------")
    print(f"Baseline Held-Out Acc:        {baseline_metrics['accuracy'] * 100:.2f}% | Macro F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"Phase 1 Improved Acc:         {phase1_metrics['accuracy'] * 100:.2f}% | Macro F1: {phase1_metrics['macro_f1']:.4f}")
    print(f"Phase 2 Improved Acc:         {phase2_metrics['accuracy'] * 100:.2f}% | Macro F1: {phase2_metrics['macro_f1']:.4f}")
    print(f"Phase 2 CPU Latency:          {phase2_metrics['avg_latency_ms']} ms/frame ({phase2_metrics['fps']} FPS)")
    print("----------------------------------------------------------------------")

    print("\nPhase 2 Confusion Matrix:")
    print(np.array(phase2_metrics["confusion_matrix"]))

    print("\nPer-Class Breakdown (Phase 2):")
    for c, stats in phase2_metrics["per_class"].items():
        print(f"  {c:20s} Prec: {stats['prec']*100:.1f}% | Rec: {stats['rec']*100:.1f}% | F1: {stats['f1']:.4f}")

    print("\nVideo-Wise Accuracy:")
    for vname, vacc in phase2_video_acc.items():
        print(f"  {vname:15s}: {vacc['correct']}/{vacc['total']} ({vacc['accuracy']*100:.1f}%)")

    # Save machine-readable results
    results_out = {
        "git_commit": "c7e7ddbd1bb27a414777a76053e578b76ac38598",
        "baseline": {
            "metrics": baseline_metrics,
            "events": baseline_events
        },
        "phase1_improved": {
            "metrics": phase1_metrics,
            "events": phase1_events
        },
        "phase2_improved": {
            "metrics": phase2_metrics,
            "events": phase2_events,
            "video_accuracy": phase2_video_acc
        }
    }

    with open(EVAL_DIR / "phase2_final_held_out_results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    print("\nSaved evaluation/phase2_final_held_out_results.json successfully!")


if __name__ == "__main__":
    main()

