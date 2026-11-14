#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Anomaly Detection Optimization & Independent Held-Out Evaluation
====================================================================
Performs:
1. Training/Tuning split parameter search.
2. Independent Held-Out test set evaluation.
3. Before-and-after comparison.
4. Export of all metrics, confusion matrices, and charts.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import json
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from ultralytics import YOLO

from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Dataset & Ground Truth Splits (Strict Train / Held-Out Test Partition)
# -----------------------------------------------------------------------------

TRAIN_EVENTS = [
    # (video, track_id, expected_flow, true_label, description)
    ('evacuation_test_corridor.avi', 1, 'LEFT', 'PROLONGED_STOPPING', 'Stationary occupant in corridor flow'),
    ('evacuation_test_corridor.avi', 2, 'LEFT', 'PROLONGED_STOPPING', 'Stationary bystander obstructing primary egress'),
    ('evacuation_test_corridor.avi', 9, 'LEFT', 'WRONG_DIRECTION', 'Counterflow traversal orthogonal DOWN'),
    ('evacuation_test_corridor.avi', 13, 'LEFT', 'WRONG_DIRECTION', 'Rapid counterflow DOWN against LEFT flow'),
    ('evacuation_test_corridor.avi', 15, 'LEFT', 'WRONG_DIRECTION', 'Counterflow traversal DOWN_RIGHT'),
    ('evacuation_test_corridor.avi', 3, 'LEFT', 'PROLONGED_STOPPING', 'Completely stationary pedestrian at corridor edge'),
    ('evacuation_test_corridor.avi', 7, 'LEFT', 'NORMAL', 'Pedestrian proceeding leftward toward exit'),
    ('running.mp4', 13, 'LEFT', 'WRONG_DIRECTION', 'Counterflow runner traversing diagonally down'),
    ('running.mp4', 24, 'LEFT', 'WRONG_DIRECTION', 'High-speed counterflow runner moving RIGHT'),
    ('running.mp4', 3, 'LEFT', 'PROLONGED_STOPPING', 'Stationary pedestrian motionless during evacuation'),
    ('intrusion.mp4', 2, 'RIGHT', 'NORMAL', 'Compliant pedestrian walking along perimeter lane'),
    ('intrusion.mp4', 9, 'RIGHT', 'WRONG_DIRECTION', 'Counterflow pedestrian moving DOWN_LEFT against RIGHT flow'),
    ('intrusion.mp4', 19, 'RIGHT', 'PROLONGED_STOPPING', 'Stationary bystander standing motionless at perimeter gate')
]

TEST_EVENTS = [
    # Independent held-out test events
    ('evacuation_test_corridor.avi', 6, 'LEFT', 'PROLONGED_STOPPING', 'Stationary occupant standing in corridor flow'),
    ('evacuation_test_corridor.avi', 11, 'LEFT', 'WRONG_DIRECTION', 'Counterflow movement DOWN'),
    ('evacuation_test_corridor.avi', 16, 'LEFT', 'WRONG_DIRECTION', 'Rapid counterflow movement DOWN'),
    ('evacuation_test_corridor.avi', 25, 'LEFT', 'WRONG_DIRECTION', 'Counterflow movement UP'),
    ('evacuation_test_corridor.avi', 4, 'LEFT', 'PROLONGED_STOPPING', 'Stationary pedestrian near corridor wall'),
    ('running.mp4', 2, 'LEFT', 'NORMAL', 'Pedestrian walking in expected leftward direction'),
    ('running.mp4', 7, 'LEFT', 'NORMAL', 'Compliant pedestrian moving toward egress'),
    ('intrusion.mp4', 1, 'RIGHT', 'NORMAL', 'Compliant pedestrian flowing in designated direction'),
    ('intrusion.mp4', 12, 'RIGHT', 'WRONG_DIRECTION', 'Counterflow movement DOWN_LEFT towards access gate'),
    ('intrusion.mp4', 42, 'RIGHT', 'PROLONGED_STOPPING', 'Prolonged stopping observed adjacent to access gate')
]

# -----------------------------------------------------------------------------
# 2. Simulation & Inference Function
# -----------------------------------------------------------------------------

def run_simulation(events_list, param_cfg, base_cfg):
    by_video = defaultdict(list)
    for v, tid, exp, label, desc in events_list:
        by_video[(v, exp)].append((tid, label))
        
    model_path = PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"
    
    predictions = {}
    delays = {"WRONG_DIRECTION": [], "PROLONGED_STOPPING": []}
    
    for (vname, exp_dir), tracks in by_video.items():
        vpath = PROJECT_ROOT / "assets" / vname
        if not vpath.exists():
            continue
            
        c = dict(base_cfg)
        c['expected_evacuation_direction'] = exp_dir
        c['stopping_detection']['threshold_sec'] = param_cfg['stopping_threshold_sec']
        c['stopping_detection']['min_frames'] = int(param_cfg['stopping_threshold_sec'] * 30)
        c['direction_min_displacement_px'] = param_cfg['direction_min_disp_px']
        c['direction_fast_displacement_px'] = param_cfg['direction_min_disp_px']
        c['direction_detection']['wrong_way_sustain_frames'] = param_cfg['wrong_way_sustain_frames']
        
        extractor = TrajectoryFeatureExtractor(c)
        engine = EvacuationAnomalyEngine(c)
        # Fresh YOLO tracker instance per video
        model = YOLO(str(model_path))
        
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        f_idx = 0
        t_counts = defaultdict(Counter)
        first_frames = defaultdict(dict)
        
        while True:
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
            
            if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
                coords = res[0].boxes.xyxy.cpu().numpy()
                tids = res[0].boxes.id.cpu().numpy().astype(int)
                for bbox, tid in zip(coords, tids):
                    feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps)
                    feat = engine.evaluate_pedestrian(feat)
                    
                    atype = feat.get('anomaly_type')
                    t_counts[tid][atype] += 1
                    if atype in ("WRONG_DIRECTION", "PROLONGED_STOPPING") and atype not in first_frames[tid]:
                        first_frames[tid][atype] = f_idx
        cap.release()
        
        # Decide event-level prediction
        for tid, gt_label in tracks:
            cnt = t_counts[tid]
            N = max(1, sum(cnt.values()))
            p_stop = cnt['PROLONGED_STOPPING']
            p_wrong = cnt['WRONG_DIRECTION']
            
            # Significant anomaly presence: >= 5 frames or >= 10% of track life
            is_stop_sig = (p_stop >= 5) or (p_stop / N >= 0.10)
            is_wrong_sig = (p_wrong >= 5) or (p_wrong / N >= 0.10)
            
            if not is_stop_sig and not is_wrong_sig:
                pred = 'NORMAL'
            elif is_stop_sig and is_wrong_sig:
                pred = 'PROLONGED_STOPPING' if p_stop >= p_wrong else 'WRONG_DIRECTION'
            elif is_stop_sig:
                pred = 'PROLONGED_STOPPING'
            else:
                pred = 'WRONG_DIRECTION'
                
            predictions[(vname, tid)] = (gt_label, pred)
            
            # Delays
            if gt_label == pred and pred in delays:
                ff = first_frames[tid].get(pred)
                if ff is not None:
                    delays[pred].append(ff / fps)
                    
    return predictions, delays

def compute_metrics(predictions, delays):
    classes = ["NORMAL", "WRONG_DIRECTION", "PROLONGED_STOPPING"]
    c_to_i = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((3, 3), dtype=int)
    
    for (v, tid), (gt, pred) in predictions.items():
        cm[c_to_i[gt], c_to_i[pred]] += 1
        
    per_class = {}
    f1_list = []
    support_list = []
    
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        
        acc = (tp + tn) / max(cm.sum(), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        spec = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6) if (prec + rec) > 0 else 0.0
        far = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
        mar = fn / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        
        per_class[c] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "far": round(far, 4),
            "mar": round(mar, 4),
            "support": int(cm[i, :].sum())
        }
        f1_list.append(f1)
        support_list.append(cm[i, :].sum())
        
    macro_f1 = sum(f1_list) / len(f1_list)
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / max(sum(support_list), 1)
    overall_acc = np.trace(cm) / max(cm.sum(), 1)
    
    avg_delays = {
        c: round(sum(delays[c]) / max(len(delays[c]), 1), 3)
        for c in delays
    }
    
    return {
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "average_delays_sec": avg_delays
    }

# -----------------------------------------------------------------------------
# 3. Main Optimization & Evaluation Pipeline
# -----------------------------------------------------------------------------

def main():
    print("="*75)
    print("AEGIS ANOMALY DETECTION OPTIMIZATION & INDEPENDENT EVALUATION")
    print("="*75)
    
    base_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml"))
    
    # ---------------------------------------------------------
    # Phase 4: Parameter Search on Training Set
    # ---------------------------------------------------------
    print("\n[Phase 4] Parameter Grid Search on Training Partition (13 Events)...")
    
    search_space = [
        {"stopping_threshold_sec": 1.5, "direction_min_disp_px": 8.0, "wrong_way_sustain_frames": 2, "event_anomaly_min_frames": 3},
        {"stopping_threshold_sec": 1.5, "direction_min_disp_px": 8.5, "wrong_way_sustain_frames": 2, "event_anomaly_min_frames": 3},
        {"stopping_threshold_sec": 2.0, "direction_min_disp_px": 8.5, "wrong_way_sustain_frames": 2, "event_anomaly_min_frames": 3},
        {"stopping_threshold_sec": 2.0, "direction_min_disp_px": 9.0, "wrong_way_sustain_frames": 3, "event_anomaly_min_frames": 3},
        {"stopping_threshold_sec": 2.5, "direction_min_disp_px": 10.0, "wrong_way_sustain_frames": 3, "event_anomaly_min_frames": 4},
        {"stopping_threshold_sec": 3.0, "direction_min_disp_px": 10.0, "wrong_way_sustain_frames": 3, "event_anomaly_min_frames": 4},
    ]
    
    search_results = []
    best_params = None
    best_macro_f1 = -1.0
    
    for i, p_cfg in enumerate(search_space, start=1):
        preds, delays = run_simulation(TRAIN_EVENTS, p_cfg, base_cfg)
        metrics = compute_metrics(preds, delays)
        search_results.append({
            "config": p_cfg,
            "accuracy": metrics["overall_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "wrong_dir_f1": metrics["per_class"]["WRONG_DIRECTION"]["f1_score"],
            "prolonged_stop_f1": metrics["per_class"]["PROLONGED_STOPPING"]["f1_score"]
        })
        print(f"  Config {i:2d} -> StopThresh: {p_cfg['stopping_threshold_sec']}s, MinDisp: {p_cfg['direction_min_disp_px']}px, Sustain: {p_cfg['wrong_way_sustain_frames']}f "
              f"| Acc: {metrics['overall_accuracy']*100:.1f}% | Macro F1: {metrics['macro_f1']:.3f} (WrongDir: {metrics['per_class']['WRONG_DIRECTION']['f1_score']:.3f}, Stop: {metrics['per_class']['PROLONGED_STOPPING']['f1_score']:.3f})")
        
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_params = p_cfg
            
    print(f"\n[Optimal Parameters Selected from Training Grid Search]:")
    print(json.dumps(best_params, indent=2))
    
    # Save search results
    with open(EVAL_DIR / "threshold_optimization_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_parameters": best_params,
            "search_space_evaluation": search_results
        }, f, indent=2)
    print(f"[Exported] {EVAL_DIR / 'threshold_optimization_results.json'}")
    
    # ---------------------------------------------------------
    # Phase 5: Independent Evaluation on Held-Out Test Set (10 Events)
    # ---------------------------------------------------------
    print("\n[Phase 5] Evaluating Locked Parameters on Held-Out Test Set (10 Events)...")
    test_preds, test_delays = run_simulation(TEST_EVENTS, best_params, base_cfg)
    test_metrics = compute_metrics(test_preds, test_delays)
    
    print("\n--- HELD-OUT TEST SET RESULTS ---")
    print(f"Overall Accuracy:  {test_metrics['overall_accuracy']*100:.1f}%")
    print(f"Macro F1-Score:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    for c, m in test_metrics["per_class"].items():
        print(f"  {c:18s} -> Prec: {m['precision']:.3f}, Rec: {m['recall']:.3f}, F1: {m['f1_score']:.3f}, TP: {m['TP']}, FP: {m['FP']}, FN: {m['FN']}, TN: {m['TN']}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")
    
    # ---------------------------------------------------------
    # Comprehensive Full Benchmark (All 23 Events)
    # ---------------------------------------------------------
    print("\nEvaluating Complete Consolidated Benchmark (23 Events)...")
    all_events = TRAIN_EVENTS + TEST_EVENTS
    full_preds, full_delays = run_simulation(all_events, best_params, base_cfg)
    full_metrics = compute_metrics(full_preds, full_delays)
    
    # Original Baseline Metrics for Before-and-After
    baseline_metrics = {
        "overall_accuracy": 0.6000,
        "macro_f1": 0.4575,
        "weighted_f1": 0.5137,
        "per_class": {
            "WRONG_DIRECTION": {"precision": 0.6000, "recall": 0.7500, "f1_score": 0.6667, "TP": 6, "FP": 4, "FN": 2, "TN": 8},
            "PROLONGED_STOPPING": {"precision": 0.0000, "recall": 0.0000, "f1_score": 0.0000, "TP": 0, "FP": 0, "FN": 5, "TN": 15},
            "NORMAL": {"precision": 0.6000, "recall": 0.8571, "f1_score": 0.7059, "TP": 6, "FP": 4, "FN": 1, "TN": 9}
        },
        "confusion_matrix": [
            [6, 1, 0],
            [2, 6, 0],
            [2, 3, 0]
        ]
    }
    
    comparison = {
        "baseline": baseline_metrics,
        "improved_full": full_metrics,
        "improved_held_out": test_metrics,
        "optimal_parameters": best_params
    }
    
    with open(EVAL_DIR / "anomaly_metrics_improved.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[Exported] {EVAL_DIR / 'anomaly_metrics_improved.json'}")
    
    # ---------------------------------------------------------
    # Generate Plots: Confusion Matrix & Before-and-After Comparison
    # ---------------------------------------------------------
    classes = ["NORMAL", "WRONG_DIRECTION", "PROLONGED_STOPPING"]
    cm = np.array(full_metrics["confusion_matrix"])
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
            
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Normal", "Wrong-Dir", "Stop"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Normal", "Wrong-Dir", "Stop"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("AEGIS Improved Anomaly Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix_improved.png")
    plt.close()
    print(f"[Exported] {EVAL_DIR / 'confusion_matrix_improved.png'}")
    
    # 2. Before-and-After Bar Chart
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    cats = ["Overall Accuracy", "Wrong-Dir F1", "Stop F1", "Macro F1"]
    base_vals = [
        baseline_metrics["overall_accuracy"],
        baseline_metrics["per_class"]["WRONG_DIRECTION"]["f1_score"],
        baseline_metrics["per_class"]["PROLONGED_STOPPING"]["f1_score"],
        baseline_metrics["macro_f1"]
    ]
    impr_vals = [
        full_metrics["overall_accuracy"],
        full_metrics["per_class"]["WRONG_DIRECTION"]["f1_score"],
        full_metrics["per_class"]["PROLONGED_STOPPING"]["f1_score"],
        full_metrics["macro_f1"]
    ]
    
    x = np.arange(len(cats))
    w = 0.35
    rects1 = ax.bar(x - w/2, [v*100 for v in base_vals], w, label="Original Baseline", color="#94a3b8")
    rects2 = ax.bar(x + w/2, [v*100 for v in impr_vals], w, label="AEGIS Improved", color="#2563eb")
    
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("AEGIS Anomaly Detection: Baseline vs Improved Performance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1d4ed8")
                    
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "anomaly_improvement_comparison.png")
    plt.close()
    print(f"[Exported] {EVAL_DIR / 'anomaly_improvement_comparison.png'}")
    
    # ---------------------------------------------------------
    # Also update production config.yaml with optimal parameters
    # ---------------------------------------------------------
    base_cfg['stopping_detection']['threshold_sec'] = best_params['stopping_threshold_sec']
    base_cfg['stopping_detection']['min_frames'] = int(best_params['stopping_threshold_sec'] * 30)
    base_cfg['direction_min_displacement_px'] = best_params['direction_min_disp_px']
    base_cfg['direction_fast_displacement_px'] = best_params['direction_min_disp_px']
    base_cfg['direction_detection']['wrong_way_sustain_frames'] = best_params['wrong_way_sustain_frames']
    with open(PROJECT_ROOT / "configs" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, default_flow_style=False)
    print(f"[Updated] Production {PROJECT_ROOT / 'configs' / 'config.yaml'} with optimal parameters.")

if __name__ == "__main__":
    main()

