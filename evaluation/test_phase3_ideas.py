#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 3 Experimental Validation on 12 Development Videos
==============================================================
Tests:
1. Multi-horizon direction estimation (6, 12, 24-32 frames)
2. Trajectory curvature & evasive motion recognition
3. False wrong-direction suppression (displacement & consistency gate)
4. Adaptive stopping threshold (1.0s / 25 frames + spatial verification)
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
        per_class[c] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "prec": prec, "rec": rec, "f1": f1}
    macro_f1 = sum(f1s) / len(f1s)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(), "per_class": per_class}


def test_configuration(stop_threshold_sec=1.0, stop_min_frames=25, min_wrong_disp=10.0):
    all_preds = {}
    fold_results = {}
    by_video = defaultdict(list)
    for e in dev_events:
        by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"]))

    for fname, fvids in FOLDS.items():
        fold_events = [e for e in dev_events if e['video_name'] in fvids]
        fold_by_video = defaultdict(list)
        for e in fold_events:
            fold_by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"]))

        fold_preds = {}
        for (vname, exp_dir), track_list in fold_by_video.items():
            vpath = ucf_dir / vname
            if not vpath.exists():
                continue

            c = dict(base_cfg)
            c['expected_evacuation_direction'] = exp_dir
            c['stopping_detection']['threshold_sec'] = stop_threshold_sec
            c['stopping_detection']['min_frames'] = stop_min_frames
            c['direction_min_displacement_px'] = 9.0
            c['direction_fast_displacement_px'] = 9.0
            c['direction_detection']['wrong_way_sustain_frames'] = 3
            c['stopping_detection']['long_horizon_verification'] = True
            c['stopping_detection']['max_cumulative_disp_px'] = 20.0
            c['enable_impulse_filter'] = True
            c['enable_confidence_ema'] = True
            c['enable_multi_horizon'] = True

            extractor = TrajectoryFeatureExtractor(c)
            engine = EvacuationAnomalyEngine(c)
            model = YOLO(str(model_path))

            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            f_idx = 0
            t_counts = defaultdict(Counter)
            t_total = defaultdict(int)
            t_wrong_disp = defaultdict(float)
            t_start_pos = {}
            t_last_pos = {}

            while f_idx < 120:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                res = model.track(source=frame, persist=True, tracker='configs/bytetrack_crowd.yaml', imgsz=800, conf=0.15, iou=0.55, classes=[0], device='cpu', verbose=False)
                if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                    tids = res[0].boxes.id.cpu().numpy().astype(int)
                    confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else [None] * len(tids)
                    for bbox, tid, conf in zip(boxes, tids, confs):
                        foot_x = (bbox[0] + bbox[2]) / 2.0
                        foot_y = bbox[3]
                        if tid not in t_start_pos:
                            t_start_pos[tid] = (foot_x, foot_y)
                        t_last_pos[tid] = (foot_x, foot_y)

                        feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps, conf=float(conf) if conf is not None else None, is_detection=True)
                        feat = engine.evaluate_pedestrian(feat)
                        atype = feat.get('anomaly_type')

                        # False wrong-direction suppression on stationary jitter
                        # If anomaly is WRONG_DIRECTION but speed is sub-threshold and cumulative disp is low, suppress
                        cur_disp = np.hypot(foot_x - t_start_pos[tid][0], foot_y - t_start_pos[tid][1])
                        speed = feat.get('speed_px_sec', 0.0)

                        if atype == 'WRONG_DIRECTION' and cur_disp < min_wrong_disp and speed < 5.0 and f_idx > 20:
                            # Transient jitter on stationary track -> treat as NORMAL or keep stopped
                            atype = 'NORMAL'

                        t_counts[tid][atype] += 1
                        t_total[tid] += 1

            cap.release()

            for tid, gt_label in track_list:
                cnt = t_counts[tid]
                N = max(1, t_total[tid])
                p_stop = cnt['PROLONGED_STOPPING']
                p_wrong = cnt['WRONG_DIRECTION']

                # Also check stationary ratio over lifespan
                # If track lived >= 25 frames and >= 75% of frames were stationary with net disp < 15px
                net_d = 0.0
                if tid in t_start_pos and tid in t_last_pos:
                    net_d = np.hypot(t_last_pos[tid][0] - t_start_pos[tid][0], t_last_pos[tid][1] - t_start_pos[tid][1])

                is_stop_sig = (p_stop >= 4) or (p_stop / N >= 0.10)
                is_wrong_sig = (p_wrong >= 3) or (p_wrong / N >= 0.08)

                # Disambiguation: if both or conflicting, check net displacement
                if is_wrong_sig and net_d < 12.0:
                    # Bounded jitter cannot be genuine counterflow
                    is_wrong_sig = False
                    if N >= 25:
                        is_stop_sig = True

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
    print(f"\n=======================================================")
    print(f"Overall Accuracy: {overall['accuracy'] * 100:.2f}% | Macro F1: {overall['macro_f1']:.4f}")
    for fname, fr in fold_results.items():
        print(f"  {fname}: Acc={fr['accuracy']*100:.1f}%, MacroF1={fr['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(overall['confusion_matrix']))
    for c, stats in overall['per_class'].items():
        print(f"  {c:20s} Prec: {stats['prec']*100:.1f}% | Rec: {stats['rec']*100:.1f}% | F1: {stats['f1']:.4f}")
    return overall, all_preds


if __name__ == "__main__":
    print("Testing Phase 3 Candidates on 12 Development Videos...")
    test_configuration(stop_threshold_sec=1.0, stop_min_frames=25, min_wrong_disp=12.0)

