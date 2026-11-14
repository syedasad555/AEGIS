import csv
import json
import yaml
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

def run_evaluation_on_held_out(mode_name, stop_verification=True, max_disp=20.0, max_frames=120):
    by_video = defaultdict(list)
    for e in test_events:
        by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"], e["notes"]))
        
    preds = {}
    event_details = []
    
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
                for bbox, tid in zip(boxes, tids):
                    feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps)
                    feat = engine.evaluate_pedestrian(feat)
                    atype = feat.get('anomaly_type')
                    t_counts[tid][atype] += 1
                    t_total[tid] += 1
        cap.release()
        
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
                
            preds[(vname, tid)] = (gt_label, pred)
            event_details.append({
                "video_name": vname,
                "track_id": tid,
                "ground_truth": gt_label,
                "predicted": pred,
                "correct": (gt_label == pred),
                "counts": dict(cnt),
                "total_frames": N,
                "notes": notes
            })
            
    metrics = calc_metrics(preds)
    return metrics, event_details

print("Running Frozen Baseline on 5 Held-Out Videos...")
base_metrics, base_events = run_evaluation_on_held_out("Baseline", stop_verification=False)
print("Baseline Held-Out Acc:", base_metrics["accuracy"], "MacroF1:", base_metrics["macro_f1"])

print("\nRunning Frozen Improved Architecture on 5 Held-Out Videos...")
imp_metrics, imp_events = run_evaluation_on_held_out("Improved", stop_verification=True, max_disp=20.0)
print("Improved Held-Out Acc:", imp_metrics["accuracy"], "MacroF1:", imp_metrics["macro_f1"])

out_data = {
    "baseline": {
        "metrics": base_metrics,
        "events": base_events
    },
    "improved": {
        "metrics": imp_metrics,
        "events": imp_events
    }
}

with open("evaluation/final_held_out_comparison.json", "w") as f:
    json.dump(out_data, f, indent=2)

print("\nSaved evaluation/final_held_out_comparison.json successfully!")

