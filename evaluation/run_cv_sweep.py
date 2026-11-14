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

with open(ann_file, "r", encoding="utf-8") as f:
    events = list(csv.DictReader(f))

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

def run_experiment(exp_name, long_horizon_enabled=True, long_horizon_threshold=20.0, resolution_norm=False, max_frames=120):
    all_preds = {}
    fold_results = {}
    
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
            
            # Read video dimensions for resolution-aware scaling if enabled
            cap = cv2.VideoCapture(str(vpath))
            vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            if resolution_norm:
                # Reference height: 360px -> 9.0px; 576px -> 14.4px
                scale = vh / 360.0
                disp_px = 9.0 * scale
                c['direction_min_displacement_px'] = disp_px
                c['direction_fast_displacement_px'] = disp_px
                c['stopping_detection']['max_cumulative_disp_px'] = long_horizon_threshold * scale
            else:
                c['direction_min_displacement_px'] = 9.0
                c['direction_fast_displacement_px'] = 9.0
                c['stopping_detection']['max_cumulative_disp_px'] = long_horizon_threshold
                
            c['direction_detection']['wrong_way_sustain_frames'] = 3
            c['stopping_detection']['long_horizon_verification'] = long_horizon_enabled
            
            extractor = TrajectoryFeatureExtractor(c)
            engine = EvacuationAnomalyEngine(c)
            
            # Inject long horizon stopping verifier into engine if not present
            engine.long_horizon_enabled = long_horizon_enabled
            engine.max_cumulative_disp_px = c['stopping_detection']['max_cumulative_disp_px']
            engine.track_foot_hist = defaultdict(list)
            
            # Monkey-patch stopping evaluation to test long-horizon rule
            orig_eval_stop = engine._eval_prolonged_stopping
            def patched_eval_stopping(feat, frame, ts):
                res = orig_eval_stop(feat, frame, ts)
                if res and engine.long_horizon_enabled:
                    tid = int(feat.get("track_id", -1))
                    foot = feat.get("foot", (0, 0))
                    # check net displacement over stopping window
                    hist = engine.track_foot_hist[tid]
                    hist.append((ts, foot[0], foot[1]))
                    # clean old points
                    target_t = ts - c['stopping_detection']['threshold_sec']
                    ref = hist[0]
                    for item in hist:
                        if item[0] >= target_t:
                            ref = item
                            break
                    net_disp = ((foot[0]-ref[1])**2 + (foot[1]-ref[2])**2)**0.5
                    if net_disp >= engine.max_cumulative_disp_px:
                        # Subject has net travel >= threshold -> slowly walking, NOT stopping!
                        return None
                return res
            engine._eval_prolonged_stopping = patched_eval_stopping
            
            model = YOLO(str(model_path))
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
    print(f"\n[{exp_name}] Overall Dev: Acc={overall['accuracy']:.4f}, MacroF1={overall['macro_f1']:.4f}")
    for fname, fr in fold_results.items():
        print(f"  {fname}: Acc={fr['accuracy']:.4f}, MacroF1={fr['macro_f1']:.4f}")
    print("  Confusion Matrix (NORMAL, WRONG_DIR, STOPPING):")
    print(np.array(overall['confusion_matrix']))
    return {"name": exp_name, "overall": overall, "folds": fold_results, "preds": all_preds}

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

if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    r_norm = (sys.argv[2] == '1') if len(sys.argv) > 2 else False
    name = f"Sweep_t={t}_norm={r_norm}"
    res = run_experiment(name, long_horizon_enabled=True, long_horizon_threshold=t, resolution_norm=r_norm)
    with open(f"evaluation/{name}.json", "w") as f:
        json.dump(res, f, indent=2)

