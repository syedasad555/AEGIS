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

# Let's inspect tracks in Dev Fold 1 that failed:
# Fold 1 videos: 2018-6_70.mov, 620-72_l.mov, 9019-13_l.mov, 3687-18_70.mov
fold1_events = [e for e in dev_events if e['video_name'] in FOLDS['Fold 1']]

by_video = defaultdict(list)
for e in fold1_events:
    by_video[(e["video_name"], e["expected_direction"])].append((int(e["pedestrian_id"]), e["ground_truth_class"], e["notes"]))

print("FOLD 1 BASELINE ERRORS INSPECTION:")
for (vname, exp_dir), track_list in by_video.items():
    vpath = ucf_dir / vname
    c = dict(base_cfg)
    c['expected_evacuation_direction'] = exp_dir
    c['stopping_detection']['threshold_sec'] = 3.0
    c['stopping_detection']['min_frames'] = 75
    c['direction_min_displacement_px'] = 9.0
    c['direction_fast_displacement_px'] = 9.0
    c['direction_detection']['wrong_way_sustain_frames'] = 3
    
    extractor = TrajectoryFeatureExtractor(c)
    engine = EvacuationAnomalyEngine(c)
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f_idx = 0
    t_counts = defaultdict(Counter)
    t_total = defaultdict(int)
    t_disps = defaultdict(list)
    
    while f_idx < 120:
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
                t_disps[tid].append(feat.get('displacement_px', 0.0))
    cap.release()
    
    for tid, gt, notes in track_list:
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
        status = "CORRECT" if pred == gt else f"ERROR: pred={pred} != gt={gt}"
        print(f"  {vname} | Track {tid} | {status} | counts={dict(cnt)} | N={N} | max_disp={max(t_disps[tid]) if t_disps[tid] else 0} | notes={notes}")

