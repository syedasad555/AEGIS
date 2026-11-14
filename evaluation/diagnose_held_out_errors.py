import csv
import json
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

EVAL_DIR = PROJECT_ROOT / "evaluation"
ucf_dir = PROJECT_ROOT / "UCF_CrowdsDataset"
ann_file = EVAL_DIR / "ucf_annotations.csv"

TEST_VIDEOS = {'688-10_l.mov', '861-13_70.mov', '879-38_l.mov', '906-17_l.mov', '913-36_l.mov'}

events = []
with open(ann_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["video_name"] in TEST_VIDEOS:
            events.append(row)

with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

# Config 4 parameters (the best validation config)
param_cfg = {
    'stopping_threshold_sec': 3.0,
    'direction_min_disp_px': 9.0,
    'wrong_way_sustain_frames': 3
}

model_path = PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"

by_video = defaultdict(list)
for e in events:
    by_video[(e["video_name"], e["expected_direction"])].append(e)

results = []

for (vname, exp_dir), ev_list in by_video.items():
    vpath = ucf_dir / vname
    c = dict(base_cfg)
    c['expected_evacuation_direction'] = exp_dir
    c['stopping_detection']['threshold_sec'] = param_cfg['stopping_threshold_sec']
    c['stopping_detection']['min_frames'] = int(param_cfg['stopping_threshold_sec'] * 25)
    c['direction_min_displacement_px'] = param_cfg['direction_min_disp_px']
    c['direction_fast_displacement_px'] = param_cfg['direction_min_disp_px']
    c['direction_detection']['wrong_way_sustain_frames'] = param_cfg['wrong_way_sustain_frames']
    
    extractor = TrajectoryFeatureExtractor(c)
    engine = EvacuationAnomalyEngine(c)
    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f_idx = 0
    
    t_counts = defaultdict(Counter)
    t_total = defaultdict(int)
    t_dirs = defaultdict(Counter)
    t_speeds = defaultdict(list)
    t_disps = defaultdict(list)
    t_anomalies_timeline = defaultdict(list)
    
    while f_idx < 120:
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1
        sec = f_idx / fps
        
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
                d = feat.get('direction')
                spd = feat.get('speed', 0.0)
                disp = feat.get('net_displacement', 0.0)
                
                t_counts[tid][atype] += 1
                t_total[tid] += 1
                t_dirs[tid][d] += 1
                t_speeds[tid].append(spd)
                t_disps[tid].append(disp)
                t_anomalies_timeline[tid].append((f_idx, sec, d, spd, disp, atype))
                
    cap.release()
    
    for ev in ev_list:
        tid = int(ev["pedestrian_id"])
        gt_label = ev["ground_truth_class"]
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
            
        results.append({
            'video_name': vname,
            'pedestrian_id': tid,
            'gt': gt_label,
            'pred': pred,
            'correct': (gt_label == pred),
            'total_frames': N,
            'counts': dict(cnt),
            'top_dirs': t_dirs[tid].most_common(3),
            'avg_speed': sum(t_speeds[tid])/len(t_speeds[tid]) if t_speeds[tid] else 0,
            'max_disp': max(t_disps[tid]) if t_disps[tid] else 0,
            'timeline': t_anomalies_timeline[tid]
        })

print("\n" + "="*80)
print("HELD-OUT TEST EVENT-BY-EVENT RESULTS & ERROR ISOLATION")
print("="*80)

for r in results:
    status = "CORRECT" if r['correct'] else f"ERROR: GT={r['gt']} vs PRED={r['pred']}"
    print(f"\nVideo: {r['video_name']} | TrackID: {r['pedestrian_id']} | {status}")
    print(f"  Frames tracked: {r['total_frames']} | Anomaly counts: {r['counts']}")
    print(f"  Top directions: {r['top_dirs']} | Avg speed: {r['avg_speed']:.2f} px/s | Max net disp: {r['max_disp']:.2f} px")
    if not r['correct']:
        print("  TIMELINE OF ANOMALIES / SAMPLES:")
        # print samples where anomaly occurred or direction was unusual
        for item in r['timeline']:
            f_idx, sec, d, spd, disp, atype = item
            if atype in ("WRONG_DIRECTION", "PROLONGED_STOPPING") or f_idx % 15 == 0:
                print(f"    Frame {f_idx:3d} ({sec:.2f}s): dir={d:12s} spd={spd:5.1f} disp={disp:5.1f} anomaly={atype}")

