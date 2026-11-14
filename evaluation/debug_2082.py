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

vname = "2082-3_70.mov"
exp_dir = "LEFT"
vpath = ucf_dir / vname

with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

c = dict(base_cfg)
c['expected_evacuation_direction'] = exp_dir
c['stopping_detection']['threshold_sec'] = 3.0
c['stopping_detection']['min_frames'] = 75
c['direction_min_displacement_px'] = 9.0
c['direction_fast_displacement_px'] = 9.0
c['direction_detection']['wrong_way_sustain_frames'] = 3

model_path = PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"
extractor = TrajectoryFeatureExtractor(c)
engine = EvacuationAnomalyEngine(c)
model = YOLO(str(model_path))

cap = cv2.VideoCapture(str(vpath))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
f_idx = 0
track_records = defaultdict(list)

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
            if tid in (1, 2, 24):
                foot = feat.get('foot', (0, 0))
                disp = feat.get('displacement_px', 0.0)
                stop_dur = feat.get('stopping_duration_sec', 0.0)
                direction = feat.get('direction', 'UNKNOWN')
                atype = feat.get('anomaly_type')
                track_records[tid].append((f_idx, sec, foot, disp, stop_dur, direction, atype))
cap.release()

print("="*80)
print(f"DEEP DIVE 2082-3_70.mov tracks 1, 2, 24:")
print("="*80)
for tid in (1, 2, 24):
    recs = track_records[tid]
    print(f"\nTrack {tid}: {len(recs)} observations")
    if recs:
        f_start, s_start, foot_start = recs[0][0], recs[0][1], recs[0][2]
        f_end, s_end, foot_end = recs[-1][0], recs[-1][1], recs[-1][2]
        net_dx = foot_end[0] - foot_start[0]
        net_dy = foot_end[1] - foot_start[1]
        net_dist = (net_dx**2 + net_dy**2)**0.5
        print(f"  Start fr {f_start} ({s_start:.2f}s) foot={foot_start} -> End fr {f_end} ({s_end:.2f}s) foot={foot_end}")
        print(f"  Net 2D dist={net_dist:.1f}px (dx={net_dx:.1f}, dy={net_dy:.1f})")
        # print sample records
        for r in recs[::max(1, len(recs)//6)]:
            print(f"    fr {r[0]:3d} ({r[1]:.2f}s): foot=({r[2][0]:.1f},{r[2][1]:.1f}) disp={r[3]:4.1f} stop_dur={r[4]:.2f}s dir={r[5]:12s} atype={r[6]}")

