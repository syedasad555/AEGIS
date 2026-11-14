import csv
import json
import yaml
from pathlib import Path
from collections import defaultdict
import cv2
from ultralytics import YOLO
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

# Targets:
targets = [
    {"video": "861-13_70.mov", "tid": 3, "exp_dir": "RIGHT", "gt": "WRONG_DIRECTION"},
    {"video": "879-38_l.mov", "tid": 37, "exp_dir": "RIGHT", "gt": "PROLONGED_STOPPING"},
    {"video": "906-17_l.mov", "tid": 4, "exp_dir": "LEFT", "gt": "NORMAL"},
    {"video": "913-36_l.mov", "tid": 1, "exp_dir": "RIGHT", "gt": "NORMAL"},
]

model_path = PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"
ucf_dir = PROJECT_ROOT / "UCF_CrowdsDataset"

for t in targets:
    vname = t["video"]
    target_tid = t["tid"]
    exp_dir = t["exp_dir"]
    gt = t["gt"]
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
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    f_idx = 0
    records = []
    
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
                if tid == target_tid:
                    records.append({
                        "frame": f_idx,
                        "sec": sec,
                        "bbox": [round(float(x), 1) for x in bbox],
                        "cx": round((bbox[0]+bbox[2])/2, 1),
                        "cy": round((bbox[1]+bbox[3])/2, 1),
                        "dir": feat.get("direction"),
                        "speed": round(float(feat.get("speed", 0.0)), 2),
                        "net_disp": round(float(feat.get("net_displacement", 0.0)), 2),
                        "smooth_vel": [round(float(v), 2) for v in feat.get("smoothed_velocity", (0, 0))],
                        "stop_dur": round(float(feat.get("stopping_duration_sec", 0.0)), 2),
                        "anomaly": feat.get("anomaly_type"),
                        "raw_angle": round(float(feat.get("raw_angle", 0.0)), 1) if feat.get("raw_angle") is not None else None,
                        "heading_deg": round(float(feat.get("heading_degrees", 0.0)), 1) if feat.get("heading_degrees") is not None else None,
                        "state": feat.get("visual_state")
                    })
    cap.release()
    
    print("\n" + "="*80)
    print(f"DEEP DIVE: Video={vname} (Res: {int(w)}x{int(h)}, FPS: {fps}) | Target TrackID={target_tid} | GT={gt} | ExpDir={exp_dir}")
    print(f"Total frames tracked: {len(records)}")
    if records:
        print(f"Start: frame {records[0]['frame']} ({records[0]['sec']:.2f}s), End: frame {records[-1]['frame']} ({records[-1]['sec']:.2f}s)")
        print(f"Start bbox: {records[0]['bbox']} -> End bbox: {records[-1]['bbox']}")
        total_dx = records[-1]['cx'] - records[0]['cx']
        total_dy = records[-1]['cy'] - records[0]['cy']
        total_disp = (total_dx**2 + total_dy**2)**0.5
        print(f"Total Trajectory Shift: dx={total_dx:.1f}px, dy={total_dy:.1f}px, Net 2D Dist={total_disp:.1f}px")
        
        # Anomaly frames
        anom_frames = [r for r in records if r['anomaly'] != 'NORMAL']
        print(f"Total Anomaly Frames: {len(anom_frames)} / {len(records)}")
        print("\nSample records throughout trajectory:")
        step = max(1, len(records)//8)
        for r in records[::step]:
            print(f"  Fr {r['frame']:3d} ({r['sec']:4.2f}s): pos=({r['cx']:5.1f},{r['cy']:5.1f}) dir={r['dir']:11s} spd={r['speed']:5.1f} disp={r['net_disp']:5.1f} stop_dur={r['stop_dur']:4.1f}s state={r['state']:10s} anom={r['anomaly']}")
        if anom_frames:
            print("\nFirst 3 anomaly occurrences:")
            for r in anom_frames[:3]:
                print(f"  Fr {r['frame']:3d} ({r['sec']:4.2f}s): pos=({r['cx']:5.1f},{r['cy']:5.1f}) dir={r['dir']:11s} spd={r['speed']:5.1f} disp={r['net_disp']:5.1f} stop_dur={r['stop_dur']:4.1f}s state={r['state']:10s} anom={r['anomaly']}")
            print("\nLast 3 anomaly occurrences:")
            for r in anom_frames[-3:]:
                print(f"  Fr {r['frame']:3d} ({r['sec']:4.2f}s): pos=({r['cx']:5.1f},{r['cy']:5.1f}) dir={r['dir']:11s} spd={r['speed']:5.1f} disp={r['net_disp']:5.1f} stop_dur={r['stop_dur']:4.1f}s state={r['state']:10s} anom={r['anomaly']}")

