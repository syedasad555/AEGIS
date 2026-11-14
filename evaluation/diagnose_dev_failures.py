import csv
import yaml
import cv2
import sys
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

ann_file = PROJECT_ROOT / "evaluation" / "ucf_annotations.csv"
with open(ann_file, "r", encoding="utf-8") as f:
    events = list(csv.DictReader(f))

targets = [
    ("2018-6_70.mov", 6),
    ("9019-13_l.mov", 78),
    ("9019-13_l.mov", 77),
    ("2082-3_70.mov", 2),
    ("2082-3_70.mov", 24),
    ("637-147_l.mov", 10),
    ("821-10_l.mov", 77),
    ("341-46_l.mov", 30)
]

with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

model = YOLO(str(PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"))

for vname, tid in targets:
    ev = next(e for e in events if e["video_name"] == vname and int(e["pedestrian_id"]) == tid)
    vpath = PROJECT_ROOT / "UCF_CrowdsDataset" / vname
    c = dict(base_cfg)
    c["expected_evacuation_direction"] = ev["expected_direction"]
    c["stopping_detection"]["threshold_sec"] = 3.0
    c["stopping_detection"]["min_frames"] = 75
    c["direction_min_displacement_px"] = 9.0
    c["direction_fast_displacement_px"] = 9.0
    c["direction_detection"]["wrong_way_sustain_frames"] = 3
    c["stopping_detection"]["long_horizon_verification"] = True
    c["stopping_detection"]["max_cumulative_disp_px"] = 20.0

    extractor = TrajectoryFeatureExtractor(c)
    engine = EvacuationAnomalyEngine(c)
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f_idx = 0
    t_feats = []
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
            for bbox, t in zip(boxes, tids):
                if t == tid:
                    feat = extractor.update_pedestrian(t, tuple(bbox), f_idx, fps=fps)
                    feat = engine.evaluate_pedestrian(feat)
                    t_feats.append({
                        "frame": f_idx,
                        "foot": feat.get("foot"),
                        "dir": feat["direction"],
                        "stop_sec": feat["stopping_duration_sec"],
                        "stop_frames": feat["stopping_duration_frames"],
                        "disp": feat.get("displacement_px", 0.0),
                        "speed": feat.get("speed_px_sec", 0.0),
                        "status": feat.get("movement_status"),
                        "anom": feat["anomaly_type"]
                    })
    cap.release()
    print(f"=== {vname} Track {tid} (GT: {ev['ground_truth_class']}, Exp: {ev['expected_direction']}) ===")
    print(f"Notes: {ev.get('notes', '')}")
    print(f"Total frames observed: {len(t_feats)}")
    if t_feats:
        anom_counts = Counter(x["anom"] for x in t_feats)
        dirs = Counter(x["dir"] for x in t_feats)
        max_stop = max(x["stop_sec"] for x in t_feats)
        max_disp = max(x["disp"] for x in t_feats)
        print(f"  Anom counts: {dict(anom_counts)}, Dirs: {dict(dirs)}, Max stop sec: {max_stop}, Max disp: {max_disp}")
        # print frames where anom occurred or key transitions
        samples = [t_feats[0], t_feats[len(t_feats)//2], t_feats[-1]]
        print(f"  Samples: {samples}")

