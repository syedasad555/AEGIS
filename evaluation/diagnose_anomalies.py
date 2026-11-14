import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cv2, yaml, json
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

curated_events = [
    ['evacuation_test_corridor.avi', '1', '1', '100', '0.03', '3.33', 'LEFT', 'STATIONARY', 'PROLONGED_STOPPING'],
    ['evacuation_test_corridor.avi', '2', '1', '120', '0.03', '4.00', 'LEFT', 'STATIONARY', 'PROLONGED_STOPPING'],
    ['evacuation_test_corridor.avi', '6', '1', '115', '0.03', '3.83', 'LEFT', 'STATIONARY', 'PROLONGED_STOPPING'],
    ['evacuation_test_corridor.avi', '9', '17', '48', '0.57', '1.60', 'LEFT', 'DOWN', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '11', '47', '60', '1.57', '2.00', 'LEFT', 'DOWN', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '13', '65', '78', '2.17', '2.60', 'LEFT', 'DOWN', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '15', '85', '90', '2.83', '3.00', 'LEFT', 'DOWN_RIGHT', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '16', '85', '90', '2.83', '3.00', 'LEFT', 'DOWN', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '25', '102', '120', '3.40', '4.00', 'LEFT', 'UP', 'WRONG_DIRECTION'],
    ['evacuation_test_corridor.avi', '3', '1', '100', '0.03', '3.33', 'LEFT', 'LEFT', 'NORMAL'],
    ['evacuation_test_corridor.avi', '4', '1', '108', '0.03', '3.60', 'LEFT', 'LEFT', 'NORMAL'],
    ['evacuation_test_corridor.avi', '7', '1', '118', '0.03', '3.93', 'LEFT', 'LEFT', 'NORMAL'],
    ['running.mp4', '1', '1', '80', '0.03', '2.67', 'RIGHT', 'RIGHT', 'NORMAL'],
    ['running.mp4', '2', '1', '120', '0.03', '4.00', 'RIGHT', 'RIGHT', 'NORMAL'],
    ['running.mp4', '3', '10', '150', '0.33', '5.00', 'RIGHT', 'LEFT', 'WRONG_DIRECTION'],
    ['running.mp4', '5', '30', '190', '1.00', '6.33', 'RIGHT', 'STATIONARY', 'PROLONGED_STOPPING'],
    ['intrusion.mp4', '1', '1', '200', '0.04', '8.00', 'RIGHT', 'RIGHT', 'NORMAL'],
    ['intrusion.mp4', '4', '20', '180', '0.80', '7.20', 'RIGHT', 'RIGHT', 'NORMAL'],
    ['intrusion.mp4', '8', '40', '160', '1.60', '6.40', 'RIGHT', 'LEFT', 'WRONG_DIRECTION'],
    ['intrusion.mp4', '12', '50', '220', '2.00', '8.80', 'RIGHT', 'STATIONARY', 'PROLONGED_STOPPING']
]

cfg = yaml.safe_load(open('configs/config.yaml'))
model = YOLO('weights/aegis_yolo11n_crowdhuman.pt')

video_events = defaultdict(list)
for row in curated_events:
    video_events[row[0]].append({
        'tid': int(row[1]), 'sf': int(row[2]), 'ef': int(row[3]),
        'exp_dir': row[6], 'act_dir': row[7], 'gt': row[8]
    })

for vname, events in video_events.items():
    cap = cv2.VideoCapture(str(Path('assets') / vname))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    exp_dir = events[0]['exp_dir']
    c = dict(cfg)
    c['expected_evacuation_direction'] = exp_dir
    extractor = TrajectoryFeatureExtractor(c)
    engine = EvacuationAnomalyEngine(c)
    model = YOLO('weights/aegis_yolo11n_crowdhuman.pt')
    
    pred_tracks = defaultdict(lambda: {'anomalies': set(), 'dirs': [], 'stops': [], 'disps': []})
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        f_idx += 1
        res = model.track(source=frame, persist=True, tracker='configs/bytetrack_crowd.yaml',
                          imgsz=800, conf=0.15, iou=0.55, classes=[0], device='cpu', verbose=False)
        if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.id is not None:
            coords = res[0].boxes.xyxy.cpu().numpy()
            tids = res[0].boxes.id.cpu().numpy().astype(int)
            for bbox, tid in zip(coords, tids):
                feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps)
                feat = engine.evaluate_pedestrian(feat)
                if feat.get('visual_state') == 'anomaly':
                    pred_tracks[tid]['anomalies'].add(feat.get('anomaly_type'))
                pred_tracks[tid]['dirs'].append((f_idx, feat.get('direction'), feat.get('direction_classification')))
                pred_tracks[tid]['stops'].append(feat.get('stopping_duration_sec', 0))
                pred_tracks[tid]['disps'].append(feat.get('displacement_px', 0))
    cap.release()
    print(f'=== Video: {vname} (FPS: {fps:.1f}) ===')
    for ev in events:
        t = ev['tid']
        gt = ev['gt']
        p = pred_tracks[t]
        max_stop = max(p['stops']) if p['stops'] else 0
        max_disp = max(p['disps']) if p['disps'] else 0
        dirs_summary = defaultdict(int)
        for _, d, _ in p['dirs']:
            dirs_summary[d] += 1
        print(f"Track {t:2d} | GT: {gt:18s} | Anomalies: {str(p['anomalies']):25s} | MaxStop: {max_stop:.2f}s | MaxDisp: {max_disp:.1f}px | Dirs: {dict(dirs_summary)}")
