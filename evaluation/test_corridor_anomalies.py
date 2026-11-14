import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
sys.path.insert(0, str(Path('scripts')))
import cv2, yaml
from ultralytics import YOLO
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

cfg = yaml.safe_load(open('configs/config.yaml'))
cfg['expected_evacuation_direction'] = 'LEFT'
cfg['stopping_detection']['threshold_sec'] = 1.5
cfg['stopping_detection']['min_frames'] = 45
cfg['direction_min_displacement_px'] = 12.0
cfg['direction_fast_displacement_px'] = 12.0
cfg['direction_detection']['wrong_way_sustain_frames'] = 4

extractor = TrajectoryFeatureExtractor(cfg)
engine = EvacuationAnomalyEngine(cfg)
model = YOLO('weights/aegis_yolo11n_crowdhuman.pt')

cap = cv2.VideoCapture('assets/evacuation_test_corridor.avi')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

f_idx = 0
track_res = {}
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
            if tid not in track_res:
                track_res[tid] = {'anomalies': set(), 'dirs': set(), 'max_stop': 0}
            if feat.get('visual_state') == 'anomaly':
                track_res[tid]['anomalies'].add(feat.get('anomaly_type'))
            track_res[tid]['dirs'].add(feat.get('direction'))
            track_res[tid]['max_stop'] = max(track_res[tid]['max_stop'], feat.get('stopping_duration_sec', 0))
cap.release()

for t in sorted(track_res):
    anoms = track_res[t]['anomalies']
    mstop = track_res[t]['max_stop']
    print(f"Track {t:2d}: anomalies={anoms} max_stop={mstop:.2f}s")

