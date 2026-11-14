import csv, json, yaml, cv2
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
import sys

PROJECT_ROOT = Path('.')
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine

ann_file = Path('evaluation/ucf_annotations.csv')
with open(ann_file) as f:
    events = list(csv.DictReader(f))

error_keys = {
    ('2018-6_70.mov', 6),
    ('2082-3_70.mov', 2),
    ('2082-3_70.mov', 24),
    ('341-46_l.mov', 30),
    ('7399-1_70.mov', 1),
    ('9019-13_l.mov', 78),
    ('913-37_l.mov', 6)
}

with open('configs/config.yaml') as f:
    base_cfg = yaml.safe_load(f)

model_path = Path('weights/aegis_yolo11n_crowdhuman.pt')
ucf_dir = Path('UCF_CrowdsDataset')

by_video = defaultdict(list)
for e in events:
    vname = e['video_name']
    tid = int(e['pedestrian_id'])
    if (vname, tid) in error_keys:
        by_video[(vname, e['expected_direction'])].append((tid, e['ground_truth_class'], e['notes']))

for (vname, exp_dir), track_list in by_video.items():
    c = dict(base_cfg)
    c['expected_evacuation_direction'] = exp_dir
    c['stopping_detection']['threshold_sec'] = 3.0
    c['stopping_detection']['min_frames'] = 75
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

    cap = cv2.VideoCapture(str(ucf_dir / vname))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f_idx = 0
    t_counts = defaultdict(Counter)
    t_total = defaultdict(int)
    t_speeds = defaultdict(list)
    t_dirs = defaultdict(list)
    t_statuses = defaultdict(list)
    t_stops = defaultdict(list)

    target_tids = {tid for tid, _, _ in track_list}

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
                if tid in target_tids:
                    feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps, conf=float(conf) if conf is not None else None, is_detection=True)
                    feat = engine.evaluate_pedestrian(feat)
                    atype = feat.get('anomaly_type')
                    t_counts[tid][atype] += 1
                    t_total[tid] += 1
                    t_speeds[tid].append(feat.get('speed_px_sec'))
                    t_dirs[tid].append(feat.get('movement_direction'))
                    t_statuses[tid].append(feat.get('movement_status'))
                    t_stops[tid].append(feat.get('stop_duration_sec', 0.0))

    cap.release()
    for tid, gt, notes in track_list:
        cnt = t_counts[tid]
        N = max(1, t_total[tid])
        p_stop = cnt['PROLONGED_STOPPING']
        p_wrong = cnt['WRONG_DIRECTION']
        print(f"==================================================")
        print(f"Video: {vname} | ID: {tid} | GT: {gt} | Expected: {exp_dir}")
        print(f"Notes: {notes}")
        print(f"Observed frames: {N} / {f_idx}")
        print(f"Counts: {dict(cnt)}")
        print(f"Statuses: {dict(Counter(t_statuses[tid]))}")
        print(f"Directions: {dict(Counter(t_dirs[tid]))}")
        speeds = [s for s in t_speeds[tid] if s is not None]
        stops = [s for s in t_stops[tid] if s is not None]
        avg_speed = sum(speeds)/len(speeds) if speeds else 0.0
        max_stop = max(stops) if stops else 0.0
        print(f"Avg speed: {avg_speed:.2f} px/s | Max stop duration: {max_stop:.2f} s")

