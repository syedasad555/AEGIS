import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
sys.path.insert(0, str(Path('scripts')))
import cv2, yaml
from ultralytics import YOLO
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine
from collections import Counter, defaultdict

test_cases = [
    # Video, track_id, expected_flow, true_label
    ('evacuation_test_corridor.avi', 1, 'LEFT', 'PROLONGED_STOPPING'),
    ('evacuation_test_corridor.avi', 2, 'LEFT', 'PROLONGED_STOPPING'),
    ('evacuation_test_corridor.avi', 6, 'LEFT', 'PROLONGED_STOPPING'),
    ('evacuation_test_corridor.avi', 9, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 11, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 13, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 15, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 16, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 25, 'LEFT', 'WRONG_DIRECTION'),
    ('evacuation_test_corridor.avi', 3, 'LEFT', 'PROLONGED_STOPPING'),
    ('evacuation_test_corridor.avi', 4, 'LEFT', 'PROLONGED_STOPPING'),
    ('evacuation_test_corridor.avi', 7, 'LEFT', 'NORMAL'),
    # running.mp4
    ('running.mp4', 2, 'LEFT', 'NORMAL'),
    ('running.mp4', 7, 'LEFT', 'NORMAL'),
    ('running.mp4', 13, 'LEFT', 'WRONG_DIRECTION'),
    ('running.mp4', 24, 'LEFT', 'WRONG_DIRECTION'),
    ('running.mp4', 3, 'LEFT', 'PROLONGED_STOPPING'),
    # intrusion.mp4
    ('intrusion.mp4', 1, 'RIGHT', 'NORMAL'),
    ('intrusion.mp4', 2, 'RIGHT', 'NORMAL'),
    ('intrusion.mp4', 9, 'RIGHT', 'WRONG_DIRECTION'),
    ('intrusion.mp4', 12, 'RIGHT', 'WRONG_DIRECTION'),
    ('intrusion.mp4', 19, 'RIGHT', 'PROLONGED_STOPPING'),
    ('intrusion.mp4', 42, 'RIGHT', 'PROLONGED_STOPPING')
]

by_video = defaultdict(list)
for v, tid, exp, label in test_cases:
    by_video[(v, exp)].append((tid, label))

thresholds = [1.5, 2.0, 3.0, 5.0, 10.0]

for th in thresholds:
    cfg = yaml.safe_load(open('configs/config.yaml'))
    cfg['stopping_detection']['threshold_sec'] = th
    cfg['stopping_detection']['min_frames'] = int(th * 30)
    cfg['direction_min_displacement_px'] = 8.5
    cfg['direction_fast_displacement_px'] = 8.5
    cfg['direction_detection']['wrong_way_sustain_frames'] = 2
    
    tp_s, fp_s, fn_s = 0, 0, 0
    tp_w, fp_w, fn_w = 0, 0, 0
    correct = 0
    total = 0
    
    for (vname, exp_dir), tracks in by_video.items():
        c = dict(cfg)
        c['expected_evacuation_direction'] = exp_dir
        extractor = TrajectoryFeatureExtractor(c)
        engine = EvacuationAnomalyEngine(c)
        model = YOLO('weights/aegis_yolo11n_crowdhuman.pt')
        
        cap = cv2.VideoCapture(str(Path('assets') / vname))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        f_idx = 0
        t_counts = defaultdict(Counter)
        while True:
            ret, frame = cap.read()
            if not ret: break
            f_idx += 1
            res = model.track(source=frame, persist=True, tracker='configs/bytetrack_crowd.yaml', imgsz=800, conf=0.15, verbose=False)
            if res and res[0].boxes.id is not None:
                coords = res[0].boxes.xyxy.cpu().numpy()
                tids = res[0].boxes.id.cpu().numpy().astype(int)
                for bbox, tid in zip(coords, tids):
                    feat = extractor.update_pedestrian(tid, tuple(bbox), f_idx, fps=fps)
                    feat = engine.evaluate_pedestrian(feat)
                    t_counts[tid][feat.get('anomaly_type')] += 1
        cap.release()
        
        for tid, gt in tracks:
            cnt = t_counts[tid]
            p_stop = cnt['PROLONGED_STOPPING']
            p_wrong = cnt['WRONG_DIRECTION']
            if p_stop == 0 and p_wrong == 0:
                pred = 'NORMAL'
            elif p_stop >= p_wrong:
                pred = 'PROLONGED_STOPPING'
            else:
                pred = 'WRONG_DIRECTION'
            
            if pred == gt:
                correct += 1
            total += 1
            
            # Category level stats
            if gt == 'PROLONGED_STOPPING' and pred == 'PROLONGED_STOPPING': tp_s += 1
            elif gt != 'PROLONGED_STOPPING' and pred == 'PROLONGED_STOPPING': fp_s += 1
            elif gt == 'PROLONGED_STOPPING' and pred != 'PROLONGED_STOPPING': fn_s += 1
            
            if gt == 'WRONG_DIRECTION' and pred == 'WRONG_DIRECTION': tp_w += 1
            elif gt != 'WRONG_DIRECTION' and pred == 'WRONG_DIRECTION': fp_w += 1
            elif gt == 'WRONG_DIRECTION' and pred != 'WRONG_DIRECTION': fn_w += 1

    f1_s = (2 * tp_s) / max(2 * tp_s + fp_s + fn_s, 1)
    f1_w = (2 * tp_w) / max(2 * tp_w + fp_w + fn_w, 1)
    acc = correct / total
    print(f"Threshold: {th:4.1f}s | Acc: {acc*100:5.1f}% ({correct}/{total}) | Stop F1: {f1_s:.3f} (TP:{tp_s}, FP:{fp_s}, FN:{fn_s}) | WrongDir F1: {f1_w:.3f} (TP:{tp_w}, FP:{fp_w}, FN:{fn_w})")

