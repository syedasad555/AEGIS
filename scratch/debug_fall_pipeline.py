import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pedestrian_pipeline import PedestrianPipeline, load_config

cfg = load_config("configs/config.yaml")
pipeline = PedestrianPipeline(cfg)

video_path = PROJECT_ROOT / "assets" / "fall.mp4"
import cv2

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Loaded {video_path.name}: {w}x{h}, {total_frames} frames, {fps} fps")

fall_detections = []
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # Run pipeline step
    results = pipeline.model.track(
        frame,
        persist=True,
        tracker=pipeline.tracker_yaml,
        conf=pipeline.conf_thresh,
        classes=[0],
        verbose=False
    )
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().numpy()
            
            for bbox, tid in zip(coords, track_ids):
                feat = pipeline.feature_extractor.update_pedestrian(
                    track_id=int(tid),
                    bbox=tuple(bbox),
                    frame_idx=frame_idx,
                    fps=fps
                )
                feat = pipeline.anomaly_engine.evaluate_pedestrian(
                    feat,
                    frame_img=frame,
                    frame_h=h,
                    frame_w=w
                )
                if feat.get("fall_status") in ("FALL_SUSPECTED", "FALL_DETECTED") or "FALL" in feat.get("anomaly_type", ""):
                    fall_detections.append((frame_idx, tid, feat.get("anomaly_type"), feat.get("fall_status"), feat.get("fall_confidence"), feat.get("fall_reasons")))

cap.release()

print(f"Total fall events recorded: {len(fall_detections)}")
for f in fall_detections[:30]:
    print(f)
if len(fall_detections) > 30:
    print("...")
    for f in fall_detections[-10:]:
        print(f)
