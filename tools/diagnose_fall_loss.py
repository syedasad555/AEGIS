#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 22 — Deep Diagnosis of Fall Track Loss
==================================================
Empirical diagnostic script investigating why the fall track is lost
in `assets/fall.mp4` across frames 220–306.

Examines:
1. Raw YOLOv8n detections at ultralow confidence thresholds (conf=0.01 to 0.50).
2. Spatial location of the fallen person (ramp corridor: x in [450, 750], y in [400, 700]).
3. ByteTrack internal state: tracked_stracks, lost_stracks, removed_stracks.
4. Pose estimation (yolov8n-pose.pt) keypoints on candidate bounding box & crop.
5. FallDetector state machine transitions and reset mechanics.
"""

import sys
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fall_detection import FallDetector
from scripts.pedestrian_pipeline import load_config


def diagnose_fall_clip(
    video_path: Path,
    start_frame: int = 218,
    end_frame: int = 250,
    conf_thresh: float = 0.05
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    pose_model = YOLO("weights/yolov8n-pose.pt")
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    detector = FallDetector(cfg)

    # Standard ByteTrack configuration used in AEGIS
    bt_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.05,
        new_track_thresh=0.15,
        track_buffer=60,
        match_thresh=0.60,
        fuse_score=True
    )
    tracker = BYTETracker(bt_args)

    frame_idx = 0
    records = []

    # Target fallen person spatial ROI on the ramp
    # In assets/fall.mp4, person slips at frame 220 near x=540, y=480 and slides down towards x=560, y=550
    target_roi = [450, 400, 750, 650]  # [x1, y1, x2, y2]

    last_known_bbox = None
    target_track_id = None

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < start_frame:
            # Warm up tracking
            res = model.predict(frame, conf=0.25, classes=[0], verbose=False)
            continue

        # 1. Raw YOLO prediction at low confidence (0.05) to check detector sensitivity
        res_raw = model.predict(frame, conf=0.03, classes=[0], verbose=False)
        raw_boxes = []
        raw_scores = []
        target_raw_detections = []

        if res_raw and len(res_raw) > 0 and res_raw[0].boxes is not None:
            boxes = res_raw[0].boxes
            b_coords = boxes.xyxy.cpu().numpy()
            b_scores = boxes.conf.cpu().numpy()
            for bx, sc in zip(b_coords, b_scores):
                raw_boxes.append(list(bx))
                raw_scores.append(float(sc))
                # Check if box is inside target corridor
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
                if target_roi[0] <= cx <= target_roi[2] and target_roi[1] <= cy <= target_roi[3]:
                    target_raw_detections.append({
                        "bbox": [round(float(x), 1) for x in bx],
                        "conf": round(float(sc), 3),
                        "w": round(float(bx[2] - bx[0]), 1),
                        "h": round(float(bx[3] - bx[1]), 1),
                        "ar": round(float((bx[2] - bx[0]) / max(1.0, bx[3] - bx[1])), 2)
                    })

        # 2. Tracking with ByteTrack via SimpleDetections
        from scripts.tiled_inference import SimpleDetections
        # Filter for tracker input (conf >= 0.05)
        flt_boxes = [b for b, s in zip(raw_boxes, raw_scores) if s >= 0.05]
        flt_scores = [s for s in raw_scores if s >= 0.05]
        flt_cls = [0] * len(flt_boxes)

        det_container = SimpleDetections(flt_boxes, flt_scores, flt_cls)
        tracks = tracker.update(det_container)

        active_tids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        target_track_info = None

        for t in tracks:
            bx = t[:4]
            tid = int(t[4])
            sc = float(t[5])
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            if target_roi[0] <= cx <= target_roi[2] and target_roi[1] <= cy <= target_roi[3]:
                target_track_id = tid
                target_track_info = {
                    "tid": tid,
                    "bbox": [round(float(x), 1) for x in bx],
                    "score": round(sc, 3),
                    "ar": round(float((bx[2] - bx[0]) / max(1.0, bx[3] - bx[1])), 2)
                }
                last_known_bbox = list(bx)

        # 3. FallDetector Evaluation
        fall_eval_output = None
        if target_track_info:
            tid = target_track_info["tid"]
            eval_res = detector.evaluate_pedestrian(
                track_id=tid,
                bbox=tuple(target_track_info["bbox"]),
                frame_idx=frame_idx,
                frame_h=h,
                frame_w=w,
                frame_img=frame
            )
            fall_eval_output = {
                "status": eval_res["status"],
                "consecutive": eval_res["consecutive_frames"],
                "ar": eval_res["aspect_ratio"],
                "reasons": eval_res["reasons"]
            }

        # 4. Pose estimation directly on candidate crop or last known location
        pose_res_info = None
        crop_box = target_track_info["bbox"] if target_track_info else last_known_bbox
        if crop_box:
            pad = 20
            cx1 = max(0, int(crop_box[0] - pad))
            cy1 = max(0, int(crop_box[1] - pad))
            cx2 = min(w, int(crop_box[2] + pad))
            cy2 = min(h, int(crop_box[3] + pad))
            if cx2 > cx1 + 10 and cy2 > cy1 + 10:
                crop = frame[cy1:cy2, cx1:cx2]
                res_p = pose_model(crop, conf=0.05, verbose=False)
                if res_p and len(res_p) > 0 and res_p[0].keypoints is not None and len(res_p[0].keypoints.xy) > 0:
                    kpts = res_p[0].keypoints.xy[0].cpu().numpy()
                    k_conf = res_p[0].keypoints.conf[0].cpu().numpy() if res_p[0].keypoints.conf is not None else None
                    valid_kpts = np.sum(kpts[:, 0] > 0)
                    mean_conf = float(np.mean(k_conf[k_conf > 0])) if (k_conf is not None and np.any(k_conf > 0)) else 0.0
                    pose_res_info = {
                        "detected": True,
                        "valid_keypoints": int(valid_kpts),
                        "mean_kpt_conf": round(mean_conf, 3)
                    }
                else:
                    pose_res_info = {"detected": False, "valid_keypoints": 0}

        rec = {
            "frame": frame_idx,
            "target_raw_detections": target_raw_detections,
            "target_track": target_track_info,
            "fall_eval": fall_eval_output,
            "pose_info": pose_res_info,
            "total_active_tracks": len(active_tids),
            "tracker_lost_tracks_count": len(tracker.lost_stracks),
            "tracker_tracked_count": len(tracker.tracked_stracks)
        }
        records.append(rec)

    cap.release()
    return {"video": str(video_path), "records": records}


if __name__ == "__main__":
    out = diagnose_fall_clip(PROJECT_ROOT / "assets" / "fall.mp4", start_frame=218, end_frame=245)
    print("=" * 80)
    print(" FRAME-BY-FRAME DEEP DIAGNOSIS ON assets/fall.mp4 (Frames 218 - 245)")
    print("=" * 80)
    for r in out["records"]:
        f = r["frame"]
        raws = r["target_raw_detections"]
        trk = r["target_track"]
        fe = r["fall_eval"]
        pose = r["pose_info"]
        print(f"Frame {f:03d}: RawDets={len(raws)} {raws} | Track={trk} | FallState={fe} | Pose={pose}")
