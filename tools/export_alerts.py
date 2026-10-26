#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
Export all anomaly alerts (frame, track_id, confidence) for ground-truth comparison.

Runs the same detection + tracking + rule pipeline as the three anomaly scripts,
collecting every alert trigger into a structured JSON file.

Usage:
    python tools/export_alerts.py --video <video_path> [--type intrusion|running|fall] [--output outputs/alerts.json]

Note: This is an advanced tool that requires YOLO weights.
      For fall detection, a pose estimation model (yolov8n-pose.pt) is also needed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Shared ──────────────────────────────────────────────────────
DET_WEIGHTS = "yolov8s.pt"       # 默认检测模型 (用户可通过 --det-weights 覆盖)
POSE_WEIGHTS = "yolov8n-pose.pt" # 默认姿态模型 (用户可通过 --pose-weights 覆盖)
TRACKER = "bytetrack.yaml"
CONF = 0.5

# ── Intrusion params (from intrusion_detection.py) ──────────────
INTRUSION_CONFIRM = 3
INTRUSION_HANGOVER = 15

# ── Running params (from running_detection.py) ──────────────────
TRAJ_MAXLEN = 60
SPEED_THRESH_NORM = 80.0   # px/s
DISPL_THRESH_NORM = 150.0   # px
RUN_CONF_TRIGGER = 0.50
ALERT_COOLDOWN = 30

# ── Fall params (from fall_detection.py) ────────────────────────
FALL_AR_THRESH = 1.0
FALL_HEAD_HIP_RATIO = 0.35
FALL_CONFIRM_WINDOW = 5
FALL_CONFIRM_MIN = 0.5
FALL_HANGOVER = 30
COCO_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

# ══════════════════════════════════════════════════════════════════
#  Intrusion detection
# ══════════════════════════════════════════════════════════════════
def point_in_polygon(px, py, polygon):
    """cv2.pointPolygonTest — returns True if point inside/on polygon."""
    pts = np.array(polygon, np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

def detect_intrusion(video_path: str, zones: List[dict], det_weights: str) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"  INTRUSION detection: {video_path}")
    print(f"{'='*60}")
    model = YOLO(det_weights)
    alerts = []
    confirm_counter = defaultdict(int)
    hangover = defaultdict(int)
    triggered = set()

    results = model.track(
        source=video_path, persist=True,
        tracker=TRACKER, classes=[0], conf=CONF,
        stream=True, verbose=False,
    )

    frame_idx = 0
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.arange(len(xyxy))
        current_ids = set()

        for i, tid in enumerate(ids):
            tid = int(tid)
            current_ids.add(tid)
            x1, y1, x2, y2 = xyxy[i]
            foot_cx = (x1 + x2) / 2
            foot_y = y2

            in_zone = False
            for zone in zones:
                if point_in_polygon(foot_cx, foot_y, zone["polygon"]):
                    in_zone = True
                    break

            if in_zone:
                confirm_counter[tid] = confirm_counter.get(tid, 0) + 1
                if confirm_counter[tid] >= INTRUSION_CONFIRM:
                    hangover[tid] = INTRUSION_HANGOVER
                    if tid not in triggered:
                        triggered.add(tid)
                        alerts.append({
                            "frame": frame_idx, "track_id": tid,
                            "type": "intrusion", "confidence": 1.0,
                        })
                        print(f"  [INTRUSION] frame={frame_idx}, tid={tid}")
            else:
                confirm_counter[tid] = 0

            if hangover.get(tid, 0) > 0:
                hangover[tid] -= 1

        for tid in list(hangover.keys()):
            if tid not in current_ids and hangover[tid] > 0:
                hangover[tid] -= 1

        frame_idx += 1

    print(f"  Done. {len(alerts)} intrusion alerts  ({frame_idx} frames)")
    return alerts


# ══════════════════════════════════════════════════════════════════
#  Running detection
# ══════════════════════════════════════════════════════════════════
def detect_running(video_path: str, det_weights: str, fps: float = 30.0) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"  RUNNING detection: {video_path}")
    print(f"{'='*60}")
    model = YOLO(det_weights)
    alerts = []
    traj = defaultdict(lambda: [])
    last_center = {}
    cooldown = {}
    alive_frames = defaultdict(int)

    results = model.track(
        source=video_path, persist=True,
        tracker=TRACKER, classes=[0], conf=CONF,
        stream=True, verbose=False,
    )

    frame_idx = 0
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.arange(len(xyxy))

        for i, tid in enumerate(ids):
            tid = int(tid)
            alive_frames[tid] += 1
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            traj[tid].append((cx, cy, frame_idx))
            if len(traj[tid]) > TRAJ_MAXLEN:
                traj[tid] = traj[tid][-TRAJ_MAXLEN:]

            if len(traj[tid]) < 10:
                last_center[tid] = (cx, cy, frame_idx)
                continue

            pts = traj[tid]
            speeds = []
            for j in range(1, len(pts)):
                dx = pts[j][0] - pts[j-1][0]
                dy = pts[j][1] - pts[j-1][1]
                dt = (pts[j][2] - pts[j-1][2]) / fps
                if dt > 0:
                    speeds.append(np.sqrt(dx**2 + dy**2) / dt)

            if not speeds:
                last_center[tid] = (cx, cy, frame_idx)
                continue

            avg_speed = np.mean(speeds)
            speed_score = min(1.0, avg_speed / SPEED_THRESH_NORM)
            displ_score = min(1.0, np.sqrt((cx - pts[0][0])**2 + (cy - pts[0][1])**2) / DISPL_THRESH_NORM)

            direc_score = 0.5
            if len(pts) >= 4:
                vecs = []
                for j in range(1, len(pts)):
                    v = np.array([pts[j][0] - pts[j-1][0], pts[j][1] - pts[j-1][1]])
                    n = np.linalg.norm(v)
                    if n > 0:
                        vecs.append(v / n)
                if len(vecs) >= 2:
                    dots = [np.dot(vecs[k], vecs[k-1]) for k in range(1, len(vecs))]
                    direc_score = min(1.0, max(0.0, (np.mean(dots) + 1) / 2))

            conf = 0.5 * speed_score + 0.3 * displ_score + 0.2 * direc_score

            if conf >= RUN_CONF_TRIGGER:
                last_log = cooldown.get(tid, -ALERT_COOLDOWN)
                if frame_idx - last_log >= ALERT_COOLDOWN:
                    alerts.append({
                        "frame": frame_idx, "track_id": tid,
                        "type": "running", "confidence": round(conf, 3),
                        "speed": round(avg_speed, 1),
                    })
                    cooldown[tid] = frame_idx
                    print(f"  [RUNNING]  frame={frame_idx}, tid={tid}, conf={conf:.3f}, speed={avg_speed:.1f}px/s")

            last_center[tid] = (cx, cy, frame_idx)

        frame_idx += 1

    print(f"  Done. {len(alerts)} running alerts  ({frame_idx} frames)")
    return alerts


# ══════════════════════════════════════════════════════════════════
#  Fall detection (pose-based)
# ══════════════════════════════════════════════════════════════════
def is_fall_pose(keypoints_xy, bbox_w, bbox_h):
    """Replicated from fall_detection.py: is_fall_by_pose()"""
    if keypoints_xy is None or len(keypoints_xy) < 17:
        return False, 0.0

    kp = dict(zip(COCO_NAMES, keypoints_xy))
    nose = kp.get("nose", (0, 0))
    l_shoulder = kp.get("left_shoulder", (0, 0))
    r_shoulder = kp.get("right_shoulder", (0, 0))
    l_hip = kp.get("left_hip", (0, 0))
    r_hip = kp.get("right_hip", (0, 0))
    l_ankle = kp.get("left_ankle", (0, 0))
    r_ankle = kp.get("right_ankle", (0, 0))

    score = 0.0

    if bbox_h > 0:
        ar = bbox_w / bbox_h
        if ar > FALL_AR_THRESH:
            score += 0.5
        elif ar > 0.8:
            score += 0.2

    head_y = nose[1] if nose[1] > 0 else None
    hip_y = (l_hip[1] + r_hip[1]) / 2 if (l_hip[1] > 0 and r_hip[1] > 0) else None
    ankle_y = (l_ankle[1] + r_ankle[1]) / 2 if (l_ankle[1] > 0 and r_ankle[1] > 0) else None

    if head_y and hip_y and ankle_y:
        body_h = abs(ankle_y - head_y) + 1e-5
        hh_ratio = abs(head_y - hip_y) / body_h
        if hh_ratio < FALL_HEAD_HIP_RATIO:
            score += 0.4
        elif hh_ratio < 0.5:
            score += 0.15

    shoulder_mid = np.array([(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2])
    hip_mid = np.array([(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2])
    torso_vec = hip_mid - shoulder_mid
    torso_len = np.linalg.norm(torso_vec) + 1e-5
    verticality = abs(torso_vec[1]) / torso_len
    if verticality < 0.45:
        score += 0.3
    elif verticality < 0.6:
        score += 0.1

    return score >= 0.45, score


def detect_fall(video_path: str, pose_weights: str) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"  FALL detection: {video_path}")
    print(f"{'='*60}")
    model = YOLO(pose_weights)
    alerts = []
    fall_history = defaultdict(list)
    last_fall_frame = {}

    FALL_CONF = 0.25
    results = model.track(
        source=video_path, persist=True,
        tracker=TRACKER, classes=[0], conf=FALL_CONF,
        stream=True, verbose=False,
    )

    frame_idx = 0
    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.arange(len(xyxy))

        for i, tid in enumerate(ids):
            tid = int(tid)
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = x2 - x1, y2 - y1

            kpts_xy = keypoints.xy[i].cpu().numpy() if keypoints is not None and len(keypoints) > i else None
            is_f, score = is_fall_pose(kpts_xy, bw, bh)

            fall_history[tid].append(is_f)
            if len(fall_history[tid]) > FALL_CONFIRM_WINDOW:
                fall_history[tid] = fall_history[tid][-FALL_CONFIRM_WINDOW:]

            if len(fall_history[tid]) >= FALL_CONFIRM_WINDOW:
                ratio = sum(fall_history[tid]) / len(fall_history[tid])
                if ratio >= FALL_CONFIRM_MIN:
                    if (tid not in last_fall_frame or
                        frame_idx - last_fall_frame[tid] > FALL_HANGOVER):
                        alerts.append({
                            "frame": frame_idx, "track_id": tid,
                            "type": "fall", "confidence": round(score, 3),
                        })
                        last_fall_frame[tid] = frame_idx
                        print(f"  [FALL]     frame={frame_idx}, tid={tid}, score={score:.3f}")

        frame_idx += 1

    print(f"  Done. {len(alerts)} fall alerts  ({frame_idx} frames)")
    return alerts


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Export anomaly alerts for evaluation")
    parser.add_argument("--video", type=str, required=True,
                        help="输入视频路径")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "intrusion", "running", "fall"],
                        help="异常类型 (默认: all)")
    parser.add_argument("--det-weights", type=str, default=DET_WEIGHTS,
                        help=f"检测模型权重路径 (默认: {DET_WEIGHTS})")
    parser.add_argument("--pose-weights", type=str, default=POSE_WEIGHTS,
                        help=f"姿态估计权重路径 (默认: {POSE_WEIGHTS})")
    parser.add_argument("--output", type=str, default="outputs/alerts_export.json",
                        help="输出 JSON 路径 (默认: outputs/alerts_export.json)")
    args = parser.parse_args()

    video_path = args.video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_alerts = []

    if args.type in ("all", "intrusion"):
        # Use default example zone (user should provide their own)
        default_zone = [{"name": "demo", "polygon": [[300, 200], [600, 200], [600, 500], [300, 500]]}]
        all_alerts.extend(detect_intrusion(video_path, default_zone, args.det_weights))

    if args.type in ("all", "running"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        all_alerts.extend(detect_running(video_path, args.det_weights, fps=fps))

    if args.type in ("all", "fall"):
        all_alerts.extend(detect_fall(video_path, args.pose_weights))

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_alerts, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Export complete: {len(all_alerts)} alerts -> {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
