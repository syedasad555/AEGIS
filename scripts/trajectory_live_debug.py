#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live trajectory / movement-estimation diagnostic for AEGIS.

Does NOT change the 3-of-8 classifier. Runs YOLO+ByteTrack on real video,
dumps per-track foot histories, and optionally writes annotated debug frames.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direction_estimation import (
    CLASSIFIER_VERSION,
    DirectionEstimator,
    angle_to_direction8,
    movement_angle_deg,
)
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine
from _common import (
    resolve_yolo_model_path,
    resolve_tracker,
    resolve_inference_imgsz,
    extract_track_ids,
)


def _draw_debug(
    frame: np.ndarray,
    ped: Dict[str, Any],
    expected: str,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in ped["bbox"]]
    vs = ped.get("visual_state")
    colour = (40, 40, 230) if vs == "anomaly" else (255, 160, 40)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    foot = ped.get("foot") or [0, 0]
    fx, fy = int(foot[0]), int(foot[1])
    cv2.circle(frame, (fx, fy), 4, (0, 255, 255), -1)

    traj = ped.get("trajectory") or []
    pts = [(int(p[0]), int(p[1])) for p in traj[-20:]]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, (0, 200, 0), -1)

    dvec = ped.get("direction_vector") or [0, 0]
    dx, dy = float(dvec[0]), float(dvec[1])
    mag = math.hypot(dx, dy) or 1.0
    tip = (int(fx + dx / mag * 40), int(fy + dy / mag * 40))
    cv2.arrowedLine(frame, (fx, fy), tip, colour, 2, tipLength=0.35)

    lines = [
        f"#{ped['track_id']} {ped.get('direction')} conf={ped.get('direction_confidence', 0):.2f}",
        f"exp={expected} Δ={ped.get('sector_distance')} {ped.get('direction_classification')}",
        f"vs={vs} anom={ped.get('anomaly_type')} sust={ped.get('direction_diagnostics', {}).get('sustain_frames')}",
        f"dx={dx:.1f} dy={dy:.1f} disp={ped.get('displacement_px')}",
    ]
    ty = max(20, y1 - 8)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x1, ty - 14 * (len(lines) - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)


def analyze(
    video: Path,
    expected: str = "LEFT",
    max_frames: int = 150,
    out_dir: Optional[Path] = None,
    save_frames: bool = True,
) -> Dict[str, Any]:
    out_dir = out_dir or (PROJECT_ROOT / "outputs" / "trajectory_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = dict(cfg)
    cfg["expected_evacuation_direction"] = expected

    print(f"CLASSIFIER_VERSION={CLASSIFIER_VERSION}")
    print(f"expected={expected}")
    print(
        f"thresholds: min_disp={cfg.get('direction_min_displacement_px')} "
        f"min_speed={cfg.get('direction_min_speed_px_sec')} "
        f"window={cfg.get('direction_window_frames')} "
        f"ema={cfg.get('direction_ema_alpha')} "
        f"sustain={ (cfg.get('direction_detection') or {}).get('wrong_way_sustain_frames') }"
    )

    from ultralytics import YOLO

    model = YOLO(resolve_yolo_model_path(cfg, require_exists=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tracker = resolve_tracker(cfg.get("tracker_config", "configs/bytetrack_crowd.yaml"))
    imgsz = resolve_inference_imgsz(cfg)
    conf = float(cfg.get("detection_confidence", 0.08))
    extractor = TrajectoryFeatureExtractor(cfg)
    engine = EvacuationAnomalyEngine(cfg)
    engine.set_expected_direction(expected)

    # Parallel raw (no EMA) estimator for comparison evidence
    raw_est = DirectionEstimator({
        **cfg,
        "direction_ema_alpha": 1.0,  # no smoothing
    })

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    classes = [int(cfg.get("person_class_id", 0))]
    rows: List[Dict[str, Any]] = []
    track_series: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    anomaly_first: Dict[int, int] = {}
    confirm_frame: Dict[int, int] = {}
    frame_idx = 0
    debug_written = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps

        results = model.track(
            source=frame, persist=True, tracker=tracker, conf=conf,
            imgsz=imgsz, classes=classes, device=device, verbose=False,
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            continue

        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        tids = extract_track_ids(boxes)
        annotated = frame.copy()
        frame_peds = []

        for bbox, tid in zip(coords, tids):
            tid = int(tid)
            x1, y1, x2, y2 = map(float, bbox)
            raw_fx = (x1 + x2) / 2.0
            raw_fy = y2

            feat = extractor.update_pedestrian(tid, (x1, y1, x2, y2), frame_idx, fps)
            feat = engine.evaluate_pedestrian(feat)
            frame_peds.append(feat)

            # Raw (unsmoothed) comparison
            raw_info = raw_est.update(tid, (raw_fx, raw_fy), frame_idx, ts)
            traj = feat.get("trajectory") or []
            prev = traj[-2] if len(traj) >= 2 else None
            dvec = feat.get("direction_vector") or [0.0, 0.0]
            dx, dy = float(dvec[0]), float(dvec[1])
            clf = feat.get("direction_classification")
            sust = (feat.get("direction_diagnostics") or {}).get("sustain_frames", 0)

            if clf == "ANOMALY" and tid not in anomaly_first:
                anomaly_first[tid] = frame_idx
            if feat.get("visual_state") == "anomaly" and feat.get("anomaly_type") == "WRONG_DIRECTION":
                if tid not in confirm_frame:
                    confirm_frame[tid] = frame_idx

            row = {
                "TRACK_ID": tid,
                "FRAME_NUMBER": frame_idx,
                "TIMESTAMP": round(ts, 3),
                "BOUNDING_BOX": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "FOOTPOINT_X": round(raw_fx, 2),
                "FOOTPOINT_Y": round(raw_fy, 2),
                "PREVIOUS_FOOTPOINT_X": None if prev is None else round(prev[0], 2),
                "PREVIOUS_FOOTPOINT_Y": None if prev is None else round(prev[1], 2),
                "DX": round(dx, 2),
                "DY": round(dy, 2),
                "DISPLACEMENT": feat.get("displacement_px"),
                "ELAPSED_TIME": round(ts, 3),
                "ESTIMATED_SPEED": feat.get("speed_px_sec"),
                "SMOOTHED_DX": round(dx, 2),
                "SMOOTHED_DY": round(dy, 2),
                "RAW_DX": raw_info.get("direction_vector", [0, 0])[0],
                "RAW_DY": raw_info.get("direction_vector", [0, 0])[1],
                "RAW_DIRECTION": raw_info.get("direction"),
                "ESTIMATED_DIRECTION": feat.get("direction"),
                "ESTIMATED_DIRECTION_CONFIDENCE": feat.get("direction_confidence"),
                "DIRECTION_RELIABLE": feat.get("direction_reliable"),
                "EXPECTED_EVACUATION_DIRECTION": feat.get("expected_direction"),
                "SECTOR_DIFFERENCE": feat.get("sector_distance"),
                "CLASSIFIER_RESULT": clf,
                "CONSECUTIVE_WRONG_DIRECTION_FRAMES": sust,
                "ANOMALY_ENGINE_RESULT": feat.get("anomaly_type"),
                "FINAL_VISUAL_STATE": feat.get("visual_state"),
                "BBOX_HEIGHT": round(y2 - y1, 1),
            }
            rows.append(row)
            track_series[tid].append(row)

            if tid in list(track_series.keys())[:8]:
                _draw_debug(annotated, feat, expected)

        if save_frames and frame_idx % 5 == 0 and frame_peds and debug_written < 25:
            path = out_dir / f"debug_{expected}_f{frame_idx:04d}.jpg"
            cv2.imwrite(str(path), annotated)
            debug_written += 1

    cap.release()

    # Pick 10 longest tracks for summary
    top = sorted(track_series.items(), key=lambda kv: -len(kv[1]))[:10]
    summary = []
    for tid, series in top:
        first, last = series[0], series[-1]
        net_dx = last["FOOTPOINT_X"] - first["FOOTPOINT_X"]
        net_dy = last["FOOTPOINT_Y"] - first["FOOTPOINT_Y"]
        net_dir = angle_to_direction8(net_dx, net_dy) if (abs(net_dx) + abs(net_dy)) > 1 else "STATIONARY"
        # Dominant estimated labels
        labels = [r["ESTIMATED_DIRECTION"] for r in series if r["DIRECTION_RELIABLE"]]
        from collections import Counter
        mode = Counter(labels).most_common(1)[0][0] if labels else "UNKNOWN"
        feet = [[r["FOOTPOINT_X"], r["FOOTPOINT_Y"]] for r in series[-20:]]
        summary.append({
            "TRACK_ID": tid,
            "N_FRAMES": len(series),
            "NET_DX": round(net_dx, 1),
            "NET_DY": round(net_dy, 1),
            "NET_DIRECTION_RAW_FOOT": net_dir,
            "MODE_ESTIMATED_DIRECTION": mode,
            "LAST_ESTIMATED": last["ESTIMATED_DIRECTION"],
            "LAST_RAW_ESTIMATED": last["RAW_DIRECTION"],
            "LAST_CLASSIFIER": last["CLASSIFIER_RESULT"],
            "LAST_VISUAL": last["FINAL_VISUAL_STATE"],
            "ANOMALY_FIRST_FRAME": anomaly_first.get(tid),
            "CONFIRM_FRAME": confirm_frame.get(tid),
            "LAST_20_FEET": feet,
            "AVG_BBOX_HEIGHT": round(sum(r["BBOX_HEIGHT"] for r in series) / len(series), 1),
            "BBOX_HEIGHT_STD": round(float(np.std([r["BBOX_HEIGHT"] for r in series])), 1),
        })

    csv_path = out_dir / f"trajectory_log_{expected}.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary_path = out_dir / f"trajectory_summary_{expected}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "classifier_version": CLASSIFIER_VERSION,
            "expected": expected,
            "video": str(video),
            "frames": frame_idx,
            "pid": os.getpid(),
            "tracks": summary,
            "n_rows": len(rows),
        }, f, indent=2)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")
    for s in summary:
        print(
            f"TID={s['TRACK_ID']} n={s['N_FRAMES']} net=({s['NET_DX']},{s['NET_DY']})→{s['NET_DIRECTION_RAW_FOOT']} "
            f"mode_est={s['MODE_ESTIMATED_DIRECTION']} last_est={s['LAST_ESTIMATED']} raw_est={s['LAST_RAW_ESTIMATED']} "
            f"clf={s['LAST_CLASSIFIER']} vs={s['LAST_VISUAL']} "
            f"anom@{s['ANOMALY_FIRST_FRAME']} confirm@{s['CONFIRM_FRAME']} "
            f"bboxHμ={s['AVG_BBOX_HEIGHT']} σ={s['BBOX_HEIGHT_STD']}"
        )
    return {"summary": summary, "csv": str(csv_path), "summary_path": str(summary_path)}


def main():
    video = PROJECT_ROOT / "assets" / "evacuation_test_corridor.avi"
    if not video.exists():
        print("Video not found:", video)
        return 1
    expected = sys.argv[1] if len(sys.argv) > 1 else "LEFT"
    analyze(video, expected=expected, max_frames=120, save_frames=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
