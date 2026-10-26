#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 28 — Compact reliability evaluation
===============================================
A) Detection: re-score Phase 27 GT frames at imgsz 800 vs 960 (crowd tracker).
B) Fall: temporal benchmark on labeled fall/ADL clips + false-alarm scan on
   corridor/running/intrusion (no fall GT — any FALL_CONFIRMED event is FA).

Does not invent IDs or GT labels. Reports unavailable metrics explicitly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker
from scripts.fall_detection import FallDetector
from tools.evaluate_phase27_recall import evaluate_config, load_gt
from tools.evaluate_temporal_fall import run_temporal_fall_benchmark


def count_confirmed_fall_events(video_rel: str, cfg: Dict[str, Any], max_frames: int = 120) -> Dict[str, Any]:
    path = PROJECT_ROOT / video_rel
    if not path.exists():
        return {"video": video_rel, "status": "MISSING"}

    model = YOLO(str(PROJECT_ROOT / cfg.get("yolo_model", "yolov8n.pt")))
    tracker = resolve_tracker(str(PROJECT_ROOT / cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")))
    imgsz = int(cfg.get("inference_imgsz", 800))
    conf = float(cfg.get("detection_confidence", 0.08))
    fall = FallDetector(cfg)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    confirmed_events = set()
    suspected_frames = 0
    confirmed_frames = 0
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        res = model.track(
            frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
            classes=[0], verbose=False
        )[0]
        tids = []
        if res.boxes is not None and res.boxes.id is not None:
            tids = extract_track_ids(res.boxes)
            xyxy = res.boxes.xyxy.cpu().numpy()
            for bbox, tid in zip(xyxy, tids):
                out = fall.evaluate_pedestrian(
                    int(tid), tuple(map(float, bbox)), frame_idx, h, w, frame, speed_px_sec=0.0
                )
                if out["status"] == FallDetector.STATE_CONFIRMED:
                    confirmed_frames += 1
                    if out.get("event_id") is not None:
                        confirmed_events.add(int(out["event_id"]))
                elif out["status"] == FallDetector.STATE_SUSPECTED:
                    suspected_frames += 1
        fall.process_frame_persistence(frame_idx, [int(t) for t in tids], [], frame, h, w)

    cap.release()
    dur = frame_idx / max(fps, 1e-6)
    return {
        "video": video_rel,
        "status": "OK",
        "frames_scanned": frame_idx,
        "duration_sec": round(dur, 2),
        "confirmed_event_count": len(confirmed_events),
        "confirmed_frames": confirmed_frames,
        "suspected_frames": suspected_frames,
        "false_alarm_events_if_no_gt_fall": len(confirmed_events),
        "eval_wall_sec": round(time.time() - t0, 1),
    }


def main():
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    out: Dict[str, Any] = {
        "suite": "AEGIS Phase 28 Reliability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": {
            "341-46_l.mov": "NOT FOUND in repository; detection metrics reuse Phase 27 GT frames instead.",
            "id_switches": "Not measured on asset clips (no person-ID GT). Available only on MOT sequences with official IDs.",
        },
    }

    # --- Task A: detection on Phase 27 GT ---
    gt = load_gt(PROJECT_ROOT / "data/phase27_gt/ground_truth.json")
    frames = gt["frames"]
    device = "cpu"
    det_results = []
    for imgsz in (800, 960):
        print(f"\n[A] Detection imgsz={imgsz} crowd tracker...", flush=True)
        res = evaluate_config(
            frames, conf=0.08, imgsz=imgsz, tracker_name="bytetrack_crowd",
            tracker_rel="configs/bytetrack_crowd.yaml", device=device, use_tiled=False
        )
        det_results.append(res)
        d = res["detection"]
        print(f"  P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"FPS~{res['latency']['approx_processing_fps']:.1f}")
    out["detection"] = det_results

    # --- Task B: labeled fall benchmark ---
    print("\n[B] Temporal fall benchmark...", flush=True)
    fall_bench = run_temporal_fall_benchmark(
        fall_dir=PROJECT_ROOT / "data" / "fall_benchmark",
        config_path=PROJECT_ROOT / "configs" / "config.yaml",
        tracker_cfg=str(cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")),
        conf_thresh=float(cfg.get("detection_confidence", 0.08)),
        imgsz=int(cfg.get("inference_imgsz", 800)),
        enable_persistence=True,
        output_json=PROJECT_ROOT / "outputs" / "reports" / "phase28_fall_benchmark.json",
    )
    out["fall_benchmark"] = {
        "frame_level": fall_bench["frame_level_metrics"],
        "event_level": fall_bench["event_level_metrics"],
    }

    # --- False-alarm scan on non-fall demos (first 90 frames) ---
    fa_scans = []
    for rel in (
        "assets/evacuation_test_corridor.avi",
        "assets/running.mp4",
        "assets/intrusion.mp4",
        "assets/341-46_l.mov",
    ):
        print(f"\n[B] FA scan {rel}...", flush=True)
        fa_scans.append(count_confirmed_fall_events(rel, cfg, max_frames=90))
    out["non_fall_false_alarm_scan"] = fa_scans

    out_path = PROJECT_ROOT / "outputs" / "reports" / "phase28_reliability_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
