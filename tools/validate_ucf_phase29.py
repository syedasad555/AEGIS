#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 29 — Run AEGIS baseline on configurable UCF Crowds clips.
Does not invent GT metrics. Records observed detections/tracks/falls/FPS only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker
from scripts.anomaly_detection import EvacuationAnomalyEngine
from scripts.feature_extraction import TrajectoryFeatureExtractor

# Crowded / diverse representatives (paths relative to project root; configurable via CLI)
DEFAULT_CLIPS = [
    "UCF_CrowdsDataset/4917-3_70.mov",   # long dense outdoor crowd
    "UCF_CrowdsDataset/4917-5_70.mov",   # long dense outdoor crowd
    "UCF_CrowdsDataset/879-38_l.mov",    # long SVQ3 crowd
    "UCF_CrowdsDataset/821-10_l.mov",    # medium-long crowd
    "UCF_CrowdsDataset/341-46_l.mov",    # previously requested clip
    "UCF_CrowdsDataset/913-37_l.mov",    # previously missing from assets
    "UCF_CrowdsDataset/RF1-12373_70.mov",  # indoor-ish RF h264
]


def load_cfg() -> Dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}


def process_clip(
    relpath: str,
    cfg: Dict[str, Any],
    max_frames: int,
    device: str,
) -> Dict[str, Any]:
    path = PROJECT_ROOT / relpath
    if not path.exists():
        return {"video": relpath, "status": "MISSING"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video": relpath, "status": "OPEN_FAILED", "error": "cv2.VideoCapture failed to open"}

    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        return {
            "video": relpath,
            "status": "DECODE_FAILED",
            "error": "Opened but cannot read frames (codec/zlib unsupported)",
        }
    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or probe.shape[1])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or probe.shape[0])
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    imgsz = int(cfg.get("inference_imgsz", 800))
    conf = float(cfg.get("detection_confidence", 0.08))
    tracker = resolve_tracker(str(PROJECT_ROOT / cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")))
    model = YOLO(str(PROJECT_ROOT / cfg.get("yolo_model", "yolov8n.pt")))
    model.to(device)

    extractor = TrajectoryFeatureExtractor(cfg)
    anomaly = EvacuationAnomalyEngine(cfg)

    frame_idx = 0
    det_counts: List[int] = []
    track_counts: List[int] = []
    unique_ids = set()
    fall_confirmed_events = set()
    fall_suspected_frames = 0
    fall_confirmed_frames = 0
    other_anomaly_frames = 0
    crashes: List[str] = []
    latencies_ms: List[float] = []
    twin_samples: List[Dict[str, Any]] = []

    t_wall0 = time.time()
    while frame_idx < max_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        try:
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                conf=conf,
                imgsz=imgsz,
                classes=[0],
                device=device,
                verbose=False,
            )
        except Exception as e:
            crashes.append(f"frame {frame_idx}: {type(e).__name__}: {e}")
            break

        pedestrians = []
        n_det = 0
        n_trk = 0
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            # raw box count before ID filter
            n_det = len(boxes)
            tids = extract_track_ids(boxes)
            coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            if tids:
                n_trk = len(tids)
                for bbox, tid in zip(coords, tids):
                    tid = int(tid)
                    unique_ids.add(tid)
                    feat = extractor.update_pedestrian(tid, tuple(map(float, bbox)), frame_idx, src_fps)
                    feat = anomaly.evaluate_pedestrian(feat, frame_img=frame, frame_h=h, frame_w=w)
                    pedestrians.append(feat)
                    if feat.get("anomaly_type") == "FALL_DETECTED":
                        fall_confirmed_frames += 1
                        if feat.get("fall_event_id") is not None:
                            fall_confirmed_events.add(int(feat["fall_event_id"]))
                    elif feat.get("anomaly_type") == "FALL_SUSPECTED":
                        fall_suspected_frames += 1
                    elif feat.get("anomaly"):
                        other_anomaly_frames += 1

        # fall persistence coasting
        observed = [p["track_id"] for p in pedestrians]
        coast = anomaly.process_frame_persistence(
            frame_idx=frame_idx,
            observed_track_ids=observed,
            raw_candidate_boxes=[],
            frame_img=frame,
            frame_h=h,
            frame_w=w,
            fps=src_fps,
        )
        for c in coast:
            pedestrians.append(c)
            if c.get("anomaly_type") == "FALL_DETECTED" or c.get("status") == "FALL_CONFIRMED":
                fall_confirmed_frames += 1
                if c.get("fall_event_id") or c.get("event_id"):
                    fall_confirmed_events.add(int(c.get("fall_event_id") or c.get("event_id")))

        det_counts.append(n_det)
        track_counts.append(n_trk)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Digital Twin sync sample: foot coords normalized by this frame's W/H
        # Sample only first / mid / last processed frame (do not append every frame)
        sample_frames = {1, max(1, max_frames // 2), max_frames}
        if frame_idx in sample_frames and pedestrians:
            sample = []
            for p in pedestrians[:8]:
                foot = p.get("foot") or [0, 0]
                sample.append({
                    "track_id": p.get("track_id"),
                    "foot_px": [float(foot[0]), float(foot[1])] if len(foot) >= 2 else [0.0, 0.0],
                    "norm_xy": [round(float(foot[0]) / max(w, 1), 4), round(float(foot[1]) / max(h, 1), 4)]
                    if len(foot) >= 2 else [0.0, 0.0],
                    "anomaly_type": p.get("anomaly_type"),
                    "video": path.name,
                    "frame": frame_idx,
                    "video_wh": [w, h],
                })
            twin_samples.append({"frame": frame_idx, "pedestrians": sample})

    cap.release()
    wall = time.time() - t_wall0
    mean_lat = sum(latencies_ms) / max(1, len(latencies_ms))
    proc_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "video": relpath,
        "status": "OK" if frame_idx > 0 and not crashes else ("CRASH" if crashes else "EMPTY"),
        "width": w,
        "height": h,
        "source_fps": round(src_fps, 2),
        "reported_frame_count": total,
        "frames_processed": frame_idx,
        "imgsz": imgsz,
        "conf": conf,
        "tracker": cfg.get("tracker_config"),
        "tiled": bool(cfg.get("tiled_inference", {}).get("enabled", False)),
        "mean_raw_detections": round(sum(det_counts) / max(1, len(det_counts)), 2),
        "mean_tracked": round(sum(track_counts) / max(1, len(track_counts)), 2),
        "unique_track_ids": len(unique_ids),
        "det_track_gap_mean": round(
            (sum(det_counts) - sum(track_counts)) / max(1, len(det_counts)), 2
        ),
        "fall_confirmed_events": len(fall_confirmed_events),
        "fall_confirmed_frames": fall_confirmed_frames,
        "fall_suspected_frames": fall_suspected_frames,
        "other_anomaly_frames": other_anomaly_frames,
        "mean_latency_ms": round(mean_lat, 1),
        "approx_processing_fps": round(proc_fps, 2),
        "wall_time_sec": round(wall, 1),
        "crashes": crashes,
        "digital_twin_sync_samples": twin_samples,
        "ground_truth": None,
        "metric_claim": (
            "No pedestrian bounding-box GT in UCF_CrowdsDataset folder. "
            "Observed counts only — precision/recall NOT claimed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 29 UCF AEGIS validation")
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Project-relative video paths (default: curated crowded set)",
    )
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase29_ucf_validation.json"),
    )
    args = parser.parse_args()

    cfg = load_cfg()
    # Force Phase 29 baseline (do not enable tiling)
    cfg["inference_imgsz"] = int(cfg.get("inference_imgsz", 800))
    cfg["tracker_config"] = cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")
    if "tiled_inference" not in cfg:
        cfg["tiled_inference"] = {}
    cfg["tiled_inference"]["enabled"] = False

    clips = args.clips if args.clips else DEFAULT_CLIPS
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print("=" * 72)
    print(" AEGIS Phase 29 — UCF Real-Video Validation")
    print(f" imgsz={cfg['inference_imgsz']} conf={cfg.get('detection_confidence')} "
          f"tracker={cfg['tracker_config']} tiled=False device={device}")
    print("=" * 72)

    results = []
    for rel in clips:
        print(f"\n>> {rel}", flush=True)
        res = process_clip(rel, cfg, max_frames=args.max_frames, device=device)
        results.append(res)
        if res.get("status") == "OK":
            print(
                f"   frames={res['frames_processed']} det={res['mean_raw_detections']} "
                f"trk={res['mean_tracked']} ids={res['unique_track_ids']} "
                f"FALL_CONF_events={res['fall_confirmed_events']} "
                f"SUSP_frames={res['fall_suspected_frames']} "
                f"{res['mean_latency_ms']:.0f}ms (~{res['approx_processing_fps']:.1f} FPS)"
            )
        else:
            print(f"   status={res.get('status')} error={res.get('error')}")

    out = {
        "suite": "AEGIS Phase 29 UCF validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": {
            "model": "yolov8n",
            "imgsz": cfg["inference_imgsz"],
            "tracker": cfg["tracker_config"],
            "tiled": False,
            "conf": cfg.get("detection_confidence", 0.08),
        },
        "dataset_root": "UCF_CrowdsDataset",
        "max_frames_per_clip": args.max_frames,
        "clips": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
