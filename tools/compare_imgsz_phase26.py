#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 26 — Reproducible imgsz & Tracker Gap Comparison
=============================================================
Compares YOLOv8n person detection at imgsz=640 vs imgsz=800 on identical
frames, and measures the detector-to-tracker gap under fall vs crowd
ByteTrack configs.

Integrity rules:
- Does NOT invent track IDs for untracked detections.
- Does NOT claim recall/precision without ground-truth files.
- Uses identical frame indices across all configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker


DEFAULT_VIDEOS = [
    "assets/evacuation_test_corridor.avi",
    "assets/running.mp4",
    "assets/intrusion.mp4",
    "assets/913-37_l.mov",
]


def load_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sample_frame_indices(total: int, max_frames: int) -> List[int]:
    """
    Consecutive 1-based frame indices from mid-clip.

    Sparse/even sampling across a long video breaks ByteTrack association
    (Kalman predictions jump), which under-reports tracked counts. Live
    console processes consecutive frames; this matcher mirrors that.
    """
    if total <= 0:
        return []
    n = min(max_frames, total)
    start = max(1, (total // 2) - (n // 2))
    end = min(total, start + n - 1)
    # If clipped at the end, shift window left
    start = max(1, end - n + 1)
    return list(range(start, end + 1))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2] - a[0]) * float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0]) * float(b[3] - b[1]))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def count_untracked(raw_xyxy: np.ndarray, tracked_xyxy: np.ndarray, iou_thresh: float = 0.5) -> int:
    """Raw detections with no tracked box at IoU >= thresh."""
    if len(raw_xyxy) == 0:
        return 0
    if len(tracked_xyxy) == 0:
        return len(raw_xyxy)
    untracked = 0
    for rb in raw_xyxy:
        matched = any(box_iou(rb, tb) >= iou_thresh for tb in tracked_xyxy)
        if not matched:
            untracked += 1
    return untracked


def evaluate_video(
    video_rel: str,
    model_path: str,
    conf: float,
    imgsz_list: List[int],
    tracker_configs: Dict[str, str],
    max_frames: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    video_path = PROJECT_ROOT / video_rel
    if not video_path.exists():
        return {
            "video": video_rel,
            "status": "MISSING",
            "error": f"File not found: {video_path}",
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": video_rel,
            "status": "UNSUPPORTED",
            "error": f"OpenCV cannot decode: {video_path.suffix}",
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idxs = sample_frame_indices(total, max_frames)

    # Preload identical frames once
    frames: Dict[int, np.ndarray] = {}
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi - 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames[fi] = frame
    cap.release()

    if not frames:
        return {
            "video": video_rel,
            "status": "EMPTY",
            "error": "No frames could be read",
        }

    used_idxs = sorted(frames.keys())
    result: Dict[str, Any] = {
        "video": video_rel,
        "status": "OK",
        "resolution": [width, height],
        "source_fps": round(src_fps, 2),
        "total_frames": total,
        "sampled_frames": used_idxs,
        "sampled_frame_count": len(used_idxs),
        "confidence": conf,
        "ground_truth": None,
        "note": "No ground-truth annotations for this clip; precision/recall not reported.",
        "configs": {},
    }

    for imgsz in imgsz_list:
        for tracker_name, tracker_rel in tracker_configs.items():
            tracker_path = resolve_tracker(str(PROJECT_ROOT / tracker_rel))
            # Separate models so predict state never contaminates ByteTrack persist state
            det_model = YOLO(model_path)
            det_model.to(device)
            track_model = YOLO(model_path)
            track_model.to(device)

            det_counts: List[int] = []
            track_counts: List[int] = []
            untracked_counts: List[int] = []
            unique_ids: set = set()
            id_appearances: Dict[int, List[int]] = {}
            infer_latencies_ms: List[float] = []
            track_latencies_ms: List[float] = []

            for fi in used_idxs:
                frame = frames[fi]

                t0 = time.perf_counter()
                pred = det_model.predict(
                    frame, conf=conf, imgsz=imgsz, classes=[0],
                    device=device, verbose=False
                )[0]
                infer_ms = (time.perf_counter() - t0) * 1000.0
                infer_latencies_ms.append(infer_ms)

                raw_xyxy = (
                    pred.boxes.xyxy.cpu().numpy()
                    if pred.boxes is not None and len(pred.boxes) > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
                det_counts.append(int(len(raw_xyxy)))

                t1 = time.perf_counter()
                tracked = track_model.track(
                    source=frame,
                    persist=True,
                    tracker=tracker_path,
                    conf=conf,
                    imgsz=imgsz,
                    classes=[0],
                    device=device,
                    verbose=False,
                )[0]
                track_ms = (time.perf_counter() - t1) * 1000.0
                track_latencies_ms.append(track_ms)

                if tracked.boxes is None or tracked.boxes.id is None:
                    tracked_xyxy = np.zeros((0, 4), dtype=np.float32)
                    tids: List[int] = []
                else:
                    tracked_xyxy = tracked.boxes.xyxy.cpu().numpy()
                    tids = extract_track_ids(tracked.boxes)

                n_tracked = len(tids)
                track_counts.append(n_tracked)
                # Approximate gap: raw predict boxes with no IoU>=0.5 match among tracked boxes
                untracked_counts.append(count_untracked(raw_xyxy, tracked_xyxy))

                cur_ids = set(int(t) for t in tids)
                unique_ids |= cur_ids
                for tid in cur_ids:
                    id_appearances.setdefault(tid, []).append(fi)

            # Fragmentation proxy: tracks that appear, disappear, then reappear
            fragmented = 0
            for tid, appearances in id_appearances.items():
                if len(appearances) < 2:
                    continue
                gaps = [appearances[i] - appearances[i - 1] for i in range(1, len(appearances))]
                # Gap > 1 sampled-frame step means missing from at least one sampled frame
                # Use actual index spacing in used_idxs
                idx_pos = {f: i for i, f in enumerate(used_idxs)}
                positions = [idx_pos[f] for f in appearances]
                for i in range(1, len(positions)):
                    if positions[i] - positions[i - 1] > 1:
                        fragmented += 1
                        break

            mean_det = float(np.mean(det_counts)) if det_counts else 0.0
            mean_trk = float(np.mean(track_counts)) if track_counts else 0.0
            mean_untrk = float(np.mean(untracked_counts)) if untracked_counts else 0.0
            mean_infer = float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0
            mean_track = float(np.mean(track_latencies_ms)) if track_latencies_ms else 0.0
            # Processing FPS approximates full predict+track cost on this machine
            mean_pipeline_ms = mean_infer + mean_track
            proc_fps = 1000.0 / mean_pipeline_ms if mean_pipeline_ms > 0 else 0.0

            key = f"imgsz_{imgsz}__{tracker_name}"
            result["configs"][key] = {
                "imgsz": imgsz,
                "tracker": tracker_name,
                "tracker_path": tracker_rel,
                "mean_raw_detections": round(mean_det, 2),
                "mean_tracked": round(mean_trk, 2),
                "mean_untracked_estimate": round(mean_untrk, 2),
                "sum_raw_detections": int(sum(det_counts)),
                "sum_tracked": int(sum(track_counts)),
                "unique_track_ids": len(unique_ids),
                "fragmented_track_ids_proxy": fragmented,
                "mean_predict_latency_ms": round(mean_infer, 1),
                "mean_track_latency_ms": round(mean_track, 1),
                "mean_pipeline_latency_ms": round(mean_pipeline_ms, 1),
                "approx_processing_fps": round(proc_fps, 2),
                "per_frame": [
                    {
                        "frame": fi,
                        "raw_detections": det_counts[i],
                        "tracked": track_counts[i],
                        "untracked_estimate": untracked_counts[i],
                        "predict_ms": round(infer_latencies_ms[i], 1),
                        "track_ms": round(track_latencies_ms[i], 1),
                    }
                    for i, fi in enumerate(used_idxs)
                ],
            }

            print(
                f"  [{video_rel}] imgsz={imgsz} tracker={tracker_name}: "
                f"det={mean_det:.1f} tracked={mean_trk:.1f} untracked~={mean_untrk:.1f} "
                f"pipeline={mean_pipeline_ms:.0f}ms (~{proc_fps:.1f} FPS)",
                flush=True,
            )

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 26 imgsz / tracker comparison")
    parser.add_argument("--max-frames", type=int, default=20, help="Evenly sampled frames per video")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 800])
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase26_imgsz_comparison.json"),
    )
    args = parser.parse_args()

    cfg = load_config()
    conf = float(args.conf if args.conf is not None else cfg.get("detection_confidence", 0.08))
    model_path = str(PROJECT_ROOT / cfg.get("yolo_model", "yolov8n.pt"))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    tracker_configs = {
        "bytetrack_fall": "configs/bytetrack_fall.yaml",
        "bytetrack_crowd": "configs/bytetrack_crowd.yaml",
    }

    print("=" * 72)
    print(" AEGIS Phase 26 — imgsz 640 vs 800 + Tracker Gap Comparison")
    print(f" conf={conf} device={device} max_frames/video={args.max_frames}")
    print("=" * 72)

    videos_out: List[Dict[str, Any]] = []
    for rel in DEFAULT_VIDEOS:
        print(f"\nEvaluating {rel} ...", flush=True)
        videos_out.append(
            evaluate_video(
                video_rel=rel,
                model_path=model_path,
                conf=conf,
                imgsz_list=list(args.imgsz),
                tracker_configs=tracker_configs,
                max_frames=args.max_frames,
                device=device,
            )
        )

    summary = {
        "suite": "AEGIS Phase 26 imgsz & tracker comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confidence": conf,
        "device": device,
        "model": cfg.get("yolo_model", "yolov8n.pt"),
        "max_frames_per_video": args.max_frames,
        "imgsz_list": list(args.imgsz),
        "trackers": tracker_configs,
        "precision_recall": "NOT_COMPUTED — no ground-truth for these clips",
        "videos": videos_out,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f" Report saved: {out_path}")
    print("=" * 72)

    # Compact console table
    print("\nCompact summary (mean per sampled frame):")
    print(f"{'video':<40} {'cfg':<28} {'det':>6} {'trk':>6} {'untrk~':>7} {'ms':>7} {'FPS':>6}")
    for v in videos_out:
        if v.get("status") != "OK":
            print(f"{v.get('video','?'):<40} {v.get('status'):<28} {'—':>6}")
            continue
        name = Path(v["video"]).name
        for key, cfg_res in v["configs"].items():
            label = f"isz{cfg_res['imgsz']}/{cfg_res['tracker'].replace('bytetrack_', '')}"
            print(
                f"{name:<40} {label:<28} "
                f"{cfg_res['mean_raw_detections']:6.1f} "
                f"{cfg_res['mean_tracked']:6.1f} "
                f"{cfg_res['mean_untracked_estimate']:7.1f} "
                f"{cfg_res['mean_pipeline_latency_ms']:7.0f} "
                f"{cfg_res['approx_processing_fps']:6.1f}"
            )


if __name__ == "__main__":
    main()
