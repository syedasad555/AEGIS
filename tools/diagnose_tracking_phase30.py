#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 30 — Tracking reliability diagnostics (observe-only by default).

Baseline frozen: YOLOv8n, imgsz=800, bytetrack_crowd, tiled OFF, conf=0.08.
Does not invent GT metrics (no P/R/IDF1/HOTA/MOTA).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker
from ultralytics import YOLO

DEFAULT_CLIPS = [
    "UCF_CrowdsDataset/4917-5_70.mov",
    "UCF_CrowdsDataset/879-38_l.mov",
    "UCF_CrowdsDataset/341-46_l.mov",
    "UCF_CrowdsDataset/913-37_l.mov",
]


def load_cfg() -> Dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}


def load_tracker_thresholds(tracker_path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(tracker_path.read_text(encoding="utf-8")) or {}
    return {
        "tracker_type": raw.get("tracker_type"),
        "track_high_thresh": float(raw.get("track_high_thresh", 0.5)),
        "track_low_thresh": float(raw.get("track_low_thresh", 0.1)),
        "new_track_thresh": float(raw.get("new_track_thresh", 0.6)),
        "track_buffer": int(raw.get("track_buffer", 30)),
        "match_thresh": float(raw.get("match_thresh", 0.8)),
        "fuse_score": bool(raw.get("fuse_score", True)),
    }


def _to_numpy(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def box_center(xyxy: np.ndarray) -> Tuple[float, float]:
    return float((xyxy[0] + xyxy[2]) * 0.5), float((xyxy[1] + xyxy[3]) * 0.5)


def box_area(xyxy: np.ndarray) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 1e-9 else 0.0


def diagnose_clip(
    relpath: str,
    max_frames: int,
    imgsz: int,
    conf: float,
    tracker_rel: str,
    device: str,
    dump_every_box: bool,
) -> Dict[str, Any]:
    path = PROJECT_ROOT / relpath
    if not path.exists():
        return {"video": relpath, "status": "MISSING"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video": relpath, "status": "OPEN_FAILED"}

    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        return {"video": relpath, "status": "DECODE_FAILED", "error": "first frame read failed"}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or probe.shape[1])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or probe.shape[0])
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    tracker_path = PROJECT_ROOT / tracker_rel
    tracker = resolve_tracker(str(tracker_path))
    thr = load_tracker_thresholds(tracker_path if tracker_path.exists() else Path(tracker))

    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.to(device)

    # Fresh model instance → fresh ByteTrack state (one clip = one tracker lifetime)
    warn_log: List[str] = []
    exc_log: List[str] = []

    per_frame: List[Dict[str, Any]] = []
    id_first_frame: Dict[int, int] = {}
    id_last_frame: Dict[int, int] = {}
    id_lifetime_frames: Dict[int, int] = defaultdict(int)
    prev_centers: Dict[int, Tuple[float, float]] = {}
    prev_boxes: Dict[int, np.ndarray] = {}
    abrupt_jumps: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    unique_ids: Set[int] = set()
    frames_with_dets_no_ids = 0
    frames_with_dets = 0
    frames_with_tracks = 0
    conf_all: List[float] = []
    conf_with_id: List[float] = []
    conf_without_id: List[float] = []
    boxes_below_new_track = 0
    boxes_total = 0
    boxes_with_id_total = 0
    ids_none_tensor_frames = 0  # boxes.id is entirely None

    frame_idx = 0
    prev_active: Set[int] = set()

    while frame_idx < max_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
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
                for cw in caught:
                    warn_log.append(f"frame {frame_idx}: {cw.category.__name__}: {cw.message}")
        except Exception as e:
            exc_log.append(f"frame {frame_idx}: {type(e).__name__}: {e}")
            break

        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        n_det = 0
        n_trk = 0
        active_ids: Set[int] = set()
        box_rows: List[Dict[str, Any]] = []
        id_is_none = True
        confs_f: List[float] = []
        classes_f: List[int] = []

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            n_det = len(boxes)
            xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
            confs = _to_numpy(boxes.conf).reshape(-1)
            clss = _to_numpy(boxes.cls).reshape(-1).astype(int) if boxes.cls is not None else np.zeros(n_det, dtype=int)
            ids_obj = boxes.id
            id_is_none = ids_obj is None
            if id_is_none:
                ids_none_tensor_frames += 1
                tid_arr = [None] * n_det
            else:
                tid_raw = _to_numpy(ids_obj).reshape(-1)
                tid_arr = [int(x) if not (isinstance(x, float) and math.isnan(x)) else None for x in tid_raw]

            tids = extract_track_ids(boxes)
            n_trk = len(tids)
            active_ids = set(tids)

            for i in range(n_det):
                c = float(confs[i]) if i < len(confs) else float("nan")
                cls_i = int(clss[i]) if i < len(clss) else -1
                tid_i = tid_arr[i] if i < len(tid_arr) else None
                bb = xyxy[i]
                conf_all.append(c)
                confs_f.append(c)
                classes_f.append(cls_i)
                boxes_total += 1
                below_new = c < thr["new_track_thresh"]
                if below_new:
                    boxes_below_new_track += 1
                if tid_i is None:
                    conf_without_id.append(c)
                else:
                    conf_with_id.append(c)
                    boxes_with_id_total += 1
                    unique_ids.add(tid_i)
                    id_lifetime_frames[tid_i] += 1
                    if tid_i not in id_first_frame:
                        id_first_frame[tid_i] = frame_idx
                    id_last_frame[tid_i] = frame_idx

                    cx, cy = box_center(bb)
                    if tid_i in prev_centers:
                        px, py = prev_centers[tid_i]
                        jump = math.hypot(cx - px, cy - py)
                        diag = math.hypot(w, h)
                        # Abrupt jump: >15% of frame diagonal in one frame (heuristic, not GT switch)
                        if jump > 0.15 * diag:
                            abrupt_jumps.append({
                                "frame": frame_idx,
                                "track_id": tid_i,
                                "jump_px": round(jump, 1),
                                "prev_center": [round(px, 1), round(py, 1)],
                                "center": [round(cx, 1), round(cy, 1)],
                                "iou_with_prev": round(iou(prev_boxes[tid_i], bb), 3) if tid_i in prev_boxes else None,
                            })
                    prev_centers[tid_i] = (cx, cy)
                    prev_boxes[tid_i] = bb.copy()

                if dump_every_box or frame_idx <= 5 or (n_det > 0 and n_trk == 0):
                    box_rows.append({
                        "i": i,
                        "xyxy": [round(float(x), 1) for x in bb.tolist()],
                        "conf": round(c, 4),
                        "cls": cls_i,
                        "track_id": tid_i,
                        "area_px": round(box_area(bb), 1),
                        "below_new_track_thresh": below_new,
                        "below_high_thresh": c < thr["track_high_thresh"],
                        "below_low_thresh": c < thr["track_low_thresh"],
                    })

        if n_det > 0:
            frames_with_dets += 1
        if n_trk > 0:
            frames_with_tracks += 1
        if n_det > 0 and n_trk == 0:
            frames_with_dets_no_ids += 1

        lost = sorted(prev_active - active_ids)
        newborn = sorted(active_ids - prev_active)
        prev_active = set(active_ids)

        # Also run predict-only on first 3 frames for scale/conf comparison (no tracker)
        predict_compare = None
        if frame_idx <= 3:
            pred = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0], device=device, verbose=False)
            if pred and pred[0].boxes is not None and len(pred[0].boxes) > 0:
                pxy = _to_numpy(pred[0].boxes.xyxy).reshape(-1, 4)
                pc = _to_numpy(pred[0].boxes.conf).reshape(-1)
                predict_compare = {
                    "n": len(pxy),
                    "confs": [round(float(x), 4) for x in pc.tolist()],
                    "xyxy_head": [[round(float(v), 1) for v in row] for row in pxy[:5].tolist()],
                }

        per_frame.append({
            "frame": frame_idx,
            "n_det": n_det,
            "n_tracked": n_trk,
            "boxes_id_tensor_is_none": id_is_none,
            "active_ids": sorted(active_ids),
            "ids_lost": lost,
            "ids_new": newborn,
            "conf_min": round(min(confs_f), 4) if confs_f else None,
            "conf_max": round(max(confs_f), 4) if confs_f else None,
            "conf_mean": round(float(np.mean(confs_f)), 4) if confs_f else None,
            "n_conf_below_new_track_thresh": sum(1 for c in confs_f if c < thr["new_track_thresh"]),
            "n_conf_gte_new_track_thresh": sum(1 for c in confs_f if c >= thr["new_track_thresh"]),
            "classes": classes_f,
            "boxes": box_rows if box_rows else None,
            "predict_compare": predict_compare,
            "latency_ms": round(latencies_ms[-1], 1),
        })

    cap.release()

    lifetimes = list(id_lifetime_frames.values())
    one_frame = sum(1 for L in lifetimes if L == 1)
    two_frame = sum(1 for L in lifetimes if L == 2)
    lifetime_hist = {
        "1": one_frame,
        "2": two_frame,
        "3_5": sum(1 for L in lifetimes if 3 <= L <= 5),
        "6_15": sum(1 for L in lifetimes if 6 <= L <= 15),
        "16_30": sum(1 for L in lifetimes if 16 <= L <= 30),
        "31_plus": sum(1 for L in lifetimes if L >= 31),
    }

    mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return {
        "video": relpath,
        "status": "OK",
        "width": w,
        "height": h,
        "source_fps": round(src_fps, 2),
        "frames_processed": frame_idx,
        "baseline": {
            "model": "yolov8n",
            "imgsz": imgsz,
            "conf": conf,
            "tracker": tracker_rel,
            "tiled": False,
            "device": device,
        },
        "tracker_thresholds": thr,
        "summary": {
            "mean_raw_detections": round(float(np.mean([f["n_det"] for f in per_frame])), 3) if per_frame else 0.0,
            "mean_tracked": round(float(np.mean([f["n_tracked"] for f in per_frame])), 3) if per_frame else 0.0,
            "unique_track_ids": len(unique_ids),
            "frames_with_detections": frames_with_dets,
            "frames_with_tracks": frames_with_tracks,
            "frames_with_dets_but_zero_ids": frames_with_dets_no_ids,
            "frames_boxes_id_tensor_none": ids_none_tensor_frames,
            "boxes_total": boxes_total,
            "boxes_with_id_total": boxes_with_id_total,
            "boxes_below_new_track_thresh": boxes_below_new_track,
            "frac_boxes_below_new_track_thresh": round(boxes_below_new_track / max(1, boxes_total), 4),
            "conf_all_mean": round(float(np.mean(conf_all)), 4) if conf_all else None,
            "conf_all_min": round(float(np.min(conf_all)), 4) if conf_all else None,
            "conf_all_max": round(float(np.max(conf_all)), 4) if conf_all else None,
            "conf_with_id_mean": round(float(np.mean(conf_with_id)), 4) if conf_with_id else None,
            "conf_without_id_mean": round(float(np.mean(conf_without_id)), 4) if conf_without_id else None,
            "n_conf_without_id": len(conf_without_id),
            "track_lifetime_hist": lifetime_hist,
            "median_track_lifetime_frames": int(np.median(lifetimes)) if lifetimes else 0,
            "mean_track_lifetime_frames": round(float(np.mean(lifetimes)), 2) if lifetimes else 0.0,
            "one_frame_tracks": one_frame,
            "two_frame_tracks": two_frame,
            "abrupt_jump_events": len(abrupt_jumps),
            "mean_latency_ms": round(mean_lat, 1),
            "approx_processing_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        },
        "note_unique_ids": (
            "unique_track_ids counts ByteTrack ID integers observed — NOT a person count. "
            "Short lifetimes and ID churn inflate this number without MOT GT."
        ),
        "abrupt_jumps_head": abrupt_jumps[:40],
        "id_lifetimes": {str(k): v for k, v in sorted(id_lifetime_frames.items(), key=lambda kv: kv[0])},
        "per_frame": per_frame,
        "warnings": warn_log[:50],
        "exceptions": exc_log,
        "metric_claim": "Observed tracking telemetry only — no P/R/IDF1/HOTA/MOTA.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 30 tracking diagnosis")
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase30_tracking_diagnosis.json"),
    )
    parser.add_argument("--dump-every-box", action="store_true")
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Optional tracker YAML relative to project root (A/B only; default = config baseline)",
    )
    args = parser.parse_args()

    cfg = load_cfg()
    imgsz = int(cfg.get("inference_imgsz", 800))
    conf = float(cfg.get("detection_confidence", 0.08))
    tracker_rel = str(
        args.tracker or cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")
    )
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    clips = args.clips if args.clips else DEFAULT_CLIPS

    print("=" * 72)
    print(" AEGIS Phase 30 — Tracking Diagnosis")
    print(f" imgsz={imgsz} conf={conf} tracker={tracker_rel} tiled=False device={device}")
    print("=" * 72)

    results = []
    for rel in clips:
        print(f"\n>> {rel}", flush=True)
        res = diagnose_clip(
            rel, args.max_frames, imgsz, conf, tracker_rel, device, args.dump_every_box
        )
        results.append(res)
        if res.get("status") == "OK":
            s = res["summary"]
            print(
                f"   det={s['mean_raw_detections']} trk={s['mean_tracked']} "
                f"uniq={s['unique_track_ids']} dets_no_id_frames={s['frames_with_dets_but_zero_ids']} "
                f"frac_conf<{res['tracker_thresholds']['new_track_thresh']}="
                f"{s['frac_boxes_below_new_track_thresh']} "
                f"life_med={s['median_track_lifetime_frames']} "
                f"1f={s['one_frame_tracks']} 2f={s['two_frame_tracks']} "
                f"~{s['approx_processing_fps']:.1f} FPS"
            )
        else:
            print(f"   status={res.get('status')} {res.get('error', '')}")

    out = {
        "suite": "AEGIS Phase 30 tracking diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_frozen": True,
        "clips": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
