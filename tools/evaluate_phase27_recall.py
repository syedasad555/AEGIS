#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 27 — Fair Detection Recall / Tracking Evaluation
============================================================
Evaluates multiple configurations on the EXACT SAME annotated frames:
  - YOLOv8n imgsz 640 / 800 / 960
  - ByteTrack fall vs crowd
  - Optional selective tiled inference (crowded ROI only)

Integrity:
  - Identical frame set for every configuration
  - Detector counts and tracker counts reported separately
  - Never invents track IDs
  - Precision/Recall/F1 only against ground_truth.json (IoU>=0.50)
  - Size-stratified metrics: small / medium / large
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker
from scripts.tiled_inference import SelectiveTiledDetector, SimpleDetections


IOU_EVAL = 0.50


def box_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = aa + ba - inter
    return inter / u if u > 0 else 0.0


def greedy_match(preds: List[List[float]], gts: List[List[float]], iou_thresh: float = IOU_EVAL):
    """One-to-one greedy IoU matching. Returns tp, fp, fn, matched_gt_indices, mean_iou."""
    if len(gts) == 0:
        return 0, len(preds), 0, set(), []
    if len(preds) == 0:
        return 0, 0, len(gts), set(), []

    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = box_iou(p, g)
            if iou >= iou_thresh:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    ious = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        ious.append(iou)
    tp = len(used_p)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, used_g, ious


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_gt(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_detect(model: YOLO, frame, conf: float, imgsz: int, device: str) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    r = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0], device=device, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000.0
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), ms
    return r.boxes.xyxy.cpu().numpy(), ms


def run_track(
    model: YOLO,
    frame,
    conf: float,
    imgsz: int,
    tracker_yaml: str,
    device: str,
) -> Tuple[np.ndarray, List[int], float]:
    t0 = time.perf_counter()
    r = model.track(
        source=frame,
        persist=True,
        tracker=tracker_yaml,
        conf=conf,
        imgsz=imgsz,
        classes=[0],
        device=device,
        verbose=False,
    )[0]
    ms = (time.perf_counter() - t0) * 1000.0
    if r.boxes is None or r.boxes.id is None:
        return np.zeros((0, 4), dtype=np.float32), [], ms
    xyxy = r.boxes.xyxy.cpu().numpy()
    tids = extract_track_ids(r.boxes)
    # Only boxes with valid IDs count as tracked (never invent IDs)
    n = min(len(xyxy), len(tids))
    return xyxy[:n], [int(t) for t in tids[:n]], ms


def run_tiled_detect(det: SelectiveTiledDetector, frame, imgsz: int) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    boxes, scores, classes, meta = det.detect(frame, imgsz=imgsz)
    ms = (time.perf_counter() - t0) * 1000.0
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), ms
    return np.asarray(boxes, dtype=np.float32), ms


def evaluate_config(
    frames_meta: Dict[str, Any],
    conf: float,
    imgsz: int,
    tracker_name: str,
    tracker_rel: str,
    device: str,
    use_tiled: bool = False,
    gt_root: Path = PROJECT_ROOT / "data" / "phase27_gt",
) -> Dict[str, Any]:
    model_path = str(PROJECT_ROOT / "yolov8n.pt")
    det_model = YOLO(model_path)
    det_model.to(device)
    track_model = YOLO(model_path)
    track_model.to(device)
    tracker_yaml = resolve_tracker(str(PROJECT_ROOT / tracker_rel))

    tiled = None
    if use_tiled:
        tiled = SelectiveTiledDetector(
            model=det_model,
            tile_size=(640, 540),
            overlap=(0.20, 0.20),
            roi_rel=(0.0, 0.10, 1.0, 0.70),
            conf_thresh=conf,
            iou_nms_thresh=0.50,
            person_class_id=0,
            mem_limit_mb=900.0,
            enabled=True,
        )

    # Aggregate counters
    tot_tp = tot_fp = tot_fn = 0
    size_stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("small", "medium", "large")}
    det_counts = []
    trk_counts = []
    untracked_est = []
    det_ms = []
    trk_ms = []
    mean_ious = []
    per_frame = []

    # Sort frames for deterministic order; trackers run per-source consecutive groups
    # For fair track metrics on sparse annotated frames: reset tracker per frame group by source
    # and only accumulate track counts (association quality measured within MOT20 consecutive subset separately)

    frame_ids = sorted(frames_meta.keys())

    # Group by video family for consecutive tracking where possible
    groups: Dict[str, List[str]] = defaultdict(list)
    for fid in frame_ids:
        prefix = fid.rsplit("_f", 1)[0]
        groups[prefix].append(fid)

    for prefix, fids in groups.items():
        # Fresh tracker state per video family
        track_model = YOLO(model_path)
        track_model.to(device)

        # If frames are sparse (manual picks), run track independently per frame
        # (persist still used but large gaps hurt — report tracked-on-same-frame association)
        sparse = True
        if prefix == "mot20" and len(fids) >= 2:
            # MOT frames 30,60,90 are sparse — treat as independent for ID assignment count
            sparse = True

        for fid in sorted(fids):
            meta = frames_meta[fid]
            frame_path = gt_root / meta["file"]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            gt_boxes = [b["bbox"] for b in meta["boxes"]]
            gt_sizes = [b.get("size_class", "medium") for b in meta["boxes"]]

            if use_tiled and tiled is not None:
                pred_xyxy, d_ms = run_tiled_detect(tiled, frame, imgsz=imgsz)
            else:
                pred_xyxy, d_ms = run_detect(det_model, frame, conf, imgsz, device)

            # For tracking: always full-frame track path (tiled track is separate experiment)
            if sparse:
                # Reset tracker each annotated frame to measure single-frame ID assignment
                # without cross-gap Kalman corruption
                track_model = YOLO(model_path)
                track_model.to(device)

            trk_xyxy, tids, t_ms = run_track(track_model, frame, conf, imgsz, tracker_yaml, device)

            preds = [list(map(float, b)) for b in pred_xyxy]
            tp, fp, fn, matched_g, ious = greedy_match(preds, gt_boxes, IOU_EVAL)
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
            mean_ious.extend(ious)

            # Size-stratified: GT recall by size; FP attributed only to unmatched preds (unscoped)
            for gi, sz in enumerate(gt_sizes):
                if gi in matched_g:
                    size_stats[sz]["tp"] += 1
                else:
                    size_stats[sz]["fn"] += 1
            # Unmatched predictions counted as FP (not size-scoped)
            size_stats["medium"]["fp"] += fp  # bookkeeping only in overall; size FP reported as overall FP

            # Untracked estimate: pred boxes without IoU>=0.5 to any tracked box
            tracked = [list(map(float, b)) for b in trk_xyxy]
            untrk = 0
            for p in preds:
                if not any(box_iou(p, t) >= 0.5 for t in tracked):
                    untrk += 1

            det_counts.append(len(preds))
            trk_counts.append(len(tids))
            untracked_est.append(untrk)
            det_ms.append(d_ms)
            trk_ms.append(t_ms)

            per_frame.append({
                "frame_id": fid,
                "gt_count": len(gt_boxes),
                "det_count": len(preds),
                "tracked_count": len(tids),
                "untracked_estimate": untrk,
                "tp": tp, "fp": fp, "fn": fn,
                "detect_ms": round(d_ms, 1),
                "track_ms": round(t_ms, 1),
            })

    overall = prf(tot_tp, tot_fp, tot_fn)
    size_out = {}
    for sz, st in size_stats.items():
        # For size bins we only have TP/FN reliably; FP not size-labeled
        rec = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) else 0.0
        size_out[sz] = {
            "tp": st["tp"],
            "fn": st["fn"],
            "recall": round(rec, 4),
            "gt_count": st["tp"] + st["fn"],
        }

    mean_det_ms = float(np.mean(det_ms)) if det_ms else 0.0
    mean_trk_ms = float(np.mean(trk_ms)) if trk_ms else 0.0
    pipeline_ms = mean_det_ms + mean_trk_ms
    return {
        "imgsz": imgsz,
        "conf": conf,
        "tracker": tracker_name,
        "tiled": use_tiled,
        "iou_eval": IOU_EVAL,
        "frames_evaluated": len(per_frame),
        "detection": {
            **overall,
            "mean_iou_matched": round(float(np.mean(mean_ious)), 4) if mean_ious else 0.0,
            "mean_det_count": round(float(np.mean(det_counts)), 2) if det_counts else 0.0,
            "sum_det_count": int(sum(det_counts)),
            "sum_gt_count": int(sum(p["gt_count"] for p in per_frame)),
        },
        "tracking": {
            "mean_tracked_count": round(float(np.mean(trk_counts)), 2) if trk_counts else 0.0,
            "sum_tracked_count": int(sum(trk_counts)),
            "mean_untracked_estimate": round(float(np.mean(untracked_est)), 2) if untracked_est else 0.0,
            "note": "Tracked = boxes with valid ByteTrack IDs only; IDs never invented",
        },
        "size_recall": size_out,
        "latency": {
            "mean_detect_ms": round(mean_det_ms, 1),
            "mean_track_ms": round(mean_trk_ms, 1),
            "mean_pipeline_ms": round(pipeline_ms, 1),
            "approx_processing_fps": round(1000.0 / pipeline_ms, 2) if pipeline_ms > 0 else 0.0,
        },
        "per_frame": per_frame,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, default=str(PROJECT_ROOT / "data/phase27_gt/ground_truth.json"))
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "outputs/reports/phase27_recall_results.json"))
    parser.add_argument("--include-tiled", action="store_true")
    parser.add_argument("--include-960", action="store_true", default=True)
    args = parser.parse_args()

    gt_doc = load_gt(Path(args.gt))
    frames = gt_doc["frames"]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    conf = args.conf

    configs = []
    for imgsz in ([640, 800, 960] if args.include_960 else [640, 800]):
        for tname, trel in [
            ("bytetrack_fall", "configs/bytetrack_fall.yaml"),
            ("bytetrack_crowd", "configs/bytetrack_crowd.yaml"),
        ]:
            configs.append((imgsz, tname, trel, False))
    if args.include_tiled:
        configs.append((800, "bytetrack_crowd", "configs/bytetrack_crowd.yaml", True))
        configs.append((640, "bytetrack_crowd", "configs/bytetrack_crowd.yaml", True))

    print("=" * 72)
    print(" AEGIS Phase 27 — Detection Recall Evaluation")
    print(f" frames={len(frames)} conf={conf} device={device} IoU>={IOU_EVAL}")
    print("=" * 72)

    results = []
    for imgsz, tname, trel, tiled in configs:
        label = f"imgsz={imgsz} tracker={tname} tiled={tiled}"
        print(f"\n>> {label}", flush=True)
        res = evaluate_config(frames, conf, imgsz, tname, trel, device, use_tiled=tiled)
        results.append(res)
        d = res["detection"]
        t = res["tracking"]
        L = res["latency"]
        print(
            f"   P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
            f"det={d['mean_det_count']:.1f} trk={t['mean_tracked_count']:.1f} "
            f"untrk~={t['mean_untracked_estimate']:.1f} "
            f"{L['mean_pipeline_ms']:.0f}ms (~{L['approx_processing_fps']:.1f} FPS)"
        )
        sr = res["size_recall"]
        print(
            f"   size recall S/M/L: "
            f"{sr['small']['recall']:.3f}/{sr['medium']['recall']:.3f}/{sr['large']['recall']:.3f}"
        )

    out = {
        "suite": "AEGIS Phase 27 Detection Recall",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": device,
        "confidence": conf,
        "iou_eval_threshold": IOU_EVAL,
        "gt_file": str(args.gt),
        "gt_method": gt_doc.get("manual_method"),
        "gt_limitation": gt_doc.get("limitation"),
        "frame_count": len(frames),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
