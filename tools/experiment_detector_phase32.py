#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 32 — Isolated far-pedestrian detector experiments.

Production configs/UI/bytetrack_crowd.yaml are NOT modified.
Compares imgsz / ROI crop / optional YOLO11s on identical frames.
Separates raw detector metrics from tracker outcomes.
No P/R without GT.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_tracker
from ultralytics import YOLO

CLIPS = [
    "UCF_CrowdsDataset/4917-5_70.mov",
    "UCF_CrowdsDataset/879-38_l.mov",
]
IMGSZ_GRID = [800, 960, 1280]
TRACK_HIGH = 0.35  # production bytetrack_crowd
CONF = 0.08
MAX_FRAMES = 60
REVIEW_FRAMES = [1, 15, 30, 45]


def mem_rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def load_cfg() -> Dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}


def conf_hist(confs: List[float]) -> Dict[str, int]:
    edges = [0.0, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 1.01]
    if not confs:
        return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": 0 for i in range(len(edges) - 1)}
    counts, _ = np.histogram(np.asarray(confs), bins=np.asarray(edges))
    return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(counts[i]) for i in range(len(edges) - 1)}


def draw_raw_dets(frame: np.ndarray, xyxy: np.ndarray, confs: np.ndarray, title: str) -> np.ndarray:
    vis = frame.copy()
    for i, bb in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, bb)
        c = float(confs[i]) if i < len(confs) else 0.0
        # Color by whether above track_high
        color = (0, 200, 0) if c >= TRACK_HIGH else (0, 0, 220)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis, f"{c:.2f}", (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )
    cv2.putText(vis, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        vis, f"green conf>={TRACK_HIGH} (high-track)  red below",
        (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
    )
    return vis


def run_predict_window(
    model: YOLO,
    frames: List[np.ndarray],
    imgsz: int,
    conf: float,
    device: str,
) -> Dict[str, Any]:
    """Raw detector only — no tracker."""
    per_frame = []
    all_confs: List[float] = []
    lat_ms: List[float] = []
    n_ge_high = 0
    n_boxes = 0
    mem0 = mem_rss_mb()

    for fi, frame in enumerate(frames, start=1):
        t0 = time.perf_counter()
        res = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0], device=device, verbose=False)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        xyxy = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros((0,), dtype=np.float32)
        if res and res[0].boxes is not None and len(res[0].boxes):
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy)
            confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
        n = len(xyxy)
        n_boxes += n
        ge = int(np.sum(confs >= TRACK_HIGH)) if n else 0
        n_ge_high += ge
        for c in confs.tolist():
            all_confs.append(float(c))
        per_frame.append({
            "frame": fi,
            "n_det": n,
            "n_conf_ge_track_high": ge,
            "conf_min": round(float(confs.min()), 4) if n else None,
            "conf_max": round(float(confs.max()), 4) if n else None,
            "conf_mean": round(float(confs.mean()), 4) if n else None,
            "boxes": [
                {
                    "xyxy": [round(float(x), 1) for x in xyxy[i].tolist()],
                    "conf": round(float(confs[i]), 4),
                    "ge_track_high": bool(float(confs[i]) >= TRACK_HIGH),
                }
                for i in range(n)
            ],
        })

    mean_lat = float(np.mean(lat_ms)) if lat_ms else 0.0
    return {
        "mode": "predict_only",
        "frames": len(frames),
        "mean_detections": round(n_boxes / max(1, len(frames)), 3),
        "total_detections": n_boxes,
        "detections_ge_track_high_thresh": n_ge_high,
        "frac_detections_ge_track_high": round(n_ge_high / max(1, n_boxes), 4),
        "conf_mean": round(float(np.mean(all_confs)), 4) if all_confs else None,
        "conf_min": round(float(np.min(all_confs)), 4) if all_confs else None,
        "conf_max": round(float(np.max(all_confs)), 4) if all_confs else None,
        "conf_histogram": conf_hist(all_confs),
        "mean_latency_ms": round(mean_lat, 1),
        "approx_processing_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        "mem_rss_mb_before": mem0,
        "mem_rss_mb_after": mem_rss_mb(),
        "per_frame": per_frame,
    }


def run_track_window(
    model: YOLO,
    frames: List[np.ndarray],
    imgsz: int,
    conf: float,
    device: str,
    tracker_rel: str,
) -> Dict[str, Any]:
    """Detector+tracker on same frames (fresh model state assumed by caller)."""
    tracker = resolve_tracker(str(PROJECT_ROOT / tracker_rel))
    per_frame = []
    n_det_total = 0
    n_trk_total = 0
    n_ge_high = 0
    unique_ids = set()
    lat_ms: List[float] = []

    for fi, frame in enumerate(frames, start=1):
        t0 = time.perf_counter()
        res = model.track(
            frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
            classes=[0], device=device, verbose=False,
        )
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        n_det = 0
        n_trk = 0
        ge = 0
        if res and res[0].boxes is not None and len(res[0].boxes):
            b = res[0].boxes
            n_det = len(b)
            confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
            ge = int(np.sum(confs >= TRACK_HIGH))
            tids = extract_track_ids(b)
            n_trk = len(tids)
            unique_ids.update(tids)
        n_det_total += n_det
        n_trk_total += n_trk
        n_ge_high += ge
        per_frame.append({
            "frame": fi,
            "n_det": n_det,
            "n_tracked": n_trk,
            "n_conf_ge_track_high": ge,
        })

    mean_lat = float(np.mean(lat_ms)) if lat_ms else 0.0
    return {
        "mode": "track",
        "tracker": tracker_rel,
        "track_high_thresh_assumed": TRACK_HIGH,
        "frames": len(frames),
        "mean_detections": round(n_det_total / max(1, len(frames)), 3),
        "mean_tracked": round(n_trk_total / max(1, len(frames)), 3),
        "unique_track_ids": len(unique_ids),
        "detections_ge_track_high_thresh": n_ge_high,
        "frac_frames_with_tracks": round(
            sum(1 for f in per_frame if f["n_tracked"] > 0) / max(1, len(per_frame)), 4
        ),
        "mean_latency_ms": round(mean_lat, 1),
        "approx_processing_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        "per_frame": per_frame,
        "note": "unique_track_ids is not person count / accuracy",
    }


def load_frames(rel: str, max_frames: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    path = PROJECT_ROOT / rel
    meta = {"video": rel, "status": "OK"}
    if not path.exists():
        return [], {"video": rel, "status": "MISSING"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"video": rel, "status": "OPEN_FAILED"}
    ok, probe = cap.read()
    if not ok:
        cap.release()
        return [], {"video": rel, "status": "DECODE_FAILED"}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or probe.shape[1])
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or probe.shape[0])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while len(frames) < max_frames:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    meta.update({"width": w, "height": h, "source_fps": round(fps, 2), "frames_loaded": len(frames)})
    return frames, meta


def save_raw_overlays(
    frames: List[np.ndarray],
    predict_result: Dict[str, Any],
    out_dir: Path,
    tag: str,
    video_stem: str,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    by_f = {p["frame"]: p for p in predict_result["per_frame"]}
    for fi in REVIEW_FRAMES:
        if fi > len(frames) or fi not in by_f:
            continue
        row = by_f[fi]
        xyxy = np.array([b["xyxy"] for b in row["boxes"]], dtype=np.float32).reshape(-1, 4)
        confs = np.array([b["conf"] for b in row["boxes"]], dtype=np.float32)
        vis = draw_raw_dets(frames[fi - 1], xyxy, confs, f"{video_stem} f={fi} RAW {tag}")
        outp = out_dir / f"{video_stem}_{tag}_raw_f{fi:03d}.jpg"
        cv2.imwrite(str(outp), vis)
        saved.append(str(outp.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return saved


def roi_crop_experiment(
    model: YOLO,
    frames: List[np.ndarray],
    imgsz: int,
    conf: float,
    device: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Isolated ROI: upper half of 4917 far scene (pedestrians often distant near horizon).
    Maps crop boxes back to full-frame coordinates. Documents duplicates vs full-frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h0, w0 = frames[0].shape[:2]
    # Crop: top 55% height, full width — far sidewalks / crossings in this camera
    y0, y1 = 0, int(h0 * 0.55)
    x0, x1 = 0, w0
    roi = (x0, y0, x1, y1)

    full_confs: List[float] = []
    roi_confs: List[float] = []
    mapped_boxes_all: List[Dict[str, Any]] = []
    n_full = 0
    n_roi = 0
    n_roi_ge_high = 0
    n_full_ge_high = 0
    lat_full: List[float] = []
    lat_roi: List[float] = []
    duplicate_pairs = 0

    def iou(a, b) -> float:
        xx1 = max(a[0], b[0])
        yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2])
        yy2 = min(a[3], b[3])
        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
        area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / area if area > 1e-9 else 0.0

    review_saved = []
    for fi, frame in enumerate(frames, start=1):
        # Full frame
        t0 = time.perf_counter()
        r_full = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0], device=device, verbose=False)
        lat_full.append((time.perf_counter() - t0) * 1000.0)
        full_boxes = []
        if r_full and r_full[0].boxes is not None and len(r_full[0].boxes):
            b = r_full[0].boxes
            xy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for i in range(len(xy)):
                full_boxes.append((xy[i].tolist(), float(cf[i])))
                full_confs.append(float(cf[i]))
                n_full += 1
                if float(cf[i]) >= TRACK_HIGH:
                    n_full_ge_high += 1

        # ROI crop
        crop = frame[y0:y1, x0:x1]
        t1 = time.perf_counter()
        r_roi = model.predict(crop, conf=conf, imgsz=imgsz, classes=[0], device=device, verbose=False)
        lat_roi.append((time.perf_counter() - t1) * 1000.0)
        mapped = []
        if r_roi and r_roi[0].boxes is not None and len(r_roi[0].boxes):
            b = r_roi[0].boxes
            xy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for i in range(len(xy)):
                # Map crop-local → full-frame
                mx1 = float(xy[i][0]) + x0
                my1 = float(xy[i][1]) + y0
                mx2 = float(xy[i][2]) + x0
                my2 = float(xy[i][3]) + y0
                # Clamp to frame
                mx1 = max(0.0, min(float(w0 - 1), mx1))
                my1 = max(0.0, min(float(h0 - 1), my1))
                mx2 = max(0.0, min(float(w0), mx2))
                my2 = max(0.0, min(float(h0), my2))
                # Boundary: boxes cut by ROI bottom edge
                touches_bottom = float(xy[i][3]) >= (y1 - y0) - 2
                c = float(cf[i])
                mapped.append({
                    "xyxy_full": [round(mx1, 1), round(my1, 1), round(mx2, 1), round(my2, 1)],
                    "conf": round(c, 4),
                    "touches_roi_bottom_edge": touches_bottom,
                    "ge_track_high": c >= TRACK_HIGH,
                })
                roi_confs.append(c)
                n_roi += 1
                if c >= TRACK_HIGH:
                    n_roi_ge_high += 1
                # Duplicate vs full-frame (IoU>=0.5)
                for fb, _ in full_boxes:
                    if iou([mx1, my1, mx2, my2], fb) >= 0.5:
                        duplicate_pairs += 1
                        break

        mapped_boxes_all.append({"frame": fi, "mapped_roi_boxes": mapped, "n_full": len(full_boxes)})

        if fi in REVIEW_FRAMES:
            vis = frame.copy()
            # draw ROI rectangle
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 2)
            for m in mapped:
                x1_, y1_, x2_, y2_ = map(int, m["xyxy_full"])
                color = (0, 200, 0) if m["ge_track_high"] else (0, 0, 220)
                cv2.rectangle(vis, (x1_, y1_), (x2_, y2_), color, 1)
                cv2.putText(vis, f"{m['conf']:.2f}", (x1_, max(12, y1_ - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(vis, f"4917 ROI crop imgsz={imgsz} f={fi}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            outp = out_dir / f"4917-5_70_roi_imgsz{imgsz}_f{fi:03d}.jpg"
            cv2.imwrite(str(outp), vis)
            review_saved.append(str(outp.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    mean_full = float(np.mean(lat_full)) if lat_full else 0.0
    mean_roi = float(np.mean(lat_roi)) if lat_roi else 0.0
    return {
        "isolated": True,
        "not_integrated_to_production": True,
        "roi_xyxy_fullframe": [x0, y0, x1, y1],
        "roi_fraction_height": 0.55,
        "coordinate_mapping": "crop_xyxy + (x0,y0) offset; clamped to full frame bounds",
        "boundary_handling": "boxes touching ROI bottom edge flagged; no merge with full-frame in this experiment",
        "duplicate_box_risk": (
            "Running ROI + full-frame together would double-count overlapping people; "
            "count of ROI boxes with IoU>=0.5 vs a full-frame det in same frame accumulated as duplicate_pair_events"
        ),
        "duplicate_pair_events": duplicate_pairs,
        "imgsz": imgsz,
        "frames": len(frames),
        "fullframe_mean_detections": round(n_full / max(1, len(frames)), 3),
        "roi_mean_detections": round(n_roi / max(1, len(frames)), 3),
        "fullframe_dets_ge_track_high": n_full_ge_high,
        "roi_dets_ge_track_high": n_roi_ge_high,
        "fullframe_conf_mean": round(float(np.mean(full_confs)), 4) if full_confs else None,
        "roi_conf_mean": round(float(np.mean(roi_confs)), 4) if roi_confs else None,
        "roi_conf_histogram": conf_hist(roi_confs),
        "fullframe_conf_histogram": conf_hist(full_confs),
        "mean_latency_ms_fullframe": round(mean_full, 1),
        "mean_latency_ms_roi_only": round(mean_roi, 1),
        "approx_fps_fullframe": round(1000.0 / mean_full, 2) if mean_full else 0.0,
        "approx_fps_roi_only": round(1000.0 / mean_roi, 2) if mean_roi else 0.0,
        "overhead_note": (
            "If ROI is an *additional* pass on top of full-frame, wall cost ≈ full+roi; "
            "if ROI *replaces* full-frame for far cameras, cost ≈ roi only (misses lower frame)."
        ),
        "review_overlays": review_saved,
        "per_frame_head": mapped_boxes_all[:5],
    }


def resolve_model_weights(name: str) -> Tuple[Optional[Path], str]:
    """Return local path or ultralytics download name."""
    candidates = [
        PROJECT_ROOT / name,
        PROJECT_ROOT / "weights" / name,
    ]
    for p in candidates:
        if p.exists():
            return p, str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    # Allow ultralytics auto-download by bare name
    return None, name


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 32 isolated detector experiment")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--skip-yolo11", action="store_true")
    parser.add_argument("--skip-1280", action="store_true", help="Skip imgsz=1280 if too slow")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase32_detector_experiment.json"),
    )
    args = parser.parse_args()

    cfg = load_cfg()
    # Assert production baseline values for documentation (do not write config)
    prod_imgsz = int(cfg.get("inference_imgsz", 800))
    prod_conf = float(cfg.get("detection_confidence", 0.08))
    tracker_rel = str(cfg.get("tracker_config", "configs/bytetrack_crowd.yaml"))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    imgsz_list = [s for s in IMGSZ_GRID if not (args.skip_1280 and s == 1280)]
    review_dir = PROJECT_ROOT / "outputs" / "reports" / "phase32_review"
    roi_dir = review_dir / "roi"
    review_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" AEGIS Phase 32 — Isolated Far-Pedestrian Detector Experiment")
    print(f" production imgsz={prod_imgsz} conf={prod_conf} tracker={tracker_rel} (unchanged)")
    print(f" experiment imgsz grid={imgsz_list} device={device}")
    print("=" * 72)

    # Load identical frames once per clip
    clip_frames: Dict[str, List[np.ndarray]] = {}
    clip_meta: Dict[str, Dict[str, Any]] = {}
    for rel in CLIPS:
        fr, meta = load_frames(rel, args.max_frames)
        clip_frames[rel] = fr
        clip_meta[rel] = meta
        print(f" loaded {rel}: {meta}")

    results: Dict[str, Any] = {
        "suite": "AEGIS Phase 32 detector experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_config_unchanged": True,
        "production": {
            "yolo_model": cfg.get("yolo_model"),
            "inference_imgsz": prod_imgsz,
            "detection_confidence": prod_conf,
            "tracker_config": tracker_rel,
            "track_high_thresh_crowd": TRACK_HIGH,
        },
        "experiment": {
            "conf": CONF,
            "max_frames": args.max_frames,
            "imgsz_grid": imgsz_list,
            "device": device,
            "ultralytics": __import__("ultralytics").__version__,
        },
        "clips": {},
        "roi_experiment": None,
        "yolo11s_comparison": None,
        "manual_review_notes_placeholder": True,
        "metric_claim": "Observed detector/tracker telemetry only — no P/R without GT.",
    }

    # --- YOLOv8n imgsz sweep ---
    v8_path, v8_label = resolve_model_weights("yolov8n.pt")
    model_v8 = YOLO(str(v8_path) if v8_path else "yolov8n.pt")
    model_v8.to(device)

    for rel in CLIPS:
        frames = clip_frames[rel]
        if not frames:
            results["clips"][rel] = {"meta": clip_meta[rel], "runs": []}
            continue
        stem = Path(rel).stem
        runs = []
        for imgsz in imgsz_list:
            print(f"\n>> {rel}  yolov8n predict imgsz={imgsz}", flush=True)
            pred = run_predict_window(model_v8, frames, imgsz, CONF, device)
            overlays = save_raw_overlays(frames, pred, review_dir, f"v8n_isz{imgsz}", stem)
            # Fresh model for track state
            model_trk = YOLO(str(v8_path) if v8_path else "yolov8n.pt")
            model_trk.to(device)
            print(f"   track imgsz={imgsz}", flush=True)
            trk = run_track_window(model_trk, frames, imgsz, CONF, device, tracker_rel)
            # Slim per_frame in export (keep summary stats)
            pred_slim = {k: v for k, v in pred.items() if k != "per_frame"}
            pred_slim["per_frame_review"] = [p for p in pred["per_frame"] if p["frame"] in REVIEW_FRAMES]
            trk_slim = {k: v for k, v in trk.items() if k != "per_frame"}
            trk_slim["per_frame_sample"] = trk["per_frame"][::10]
            runs.append({
                "model": "yolov8n",
                "weights": v8_label,
                "imgsz": imgsz,
                "detector": pred_slim,
                "tracker": trk_slim,
                "raw_overlays": overlays,
                "dets_ge_high_vs_unique_ids": {
                    "detector_boxes_ge_track_high": pred["detections_ge_track_high_thresh"],
                    "unique_track_ids": trk["unique_track_ids"],
                    "mean_tracked": trk["mean_tracked"],
                },
            })
            print(
                f"   det_mean={pred['mean_detections']} conf_mean={pred['conf_mean']} "
                f"ge_high={pred['detections_ge_track_high_thresh']} "
                f"trk_mean={trk['mean_tracked']} uniq={trk['unique_track_ids']} "
                f"~{pred['approx_processing_fps']:.1f} predFPS / ~{trk['approx_processing_fps']:.1f} trkFPS"
            )
        results["clips"][rel] = {"meta": clip_meta[rel], "runs": runs}

    # --- ROI on 4917 only @ 800 and 960 ---
    far = "UCF_CrowdsDataset/4917-5_70.mov"
    if clip_frames.get(far):
        roi_runs = []
        for imgsz in [800, 960]:
            print(f"\n>> ROI crop experiment 4917 imgsz={imgsz}", flush=True)
            model_roi = YOLO(str(v8_path) if v8_path else "yolov8n.pt")
            model_roi.to(device)
            roi_runs.append(roi_crop_experiment(model_roi, clip_frames[far], imgsz, CONF, device, roi_dir))
        results["roi_experiment"] = {
            "video": far,
            "runs": roi_runs,
        }

    # --- YOLO11s optional ---
    if not args.skip_yolo11:
        print("\n>> Attempting YOLO11s comparison (may auto-download)", flush=True)
        y11_info: Dict[str, Any] = {"attempted": True, "weights": "yolo11s.pt"}
        try:
            model_11 = YOLO("yolo11s.pt")
            model_11.to(device)
            y11_info["loaded"] = True
            y11_info["weights_resolved"] = "yolo11s.pt (ultralytics cache/download)"
            y11_runs = []
            for rel in CLIPS:
                frames = clip_frames[rel]
                if not frames:
                    continue
                stem = Path(rel).stem
                # Compare at production imgsz 800 and 960 only for time
                for imgsz in [800, 960]:
                    print(f"   yolo11s {rel} imgsz={imgsz}", flush=True)
                    pred = run_predict_window(model_11, frames, imgsz, CONF, device)
                    overlays = save_raw_overlays(frames, pred, review_dir, f"y11s_isz{imgsz}", stem)
                    model_trk = YOLO("yolo11s.pt")
                    model_trk.to(device)
                    trk = run_track_window(model_trk, frames, imgsz, CONF, device, tracker_rel)
                    pred_slim = {k: v for k, v in pred.items() if k != "per_frame"}
                    pred_slim["per_frame_review"] = [p for p in pred["per_frame"] if p["frame"] in REVIEW_FRAMES]
                    trk_slim = {k: v for k, v in trk.items() if k != "per_frame"}
                    y11_runs.append({
                        "video": rel,
                        "model": "yolo11s",
                        "imgsz": imgsz,
                        "detector": pred_slim,
                        "tracker": trk_slim,
                        "raw_overlays": overlays,
                    })
                    print(
                        f"     det={pred['mean_detections']} ge_high={pred['detections_ge_track_high_thresh']} "
                        f"trk={trk['mean_tracked']} uniq={trk['unique_track_ids']} ~{pred['approx_processing_fps']} FPS"
                    )
            y11_info["runs"] = y11_runs
        except Exception as e:
            y11_info["loaded"] = False
            y11_info["error"] = f"{type(e).__name__}: {e}"
            print(f"   YOLO11s skipped: {y11_info['error']}")
        results["yolo11s_comparison"] = y11_info

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Summary without heavy fields
    summary = {
        "suite": results["suite"],
        "timestamp": results["timestamp"],
        "production_config_unchanged": True,
        "production": results["production"],
        "experiment": results["experiment"],
        "clip_summaries": {},
        "roi_summary": None,
        "yolo11s_summary": None,
    }
    for rel, block in results["clips"].items():
        summary["clip_summaries"][rel] = []
        for run in block.get("runs") or []:
            summary["clip_summaries"][rel].append({
                "model": run["model"],
                "imgsz": run["imgsz"],
                "mean_detections": run["detector"]["mean_detections"],
                "conf_mean": run["detector"]["conf_mean"],
                "conf_max": run["detector"]["conf_max"],
                "dets_ge_track_high": run["detector"]["detections_ge_track_high_thresh"],
                "frac_ge_track_high": run["detector"]["frac_detections_ge_track_high"],
                "conf_histogram": run["detector"]["conf_histogram"],
                "pred_fps": run["detector"]["approx_processing_fps"],
                "pred_mem_after_mb": run["detector"].get("mem_rss_mb_after"),
                "mean_tracked": run["tracker"]["mean_tracked"],
                "unique_track_ids": run["tracker"]["unique_track_ids"],
                "track_fps": run["tracker"]["approx_processing_fps"],
                "raw_overlays": run["raw_overlays"],
            })
    if results.get("roi_experiment"):
        summary["roi_summary"] = [
            {k: r[k] for k in r if k not in ("per_frame_head",)}
            for r in results["roi_experiment"]["runs"]
        ]
    if results.get("yolo11s_comparison"):
        y = results["yolo11s_comparison"]
        summary["yolo11s_summary"] = {
            "loaded": y.get("loaded"),
            "error": y.get("error"),
            "runs": [
                {
                    "video": r["video"],
                    "imgsz": r["imgsz"],
                    "mean_detections": r["detector"]["mean_detections"],
                    "conf_mean": r["detector"]["conf_mean"],
                    "dets_ge_track_high": r["detector"]["detections_ge_track_high_thresh"],
                    "mean_tracked": r["tracker"]["mean_tracked"],
                    "unique_track_ids": r["tracker"]["unique_track_ids"],
                    "pred_fps": r["detector"]["approx_processing_fps"],
                }
                for r in (y.get("runs") or [])
            ],
        }

    slim_path = out_path.with_name("phase32_detector_experiment_summary.json")
    slim_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Saved {slim_path}")


if __name__ == "__main__":
    main()
