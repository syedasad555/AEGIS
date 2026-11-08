#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS — Reproducible person detection/tracking benchmark (inspect → baseline → candidates).

Uses production bytetrack_crowd. Does not modify configs/config.yaml.
No P/R/IDF1 without bbox/ID GT. No density/count models.
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
    ("far", "UCF_CrowdsDataset/4917-5_70.mov"),
    ("far", "UCF_CrowdsDataset/4917-3_70.mov"),
    ("far", "UCF_CrowdsDataset/RF1-12373_70.mov"),
    ("dense", "UCF_CrowdsDataset/879-38_l.mov"),
    ("dense", "UCF_CrowdsDataset/913-37_l.mov"),
    ("medium", "UCF_CrowdsDataset/341-46_l.mov"),
    ("medium", "UCF_CrowdsDataset/821-10_l.mov"),
]

# Isolated candidates — production remains yolov8n @ 800
CANDIDATES = [
    {"name": "prod_v8n_800", "model": "yolov8n.pt", "imgsz": 800, "role": "baseline"},
    {"name": "v8n_960", "model": "yolov8n.pt", "imgsz": 960, "role": "imgsz_candidate"},
    {"name": "y11s_800", "model": "yolo11s.pt", "imgsz": 800, "role": "model_candidate"},
]

TRACK_HIGH = 0.35
REVIEW_FRAMES = [1, 15, 30, 45]
CONF = 0.08
TRACKER = "configs/bytetrack_crowd.yaml"


def mem_rss_mb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def load_frames(rel: str, max_frames: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    path = PROJECT_ROOT / rel
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
    frames = []
    while len(frames) < max_frames:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    meta = {
        "video": rel,
        "status": "OK",
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or probe.shape[1]),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or probe.shape[0]),
        "source_fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 2),
        "frames_loaded": len(frames),
    }
    cap.release()
    return frames, meta


def resolve_weights(name: str) -> Optional[str]:
    for p in (PROJECT_ROOT / name, PROJECT_ROOT / "weights" / name):
        if p.exists():
            return str(p)
    # Allow ultralytics download by basename for isolated bench only
    return name


def run_one(
    frames: List[np.ndarray],
    model_path: str,
    imgsz: int,
    device: str,
    save_dir: Optional[Path],
    tag: str,
    video_stem: str,
) -> Dict[str, Any]:
    model = YOLO(model_path)
    model.to(device)
    tracker = resolve_tracker(str(PROJECT_ROOT / TRACKER))

    det_counts: List[int] = []
    trk_counts: List[int] = []
    unique: set = set()
    life: Dict[int, int] = {}
    confs_all: List[float] = []
    n_ge_high = 0
    lat: List[float] = []
    errors: List[str] = []
    overlays: List[str] = []
    mem0 = mem_rss_mb()

    for fi, frame in enumerate(frames, start=1):
        t0 = time.perf_counter()
        try:
            res = model.track(
                frame, persist=True, tracker=tracker, conf=CONF, imgsz=imgsz,
                classes=[0], device=device, verbose=False,
            )
        except Exception as e:
            errors.append(f"frame {fi}: {type(e).__name__}: {e}")
            break
        lat.append((time.perf_counter() - t0) * 1000.0)

        n_det = n_trk = 0
        xyxy = np.zeros((0, 4))
        confs = np.zeros((0,))
        id_arr = None
        if res and res[0].boxes is not None and len(res[0].boxes):
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy)
            confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
            n_det = len(xyxy)
            n_ge_high += int(np.sum(confs >= TRACK_HIGH))
            confs_all.extend(float(c) for c in confs.tolist())
            tids = extract_track_ids(b)
            n_trk = len(tids)
            for tid in tids:
                unique.add(int(tid))
                life[int(tid)] = life.get(int(tid), 0) + 1
            if b.id is not None:
                id_arr = b.id.cpu().numpy() if hasattr(b.id, "cpu") else np.asarray(b.id)

        det_counts.append(n_det)
        trk_counts.append(n_trk)

        if save_dir is not None and fi in REVIEW_FRAMES:
            save_dir.mkdir(parents=True, exist_ok=True)
            vis = frame.copy()
            for i, bb in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, bb)
                c = float(confs[i]) if i < len(confs) else 0.0
                tid = int(id_arr[i]) if id_arr is not None and i < len(id_arr) else None
                color = (0, 200, 0) if tid is not None else (0, 0, 220)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                lab = f"id={tid} {c:.2f}" if tid is not None else f"NO_ID {c:.2f}"
                cv2.putText(vis, lab, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(vis, f"{video_stem} {tag} f={fi}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            outp = save_dir / f"{video_stem}_{tag}_f{fi:03d}.jpg"
            cv2.imwrite(str(outp), vis)
            overlays.append(str(outp.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    n_f = max(1, len(det_counts))
    lives = list(life.values())
    short = sum(1 for L in lives if L <= 2)
    mean_lat = float(np.mean(lat)) if lat else 0.0
    return {
        "frames_processed": len(det_counts),
        "mean_detections": round(float(np.mean(det_counts)), 3) if det_counts else 0.0,
        "mean_tracked": round(float(np.mean(trk_counts)), 3) if trk_counts else 0.0,
        "max_concurrent_tracks": int(max(trk_counts)) if trk_counts else 0,
        "unique_track_ids": len(unique),
        "median_track_lifetime": int(np.median(lives)) if lives else 0,
        "mean_track_lifetime": round(float(np.mean(lives)), 2) if lives else 0.0,
        "short_lived_1_2f": short,
        "short_lived_proportion": round(short / max(1, len(unique)), 4) if unique else None,
        "id_inflation": round(len(unique) / max(float(np.mean(trk_counts) if trk_counts else 0), 1e-6), 3),
        "dets_ge_track_high": n_ge_high,
        "conf_mean": round(float(np.mean(confs_all)), 4) if confs_all else None,
        "conf_max": round(float(np.max(confs_all)), 4) if confs_all else None,
        "mean_latency_ms": round(mean_lat, 1),
        "approx_processing_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        "mem_rss_mb_before": mem0,
        "mem_rss_mb_after": mem_rss_mb(),
        "errors": errors,
        "overlays": overlays,
        "note": "Observed telemetry only — not precision/recall/IDF1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "detection_tracking_benchmark.json"),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    review = PROJECT_ROOT / "outputs" / "reports" / "detection_tracking_review"
    cands = [CANDIDATES[0]] if args.baseline_only else CANDIDATES

    print("=" * 72)
    print(" AEGIS Detection/Tracking Benchmark")
    print(f" production: model={cfg.get('yolo_model')} imgsz={cfg.get('inference_imgsz')} "
          f"tracker={cfg.get('tracker_config')} tiled={cfg.get('tiled_inference',{}).get('enabled')} "
          f"far_enabled={bool((cfg.get('far_camera_profile') or {}).get('enabled'))}")
    print(f" device={device} max_frames={args.max_frames} candidates={[c['name'] for c in cands]}")
    print("=" * 72)

    # Preload frames once per clip
    loaded: Dict[str, Tuple[List[np.ndarray], Dict[str, Any]]] = {}
    for _, rel in CLIPS:
        loaded[rel] = load_frames(rel, args.max_frames)

    results = {
        "suite": "AEGIS detection/tracking improvement benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_config": {
            "yolo_model": cfg.get("yolo_model"),
            "inference_imgsz": cfg.get("inference_imgsz"),
            "detection_confidence": cfg.get("detection_confidence"),
            "tracker_config": cfg.get("tracker_config"),
            "tiled_enabled": bool(cfg.get("tiled_inference", {}).get("enabled", False)),
            "far_camera_profile_enabled": bool((cfg.get("far_camera_profile") or {}).get("enabled", False)),
        },
        "benchmark_settings": {
            "conf": CONF,
            "tracker": TRACKER,
            "track_high_thresh_reference": TRACK_HIGH,
            "max_frames": args.max_frames,
            "device": device,
            "ultralytics": __import__("ultralytics").__version__,
        },
        "candidates": [],
        "metric_claim": "No P/R/F1/IDF1 — UCF folder has no pedestrian bbox/ID GT.",
    }

    for cand in cands:
        w = resolve_weights(cand["model"])
        print(f"\n## Candidate {cand['name']} model={w} imgsz={cand['imgsz']}", flush=True)
        # Skip missing optional weights without aborting baseline
        if cand["model"] != "yolov8n.pt":
            local = PROJECT_ROOT / cand["model"]
            local_w = PROJECT_ROOT / "weights" / cand["model"]
            if not local.exists() and not local_w.exists():
                # try load once; if fails, record skip
                try:
                    YOLO(cand["model"])
                except Exception as e:
                    results["candidates"].append({
                        "name": cand["name"], "role": cand["role"], "status": "SKIPPED",
                        "error": f"{type(e).__name__}: {e}",
                    })
                    print(f"   SKIPPED: {e}")
                    continue

        clip_rows = []
        for group, rel in CLIPS:
            frames, meta = loaded[rel]
            print(f"  >> [{group}] {rel} status={meta.get('status')}", flush=True)
            if meta.get("status") != "OK":
                clip_rows.append({"group": group, "meta": meta})
                continue
            row = run_one(
                frames, w, cand["imgsz"], device, review,
                cand["name"], Path(rel).stem,
            )
            print(
                f"     det={row['mean_detections']} trk={row['mean_tracked']} "
                f"max_trk={row['max_concurrent_tracks']} uniq={row['unique_track_ids']} "
                f"ge_high={row['dets_ge_track_high']} life_med={row['median_track_lifetime']} "
                f"short%={row['short_lived_proportion']} ~{row['approx_processing_fps']} FPS"
            )
            clip_rows.append({"group": group, "meta": meta, "metrics": row})

        results["candidates"].append({
            "name": cand["name"],
            "role": cand["role"],
            "model": cand["model"],
            "weights_resolved": w,
            "imgsz": cand["imgsz"],
            "status": "OK",
            "clips": clip_rows,
        })

    # Decision (not accuracy): prefer measurable far-scene high-score/track gain without dense short-lived explosion
    by_name = {c["name"]: c for c in results["candidates"] if c.get("status") == "OK"}
    baseline = by_name.get("prod_v8n_800")
    decision = {
        "integrate": False,
        "recommended_candidate": None,
        "reason": "Insufficient improvement or baseline only.",
        "production_default_unchanged": True,
        "optional_far_camera_profile": {
            "already_in_config": True,
            "keep_enabled_false": True,
            "note": "Operator may enable far_camera_profile for known far outdoor sessions if v8n_960 helps far clips.",
        },
    }
    if baseline and "v8n_960" in by_name:
        def far_metrics(cand):
            rows = [x for x in cand["clips"] if x.get("group") == "far" and "metrics" in x]
            return rows

        b_far = far_metrics(baseline)
        c_far = far_metrics(by_name["v8n_960"])
        b_dense = [x for x in baseline["clips"] if x.get("group") == "dense" and "metrics" in x]
        c_dense = [x for x in by_name["v8n_960"]["clips"] if x.get("group") == "dense" and "metrics" in x]

        far_better = False
        if b_far and c_far:
            # any far clip gains ge_high and mean_tracked
            for bf, cf in zip(b_far, c_far):
                bm, cm = bf["metrics"], cf["metrics"]
                if cm["dets_ge_track_high"] > bm["dets_ge_track_high"] and cm["mean_tracked"] > bm["mean_tracked"]:
                    far_better = True
                    break

        dense_ok = True
        for bd, cd in zip(b_dense, c_dense):
            bm, cm = bd["metrics"], cd["metrics"]
            sp_b = bm.get("short_lived_proportion") or 0
            sp_c = cm.get("short_lived_proportion") or 0
            if sp_c > sp_b + 0.20 or cm["approx_processing_fps"] < 2.0:
                dense_ok = False

        y11 = by_name.get("y11s_800")
        y11_better_far = False
        if y11:
            for bf, yf in zip(b_far, [x for x in y11["clips"] if x.get("group") == "far" and "metrics" in x]):
                if yf["metrics"]["mean_tracked"] > bf["metrics"]["mean_tracked"] + 0.5:
                    y11_better_far = True

        if far_better and dense_ok:
            decision.update({
                "integrate": False,  # do not change global default
                "recommended_candidate": "v8n_960_via_far_camera_profile",
                "reason": (
                    "YOLOv8n imgsz=960 improves far-scene high-score dets/tracks vs production 800 "
                    "without catastrophic dense short-lived growth. Keep production default at 800; "
                    "use existing far_camera_profile (disabled) for selected far cameras only."
                ),
                "optional_far_camera_profile": {
                    "already_in_config": True,
                    "keep_enabled_false": True,
                    "allow_manual_enable_for_far_cameras": True,
                },
            })
        elif y11_better_far:
            decision.update({
                "recommended_candidate": None,
                "reason": "YOLO11s showed some far track gain but prior Phase 32 found it weaker overall; do not swap production model without more evidence.",
            })
        else:
            decision["reason"] = "No candidate showed clear practical far-scene benefit with acceptable dense cost."

    results["decision"] = decision

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Slim summary
    slim = {
        "suite": results["suite"],
        "timestamp": results["timestamp"],
        "production_config": results["production_config"],
        "decision": decision,
        "candidates": [],
    }
    for c in results["candidates"]:
        if c.get("status") != "OK":
            slim["candidates"].append(c)
            continue
        slim["candidates"].append({
            "name": c["name"], "role": c["role"], "model": c["model"], "imgsz": c["imgsz"],
            "clips": [
                {
                    "group": r.get("group"),
                    "video": r.get("meta", {}).get("video"),
                    "status": r.get("meta", {}).get("status"),
                    **({k: r["metrics"][k] for k in (
                        "mean_detections", "mean_tracked", "max_concurrent_tracks",
                        "unique_track_ids", "median_track_lifetime", "short_lived_proportion",
                        "dets_ge_track_high", "conf_mean", "approx_processing_fps",
                        "mem_rss_mb_after", "errors",
                    )} if "metrics" in r else {}),
                }
                for r in c.get("clips") or []
            ],
        })
    slim_path = out.with_name("detection_tracking_benchmark_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("\nDECISION:", json.dumps(decision, indent=2))
    print("Saved", out)
    print("Saved", slim_path)


if __name__ == "__main__":
    main()
