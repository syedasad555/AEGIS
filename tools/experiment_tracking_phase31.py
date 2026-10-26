#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 31 — Isolated confidence-aware ByteTrack parameter sweep.

Does NOT modify configs/config.yaml or production bytetrack_crowd.yaml.
Candidate YAMLs live under outputs/experiments/phase31/trackers/.
Observed metrics only — no P/R/IDF1/HOTA/MOTA claims.
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

import importlib.util

_diag_path = PROJECT_ROOT / "tools" / "diagnose_tracking_phase30.py"
_spec = importlib.util.spec_from_file_location("diagnose_tracking_phase30", _diag_path)
_diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_diag)
diagnose_clip = _diag.diagnose_clip
load_cfg = _diag.load_cfg

CLIPS = [
    "UCF_CrowdsDataset/4917-5_70.mov",
    "UCF_CrowdsDataset/879-38_l.mov",
    "UCF_CrowdsDataset/341-46_l.mov",
    "UCF_CrowdsDataset/913-37_l.mov",
]

# Justified grid (high, low, new). Names document intent.
# Ultralytics 8.4: NEW tracks only from high-score pool (>= track_high), then new_track_thresh.
SWEEP: List[Dict[str, Any]] = [
    {
        "name": "baseline_crowd",
        "track_high_thresh": 0.35,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.30,
        "note": "Production Phase 29/30 crowd baseline",
    },
    {
        "name": "fall_like",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.15,
        "note": "Mirrors bytetrack_fall.yaml (not production)",
    },
    {
        "name": "new_only_lower",
        "track_high_thresh": 0.35,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.12,
        "note": "Phase 30 negative control: lower new_track alone cannot mint far IDs",
    },
    {
        "name": "moderate_high",
        "track_high_thresh": 0.20,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.20,
        "note": "Allow mid-conf into high-score init path; new == high",
    },
    {
        "name": "far_aggressive",
        "track_high_thresh": 0.10,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.10,
        "note": "Phase 30 far candidate; expect dense-scene ID churn risk",
    },
]


def mem_rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        try:
            import resource

            # Linux max RSS in KB; on Windows resource may be missing
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return None


def write_tracker_yaml(out_dir: Path, spec: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['name']}.yaml"
    body = {
        "tracker_type": "bytetrack",
        "track_high_thresh": float(spec["track_high_thresh"]),
        "track_low_thresh": float(spec["track_low_thresh"]),
        "new_track_thresh": float(spec["new_track_thresh"]),
        "track_buffer": 60,
        "match_thresh": 0.70,
        "fuse_score": True,
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def conf_histogram(confs: List[float], edges: List[float]) -> Dict[str, int]:
    if not confs:
        return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": 0 for i in range(len(edges) - 1)}
    arr = np.asarray(confs, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=np.asarray(edges, dtype=np.float64))
    return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(counts[i]) for i in range(len(edges) - 1)}


def extract_confs_from_result(res: Dict[str, Any]) -> List[float]:
    confs: List[float] = []
    for f in res.get("per_frame") or []:
        boxes = f.get("boxes") or []
        # boxes only dumped on early / zero-id frames; also use conf_mean * n as fallback weak
        for b in boxes:
            if b.get("conf") is not None:
                confs.append(float(b["conf"]))
        # Always pull from frame-level if we stored nothing — recompute from min/max weak
    return confs


def pool_split_counts(confs: List[float], high: float, low: float) -> Dict[str, int]:
    """How many dets would enter high vs low pools under Ultralytics split rules."""
    n_high = sum(1 for c in confs if c >= high)
    n_low = sum(1 for c in confs if (c > low and c < high))
    n_drop = sum(1 for c in confs if c <= low)
    return {"n_high_pool": n_high, "n_low_pool": n_low, "n_at_or_below_low": n_drop}


def score_candidate(clip_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic usefulness score — NOT accuracy.
    Prefer: far clip any stable tracks; dense clip longer life / fewer 1f tracks / less uniq inflation.
    Penalize: dense short-lived explosion; far zero tracks.
    """
    far = clip_summaries.get("4917-5_70.mov") or clip_summaries.get("UCF_CrowdsDataset/4917-5_70.mov")
    dense = clip_summaries.get("879-38_l.mov") or clip_summaries.get("UCF_CrowdsDataset/879-38_l.mov")
    # keys may be full relpaths
    if far is None:
        for k, v in clip_summaries.items():
            if "4917-5" in k:
                far = v
            if "879-38" in k:
                dense = v

    notes = []
    score = 0.0
    if far:
        if far["unique_track_ids"] == 0:
            notes.append("far: still zero IDs")
            score -= 5.0
        else:
            # reward modest tracks, not explosion
            uniq = far["unique_track_ids"]
            mean_trk = far["mean_tracked"]
            life = far["median_track_lifetime_frames"]
            if mean_trk >= 0.5 and life >= 5:
                score += 3.0
                notes.append("far: modest continuous tracks")
            elif mean_trk > 0:
                score += 1.0
                notes.append("far: sparse tracks only")
            if uniq > 20 and mean_trk < 3:
                score -= 2.0
                notes.append("far: high uniq vs low concurrent → unstable")

    if dense:
        uniq = dense["unique_track_ids"]
        mean_trk = max(dense["mean_tracked"], 1e-6)
        inflation = uniq / mean_trk
        life = dense["median_track_lifetime_frames"]
        short = dense["one_frame_tracks"] + dense["two_frame_tracks"]
        # Baseline-ish: uniq~85, mean~32, inflation~2.65, life~14, short~6
        if inflation <= 3.0 and life >= 12:
            score += 3.0
            notes.append("dense: continuity OK vs baseline band")
        elif inflation <= 4.0 and life >= 8:
            score += 1.0
            notes.append("dense: mild degradation")
        else:
            score -= 4.0
            notes.append(f"dense: churn risk inflation={inflation:.2f} life_med={life} short={short}")
        if short > 20:
            score -= 3.0
            notes.append("dense: many 1–2 frame tracks")

    return {"usefulness_score": round(score, 2), "notes": notes}


def save_review_overlays(
    video_rel: str,
    tracker_rel: str,
    out_dir: Path,
    tag: str,
    frame_indices: List[int],
    imgsz: int,
    conf: float,
    device: str,
) -> List[str]:
    """Write a few annotated frames for manual review (det boxes + track IDs)."""
    from ultralytics import YOLO
    from _common import extract_track_ids, resolve_tracker

    path = PROJECT_ROOT / video_rel
    if not path.exists():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.to(device)
    tracker = resolve_tracker(str(PROJECT_ROOT / tracker_rel))
    want = set(frame_indices)
    saved: List[str] = []
    fi = 0
    max_f = max(frame_indices) if frame_indices else 0
    while fi < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        results = model.track(
            frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
            classes=[0], device=device, verbose=False,
        )
        if fi not in want:
            continue
        vis = frame.copy()
        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            tids = extract_track_ids(boxes)
            id_arr = None
            if boxes.id is not None:
                id_arr = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else np.asarray(boxes.id)
            for i, bb in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, bb)
                c = float(confs[i]) if i < len(confs) else 0.0
                tid = int(id_arr[i]) if id_arr is not None and i < len(id_arr) else None
                color = (0, 200, 0) if tid is not None else (0, 0, 220)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                label = f"id={tid} {c:.2f}" if tid is not None else f"NO_ID {c:.2f}"
                cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(
            vis, f"{Path(video_rel).name} f={fi} {tag}", (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        outp = out_dir / f"{Path(video_rel).stem}_{tag}_f{fi:03d}.jpg"
        cv2.imwrite(str(outp), vis)
        saved.append(str(outp.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    cap.release()
    return saved


def summarize_for_report(res: Dict[str, Any]) -> Dict[str, Any]:
    s = res["summary"]
    pf = res.get("per_frame") or []
    ids_new = sum(len(f.get("ids_new") or []) for f in pf)
    ids_lost = sum(len(f.get("ids_lost") or []) for f in pf)
    short = s["one_frame_tracks"] + s["two_frame_tracks"]
    uniq = max(s["unique_track_ids"], 1)
    return {
        "video": res["video"],
        "status": res["status"],
        "frames_processed": res.get("frames_processed"),
        "mean_raw_detections": s["mean_raw_detections"],
        "mean_tracked": s["mean_tracked"],
        "unique_track_ids": s["unique_track_ids"],
        "median_track_lifetime_frames": s["median_track_lifetime_frames"],
        "mean_track_lifetime_frames": s["mean_track_lifetime_frames"],
        "track_lifetime_hist": s["track_lifetime_hist"],
        "one_frame_tracks": s["one_frame_tracks"],
        "two_frame_tracks": s["two_frame_tracks"],
        "short_lived_track_proportion": round(short / uniq, 4) if s["unique_track_ids"] else None,
        "ids_new_events_sum": ids_new,
        "ids_lost_events_sum": ids_lost,
        "frames_with_dets_but_zero_ids": s["frames_with_dets_but_zero_ids"],
        "frac_boxes_below_new_track_thresh": s["frac_boxes_below_new_track_thresh"],
        "conf_all_mean": s["conf_all_mean"],
        "conf_all_min": s["conf_all_min"],
        "conf_all_max": s["conf_all_max"],
        "abrupt_jump_events": s["abrupt_jump_events"],
        "approx_processing_fps": s["approx_processing_fps"],
        "mean_latency_ms": s["mean_latency_ms"],
        "id_inflation_vs_mean_tracks": round(
            s["unique_track_ids"] / max(s["mean_tracked"], 1e-6), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 31 isolated ByteTrack sweep")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase31_tracking_experiment.json"),
    )
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg()
    imgsz = int(cfg.get("inference_imgsz", 800))
    conf = float(cfg.get("detection_confidence", 0.08))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    exp_root = PROJECT_ROOT / "outputs" / "experiments" / "phase31"
    tracker_dir = exp_root / "trackers"
    review_dir = PROJECT_ROOT / "outputs" / "reports" / "phase31_review"

    # Semantics note from installed ultralytics
    import ultralytics

    semantics = {
        "ultralytics_version": ultralytics.__version__,
        "byte_tracker_file": str(
            Path(ultralytics.__file__).resolve().parent / "trackers" / "byte_tracker.py"
        ),
        "detector_conf": conf,
        "imgsz": imgsz,
        "rules": [
            "YOLO conf filters detections before ByteTrack sees them.",
            "High pool: score >= track_high_thresh → first association + eligible for NEW track init.",
            "Low pool: track_low_thresh < score < track_high_thresh → second association to EXISTING tracks only.",
            "NEW tracks: unmatched high-pool dets with score >= new_track_thresh (low pool cannot create IDs).",
            "Changing new_track_thresh alone does not move dets into the high pool.",
            "track_low_thresh does not enable track creation for low-confidence dets.",
        ],
        "low_conf_track_creation_supported": False,
    }

    print("=" * 72)
    print(" AEGIS Phase 31 — Isolated ByteTrack Confidence Sweep")
    print(f" imgsz={imgsz} conf={conf} device={device} max_frames={args.max_frames}")
    print(" production configs/config.yaml and bytetrack_crowd.yaml NOT modified")
    print("=" * 72)

    results_by_cfg: List[Dict[str, Any]] = []
    mem_before = mem_rss_mb()

    for spec in SWEEP:
        yaml_path = write_tracker_yaml(tracker_dir, spec)
        # diagnose_clip expects project-relative tracker path
        tracker_rel = str(yaml_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        print(f"\n## {spec['name']}  high={spec['track_high_thresh']} "
              f"low={spec['track_low_thresh']} new={spec['new_track_thresh']}", flush=True)

        clip_rows = []
        clip_sum_for_score: Dict[str, Dict[str, Any]] = {}
        for rel in CLIPS:
            print(f"  >> {rel}", flush=True)
            t0 = time.time()
            mem0 = mem_rss_mb()
            res = diagnose_clip(
                relpath=rel,
                max_frames=args.max_frames,
                imgsz=imgsz,
                conf=conf,
                tracker_rel=tracker_rel,
                device=device,
                dump_every_box=True,  # need confs for histograms on all frames with dumps
            )
            mem1 = mem_rss_mb()
            wall = time.time() - t0
            if res.get("status") != "OK":
                clip_rows.append({"video": rel, "status": res.get("status"), "error": res.get("error")})
                print(f"     status={res.get('status')}")
                continue

            # Collect confs from per-frame box dumps (diagnose dumps all boxes when dets&no ids
            # or early frames — for tracked scenes dump_every_box forces all)
            confs: List[float] = []
            for f in res.get("per_frame") or []:
                for b in f.get("boxes") or []:
                    if b.get("conf") is not None:
                        confs.append(float(b["conf"]))

            thr = res["tracker_thresholds"]
            pool = pool_split_counts(confs, thr["track_high_thresh"], thr["track_low_thresh"])
            row = summarize_for_report(res)
            row["wall_time_sec"] = round(wall, 1)
            row["mem_rss_mb_before"] = mem0
            row["mem_rss_mb_after"] = mem1
            row["conf_histogram"] = conf_histogram(
                confs, [0.0, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 1.01]
            )
            row["pool_split_observed"] = pool
            row["n_conf_samples"] = len(confs)
            clip_rows.append(row)
            clip_sum_for_score[rel] = res["summary"]
            print(
                f"     det={row['mean_raw_detections']} trk={row['mean_tracked']} "
                f"uniq={row['unique_track_ids']} life_med={row['median_track_lifetime_frames']} "
                f"short%={row['short_lived_track_proportion']} "
                f"infl={row['id_inflation_vs_mean_tracks']} ~{row['approx_processing_fps']} FPS"
            )

        usefulness = score_candidate(clip_sum_for_score)
        results_by_cfg.append({
            "name": spec["name"],
            "note": spec["note"],
            "thresholds": {
                "track_high_thresh": spec["track_high_thresh"],
                "track_low_thresh": spec["track_low_thresh"],
                "new_track_thresh": spec["new_track_thresh"],
            },
            "tracker_yaml": tracker_rel,
            "clips": clip_rows,
            "usefulness": usefulness,
        })
        print(f"  usefulness={usefulness}")

    # Overlays: baseline + best non-baseline by usefulness (if any positive) + far_aggressive
    review_files: Dict[str, List[str]] = {}
    if not args.skip_overlays:
        overlay_cfgs = ["baseline_crowd", "moderate_high", "far_aggressive"]
        for name in overlay_cfgs:
            spec = next(s for s in SWEEP if s["name"] == name)
            ypath = tracker_dir / f"{name}.yaml"
            trel = str(ypath.relative_to(PROJECT_ROOT)).replace("\\", "/")
            review_files[name] = []
            for rel, frames in [
                ("UCF_CrowdsDataset/4917-5_70.mov", [1, 15, 30, 45]),
                ("UCF_CrowdsDataset/879-38_l.mov", [1, 20, 40, 55]),
            ]:
                saved = save_review_overlays(
                    rel, trel, review_dir, name, frames, imgsz, conf, device
                )
                review_files[name].extend(saved)

    # Recommendation: only if some candidate beats baseline on usefulness without dense collapse
    baseline_u = next(r["usefulness"]["usefulness_score"] for r in results_by_cfg if r["name"] == "baseline_crowd")
    ranked = sorted(results_by_cfg, key=lambda r: r["usefulness"]["usefulness_score"], reverse=True)
    best = ranked[0]
    recommend = None
    recommend_reason = ""
    if best["name"] != "baseline_crowd" and best["usefulness"]["usefulness_score"] > baseline_u + 1.0:
        # sanity: dense short-lived not exploded
        dense_row = next((c for c in best["clips"] if "879-38" in c.get("video", "")), None)
        if dense_row and (dense_row.get("short_lived_track_proportion") or 0) < 0.35:
            recommend = best["name"]
            recommend_reason = (
                f"Higher usefulness ({best['usefulness']['usefulness_score']} vs baseline {baseline_u}) "
                "without extreme short-lived proportion on dense clip."
            )
        else:
            recommend_reason = (
                f"Top score {best['name']} rejected: dense short-lived tracks or insufficient margin; "
                "keep production baseline."
            )
    else:
        recommend_reason = (
            "No candidate improves useful continuity enough vs baseline without dense-scene churn; "
            "keep production bytetrack_crowd.yaml. Low-conf dets cannot create tracks under Ultralytics 8.4 ByteTrack."
        )

    out = {
        "suite": "AEGIS Phase 31 confidence-aware tracking experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_config_unchanged": True,
        "baseline": {
            "model": "yolov8n",
            "imgsz": imgsz,
            "conf": conf,
            "production_tracker": "configs/bytetrack_crowd.yaml",
            "tiled": False,
            "device": device,
            "max_frames": args.max_frames,
        },
        "threshold_semantics": semantics,
        "mem_rss_mb_process_start": mem_before,
        "mem_rss_mb_process_end": mem_rss_mb(),
        "selection_policy": (
            "Do not pick by raw track count or unique ID count alone. "
            "Prefer far-scene real continuous tracks + dense-scene longer lifetimes / lower inflation / fewer 1–2f IDs."
        ),
        "sweep": results_by_cfg,
        "ranking": [{"name": r["name"], "score": r["usefulness"]["usefulness_score"]} for r in ranked],
        "recommendation": recommend,
        "recommendation_reason": recommend_reason,
        "architecture_next_steps_if_rejected": [
            "Higher imgsz or ROI crop for far outdoor (raise detector conf before tracker)",
            "Separate tracker profiles per scene type (crowd vs far) selected by camera metadata — not a silent global change",
            "Optional custom tracker that can initialize from low-score dets (not supported by stock Ultralytics ByteTrack)",
        ],
        "manual_review_overlays": review_files,
        "metric_claim": "Observed tracking telemetry only — no accuracy / IDF1 / MOTA claims.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Slim summary without huge nested noise
    slim = {k: out[k] for k in out if k != "sweep"}
    slim["sweep_summary"] = []
    for r in results_by_cfg:
        slim["sweep_summary"].append({
            "name": r["name"],
            "thresholds": r["thresholds"],
            "usefulness": r["usefulness"],
            "clips": [
                {k: c.get(k) for k in (
                    "video", "status", "mean_raw_detections", "mean_tracked",
                    "unique_track_ids", "median_track_lifetime_frames",
                    "short_lived_track_proportion", "id_inflation_vs_mean_tracks",
                    "approx_processing_fps", "frames_with_dets_but_zero_ids",
                    "conf_all_mean",
                )}
                for c in r["clips"]
            ],
        })
    slim_path = out_path.with_name("phase31_tracking_experiment_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(" RECOMMENDATION:", recommend or "KEEP BASELINE")
    print(" ", recommend_reason)
    print(f" Saved {out_path}")
    print(f" Saved {slim_path}")


if __name__ == "__main__":
    main()
