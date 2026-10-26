#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 33 — Baseline (imgsz=800) vs optional far-camera profile (imgsz=960).

Uses production tracker (bytetrack_crowd). Does not enable far profile in config.yaml.
No P/R claims without GT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids, resolve_inference_imgsz, resolve_tracker
from ultralytics import YOLO

FAR_CLIPS = [
    "UCF_CrowdsDataset/4917-5_70.mov",
    "UCF_CrowdsDataset/4917-3_70.mov",
    "UCF_CrowdsDataset/RF1-12373_70.mov",
]
DENSE_CLIPS = [
    "UCF_CrowdsDataset/879-38_l.mov",
    "UCF_CrowdsDataset/913-37_l.mov",
    "UCF_CrowdsDataset/341-46_l.mov",
]
TRACK_HIGH = 0.35
REVIEW_FRAMES = [1, 15, 30, 45]


def mem_rss_mb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def load_cfg() -> Dict[str, Any]:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}


def load_frames(rel: str, max_frames: int):
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
        "frames_loaded": len(frames),
    }
    cap.release()
    return frames, meta


def run_profile(
    frames: List[np.ndarray],
    imgsz: int,
    conf: float,
    tracker_rel: str,
    device: str,
    profile_name: str,
    save_dir: Optional[Path],
    video_stem: str,
) -> Dict[str, Any]:
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.to(device)
    tracker = resolve_tracker(str(PROJECT_ROOT / tracker_rel))

    all_confs: List[float] = []
    n_det = n_trk = n_ge = 0
    unique = set()
    id_life: Dict[int, int] = {}
    lat: List[float] = []
    mem0 = mem_rss_mb()
    review_paths = []

    for fi, frame in enumerate(frames, start=1):
        t0 = time.perf_counter()
        res = model.track(
            frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
            classes=[0], device=device, verbose=False,
        )
        lat.append((time.perf_counter() - t0) * 1000.0)

        xyxy = np.zeros((0, 4))
        confs = np.zeros((0,))
        tids: List[int] = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy)
            confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.asarray(b.conf)
            tids = extract_track_ids(b)
            n_det += len(xyxy)
            n_ge += int(np.sum(confs >= TRACK_HIGH))
            n_trk += len(tids)
            for c in confs.tolist():
                all_confs.append(float(c))
            for tid in tids:
                unique.add(tid)
                id_life[tid] = id_life.get(tid, 0) + 1

        if save_dir is not None and fi in REVIEW_FRAMES:
            save_dir.mkdir(parents=True, exist_ok=True)
            vis = frame.copy()
            id_arr = None
            if res and res[0].boxes is not None and res[0].boxes.id is not None:
                id_arr = res[0].boxes.id.cpu().numpy()
            for i, bb in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, bb)
                c = float(confs[i]) if i < len(confs) else 0.0
                tid = int(id_arr[i]) if id_arr is not None and i < len(id_arr) else None
                color = (0, 200, 0) if tid is not None else ((0, 200, 0) if c >= TRACK_HIGH else (0, 0, 220))
                if tid is None:
                    color = (0, 0, 220) if c < TRACK_HIGH else (0, 165, 255)  # orange = high conf no id yet
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                label = f"id={tid} {c:.2f}" if tid is not None else f"NO_ID {c:.2f}"
                cv2.putText(vis, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(
                vis, f"{video_stem} f={fi} {profile_name} imgsz={imgsz}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            outp = save_dir / f"{video_stem}_{profile_name}_f{fi:03d}.jpg"
            cv2.imwrite(str(outp), vis)
            review_paths.append(str(outp.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    n_f = max(1, len(frames))
    lives = list(id_life.values())
    short = sum(1 for L in lives if L <= 2)
    mean_lat = float(np.mean(lat)) if lat else 0.0
    return {
        "profile": profile_name,
        "imgsz": imgsz,
        "frames": len(frames),
        "mean_detections": round(n_det / n_f, 3),
        "mean_tracked": round(n_trk / n_f, 3),
        "unique_track_ids": len(unique),
        "dets_ge_track_high": n_ge,
        "frac_dets_ge_track_high": round(n_ge / max(1, n_det), 4),
        "conf_mean": round(float(np.mean(all_confs)), 4) if all_confs else None,
        "conf_max": round(float(np.max(all_confs)), 4) if all_confs else None,
        "median_track_lifetime": int(np.median(lives)) if lives else 0,
        "short_lived_ids_1_2f": short,
        "short_lived_proportion": round(short / max(1, len(unique)), 4) if unique else None,
        "id_inflation": round(len(unique) / max(float(n_trk) / n_f, 1e-6), 3),
        "mean_latency_ms": round(mean_lat, 1),
        "approx_processing_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        "mem_rss_mb_before": mem0,
        "mem_rss_mb_after": mem_rss_mb(),
        "review_overlays": review_paths,
        "note": "unique_track_ids is not person count; no P/R claimed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "phase33_scene_resolution.json"),
    )
    args = parser.parse_args()

    cfg = load_cfg()
    conf = float(cfg.get("detection_confidence", 0.08))
    tracker = str(cfg.get("tracker_config", "configs/bytetrack_crowd.yaml"))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    baseline_isz = resolve_inference_imgsz(cfg)  # should be 800 with profile disabled
    far_cfg = dict(cfg)
    far_cfg["far_camera_profile"] = {
        "enabled": True,
        "inference_imgsz": int((cfg.get("far_camera_profile") or {}).get("inference_imgsz", 960)),
    }
    far_isz = resolve_inference_imgsz(far_cfg)

    review_dir = PROJECT_ROOT / "outputs" / "reports" / "phase33_review"

    print("=" * 72)
    print(" AEGIS Phase 33 — Baseline vs Far-Camera Profile")
    print(f" baseline imgsz={baseline_isz}  far_profile imgsz={far_isz}  "
          f"config far.enabled={bool((cfg.get('far_camera_profile') or {}).get('enabled'))}")
    print("=" * 72)

    assert baseline_isz == 800, "Production resolve must stay 800 when profile disabled"
    assert far_isz == 960, "Far profile must resolve to 960 when enabled in experiment cfg copy"

    out = {
        "suite": "AEGIS Phase 33 scene resolution",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "production_far_profile_enabled": bool((cfg.get("far_camera_profile") or {}).get("enabled", False)),
        "baseline_imgsz": baseline_isz,
        "far_profile_imgsz": far_isz,
        "tracker": tracker,
        "model": "yolov8n",
        "conf": conf,
        "device": device,
        "max_frames": args.max_frames,
        "clips": [],
        "metric_claim": "Observed telemetry only — no precision/recall without GT.",
    }

    for group, clips in (("far", FAR_CLIPS), ("dense", DENSE_CLIPS)):
        for rel in clips:
            print(f"\n>> [{group}] {rel}", flush=True)
            frames, meta = load_frames(rel, args.max_frames)
            if meta.get("status") != "OK":
                out["clips"].append({"group": group, "meta": meta})
                print(f"   status={meta.get('status')}")
                continue
            stem = Path(rel).stem
            base = run_profile(
                frames, baseline_isz, conf, tracker, device, "baseline800",
                review_dir, stem,
            )
            print(
                f"   baseline: det={base['mean_detections']} ge_high={base['dets_ge_track_high']} "
                f"trk={base['mean_tracked']} uniq={base['unique_track_ids']} "
                f"life_med={base['median_track_lifetime']} ~{base['approx_processing_fps']} FPS"
            )
            far = run_profile(
                frames, far_isz, conf, tracker, device, "far960",
                review_dir, stem,
            )
            print(
                f"   far960:   det={far['mean_detections']} ge_high={far['dets_ge_track_high']} "
                f"trk={far['mean_tracked']} uniq={far['unique_track_ids']} "
                f"life_med={far['median_track_lifetime']} ~{far['approx_processing_fps']} FPS"
            )
            out["clips"].append({
                "group": group,
                "meta": meta,
                "baseline": base,
                "far_profile": far,
                "deltas": {
                    "mean_detections": round(far["mean_detections"] - base["mean_detections"], 3),
                    "dets_ge_track_high": far["dets_ge_track_high"] - base["dets_ge_track_high"],
                    "mean_tracked": round(far["mean_tracked"] - base["mean_tracked"], 3),
                    "unique_track_ids": far["unique_track_ids"] - base["unique_track_ids"],
                    "approx_processing_fps": round(
                        far["approx_processing_fps"] - base["approx_processing_fps"], 2
                    ),
                },
            })

    # Recommendation heuristic (not accuracy)
    far_rows = [c for c in out["clips"] if c.get("group") == "far" and "baseline" in c]
    dense_rows = [c for c in out["clips"] if c.get("group") == "dense" and "baseline" in c]
    far_gain = any(
        (c["far_profile"]["dets_ge_track_high"] > c["baseline"]["dets_ge_track_high"]
         and c["far_profile"]["mean_tracked"] > c["baseline"]["mean_tracked"])
        for c in far_rows
    )
    dense_churn_ok = all(
        (c["far_profile"].get("short_lived_proportion") or 0) <= (
            (c["baseline"].get("short_lived_proportion") or 0) + 0.15
        )
        and c["far_profile"]["approx_processing_fps"] >= 2.0
        for c in dense_rows
    ) if dense_rows else False

    # Keep production disabled; allow enable-for-selected-cameras if far_gain and dense not wrecked
    if far_gain and dense_churn_ok:
        decision = "OPTIONAL_ENABLE_FOR_SELECTED_FAR_CAMERAS"
        reason = (
            "Far clips gain high-score dets and tracks at 960; dense short-lived proportion "
            "does not explode beyond +0.15 and FPS stays usable. Keep production default disabled; "
            "operators may enable far_camera_profile per known far outdoor session."
        )
    elif far_gain:
        decision = "KEEP_DISABLED_DENSE_COST"
        reason = (
            "Far clips improve, but dense-scene continuity/FPS trade-off is concerning; "
            "leave far_camera_profile.enabled=false globally."
        )
    else:
        decision = "KEEP_DISABLED"
        reason = "No clear practical far-scene benefit vs baseline on this sample set."

    out["decision"] = decision
    out["decision_reason"] = reason
    out["production_recommendation"] = {
        "leave_far_camera_profile_enabled": False,
        "allow_manual_enable_for_selected_far_cameras": decision == "OPTIONAL_ENABLE_FOR_SELECTED_FAR_CAMERAS",
        "default_inference_imgsz": 800,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    slim = {k: out[k] for k in out if k != "clips"}
    slim["clips"] = []
    for c in out["clips"]:
        if "baseline" not in c:
            slim["clips"].append(c)
            continue
        slim["clips"].append({
            "group": c["group"],
            "video": c["meta"]["video"],
            "status": c["meta"]["status"],
            "baseline": {k: c["baseline"][k] for k in c["baseline"] if k != "review_overlays"},
            "far_profile": {k: c["far_profile"][k] for k in c["far_profile"] if k != "review_overlays"},
            "deltas": c["deltas"],
            "review_overlays": {
                "baseline": c["baseline"].get("review_overlays"),
                "far": c["far_profile"].get("review_overlays"),
            },
        })
    slim_path = out_path.with_name("phase33_scene_resolution_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("\nDECISION:", decision)
    print(reason)
    print("Saved", out_path)
    print("Saved", slim_path)


if __name__ == "__main__":
    main()
