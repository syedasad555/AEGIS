#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 29 — Audit UCF_CrowdsDataset without modifying original files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "UCF_CrowdsDataset"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mpeg", ".mpg", ".mkv", ".webm"}
ANNOT_HINTS = {".txt", ".csv", ".json", ".xml", ".mat", ".vtt", ".srt", ".yaml", ".yml"}


def fourcc_to_str(v: int) -> str:
    if not v:
        return ""
    chars = [chr((int(v) >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in chars)


def main() -> Dict[str, Any]:
    if not DATASET_ROOT.exists():
        raise SystemExit(f"Dataset not found: {DATASET_ROOT}")

    videos: List[Path] = []
    other: List[Path] = []
    seq_dirs: List[Dict[str, Any]] = []

    for p in sorted(DATASET_ROOT.rglob("*")):
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append(p)
            else:
                other.append(p)
        elif p.is_dir():
            imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.bmp"))
            if len(imgs) >= 5:
                seq_dirs.append({"dir": str(p.relative_to(DATASET_ROOT)), "image_count": len(imgs)})

    rows = []
    playable = 0
    for p in videos:
        cap = cv2.VideoCapture(str(p))
        opened = bool(cap.isOpened())
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
        fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0)) if opened else ""
        ret = False
        shape = None
        err = None
        if opened:
            ret, fr = cap.read()
            if ret and fr is not None:
                shape = [int(x) for x in fr.shape]
                playable += 1
            else:
                err = "VideoCapture opened but first frame read failed"
        else:
            err = "cv2.VideoCapture failed to open"
        cap.release()
        dur = (n / fps) if fps > 1e-6 else 0.0
        row = {
            "file": p.name,
            "relpath": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "opened": opened,
            "frame_readable": bool(ret),
            "width": w,
            "height": h,
            "fps": round(fps, 3),
            "frame_count": n,
            "duration_sec": round(dur, 2),
            "fourcc": fourcc or None,
            "frame_shape": shape,
            "error": err,
        }
        rows.append(row)
        print(
            f"{p.name}: open={opened} read={ret} {w}x{h} fps={fps:.2f} "
            f"n={n} dur={dur:.1f}s size={row['size_mb']}MB fourcc={fourcc!r}"
        )

    annot_like = [str(x.relative_to(DATASET_ROOT)) for x in other if x.suffix.lower() in ANNOT_HINTS]
    meta = {
        "dataset_root": str(DATASET_ROOT.resolve()),
        "layout": "flat directory of QuickTime .mov clips (no nested class folders found)",
        "video_file_count": len(videos),
        "playable_first_frame_count": playable,
        "non_video_file_count": len(other),
        "non_video_files": [str(x.relative_to(DATASET_ROOT)) for x in other],
        "annotation_like_files": annot_like,
        "image_sequence_dirs": seq_dirs,
        "ground_truth_present": False,
        "ground_truth_notes": (
            "No annotation/CSV/JSON/XML/MAT ground-truth files found beside the videos. "
            "UCF Crowds / similar crowd datasets typically provide density maps or count labels "
            "elsewhere; this folder contains only .mov media. Do not treat presence of 'anomaly' "
            "in a filename as pedestrian bounding-box GT."
        ),
        "videos": rows,
        "totals": {
            "total_size_mb": round(sum(r["size_mb"] for r in rows), 2),
            "total_duration_sec": round(sum(r["duration_sec"] for r in rows), 2),
            "unique_resolutions": sorted({f"{r['width']}x{r['height']}" for r in rows if r["frame_readable"]}),
            "fps_values": sorted({r["fps"] for r in rows if r["frame_readable"]}),
        },
    }

    out = PROJECT_ROOT / "outputs" / "reports" / "phase29_ucf_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nplayable={playable}/{len(videos)}")
    print(f"saved {out}")
    return meta


if __name__ == "__main__":
    main()
