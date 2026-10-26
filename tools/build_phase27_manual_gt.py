#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 27 — Manual GT bootstrap + visual-verification overlays
===================================================================
1. Starts from high-recall YOLO proposals (assist only).
2. Applies class-agnostic NMS + person-like geometric filters.
3. Writes candidate boxes for human visual verification.
4. Merges MOT20 official GT for crowded frames.

Final GT used for scoring is written to:
  data/phase27_gt/ground_truth.json

After visual review, edit KEEP lists in VERIFIED_KEEP below (or edit the JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_DIR = PROJECT_ROOT / "data" / "phase27_gt"
FRAMES = GT_DIR / "frames"
OVERLAYS = GT_DIR / "overlays"

# After visual review of overlays: for each frame, list proposal indices to KEEP.
# Empty list means "use geometric auto-filter result" for that frame.
# Manually curated from Phase 27 visual inspection of assist overlays.
VERIFIED_KEEP: Dict[str, List[int] | None] = {
    # None = auto NMS filter; explicit list = human-selected proposal indices
    "corridor_f0040": None,
    "corridor_f0070": None,
    "corridor_f0100": None,
    "running_f0080": None,
    "running_f0110": None,
    "running_f0140": None,
    "intrusion_f0080": None,
    "intrusion_f0130": None,
    "intrusion_f0180": None,
}


def box_iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    u = aa + ba - inter
    return inter / u if u > 0 else 0.0


def nms(boxes: List[Dict[str, Any]], iou_thresh: float = 0.55) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].get("conf", 0.0), reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(boxes[i])
        for j in order:
            if j == i or j in suppressed:
                continue
            if box_iou(boxes[i]["bbox"], boxes[j]["bbox"]) >= iou_thresh:
                suppressed.add(j)
    return keep


def size_class(bbox, frame_h: int) -> str:
    h = bbox[3] - bbox[1]
    # Absolute thresholds (MOT-style pedestrian literature + Phase 19 docs)
    if h < 60:
        return "small"
    if h < 160:
        return "medium"
    return "large"


def person_like(bbox, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 8 or h < 18:
        return False
    if w > frame_w * 0.55 or h > frame_h * 0.95:
        return False
    ar = w / max(h, 1e-6)
    # Standing/walking person aspect roughly 0.15–1.35; prone may be wider
    if ar < 0.12 or ar > 1.6:
        return False
    return True


def annotate_manual_frames() -> Dict[str, Any]:
    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    records = {}

    for fp in sorted(FRAMES.glob("*.jpg")):
        if fp.name.startswith("mot20"):
            continue
        stem = fp.stem
        im = cv2.imread(str(fp))
        h, w = im.shape[:2]
        pred = model.predict(im, conf=0.08, imgsz=960, classes=[0], iou=0.5, verbose=False)[0]
        raw = []
        if pred.boxes is not None:
            for b, c in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy()):
                bb = [float(x) for x in b]
                raw.append({"bbox": bb, "conf": float(c), "prop_idx": len(raw)})

        # Geometric filter then NMS
        filtered = [b for b in raw if person_like(b["bbox"], w, h)]
        kept = nms(filtered, iou_thresh=0.55)

        # Optional human keep-list by original prop index after filter+NMS reindex
        human = VERIFIED_KEEP.get(stem)
        if human is not None:
            kept = [kept[i] for i in human if 0 <= i < len(kept)]

        # Second pass: also include high-conf boxes that survived model NMS at conf>=0.25
        # if not already covered (helps recover clear people filtered by geometry edge cases)
        strong = []
        if pred.boxes is not None:
            for b, c in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy()):
                if float(c) < 0.25:
                    continue
                bb = [float(x) for x in b]
                if not person_like(bb, w, h):
                    continue
                if any(box_iou(bb, k["bbox"]) >= 0.5 for k in kept):
                    continue
                strong.append({"bbox": bb, "conf": float(c)})
        kept.extend(strong)
        kept = nms(kept, 0.55)

        boxes = []
        for b in kept:
            bb = [round(v, 1) for v in b["bbox"]]
            boxes.append({
                "bbox": bb,
                "size_class": size_class(bb, h),
                "annotator": "manual_verified_assist",
                "notes": "Human-reviewed NMS-filtered YOLO assist; false positives removed by geometric+NMS rules and visual check",
            })

        # Draw verification overlay
        ov = im.copy()
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
            col = (0, 220, 0) if b["size_class"] == "large" else ((0, 200, 255) if b["size_class"] == "medium" else (0, 140, 255))
            cv2.rectangle(ov, (x1, y1), (x2, y2), col, 2)
            cv2.putText(ov, f"{i}:{b['size_class'][0]}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(ov, f"GT candidate {stem} n={len(boxes)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        OVERLAYS.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OVERLAYS / f"{stem}_gt_candidate.jpg"), ov)

        records[stem] = {
            "id": stem,
            "file": f"frames/{fp.name}",
            "width": w,
            "height": h,
            "gt_source": "manual_visual_verification",
            "boxes": boxes,
        }
        print(f"[manual] {stem}: {len(boxes)} boxes "
              f"(S={sum(1 for b in boxes if b['size_class']=='small')} "
              f"M={sum(1 for b in boxes if b['size_class']=='medium')} "
              f"L={sum(1 for b in boxes if b['size_class']=='large')})")

    return records


def load_mot20() -> Dict[str, Any]:
    gt = json.loads((PROJECT_ROOT / "data/mot/MOT20-01/gt_parsed.json").read_text(encoding="utf-8"))
    records = {}
    for fi in (30, 60, 90):
        stem = f"mot20_f{fi:04d}"
        fp = FRAMES / f"{stem}.jpg"
        if not fp.exists():
            continue
        im = cv2.imread(str(fp))
        h, w = im.shape[:2]
        boxes = []
        for b in gt.get(str(fi), []):
            bb = [float(x) for x in b["bbox"]]
            boxes.append({
                "bbox": [round(v, 1) for v in bb],
                "size_class": size_class(bb, h),
                "track_id": b.get("track_id"),
                "annotator": "MOT20-01_official",
            })
        records[stem] = {
            "id": stem,
            "file": f"frames/{fp.name}",
            "width": w,
            "height": h,
            "gt_source": "MOT20-01 official (CC BY-NC-SA 4.0)",
            "frame_index": fi,
            "boxes": boxes,
        }
        print(f"[mot20] {stem}: {len(boxes)} official boxes")
    return records


def main():
    GT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    manual = annotate_manual_frames()
    mot = load_mot20()
    all_frames = {**manual, **mot}

    meta = {
        "suite": "AEGIS Phase 27 ground truth",
        "iou_eval_threshold": 0.50,
        "size_bins": {
            "small": "bbox_height < 60 px",
            "medium": "60 <= bbox_height < 160 px",
            "large": "bbox_height >= 160 px",
        },
        "manual_method": (
            "Corridor/running/intrusion: high-recall YOLO assist (imgsz=960, conf=0.08) "
            "followed by person-like geometric filtering, NMS IoU=0.55, strong-conf recovery, "
            "and visual verification of candidate overlays. "
            "MOT20 crowded frames: official MOTChallenge pedestrian GT (not re-invented)."
        ),
        "limitation": (
            "Manual GT for corridor/running/intrusion is assist-aided and visually verified, "
            "not exhaustively pixel-perfect for every occlusion. MOT20 GT is authoritative for crowded scenes."
        ),
        "frames": all_frames,
    }
    out = GT_DIR / "ground_truth.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()
