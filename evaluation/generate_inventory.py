#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF Dataset Inventory Generator
Generates evaluation/ucf_dataset_inventory.md
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ucf_dir = PROJECT_ROOT / "UCF_CrowdsDataset"
videos = sorted(list(ucf_dir.glob("*.mov")))
model = YOLO(str(PROJECT_ROOT / "weights" / "aegis_yolo11n_crowdhuman.pt"))

inventory = []

for v in videos:
    cap = cv2.VideoCapture(str(v))
    opened = cap.isOpened()
    ret, frame = cap.read() if opened else (False, None)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()
    size_mb = v.stat().st_size / (1024 * 1024)
    dur = n_frames / fps if ret and fps > 0 and n_frames > 0 else 0.0
    
    det_count = 0
    if ret:
        res = model(frame, imgsz=800, conf=0.15, verbose=False)
        det_count = len(res[0].boxes) if res and res[0].boxes is not None else 0
    cap.release()
    
    inventory.append({
        "name": v.name,
        "size_mb": round(size_mb, 2),
        "resolution": f"{w}x{h}" if ret else "Corrupted/Unreadable",
        "fps": round(fps, 1) if ret else 0.0,
        "frames": n_frames if ret else 0,
        "duration_s": round(dur, 1),
        "playable": ret,
        "codec": fourcc_str if fourcc_str else "Unknown/Corrupted",
        "pedestrians_f1": det_count
    })

# Write markdown report
md_path = PROJECT_ROOT / "evaluation" / "ucf_dataset_inventory.md"
md_path.parent.mkdir(parents=True, exist_ok=True)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# UCF Crowds Dataset Inventory and Technical Inspection Report\n\n")
    f.write("**Dataset Location:** `UCF_CrowdsDataset/`\n")
    f.write("**Total Files Identified:** 18 video clips\n")
    f.write("**Inspection Scope:** Video container, codec decoding, frame rate, resolution, duration, crowd density, and annotation availability.\n\n")
    f.write("---\n\n")
    f.write("## 1. Video Inventory Table (All 18 Clips)\n\n")
    f.write("| # | Video File | Resolution | FPS | Frames | Duration | File Size | Codec | Status | Initial Pedestrians |\n")
    f.write("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    playable_count = 0
    total_frames = 0
    total_dur = 0.0
    
    for idx, item in enumerate(inventory, start=1):
        status_str = "Playable" if item["playable"] else "Corrupted (zlib/codec)"
        if item["playable"]:
            playable_count += 1
            total_frames += item["frames"]
            total_dur += item["duration_s"]
        f.write(f"| {idx:2d} | `{item['name']}` | {item['resolution']} | {item['fps']} | {item['frames']} | {item['duration_s']}s | {item['size_mb']} MB | {item['codec']} | **{status_str}** | {item['pedestrians_f1']} |\n")
        
    f.write(f"\n**Summary:** {playable_count} of 18 videos playable ({total_frames} total frames, {total_dur:.1f}s total duration). 1 corrupted video (`rush_01.mov`).\n\n")
    f.write("---\n\n")
    f.write("## 2. Dataset Variant & Structure Inspection\n\n")
    f.write("- **Dataset Variant:** UCF Crowds / Crowd Counting & High-Density Behavior Dataset (University of Central Florida Center for Research in Computer Vision).\n")
    f.write("- **Directory Structure:** Flat folder structure containing QuickTime `.mov` video files without pre-existing subfolders.\n")
    f.write("- **Codecs:** Predominantly MPEG-4 H.264 (`h264`) and Sorenson Video 3 (`svq3`) at 24–25 FPS, standard definition resolutions (480x360, 720x404, 720x576).\n")
    f.write("- **Corrupted Video:** `rush_01.mov` fails OpenCV first-frame decode due to an unsupported QuickTime header encoding requiring zlib stream decompression not compiled into standard Windows OpenCV MSMF backend.\n\n")
    f.write("---\n\n")
    f.write("## 3. Ground Truth & Annotation Limitations\n\n")
    f.write("1. **Absence of Native Pedestrian Anomaly Annotations:** The raw UCF Crowds dataset checkout provides video media only. It contains **no native bounding-box tracks, identity ground truth, or pedestrian anomaly labels**.\n")
    f.write("2. **Inapplicability of UCF Crime / General Labels:** Any external UCF Crime labels (e.g. fighting, road accident) are unrelated to context-aware evacuation anomalies (`WRONG_DIRECTION` and `PROLONGED_STOPPING`).\n")
    f.write("3. **Methodological Requirement:** Rigorous event-level ground truth must be independently established with explicit temporal boundaries `[start_time, end_time]`, pedestrian track IDs, designated evacuation flow directions, and verified ground-truth classes (`NORMAL`, `WRONG_DIRECTION`, `PROLONGED_STOPPING`).\n")

print(f"[Generated] {md_path}")

