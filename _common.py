#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
Internal shared utilities for detection scripts.

This module centralises common boilerplate used across `scripts/fall_detection.py`,
`scripts/running_detection.py` and `scripts/intrusion_detection.py`:

- PROJECT_ROOT detection
- Video source opening & validation
- VideoWriter creation
- Track ID extraction (torch / numpy compatibility)
- Keyboard display loop helper
- Keyframe extraction
- Tracker config fallback

⚠️  Internal module — not part of the public API. Subject to change without notice.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# ------------------------------------------------------------------
#  Project root
# ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ------------------------------------------------------------------
#  Video I/O helpers
# ------------------------------------------------------------------
def open_video(source: Union[str, Path]) -> cv2.VideoCapture:
    """Open a video file and return a VideoCapture handle.
    Exits with code 1 on failure."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[X] 无法打开视频: {source}")
        sys.exit(1)
    return cap


def get_video_props(cap: cv2.VideoCapture) -> dict:
    """Read FPS, width, height and total frame count from an opened capture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fps": fps, "width": w, "height": h, "total_frames": total}


def create_video_writer(output_path: Union[str, Path],
                        fps: float,
                        size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter. Exits with code 1 on failure."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, fps, size)
    if not writer.isOpened():
        print(f"[X] 无法创建输出视频: {output_path}")
        sys.exit(1)
    return writer


# ------------------------------------------------------------------
#  Detector weights resolution
# ------------------------------------------------------------------
DEFAULT_YOLO_MODEL = "weights/aegis_yolo11n_crowdhuman.pt"
ROLLBACK_YOLO_MODEL = "yolov8n.pt"


def resolve_yolo_model_path(
    config: Optional[Dict[str, Any]] = None,
    *,
    project_root: Optional[Path] = None,
    require_exists: bool = True,
) -> str:
    """Resolve detector weights path from config (absolute string).

    Default production model is the CrowdHuman fine-tune under ``weights/``.
    Raises ``FileNotFoundError`` with a clear message when weights are missing
    and ``require_exists`` is True.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    cfg = config or {}
    raw = str(cfg.get("yolo_model", DEFAULT_YOLO_MODEL) or DEFAULT_YOLO_MODEL).strip()
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)
        candidates.append(root / "weights" / p.name)
        candidates.append(root / p.name)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())

    if not require_exists:
        return str((root / raw).resolve()) if not p.is_absolute() else str(p)

    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "AEGIS detector weights not found.\n"
        f"  Config yolo_model: {raw}\n"
        f"  Looked in:\n  - {tried}\n"
        f"Place the fine-tuned weights at '{DEFAULT_YOLO_MODEL}' "
        f"or set yolo_model to an existing file (rollback: '{ROLLBACK_YOLO_MODEL}')."
    )


# ------------------------------------------------------------------
#  Session inference resolution (Phase 33)
# ------------------------------------------------------------------
def resolve_inference_imgsz(config: Optional[Dict[str, Any]] = None) -> int:
    """Return imgsz for the current camera/session.

    Production default remains ``inference_imgsz`` (800).
    Optional ``far_camera_profile.enabled: true`` overrides to the far
    profile imgsz (960) for the whole session — never per-frame, and never
    via automatic scene classification.
    """
    cfg = config or {}
    baseline = int(cfg.get("inference_imgsz", 800))
    far = cfg.get("far_camera_profile") or {}
    if not isinstance(far, dict):
        return baseline
    if bool(far.get("enabled", False)):
        return int(far.get("inference_imgsz", 960))
    return baseline


# ------------------------------------------------------------------
#  Track ID extraction (works with torch.Tensor or numpy.ndarray)
# ------------------------------------------------------------------
def extract_track_ids(boxes) -> List[int]:
    """Extract integer track IDs from ultralytics boxes. Never fabricate sequential IDs."""
    ids_obj = boxes.id
    if ids_obj is None:
        return []

    try:
        cpu_fn = getattr(ids_obj, "cpu", None)
        if cpu_fn is not None:
            arr = cpu_fn()
            numpy_fn = getattr(arr, "numpy", None)
            if numpy_fn is not None:
                arr = numpy_fn()
            return [int(x) for x in np.asarray(arr).flat]
        return [int(x) for x in np.asarray(ids_obj).flat]
    except Exception:
        return []


# ------------------------------------------------------------------
#  Tracker config fallback
# ------------------------------------------------------------------
def resolve_tracker(tracker_candidate: Union[str, Path],
                    default: str = "bytetrack.yaml") -> str:
    """Return the tracker YAML path if it exists, otherwise fall back to builtin."""
    p = Path(str(tracker_candidate))
    if p.exists():
        print(f"[i] 追踪器配置: {tracker_candidate} (自定义)")
        return str(tracker_candidate)
    print(f"[i] 追踪器配置 '{tracker_candidate}' 不存在，回退到内置 {default}")
    return default


# ------------------------------------------------------------------
#  Keyboard display loop helper
# ------------------------------------------------------------------
def show_frame(frame: np.ndarray,
               window_name: str = "Detection",
               quit_key: str = "q") -> bool:
    """Show a frame and check for quit key. Returns False when user wants to quit."""
    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(quit_key):
            return False
    except cv2.error:
        pass
    return True


# ------------------------------------------------------------------
#  Keyframe extraction (generic version)
# ------------------------------------------------------------------
def extract_event_keyframes(video_path: Union[str, Path],
                            event_frames: List[int],
                            output_dir: Union[str, Path],
                            label: str = "event",
                            range_frames: int = 10,
                            step: int = 2) -> List[Path]:
    """
    Extract keyframes around detected events from an already-written output video.

    Parameters
    ----------
    video_path : Path to the output video.
    event_frames : Sorted list of frame numbers where events occurred.
    output_dir : Directory to save PNG keyframes.
    label : Filename prefix (e.g. "fall" → fall_0001.png).
    range_frames : How many frames before/after the first event frame to include.
    step : Frame step when saving.

    Returns
    -------
    List of saved file paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[X] 关键帧抽取失败: 视频不存在 {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_event = min(event_frames)
    start = max(0, first_event - range_frames)
    end = min(total - 1, first_event + range_frames)

    print(f"\n[关键帧抽取] 首个告警帧 #{first_event}, 范围 [{start}, {end}], 步长 {step}")
    extracted = []
    for fn in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"  警告: 无法读取帧 #{fn}")
            continue
        out_path = output_dir / f"{label}_{fn:04d}.png"
        cv2.imwrite(str(out_path), frame)
        extracted.append(out_path)
        print(f"  已保存: {out_path.name}")

    cap.release()
    print(f"[关键帧抽取] 完成, 共 {len(extracted)} 张 → {output_dir}/")
    return extracted


# ------------------------------------------------------------------
#  Resource cleanup
# ------------------------------------------------------------------
def cleanup_resources(cap: Optional[cv2.VideoCapture] = None,
                      out: Optional[cv2.VideoWriter] = None) -> None:
    """Safely release capture / writer and destroy OpenCV windows."""
    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
