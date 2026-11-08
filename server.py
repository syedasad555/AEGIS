#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Communication & Web Server for 2D Digital Twin
=====================================================
Features:
  - Multi-Video Input Support (MP4, AVI, MOV, MKV, etc.)
  - Real-time CCTV Frame & Digital Twin State Streaming over WebSocket
  - REST API for Dynamic Video Selection, Uploads, and Zone Configuration
  - Serves Web UI frontend at http://localhost:8080
"""

import asyncio
import base64
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
import websockets
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import (
    open_video,
    get_video_props,
    extract_track_ids,
    resolve_tracker,
    resolve_inference_imgsz,
    resolve_yolo_model_path,
)
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine
from scripts import direction_estimation as _direction_estimation_mod
from scripts.direction_estimation import (
    CLASSIFIER_VERSION,
    ADMIN_CARDINALS,
    allowed_directions,
    coerce_admin_direction,
)

HTTP_PORT = 8080
WS_PORT = 8765
RUNTIME_ID = f"AEGIS-{CLASSIFIER_VERSION}"

CONNECTED_CLIENTS: Set[Any] = set()

def load_yaml_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


GLOBAL_CONFIG = load_yaml_config()

# Initialize STREAM_STATE expected direction from config.yaml so it is consistent
# from the very first frame — avoids the 1-frame mismatch when server starts.
_cfg_expected_dir = GLOBAL_CONFIG.get("expected_evacuation_direction", "LEFT")
try:
    _cfg_expected_dir = coerce_admin_direction(_cfg_expected_dir)
except Exception:
    pass

# Global streaming state — expected_direction seeded from config.yaml
STREAM_STATE = {
    "is_streaming": False,
    "is_paused": False,
    "current_video": "assets/intrusion.mp4",
    "expected_direction": _cfg_expected_dir,
    "direction_diagnostic_mode": False,
    "playback_speed": 1.0,
    "detection_profile": "balanced",
    "custom_zones": None,
    "selected_track_id": None,
    "thread": None,
    "stop_event": threading.Event()
}

DETECTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "performance": {
        "name": "Performance",
        "description": "High framerate priority, low false positives (640px, conf=0.22, iou=0.50)",
        "conf": 0.22,
        "imgsz": 640,
        "iou": 0.50,
        "tracker": "configs/bytetrack_crowd.yaml",
    },
    "balanced": {
        "name": "Balanced",
        "description": "Optimized precision-recall balance (800px, conf=0.15, iou=0.55)",
        "conf": 0.15,
        "imgsz": 800,
        "iou": 0.55,
        "tracker": "configs/bytetrack_crowd.yaml",
    },
    "accuracy": {
        "name": "Accuracy",
        "description": "High recall for dense crowds, small & occluded persons (960px, conf=0.10, iou=0.55)",
        "conf": 0.10,
        "imgsz": 960,
        "iou": 0.55,
        "tracker": "configs/bytetrack_crowd.yaml",
    },
}

# Shared event loop for WebSocket broadcasting
ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None


def get_available_videos() -> List[Dict[str, str]]:
    """Scan assets/ and assets/uploads/ for video files supported by OpenCV."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []

    # Priority preset videos
    preset_files = [
        ("intrusion.mp4", "Intrusion Detection Scenario (intrusion.mp4)"),
        ("running.mp4", "Evacuation Movement Scenario (running.mp4)"),
        ("evacuation_test_corridor.avi", "Evacuation Corridor Test (evacuation_test_corridor.avi)"),
    ]

    found_paths = set()
    for filename, label in preset_files:
        p = PROJECT_ROOT / "assets" / filename
        if p.exists():
            videos.append({
                "name": label,
                "path": f"assets/{filename}",
                "format": p.suffix.upper().replace(".", "")
            })
            found_paths.add(str(p.resolve()))

    # Scan any other videos in assets and uploads
    scan_dirs = [PROJECT_ROOT / "assets", PROJECT_ROOT / "assets" / "uploads"]
    for sdir in scan_dirs:
        if sdir.exists():
            for f in sdir.glob("*.*"):
                if f.suffix.lower() in allowed_exts and str(f.resolve()) not in found_paths:
                    rel = f.relative_to(PROJECT_ROOT).as_posix()
                    videos.append({
                        "name": f"{f.stem} ({f.name})",
                        "path": rel,
                        "format": f.suffix.upper().replace(".", "")
                    })
                    found_paths.add(str(f.resolve()))

    return videos


def broadcast_payload(payload_dict: Dict[str, Any]):
    """Safely broadcasts JSON payload to all connected WebSocket clients."""
    if not CONNECTED_CLIENTS or ASYNC_LOOP is None:
        return
    msg = json.dumps(payload_dict)
    asyncio.run_coroutine_threadsafe(_send_to_clients(msg), ASYNC_LOOP)


async def _send_to_clients(message: str):
    if CONNECTED_CLIENTS:
        await asyncio.gather(*[client.send(message) for client in CONNECTED_CLIENTS], return_exceptions=True)


class VideoStreamWorker:
    """Processes video in a background thread and streams frames and digital twin states."""

    @staticmethod
    def run(video_rel_path: str, expected_dir: str = "RIGHT", custom_zones: Optional[Dict] = None, stop_event: Optional[threading.Event] = None):
        if stop_event is None:
            stop_event = STREAM_STATE["stop_event"]

        video_full_path = (PROJECT_ROOT / video_rel_path).resolve()
        if not video_full_path.exists():
            if (PROJECT_ROOT / "assets" / video_rel_path).resolve().exists():
                video_full_path = (PROJECT_ROOT / "assets" / video_rel_path).resolve()
                video_rel_path = f"assets/{video_rel_path}"
            elif (PROJECT_ROOT / "assets" / "uploads" / video_rel_path).resolve().exists():
                video_full_path = (PROJECT_ROOT / "assets" / "uploads" / video_rel_path).resolve()
                video_rel_path = f"assets/uploads/{video_rel_path}"
            else:
                broadcast_payload({
                    "type": "error",
                    "message": f"Video file not found: {video_rel_path}"
                })
                STREAM_STATE["is_streaming"] = False
                return

        # Validate video format with OpenCV
        cap = cv2.VideoCapture(str(video_full_path))
        if not cap.isOpened():
            broadcast_payload({
                "type": "error",
                "message": f"OpenCV cannot decode video format: {video_full_path.suffix}"
            })
            STREAM_STATE["is_streaming"] = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Streamer] Starting live stream: {video_full_path.name} ({width}x{height}, {total_frames} frames, {fps:.1f} FPS)")
        print(f"[Streamer] Evacuation direction: {expected_dir}")
        print(
            f"[Streamer] RUNTIME_ID={RUNTIME_ID} pid={os.getpid()} "
            f"server.py={Path(__file__).resolve()} "
            f"direction_module={Path(_direction_estimation_mod.__file__).resolve()} "
            f"classifier_version={CLASSIFIER_VERSION} "
            f"allowed_for_expected={list(allowed_directions(expected_dir))}",
            flush=True,
        )

        # Initialize detector & tracker (single pipeline — CrowdHuman YOLO11n by default)
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg_copy = dict(GLOBAL_CONFIG)
        cfg_copy["expected_evacuation_direction"] = expected_dir
        if custom_zones:
            if isinstance(custom_zones, dict) and custom_zones.get("evacuation_corridor"):
                cfg_copy["evacuation_corridor"] = custom_zones["evacuation_corridor"]
                cfg_copy["zones"] = {}
            elif isinstance(custom_zones, dict) and custom_zones.get("polygon"):
                cfg_copy["evacuation_corridor"] = custom_zones
                cfg_copy["zones"] = {}
            else:
                # Ignore obsolete multi-zone payloads
                cfg_copy["zones"] = {}

        model_path = resolve_yolo_model_path(cfg_copy, require_exists=True)
        model = YOLO(model_path)
        model.to(device)

        extractor = TrajectoryFeatureExtractor(cfg_copy)
        anomaly_engine = EvacuationAnomalyEngine(cfg_copy)
        anomaly_engine.set_expected_direction(expected_dir)

        # Active detection profile settings
        active_profile_key = STREAM_STATE.get("detection_profile", "balanced")
        if active_profile_key not in DETECTION_PROFILES:
            active_profile_key = "balanced"
        prof_info = DETECTION_PROFILES[active_profile_key]
        conf_thresh = float(prof_info["conf"])
        inference_imgsz = int(prof_info["imgsz"])
        nms_iou_thresh = float(prof_info.get("iou", 0.55))
        tracker_yaml = resolve_tracker(prof_info.get("tracker", "configs/bytetrack_crowd.yaml"))

        far_prof = cfg_copy.get("far_camera_profile") or {}
        person_class_id = int(cfg_copy.get("person_class_id", 0))
        # Single-class CrowdHuman fine-tune and COCO person both use class 0; skip filter for pose models
        person_classes = None if "pose" in str(cfg_copy.get("yolo_model", "")).lower() else [person_class_id]
        show_untracked = bool(cfg_copy.get("show_untracked_detections", False))
        overlay_cfg = cfg_copy.get("overlay") or {}

        print(
            f"[Streamer] Inference settings: model={model_path} profile={active_profile_key} "
            f"imgsz={inference_imgsz} conf={conf_thresh} iou={nms_iou_thresh} classes={person_classes} "
            f"tracker={tracker_yaml} device={device} "
            f"far_camera_profile.enabled={bool(far_prof.get('enabled', False))}",
            flush=True
        )

        base_frame_interval = 1.0 / max(fps, 10.0)
        frame_idx = 0
        consecutive_empty_frames = 0
        # Rolling processing throughput (excludes pacing sleep)
        proc_latency_ema_ms = None
        proc_fps_ema = None

        # Playback metrics and wall-clock synchronization
        wall_sync_start = time.time()
        video_sync_frame = 0
        current_requested_speed = float(STREAM_STATE.get("playback_speed", 1.0))
        last_synced_speed = current_requested_speed

        playback_window_start = time.time()
        window_start_frame_idx = 0
        frames_skipped = 0
        frames_processed = 0
        frames_skipped_this_step = 0
        achieved_playback_speed = current_requested_speed
        effective_playback_fps = fps * current_requested_speed
        monitoring_mode = "NORMAL"
        current_max_skip = 4

        while not stop_event.is_set():
            if STREAM_STATE["is_paused"]:
                time.sleep(0.05)
                # Keep wall-clock reference updated so pause duration does not cause a sudden forward jump upon resume
                wall_sync_start = time.time()
                video_sync_frame = frame_idx
                playback_window_start = time.time()
                window_start_frame_idx = frame_idx
                continue

            # Hot-reload playback speed if changed live by administrator
            req_speed = float(STREAM_STATE.get("playback_speed", 1.0))
            if abs(req_speed - last_synced_speed) > 1e-4:
                wall_sync_start = time.time()
                video_sync_frame = frame_idx
                last_synced_speed = req_speed
                playback_window_start = time.time()
                window_start_frame_idx = frame_idx
            current_requested_speed = req_speed

            # Determine tracking-aware adaptive frame skipping limit based on scene state
            adaptive_cfg = cfg_copy.get("adaptive_frame_skipping", {})
            adaptive_enabled = bool(adaptive_cfg.get("enabled", True))

            if not adaptive_enabled:
                current_max_skip = max(1, int(round(current_requested_speed * 4)))
            else:
                max_skip_empty = int(adaptive_cfg.get("empty_scene_max_skip", 15))
                max_skip_normal = int(adaptive_cfg.get("normal_flow_max_skip", 4))
                max_skip_suspect = int(adaptive_cfg.get("suspect_anomaly_max_skip", 1))
                max_skip_alert = int(adaptive_cfg.get("active_anomaly_max_skip", 2))
                max_skip_stopped = int(adaptive_cfg.get("prolonged_stopping_max_skip", 3))

                if monitoring_mode == "EMPTY":
                    current_max_skip = min(max_skip_empty, max(1, int(round(current_requested_speed * 5))))
                elif monitoring_mode == "SUSPECT":
                    # Dense sampling for counterflow confirmation: maximum 1 frame skip (or 0 if speed <= 1.5x)
                    current_max_skip = max_skip_suspect
                    if current_requested_speed <= 1.5:
                        current_max_skip = 0
                elif monitoring_mode == "ALERT":
                    # Confirmed wrong direction: preserve recovery and trail tracking
                    current_max_skip = min(max_skip_alert, max(1, int(round(current_requested_speed * 1.0))))
                elif monitoring_mode == "STOPPED":
                    current_max_skip = min(max_skip_stopped, max(1, int(round(current_requested_speed * 1.5))))
                else:  # NORMAL
                    current_max_skip = min(max_skip_normal, max(1, int(round(current_requested_speed * 1.5))))

            # Pacing & frame selection based on elapsed wall-clock time
            now = time.time()
            elapsed_wall = now - wall_sync_start
            target_video_frame = video_sync_frame + int(round(elapsed_wall * current_requested_speed * fps))

            frames_skipped_this_step = 0
            # Case A: If target frame is <= frame_idx, we are ahead of wall-clock pacing (e.g. at 0.25x or 0.5x,
            # or inference finished faster than target interval). Sleep smoothly until wall-clock reaches next frame.
            if target_video_frame <= frame_idx:
                next_frame = frame_idx + 1
                target_wall_time = wall_sync_start + ((next_frame - video_sync_frame) / max(fps, 1.0)) / max(0.1, current_requested_speed)
                sleep_sec = target_wall_time - time.time()
                if sleep_sec > 0.001:
                    time.sleep(min(0.25, sleep_sec))

            # Case B: If target frame > frame_idx + 1, we are behind wall-clock pacing because inference latency
            # is higher than the requested frame interval (e.g. at 1.5x, 2x, 3x, 5x, or CPU-bound 1x).
            # Advance video using cap.grab() up to tracking-aware current_max_skip.
            elif target_video_frame > frame_idx + 1:
                skip_count = min(target_video_frame - (frame_idx + 1), current_max_skip)
                for _ in range(skip_count):
                    if not cap.grab():
                        break
                    frame_idx += 1
                    frames_skipped += 1
                    frames_skipped_this_step += 1

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                # Loop video seamlessly and reset feature tracking state
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                wall_sync_start = time.time()
                video_sync_frame = 0
                playback_window_start = time.time()
                window_start_frame_idx = 0
                extractor.reset()
                anomaly_engine.reset()
                try:
                    if hasattr(model, "predictor") and model.predictor and hasattr(model.predictor, "trackers"):
                        for trk in model.predictor.trackers:
                            if hasattr(trk, "reset"):
                                trk.reset()
                            else:
                                trk.tracked_stracks = []
                                trk.lost_stracks = []
                                trk.removed_stracks = []
                                trk.frame_id = 0
                except Exception:
                    pass
                ret, frame = cap.read()
                if not ret:
                    break

            frame_idx += 1
            frames_processed += 1

            # Hot-reload admin expected direction + diagnostic flag each frame
            live_dir = STREAM_STATE.get("expected_direction") or expected_dir
            if live_dir != expected_dir:
                old_dir = expected_dir
                expected_dir = live_dir
                cfg_copy["expected_evacuation_direction"] = expected_dir
                if hasattr(anomaly_engine, "set_expected_direction"):
                    anomaly_engine.set_expected_direction(expected_dir)
                print(
                    f"[Streamer] EXPECTED_DIRECTION_CHANGED: OLD_DIRECTION={old_dir} "
                    f"NEW_DIRECTION={expected_dir} "
                    f"ACTIVE_CLASSIFIER_DIRECTION={anomaly_engine.classifier.expected} "
                    f"ACTIVE_CONFIG_VALUE={cfg_copy.get('expected_evacuation_direction')} "
                    f"allowed={list(allowed_directions(expected_dir))}",
                    flush=True,
                )

            # Hot-reload detection profile if changed live
            current_profile = STREAM_STATE.get("detection_profile", "balanced")
            if current_profile in DETECTION_PROFILES:
                p_cfg = DETECTION_PROFILES[current_profile]
                conf_thresh = float(p_cfg["conf"])
                inference_imgsz = int(p_cfg["imgsz"])
                nms_iou_thresh = float(p_cfg.get("iou", 0.55))

            # Diagnostics always useful; force on when a track is selected or admin enables mode
            anomaly_engine.diagnostic_mode = bool(
                STREAM_STATE.get("direction_diagnostic_mode", cfg_copy.get("direction_diagnostic_mode", False))
                or STREAM_STATE.get("selected_track_id") is not None
            )

            # YOLO11n + ByteTrack Inference for person class only (explicit imgsz + iou)
            t_infer0 = time.time()
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_yaml,
                conf=conf_thresh,
                iou=nms_iou_thresh,
                imgsz=inference_imgsz,
                classes=person_classes,
                device=device,
                verbose=False
            )
            infer_ms = (time.time() - t_infer0) * 1000.0

            # Optional diagnostic: raw person detections (no fabricated track IDs)
            raw_det_count = 0
            untracked_boxes = []
            if show_untracked:
                raw_res = model.predict(
                    frame, conf=conf_thresh, imgsz=inference_imgsz,
                    classes=person_classes, device=device, verbose=False
                )
                if raw_res and raw_res[0].boxes is not None and len(raw_res[0].boxes) > 0:
                    raw_coords = raw_res[0].boxes.xyxy.cpu().numpy()
                    raw_det_count = len(raw_coords)
                    tracked_coords = []
                    if results and results[0].boxes is not None and results[0].boxes.id is not None:
                        tracked_coords = results[0].boxes.xyxy.cpu().numpy()
                    for rb in raw_coords:
                        is_matched = False
                        for tb in tracked_coords:
                            ix1 = max(rb[0], tb[0]); iy1 = max(rb[1], tb[1])
                            ix2 = min(rb[2], tb[2]); iy2 = min(rb[3], tb[3])
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            a1 = max(0.0, (rb[2] - rb[0]) * (rb[3] - rb[1]))
                            a2 = max(0.0, (tb[2] - tb[0]) * (tb[3] - tb[1]))
                            union = a1 + a2 - inter
                            if union > 0 and (inter / union) >= 0.5:
                                is_matched = True
                                break
                        if not is_matched:
                            untracked_boxes.append(rb.tolist())
                else:
                    raw_det_count = 0

            current_pedestrians = []
            active_alerts = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                track_ids = extract_track_ids(boxes)
                coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)

                for bbox, tid in zip(coords, track_ids):
                    tid = int(tid)
                    feat = extractor.update_pedestrian(
                        track_id=tid,
                        bbox=tuple(bbox),
                        frame_idx=frame_idx,
                        fps=fps
                    )
                    feat = anomaly_engine.evaluate_pedestrian(
                        feat,
                        frame_img=frame,
                        frame_h=height,
                        frame_w=width
                    )
                    if feat["anomaly"]:
                        active_alerts.append({
                            "track_id": tid,
                            "type": feat["anomaly_type"],
                            "severity": feat.get("severity", "WARNING"),
                            "duration_sec": feat.get("duration_sec", 0.0),
                            "timestamp": feat.get("timestamp", 0.0),
                            "speed": feat.get("speed_px_sec", 0.0),
                            "direction": feat.get("direction", "STATIONARY"),
                            "zone": feat.get("zone_type", "NORMAL_ZONE"),
                            "zone_id": feat.get("zone_id"),
                            "zone_name": feat.get("zone_name"),
                            "event_id": feat.get("event_id"),
                            "evidence": feat.get("evidence", {}),
                            "status": "ACTIVE",
                            "is_live_observation": True,
                        })
                    current_pedestrians.append(feat)

            if len(current_pedestrians) == 0:
                consecutive_empty_frames += 1
                monitoring_mode = "EMPTY"
            else:
                consecutive_empty_frames = 0
                has_active_wrong_dir = any(
                    p.get("visual_state") == "anomaly" and p.get("anomaly_type") == "WRONG_DIRECTION"
                    for p in current_pedestrians
                ) or any(a.get("type") == "WRONG_DIRECTION" for a in active_alerts)

                if has_active_wrong_dir:
                    monitoring_mode = "ALERT"
                else:
                    has_suspect_wrong_dir = False
                    for p in current_pedestrians:
                        tid = p.get("track_id")
                        sust = anomaly_engine.wrong_dir_counters.get(tid, 0)
                        cls_res = p.get("direction_classification")
                        sec_dist = p.get("sector_distance")
                        if sust > 0 or cls_res == "ANOMALY" or (sec_dist is not None and sec_dist >= 2):
                            has_suspect_wrong_dir = True
                            break
                    if has_suspect_wrong_dir:
                        monitoring_mode = "SUSPECT"
                    elif any(
                        p.get("anomaly_type") == "PROLONGED_STOPPING" or p.get("stopping_duration_sec", 0.0) >= 3.0
                        for p in current_pedestrians
                    ):
                        monitoring_mode = "STOPPED"
                    else:
                        monitoring_mode = "NORMAL"

            # Live runtime trace (every ~1s): up to 5 tracks with full classification chain
            if frame_idx % max(1, int(fps)) == 0 and current_pedestrians:
                for ped in current_pedestrians[:5]:
                    dvec = ped.get("direction_vector") or [0, 0]
                    traj = ped.get("trajectory") or []
                    prev_foot = traj[-2][:2] if len(traj) >= 2 else None
                    vs = ped.get("visual_state")
                    colour = "RED" if vs == "anomaly" else ("BLUE" if vs in ("normal", "unknown") else str(vs))
                    print(
                        f"[LIVE_TRACE] TRACK_ID={ped.get('track_id')} "
                        f"FOOTPOINT={ped.get('foot')} PREVIOUS_FOOTPOINT={prev_foot} "
                        f"DX={dvec[0] if len(dvec) else None} DY={dvec[1] if len(dvec) > 1 else None} "
                        f"ESTIMATED_DIRECTION={ped.get('direction')} "
                        f"ESTIMATED_SECTOR={ped.get('direction')} "
                        f"EXPECTED_DIRECTION={ped.get('expected_direction')} "
                        f"CIRCULAR_SECTOR_DIFFERENCE={ped.get('sector_distance')} "
                        f"CLASSIFIER_RESULT={ped.get('direction_classification')} "
                        f"ANOMALY_ENGINE_RESULT={ped.get('anomaly_type')} "
                        f"FINAL_VISUAL_STATE={vs} FINAL_DISPLAY_COLOUR={colour} "
                        f"CLASSIFICATION_REASON={ped.get('direction_reason')} "
                        f"RUNTIME_ID={RUNTIME_ID}",
                        flush=True,
                    )

            # Render annotations onto CCTV Frame
            annotated_frame = frame.copy()

            # Draw ONLY main evacuation corridor + flow line (subtle, no fill clutter)
            corr = cfg_copy.get("evacuation_corridor") or {}
            corr_color = tuple(int(c) for c in (overlay_cfg.get("color_corridor_bgr") or corr.get("color") or [40, 200, 80]))
            if len(corr_color) != 3:
                corr_color = (40, 200, 80)
            poly = np.array(corr.get("polygon") or [], dtype=np.int32)
            if len(poly) >= 3:
                cv2.polylines(annotated_frame, [poly], isClosed=True, color=corr_color, thickness=1, lineType=cv2.LINE_AA)
            flow = np.array(corr.get("flow_line") or [], dtype=np.int32)
            if len(flow) >= 2:
                for i in range(len(flow) - 1):
                    cv2.arrowedLine(
                        annotated_frame,
                        tuple(flow[i]),
                        tuple(flow[i + 1]),
                        corr_color,
                        2,
                        tipLength=0.04,
                        line_type=cv2.LINE_AA,
                    )

            # Legend colours: blue=normal, red=confirmed anomaly, amber=selection outline
            col_normal = tuple(int(c) for c in (overlay_cfg.get("color_normal_bgr") or [255, 160, 40]))
            col_anomaly = tuple(int(c) for c in (overlay_cfg.get("color_anomaly_bgr") or [40, 40, 230]))
            col_select = tuple(int(c) for c in (overlay_cfg.get("color_selection_bgr") or [0, 180, 255]))

            selected_tid = STREAM_STATE.get("selected_track_id", None)
            occupied_label_rects = []
            sorted_peds = sorted(current_pedestrians, key=lambda p: p["bbox"][1])

            for ped in sorted_peds:
                tid = ped["track_id"]
                x1, y1, x2, y2 = [int(v) for v in ped["bbox"]]
                bw = max(10, x2 - x1)
                bh = max(10, y2 - y1)
                # Authoritative backend visual_state: anomaly→red; normal/unknown→blue
                vs = ped.get("visual_state") or (
                    "anomaly" if ped.get("anomaly") else "normal"
                )
                is_anom = vs == "anomaly" and ped.get("anomaly_type") in (
                    "WRONG_DIRECTION", "PROLONGED_STOPPING"
                )
                anom_type = ped.get("anomaly_type", "")
                is_selected = (selected_tid is not None and tid == selected_tid)

                t_normal = int(overlay_cfg.get("box_thickness_normal", 1))
                t_alert = int(overlay_cfg.get("box_thickness_alert", 2))
                # UNKNOWN stays blue — never red without confirmed anomaly
                box_color = col_anomaly if is_anom else col_normal
                thickness = t_alert if is_anom else t_normal

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)

                # Amber selection outline (independent of anomaly colour)
                if is_selected:
                    pad = 3
                    cv2.rectangle(
                        annotated_frame,
                        (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                        col_select, 2, cv2.LINE_AA,
                    )

                foot_x, foot_y = (x1 + x2) // 2, y2
                foot_r = int(overlay_cfg.get("foot_marker_radius", 2))
                cv2.circle(annotated_frame, (foot_x, foot_y), foot_r, box_color, -1, cv2.LINE_AA)

                show_arrows = bool(overlay_cfg.get("show_direction_arrows", True))
                pdir = ped.get("direction", "STATIONARY")
                dvec = ped.get("direction_vector") or [0, 0]
                reliable = bool(ped.get("direction_reliable", pdir not in ("STATIONARY", "UNKNOWN")))
                if show_arrows and (is_selected or is_anom) and reliable:
                    dx, dy = float(dvec[0]), float(dvec[1])
                    mag = float(np.hypot(dx, dy)) or 1.0
                    scale = 36.0 / mag
                    tip = (int(foot_x + dx * scale), int(foot_y + dy * scale))
                    cv2.arrowedLine(annotated_frame, (foot_x, foot_y), tip, box_color, 2, tipLength=0.35)

                # Selected-track movement HUD: compare visible trail vs estimated arrow
                if is_selected:
                    hud = (
                        f"est={pdir} exp={ped.get('expected_direction')} "
                        f"dSec={ped.get('sector_distance')} "
                        f"{ped.get('direction_classification')} "
                        f"sust={(ped.get('direction_diagnostics') or {}).get('sustain_frames', 0)} "
                        f"vs={vs}"
                    )
                    cv2.putText(
                        annotated_frame, hud, (max(8, x1), min(height - 8, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA,
                    )

                show_labels = bool(overlay_cfg.get("show_track_labels", True))
                compact = bool(overlay_cfg.get("compact_labels", True))
                show_dir = bool(overlay_cfg.get("show_direction_in_label", False))
                font_scale = float(overlay_cfg.get("label_font_scale", 0.32))

                if is_anom:
                    anom_map = {"PROLONGED_STOPPING": "STOP", "WRONG_DIRECTION": "DIR"}
                    lbl = f"#{tid} {anom_map.get(anom_type, 'ALERT')}"
                elif is_selected and reliable:
                    lbl = f"#{tid} {pdir}"
                else:
                    if compact and not show_dir:
                        lbl = f"#{tid}"
                    else:
                        lbl = f"#{tid} {pdir}" if show_dir and reliable else f"#{tid}"

                draw_label = show_labels or is_anom or is_selected
                if draw_label:
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    lw = tw + 8
                    lh = th + 6

                    # Candidate slots: (lx1, ly1, lx2, ly2, tx, ty, leader_target)
                    candidates = [
                        (x1, y1 - lh, x1 + lw, y1, x1 + 4, y1 - 4, None),
                        (x2 - lw, y1 - lh, x2, y1, x2 - lw + 4, y1 - 4, None),
                        (x1, y2, x1 + lw, y2 + lh, x1 + 4, y2 + lh - 4, None),
                        (x1, y1 - lh - 12, x1 + lw, y1 - 12, x1 + 4, y1 - 16, (x1 + bw // 2, y1)),
                        (x1, y1 - (lh * 2) - 16, x1 + lw, y1 - lh - 16, x1 + 4, y1 - lh - 20, (x1 + bw // 2, y1)),
                        (x2 + 6, y1, x2 + 6 + lw, y1 + lh, x2 + 10, y1 + lh - 4, (x2, y1 + bh // 4)),
                    ]

                    chosen = None
                    for cx1, cy1, cx2, cy2, tx, ty, leader in candidates:
                        if cx1 < 4:
                            diff = 4 - cx1
                            cx1 += diff; cx2 += diff; tx += diff
                        if cx2 > width - 4:
                            diff = cx2 - (width - 4)
                            cx1 -= diff; cx2 -= diff; tx -= diff
                        if cy1 < 36:
                            continue

                        collision = False
                        for ox1, oy1, ox2, oy2 in occupied_label_rects:
                            if not (cx2 < ox1 or cx1 > ox2 or cy2 < oy1 or cy1 > oy2):
                                collision = True
                                break

                        if not collision:
                            chosen = (cx1, cy1, cx2, cy2, tx, ty, leader)
                            break

                    if chosen is None:
                        chosen = (x1, max(28, y1 - lh), x1 + lw, max(28 + lh, y1), x1 + 4, max(28 + lh - 3, y1 - 3), None)

                    lx1, ly1, lx2, ly2, tx, ty, leader = chosen
                    occupied_label_rects.append((lx1, ly1, lx2, ly2))

                    if leader is not None:
                        cv2.line(annotated_frame, (lx1 + lw // 2, ly2), leader, box_color, 1, cv2.LINE_AA)

                    cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), (12, 18, 30), -1)
                    cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), box_color, 1)
                    cv2.putText(annotated_frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 245, 255), 1, cv2.LINE_AA)

                # Subtle recent trajectory trail (max 18 points)
                traj = ped.get("trajectory", [])
                if len(traj) >= 2:
                    recent_traj = traj[-18:]
                    segments = []
                    curr_seg = []
                    for pt in recent_traj:
                        px, py = float(pt[0]), float(pt[1])
                        if not curr_seg:
                            curr_seg.append([px, py])
                        else:
                            last_x, last_y = curr_seg[-1]
                            dist = float(np.hypot(px - last_x, py - last_y))
                            if dist > 65.0:
                                if len(curr_seg) >= 2:
                                    segments.append(curr_seg)
                                curr_seg = [[px, py]]
                            else:
                                curr_seg.append([px, py])
                    if len(curr_seg) >= 2:
                        segments.append(curr_seg)

                    for seg in segments:
                        t_pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [t_pts], isClosed=False, color=box_color, thickness=1, lineType=cv2.LINE_AA)

            # Diagnostic-only: dashed boxes for untracked raw detections (never used for anomalies)
            for ub in untracked_boxes:
                ux1, uy1, ux2, uy2 = [int(v) for v in ub]
                dash = 5
                ucol = (180, 180, 180)
                for dx in range(ux1, ux2, dash * 2):
                    cv2.line(annotated_frame, (dx, uy1), (min(dx + dash, ux2), uy1), ucol, 1)
                    cv2.line(annotated_frame, (dx, uy2), (min(dx + dash, ux2), uy2), ucol, 1)
                for dy in range(uy1, uy2, dash * 2):
                    cv2.line(annotated_frame, (ux1, dy), (ux1, min(dy + dash, uy2)), ucol, 1)
                    cv2.line(annotated_frame, (ux2, dy), (ux2, min(dy + dash, uy2)), ucol, 1)
                cv2.putText(annotated_frame, "UNTRACKED", (ux1, max(14, uy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, ucol, 1, cv2.LINE_AA)

            # Rolling processing throughput (excludes pacing sleep)
            pipeline_ms = (time.time() - t0) * 1000.0
            if proc_latency_ema_ms is None:
                proc_latency_ema_ms = pipeline_ms
            else:
                proc_latency_ema_ms = 0.85 * proc_latency_ema_ms + 0.15 * pipeline_ms
            instant_proc_fps = 1000.0 / max(pipeline_ms, 1e-3)
            if proc_fps_ema is None:
                proc_fps_ema = instant_proc_fps
            else:
                proc_fps_ema = 0.85 * proc_fps_ema + 0.15 * instant_proc_fps

            # Actual achieved playback speed calculation
            # Measured from actual video frames elapsed divided by wall-clock time
            current_requested_speed = float(STREAM_STATE.get("playback_speed", 1.0))
            wall_elapsed = time.time() - playback_window_start
            if wall_elapsed >= 0.5:
                delta_video_frames = frame_idx - window_start_frame_idx
                # Effective playback FPS achieved over window (video frames elapsed per wall second)
                effective_playback_fps = delta_video_frames / max(1e-3, wall_elapsed)
                # Achieved speed multiplier vs source video FPS
                achieved_playback_speed = round(effective_playback_fps / max(1.0, fps), 2)
                # Cap achieved speed display to requested speed to prevent measurement jitter
                if achieved_playback_speed > current_requested_speed * 1.05:
                    achieved_playback_speed = current_requested_speed
                # Reset window
                playback_window_start = time.time()
                window_start_frame_idx = frame_idx

            # Compact, clean top status pill on CCTV
            cv2.rectangle(annotated_frame, (0, 0), (width, 24), (10, 15, 26), -1)
            cv2.line(annotated_frame, (0, 24), (width, 24), (30, 42, 60), 1)
            proc_val = f"{proc_fps_ema:.1f}" if proc_fps_ema is not None else "--"
            header_txt = (
                f"{video_full_path.name} · F{frame_idx}/{total_frames} · Flow:{expected_dir} · "
                f"Peds:{len(current_pedestrians)} · Mode:{monitoring_mode} · "
                f"Req:{current_requested_speed}x · Act:{achieved_playback_speed:.2f}x ({effective_playback_fps:.1f} FPS) · "
                f"Proc:{proc_val} FPS · Skp:{frames_skipped}"
            )
            cv2.putText(annotated_frame, header_txt, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 215, 235), 1, cv2.LINE_AA)

            # Encode frame to Base64 JPEG for browser rendering
            render_w = min(width, 1280)
            render_h = int(height * (render_w / width))
            resized = cv2.resize(annotated_frame, (render_w, render_h)) if (width > 1280) else annotated_frame
            _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_image = "data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii')

            # Broadcast frame state & Digital Twin data
            broadcast_payload({
                "type": "frame_update",
                "runtime_id": RUNTIME_ID,
                "classifier_version": CLASSIFIER_VERSION,
                "frame": frame_idx,
                "total_frames": total_frames,
                "timestamp": round(frame_idx / fps, 2),
                "fps": round(fps, 1),  # source video FPS (nominal)
                "processing_fps": round(float(proc_fps_ema), 2) if proc_fps_ema is not None else 0.0,
                "latency_ms": round(float(proc_latency_ema_ms), 1) if proc_latency_ema_ms is not None else 0.0,
                "infer_ms": round(float(infer_ms), 1),
                "inference_imgsz": inference_imgsz,
                "detection_confidence": conf_thresh,
                "detection_profile": current_profile,
                "playback_speed": current_requested_speed,
                "requested_playback_speed": current_requested_speed,
                "achieved_playback_speed": achieved_playback_speed,
                "effective_playback_fps": round(float(effective_playback_fps), 1),
                "frames_processed": frames_processed,
                "frames_skipped": frames_skipped,
                "monitoring_mode": monitoring_mode,
                "adaptive_policy": {
                    "enabled": adaptive_enabled,
                    "mode": monitoring_mode,
                    "max_skip_allowed": current_max_skip,
                    "skipped_this_step": frames_skipped_this_step,
                },
                "raw_detection_count": raw_det_count if show_untracked else None,
                "tracked_count": len(current_pedestrians),
                "untracked_count": len(untracked_boxes) if show_untracked else None,
                "width": width,
                "height": height,
                "video_name": video_full_path.name,
                "frame_image": b64_image,
                "pedestrians": current_pedestrians,
                "active_alerts": active_alerts,
                "expected_direction": expected_dir,
                "evacuation_corridor": cfg_copy.get("evacuation_corridor") or {},
                "zones": {},
                "legend": {
                    "normal": "blue",
                    "anomaly": "red",
                    "selected": "amber",
                    "corridor": "green",
                },
                "has_pedestrians": len(current_pedestrians) > 0,
                "empty_warning": consecutive_empty_frames > 45
            })

            # Yield thread briefly to prevent CPU spinning between iterations
            time.sleep(0.001)

        cap.release()
        STREAM_STATE["is_streaming"] = False
        print(f"[Streamer] Stream finished: {video_full_path.name}", flush=True)


def stop_streaming():
    """Stops the active video stream cleanly before any new stream starts."""
    if STREAM_STATE.get("is_streaming") or STREAM_STATE.get("thread"):
        STREAM_STATE["stop_event"].set()
        th = STREAM_STATE.get("thread")
        if th and th.is_alive():
            th.join(timeout=2.0)
        STREAM_STATE["is_streaming"] = False
        STREAM_STATE["thread"] = None
        broadcast_payload({"type": "stream_stopped"})
        print("[Streamer] Stream stopped.", flush=True)


def start_streaming(video_path: str, direction: str = "RIGHT", zones: Optional[Dict] = None):
    """Starts or switches the active video stream."""
    stop_streaming()
    direction = coerce_admin_direction(direction)

    # Create a fresh stop event specifically for this new worker thread
    worker_stop_event = threading.Event()
    STREAM_STATE["stop_event"] = worker_stop_event
    STREAM_STATE["is_streaming"] = True
    STREAM_STATE["is_paused"] = False
    STREAM_STATE["current_video"] = video_path
    STREAM_STATE["expected_direction"] = direction
    STREAM_STATE["custom_zones"] = zones

    # Notify connected clients immediately that a video switch has occurred
    broadcast_payload({
        "type": "video_switched",
        "video_name": Path(video_path).name,
        "video_path": video_path,
        "direction": direction
    })

    t = threading.Thread(
        target=VideoStreamWorker.run,
        args=(video_path, direction, zones, worker_stop_event),
        daemon=True
    )
    STREAM_STATE["thread"] = t
    t.start()


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Handler with CORS and REST API support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/digital_twin/index.html"
            return super().do_GET()

        # API: Get available videos
        if parsed.path == "/api/videos":
            videos = get_available_videos()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(videos).encode("utf-8"))
            return

        # API: Get current config & zones
        if parsed.path == "/api/config":
            cfg = load_yaml_config()
            cfg["current_streaming"] = {
                "video": STREAM_STATE["current_video"],
                "direction": STREAM_STATE["expected_direction"],
                "is_streaming": STREAM_STATE["is_streaming"],
                "is_paused": STREAM_STATE["is_paused"]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cfg).encode("utf-8"))
            return

        # HTTP 206 Partial Content for video streaming
        range_header = self.headers.get("Range")
        req_file = PROJECT_ROOT / parsed.path.lstrip("/")
        if range_header and req_file.is_file() and req_file.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}:
            try:
                file_size = req_file.stat().st_size
                byte_range = range_header.strip().replace("bytes=", "").split("-")
                start = int(byte_range[0]) if byte_range[0] else 0
                end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
                end = min(end, file_size - 1)
                content_len = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "video/mp4" if req_file.suffix.lower() == ".mp4" else "video/webm")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_len))
                self.end_headers()

                with open(req_file, "rb") as f:
                    f.seek(start)
                    bytes_remaining = content_len
                    chunk_size = 64 * 1024
                    while bytes_remaining > 0:
                        chunk = f.read(min(chunk_size, bytes_remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        bytes_remaining -= len(chunk)
                return
            except Exception:
                pass

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        # API: Validate selected video path
        if parsed.path == "/api/validate_video":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body) if body else {}
            video_rel = data.get("video_path", "").strip()

            video_full_path = (PROJECT_ROOT / video_rel).resolve()
            if not video_full_path.exists():
                if (PROJECT_ROOT / "assets" / video_rel).resolve().exists():
                    video_full_path = (PROJECT_ROOT / "assets" / video_rel).resolve()
                    video_rel = f"assets/{video_rel}"
                elif (PROJECT_ROOT / "assets" / "uploads" / video_rel).resolve().exists():
                    video_full_path = (PROJECT_ROOT / "assets" / "uploads" / video_rel).resolve()
                    video_rel = f"assets/uploads/{video_rel}"

            if not video_full_path.exists():
                resp = {
                    "valid": False,
                    "error": f"File does not exist: {video_rel}"
                }
            else:
                cap = cv2.VideoCapture(str(video_full_path))
                if not cap.isOpened():
                    resp = {
                        "valid": False,
                        "error": f"OpenCV cannot open this video. Unsupported format or codec: {video_full_path.suffix}"
                    }
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        resp = {
                            "valid": False,
                            "error": "Video opened but contains no readable frames (file may be empty or corrupted)."
                        }
                    else:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        resp = {
                            "valid": True,
                            "filename": video_full_path.name,
                            "path": video_rel,
                            "width": w,
                            "height": h,
                            "fps": round(fps, 1),
                            "total_frames": total
                        }
                    cap.release()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # API: Upload new user video
        if parsed.path == "/api/upload_video":
            content_type = self.headers.get("Content-Type", "")
            content_len = int(self.headers.get("Content-Length", 0))

            upload_dir = PROJECT_ROOT / "assets" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            query = parse_qs(parsed.query)
            filename = query.get("filename", ["uploaded_video.mp4"])[0]
            filename = Path(filename).name
            target_path = upload_dir / filename

            # Read stream in chunks to handle arbitrary video sizes
            bytes_left = content_len
            with open(target_path, "wb") as f:
                while bytes_left > 0:
                    chunk = self.rfile.read(min(bytes_left, 64 * 1024))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_left -= len(chunk)

            # Test readability with OpenCV
            cap = cv2.VideoCapture(str(target_path))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    rel_path = f"assets/uploads/{filename}"
                    resp = {
                        "status": "ok",
                        "valid": True,
                        "path": rel_path,
                        "filename": filename,
                        "frames": total,
                        "fps": round(fps, 1),
                        "width": w,
                        "height": h
                    }
                else:
                    cap.release()
                    target_path.unlink(missing_ok=True)
                    resp = {"status": "error", "valid": False, "error": "Uploaded video contains no readable frames."}
            else:
                target_path.unlink(missing_ok=True)
                resp = {"status": "error", "valid": False, "error": "Uploaded file is not a supported video format or is corrupted."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # API: Start video stream
        if parsed.path == "/api/stream_video":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body) if body else {}

            video_path = data.get("video_path", "assets/intrusion.mp4")
            direction = data.get("direction", "RIGHT").upper()
            zones = data.get("zones", None)

            start_streaming(video_path, direction, zones)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "streaming": video_path, "direction": direction}).encode("utf-8"))
            return

        # API: Control stream (pause / resume / restart / stop)
        if parsed.path == "/api/stream_control":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body) if body else {}

            action = data.get("action", "")
            if action == "pause":
                STREAM_STATE["is_paused"] = True
            elif action == "resume":
                STREAM_STATE["is_paused"] = False
            elif action == "stop":
                stop_streaming()
            elif action == "restart":
                start_streaming(STREAM_STATE["current_video"], STREAM_STATE["expected_direction"], STREAM_STATE["custom_zones"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "action": action}).encode("utf-8"))
            return

        # API: Save zone & rule configuration
        if parsed.path == "/api/save_zones":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body) if body else {}

            zones = data.get("zones", None)
            direction = data.get("direction", None)
            camera_id = str(data.get("camera_id") or Path(STREAM_STATE.get("current_video") or "default").stem)

            corridor = data.get("evacuation_corridor") or zones
            if corridor:
                # Accept corridor object or wrap polygon under evacuation_corridor
                if isinstance(corridor, dict) and (
                    "polygon" in corridor or "flow_line" in corridor or "expected_direction" in corridor
                ):
                    corr_obj = corridor
                elif isinstance(corridor, dict) and corridor.get("evacuation_corridor"):
                    corr_obj = corridor["evacuation_corridor"]
                    if isinstance(corr_obj, list):
                        corr_obj = corr_obj[0] if corr_obj else {}
                else:
                    corr_obj = None

                try:
                    from scripts.feature_extraction import EvacuationCorridor
                    check_cfg = {"evacuation_corridor": corr_obj} if corr_obj else {"zones": zones or {}}
                    ok, msg = EvacuationCorridor.validate_config(check_cfg)
                except Exception as e:
                    ok, msg = False, str(e)
                if not ok:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": msg}).encode("utf-8"))
                    return

                STREAM_STATE["custom_zones"] = {"evacuation_corridor": corr_obj} if corr_obj else {}
                cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cur_cfg = yaml.safe_load(f) or {}
                    if corr_obj:
                        cur_cfg["evacuation_corridor"] = corr_obj
                        if corr_obj.get("expected_direction"):
                            cur_cfg["expected_evacuation_direction"] = corr_obj["expected_direction"]
                    cur_cfg["zones"] = {}
                    if direction:
                        cur_cfg["expected_evacuation_direction"] = direction
                    if data.get("speed_threshold") is not None:
                        cur_cfg["stationary_speed_threshold"] = float(data["speed_threshold"])
                    if data.get("stopping_frames") is not None:
                        cur_cfg["prolonged_stopping_frames"] = int(data["stopping_frames"])
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.dump(cur_cfg, f, default_flow_style=False, sort_keys=False)
                    cam_dir = PROJECT_ROOT / "configs" / "zones"
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    safe_cam = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_id)
                    with open(cam_dir / f"{safe_cam}.json", "w", encoding="utf-8") as f:
                        json.dump({"camera_id": camera_id, "evacuation_corridor": corr_obj}, f, indent=2)
                    global GLOBAL_CONFIG
                    GLOBAL_CONFIG = cur_cfg
                except Exception as e:
                    print(f"[Config] Error updating config.yaml: {e}", flush=True)

            if direction:
                STREAM_STATE["expected_direction"] = direction

            if STREAM_STATE["is_streaming"]:
                start_streaming(STREAM_STATE["current_video"], STREAM_STATE["expected_direction"], STREAM_STATE["custom_zones"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "message": "Zones updated and live pipeline reconfigured.",
                "camera_id": camera_id,
            }).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


async def register_client(websocket):
    """Register WebSocket client and listen for client commands."""
    CONNECTED_CLIENTS.add(websocket)
    print(f"[WebSocket] Digital Twin Client Connected ({len(CONNECTED_CLIENTS)} active)", flush=True)

    # Send initial available videos and status
    init_msg = json.dumps({
        "type": "init",
        "videos": get_available_videos(),
        "current_video": STREAM_STATE["current_video"],
        "expected_direction": STREAM_STATE["expected_direction"],
        "is_streaming": STREAM_STATE["is_streaming"],
        "playback_speed": STREAM_STATE.get("playback_speed", 1.0),
        "detection_profile": STREAM_STATE.get("detection_profile", "balanced"),
        "detection_profiles": {
            k: {"name": v["name"], "description": v["description"], "conf": v["conf"], "imgsz": v["imgsz"]}
            for k, v in DETECTION_PROFILES.items()
        }
    })
    await websocket.send(init_msg)

    # Note: We do NOT auto-start streaming here.
    # The application waits for the user to select video and click Start Analysis.

    try:
        async for raw_message in websocket:
            try:
                cmd = json.loads(raw_message)
                action = cmd.get("action", "")
                if action in ("start_stream", "switch_video"):
                    vpath = cmd.get("video_path", "assets/intrusion.mp4")
                    vdir = cmd.get("direction", "RIGHT")
                    vzones = cmd.get("zones", None)
                    if "speed" in cmd:
                        try:
                            STREAM_STATE["playback_speed"] = max(0.1, min(5.0, float(cmd["speed"])))
                        except Exception:
                            pass
                    if "profile" in cmd and cmd["profile"] in DETECTION_PROFILES:
                        STREAM_STATE["detection_profile"] = cmd["profile"]
                    start_streaming(vpath, vdir, vzones)
                elif action == "stop_stream":
                    stop_streaming()
                elif action == "toggle_pause":
                    STREAM_STATE["is_paused"] = not STREAM_STATE["is_paused"]
                elif action == "set_speed":
                    # Dynamic playback speed control: 0.25x, 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x, 3x, or custom
                    try:
                        speed_val = max(0.1, min(5.0, float(cmd.get("speed", 1.0))))
                        STREAM_STATE["playback_speed"] = speed_val
                        broadcast_payload({
                            "type": "speed_updated",
                            "playback_speed": speed_val
                        })
                        print(f"[WebSocket] Playback speed updated -> {speed_val}x", flush=True)
                    except Exception as e:
                        print(f"[WebSocket] Invalid speed value: {e}", flush=True)
                elif action == "set_detection_profile":
                    # Dynamic detection profile: performance, balanced, accuracy
                    prof = str(cmd.get("profile", "balanced")).lower()
                    if prof in DETECTION_PROFILES:
                        STREAM_STATE["detection_profile"] = prof
                        p_info = DETECTION_PROFILES[prof]
                        broadcast_payload({
                            "type": "profile_updated",
                            "detection_profile": prof,
                            "profile_info": p_info
                        })
                        print(f"[WebSocket] Detection profile updated -> {prof} (conf={p_info['conf']}, imgsz={p_info['imgsz']})", flush=True)
                elif action in ("set_direction", "update_direction"):
                    # Live cardinal expected-direction — no restart; persist for restarts
                    raw = str(cmd.get("direction") or STREAM_STATE["expected_direction"]).upper()
                    vdir = coerce_admin_direction(raw)
                    STREAM_STATE["expected_direction"] = vdir
                    try:
                        cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
                        if cfg_path.exists():
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                cur_cfg = yaml.safe_load(f) or {}
                            cur_cfg["expected_evacuation_direction"] = vdir
                            with open(cfg_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(cur_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    except Exception as e:
                        print(f"[Config] Failed to persist expected direction: {e}", flush=True)
                    broadcast_payload({
                        "type": "direction_updated",
                        "expected_direction": vdir,
                        "admin_cardinals": list(ADMIN_CARDINALS),
                    })
                    print(f"[WebSocket] Expected direction set → {vdir} (raw={raw})", flush=True)
                elif action == "update_zones":
                    vzones = cmd.get("zones", None)
                    vdir = cmd.get("direction", STREAM_STATE["expected_direction"])
                    if vdir:
                        STREAM_STATE["expected_direction"] = str(vdir).upper()
                    if STREAM_STATE["is_streaming"]:
                        start_streaming(STREAM_STATE["current_video"], STREAM_STATE["expected_direction"], vzones)
                elif action == "select_track":
                    STREAM_STATE["selected_track_id"] = cmd.get("track_id", None)
                elif action == "set_diagnostic":
                    STREAM_STATE["direction_diagnostic_mode"] = bool(cmd.get("enabled", False))
            except Exception as e:
                print(f"[WebSocket] Error handling command: {e}", flush=True)
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"[WebSocket] Digital Twin Client Disconnected ({len(CONNECTED_CLIENTS)} active)", flush=True)


def start_http_server():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    handler = CORSHTTPRequestHandler
    try:
        with socketserver.ThreadingTCPServer(("", HTTP_PORT), handler) as httpd:
            print("=======================================================")
            print(" 2D Evacuation Digital Twin Dashboard Server")
            print(f" Dashboard UI: http://localhost:{HTTP_PORT}/")
            print(f" WebSocket Stream: ws://localhost:{WS_PORT}/")
            print("=======================================================\n")
            httpd.serve_forever()
    except OSError as e:
        print(f"[Warning] HTTP Server could not bind port {HTTP_PORT}: {e}")


async def start_ws_server():
    global ASYNC_LOOP
    ASYNC_LOOP = asyncio.get_running_loop()
    try:
        async with websockets.serve(register_client, "0.0.0.0", WS_PORT):
            await asyncio.Future()  # Run forever
    except OSError as e:
        print(f"[Warning] WebSocket Server could not bind port {WS_PORT}: {e}")


def main():
    # Startup check: fail fast if configured detector weights are missing
    try:
        model_path = resolve_yolo_model_path(GLOBAL_CONFIG, require_exists=True)
        print(f"[Startup] Detector weights OK: {model_path}", flush=True)
        print(
            f"[Startup] Tracker: {GLOBAL_CONFIG.get('tracker_config', 'configs/bytetrack_crowd.yaml')} | "
            f"imgsz={resolve_inference_imgsz(GLOBAL_CONFIG)} | "
            f"conf={GLOBAL_CONFIG.get('detection_confidence', 0.08)}",
            flush=True,
        )
        exp = GLOBAL_CONFIG.get("expected_evacuation_direction", STREAM_STATE.get("expected_direction"))
        print(
            f"[Startup] RUNTIME_ID={RUNTIME_ID} pid={os.getpid()}\n"
            f"[Startup] server.py={Path(__file__).resolve()}\n"
            f"[Startup] direction_module={Path(_direction_estimation_mod.__file__).resolve()}\n"
            f"[Startup] classifier_version={CLASSIFIER_VERSION}\n"
            f"[Startup] expected_direction={exp} "
            f"allowed={list(allowed_directions(str(exp)))} "
            f"max_sector_distance_normal="
            f"{(GLOBAL_CONFIG.get('direction_detection') or {}).get('max_sector_distance_normal', 1)}",
            flush=True,
        )
    except FileNotFoundError as e:
        print(f"[Startup] FATAL: {e}", flush=True)
        raise SystemExit(1) from e

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    try:
        asyncio.run(start_ws_server())
    except KeyboardInterrupt:
        STREAM_STATE["stop_event"].set()
        print("\n[Server] Shutting down Digital Twin server.")


if __name__ == "__main__":
    main()
