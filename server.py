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

from _common import open_video, get_video_props, extract_track_ids, resolve_tracker, resolve_inference_imgsz
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine

HTTP_PORT = 8080
WS_PORT = 8765

CONNECTED_CLIENTS: Set[Any] = set()

# Global streaming state
STREAM_STATE = {
    "is_streaming": False,
    "is_paused": False,
    "current_video": "assets/intrusion.mp4",
    "expected_direction": "RIGHT",
    "custom_zones": None,
    "selected_track_id": None,
    "thread": None,
    "stop_event": threading.Event()
}

# Shared event loop for WebSocket broadcasting
ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None


def load_yaml_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


GLOBAL_CONFIG = load_yaml_config()


def get_available_videos() -> List[Dict[str, str]]:
    """Scan assets/ and assets/uploads/ for video files supported by OpenCV."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []

    # Priority preset videos
    preset_files = [
        ("intrusion.mp4", "Intrusion Detection Scenario (intrusion.mp4)"),
        ("running.mp4", "Evacuation Movement Scenario (running.mp4)"),
        ("fall.mp4", "Pedestrian Fall Scenario (fall.mp4)"),
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

        # Initialize detector & tracker
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(GLOBAL_CONFIG.get("yolo_model", "yolov8n.pt"))
        model.to(device)

        cfg_copy = dict(GLOBAL_CONFIG)
        cfg_copy["expected_evacuation_direction"] = expected_dir
        if custom_zones:
            cfg_copy["zones"] = custom_zones

        extractor = TrajectoryFeatureExtractor(cfg_copy)
        anomaly_engine = EvacuationAnomalyEngine(cfg_copy)
        tracker_yaml = resolve_tracker(cfg_copy.get("tracker_config", "configs/bytetrack.yaml"))
        conf_thresh = float(cfg_copy.get("detection_confidence", 0.08))
        # Phase 26/33: session imgsz — production default 800; optional far_camera_profile → 960
        # Resolved once per stream session (not per frame). No automatic scene classification.
        inference_imgsz = resolve_inference_imgsz(cfg_copy)
        far_prof = cfg_copy.get("far_camera_profile") or {}
        person_classes = [0] if "pose" not in str(cfg_copy.get("yolo_model", "")) else None
        show_untracked = bool(cfg_copy.get("show_untracked_detections", False))

        print(
            f"[Streamer] Inference settings: model={cfg_copy.get('yolo_model', 'yolov8n.pt')} "
            f"imgsz={inference_imgsz} conf={conf_thresh} classes={person_classes} "
            f"tracker={tracker_yaml} device={device} "
            f"far_camera_profile.enabled={bool(far_prof.get('enabled', False))}",
            flush=True
        )

        frame_interval = 1.0 / max(fps, 10.0)
        frame_idx = 0
        consecutive_empty_frames = 0
        # Rolling processing throughput (excludes pacing sleep)
        proc_latency_ema_ms = None
        proc_fps_ema = None

        while not stop_event.is_set():
            if STREAM_STATE["is_paused"]:
                time.sleep(0.1)
                continue

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                # Loop video seamlessly and reset feature tracking state
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
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

            # YOLOv8 + ByteTrack Inference for person class only (explicit imgsz)
            t_infer0 = time.time()
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_yaml,
                conf=conf_thresh,
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
                            "confidence": feat.get("fall_confidence", 0.0),
                            "event_id": feat.get("fall_event_id"),
                            "tracking_state": feat.get("fall_tracking_state", "TRACKED"),
                            "is_live_observation": feat.get("is_live_observation", True)
                        })
                    current_pedestrians.append(feat)

            # Post-frame fall persistence & spatial re-association for active falls
            observed_tids = [p["track_id"] for p in current_pedestrians]
            raw_candidates = []
            if hasattr(anomaly_engine, "fall_detector") and anomaly_engine.fall_detector:
                has_active_fall = any(s.get("status") in ("FALL_CONFIRMED", "FALL_SUSPECTED") for s in anomaly_engine.fall_detector.fall_state.values())
                if has_active_fall:
                    res_raw = model.predict(frame, conf=0.04, classes=[0], verbose=False)
                    if res_raw and len(res_raw) > 0 and res_raw[0].boxes is not None and len(res_raw[0].boxes) > 0:
                        raw_candidates = [list(c) for c in res_raw[0].boxes.xyxy.cpu().numpy()]

            coasting_feats = anomaly_engine.process_frame_persistence(
                frame_idx=frame_idx,
                observed_track_ids=observed_tids,
                raw_candidate_boxes=raw_candidates,
                frame_img=frame,
                frame_h=height,
                frame_w=width,
                fps=fps
            )
            for c_feat in coasting_feats:
                current_pedestrians.append(c_feat)
                if c_feat.get("anomaly"):
                    active_alerts.append({
                        "track_id": c_feat["track_id"],
                        "type": c_feat["anomaly_type"],
                        "severity": c_feat.get("severity", "CRITICAL"),
                        "duration_sec": c_feat.get("duration_sec", 0.0),
                        "timestamp": c_feat.get("timestamp", 0.0),
                        "speed": 0.0,
                        "direction": "STATIONARY",
                        "zone": c_feat.get("zone_type", "NORMAL_ZONE"),
                        "confidence": c_feat.get("fall_confidence", 0.60),
                        "event_id": c_feat.get("fall_event_id"),
                        "tracking_state": c_feat.get("fall_tracking_state", "LOST_COASTING"),
                        "is_live_observation": c_feat.get("is_live_observation", False)
                    })

            if len(current_pedestrians) == 0:
                consecutive_empty_frames += 1
            else:
                consecutive_empty_frames = 0

            # Render annotations onto CCTV Frame
            annotated_frame = frame.copy()

            # Draw Zones on video with subtle styling
            zones_data = cfg_copy.get("zones", {})
            for ztype, color in [("restricted_zones", (50, 50, 220)), ("evacuation_routes", (40, 180, 40)), ("safe_zones", (220, 180, 40)), ("exit_zones", (40, 230, 130))]:
                for zinfo in zones_data.get(ztype, []):
                    pts = np.array(zinfo.get("polygon", []), dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [pts], color=color)
                        cv2.addWeighted(overlay, 0.08, annotated_frame, 0.92, 0, annotated_frame)
                        cv2.putText(annotated_frame, zinfo.get("name", ztype).upper(), (pts[0][0], max(18, pts[0][1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # Anti-Collision Label Placement for Pedestrians
            selected_tid = STREAM_STATE.get("selected_track_id", None)
            occupied_label_rects = []
            sorted_peds = sorted(current_pedestrians, key=lambda p: p["bbox"][1])

            for ped in sorted_peds:
                tid = ped["track_id"]
                x1, y1, x2, y2 = [int(v) for v in ped["bbox"]]
                bw = max(10, x2 - x1)
                bh = max(10, y2 - y1)
                is_anom = ped["anomaly"]
                anom_type = ped.get("anomaly_type", "")
                is_selected = (selected_tid is not None and tid == selected_tid)
                is_coasting = (ped.get("fall_tracking_state") == "LOST_COASTING" or not ped.get("is_live_observation", True))

                # Clean restrained color palette:
                # Cyan/Blue for normal, Red for critical/fall, Amber for warning/fall suspected, Bright cyan for selection
                if is_selected:
                    box_color = (0, 220, 255)
                    thickness = 2
                elif is_coasting:
                    box_color = (0, 140, 255)  # Coasting Amber-Orange
                    thickness = 2
                elif anom_type == "FALL_DETECTED":
                    box_color = (50, 50, 235)  # Critical Red
                    thickness = 2
                elif anom_type == "FALL_SUSPECTED":
                    box_color = (0, 165, 255)  # Warning Amber
                    thickness = 2
                elif is_anom:
                    box_color = (50, 50, 235)  # Restrained Red
                    thickness = 2
                else:
                    box_color = (235, 175, 50)  # Restrained Blue/Cyan
                    thickness = 1

                # Draw pedestrian bounding box (dashed if coasting)
                if is_coasting:
                    dash_len = 6
                    for dx in range(x1, x2, dash_len * 2):
                        cv2.line(annotated_frame, (dx, y1), (min(dx + dash_len, x2), y1), box_color, thickness)
                        cv2.line(annotated_frame, (dx, y2), (min(dx + dash_len, x2), y2), box_color, thickness)
                    for dy in range(y1, y2, dash_len * 2):
                        cv2.line(annotated_frame, (x1, dy), (x1, min(dy + dash_len, y2)), box_color, thickness)
                        cv2.line(annotated_frame, (x2, dy), (x2, min(dy + dash_len, y2)), box_color, thickness)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)

                # If selected or fall detected, draw corner accent brackets
                if is_selected or anom_type == "FALL_DETECTED":
                    corner_col = (0, 240, 255) if is_selected else (50, 50, 235)
                    c_len = min(12, bw // 3, bh // 3)
                    cv2.line(annotated_frame, (x1 - 3, y1 - 3), (x1 - 3 + c_len, y1 - 3), corner_col, 2)
                    cv2.line(annotated_frame, (x1 - 3, y1 - 3), (x1 - 3, y1 - 3 + c_len), corner_col, 2)
                    cv2.line(annotated_frame, (x2 + 3, y2 + 3), (x2 + 3 - c_len, y2 + 3), corner_col, 2)
                    cv2.line(annotated_frame, (x2 + 3, y2 + 3), (x2 + 3 - c_len, y2 + 3), corner_col, 2)

                # Draw bottom foot position marker
                foot_x, foot_y = (x1 + x2) // 2, y2
                cv2.circle(annotated_frame, (foot_x, foot_y), 3, box_color, -1, cv2.LINE_AA)

                # Clean, compact label
                if is_coasting:
                    lbl = f"#{tid} - FALLEN (COASTING)"
                elif is_anom:
                    anom_map = {
                        "FALL_DETECTED": "FALL DETECTED",
                        "FALL_SUSPECTED": "FALL SUSPECTED",
                        "PROLONGED_STOPPING": "STOPPING",
                        "WRONG_DIRECTION": "WRONG DIR",
                        "RESTRICTED_ZONE_ENTRY": "RESTRICTED"
                    }
                    status_str = anom_map.get(ped.get("anomaly_type", ""), "ALERT")
                    lbl = f"#{tid} - {status_str}"
                else:
                    pdir = ped.get("direction", "STATIONARY")
                    lbl = f"#{tid} - {pdir}" if pdir != "STATIONARY" else f"#{tid}"

                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
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

                # Draw leader line if staggered
                if leader is not None:
                    cv2.line(annotated_frame, (lx1 + lw // 2, ly2), leader, box_color, 1, cv2.LINE_AA)

                # Draw dark label badge with refined border
                cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), (12, 18, 30), -1)
                cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), box_color, 1)
                cv2.putText(annotated_frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)

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

            # Processing latency / FPS (pipeline wall time, excludes pacing sleep)
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

            # Compact, clean top status pill on CCTV
            cv2.rectangle(annotated_frame, (0, 0), (width, 24), (10, 15, 26), -1)
            cv2.line(annotated_frame, (0, 24), (width, 24), (30, 42, 60), 1)
            header_txt = (
                f"{video_full_path.name} · Frame {frame_idx}/{total_frames} · Flow: {expected_dir} · "
                f"Tracks: {len(current_pedestrians)} · imgsz={inference_imgsz} · "
                f"{proc_fps_ema:.1f} procFPS · {proc_latency_ema_ms:.0f}ms"
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
                "frame": frame_idx,
                "total_frames": total_frames,
                "timestamp": round(frame_idx / fps, 2),
                "fps": round(fps, 1),  # source video FPS (nominal)
                "processing_fps": round(float(proc_fps_ema), 2),
                "latency_ms": round(float(proc_latency_ema_ms), 1),
                "infer_ms": round(float(infer_ms), 1),
                "inference_imgsz": inference_imgsz,
                "detection_confidence": conf_thresh,
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
                "zones": cfg_copy.get("zones", {}),
                "has_pedestrians": len(current_pedestrians) > 0,
                "empty_warning": consecutive_empty_frames > 45
            })

            # Control streaming speed
            elapsed = time.time() - t0
            sleep_time = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_time)

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

            if zones:
                STREAM_STATE["custom_zones"] = zones
                cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cur_cfg = yaml.safe_load(f) or {}
                    cur_cfg["zones"] = zones
                    if direction:
                        cur_cfg["expected_evacuation_direction"] = direction
                    if data.get("speed_threshold") is not None:
                        cur_cfg["stationary_speed_threshold"] = float(data["speed_threshold"])
                    if data.get("stopping_frames") is not None:
                        cur_cfg["prolonged_stopping_frames"] = int(data["stopping_frames"])
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.dump(cur_cfg, f, default_flow_style=False, sort_keys=False)
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
            self.wfile.write(json.dumps({"status": "ok", "message": "Zones updated and live pipeline reconfigured."}).encode("utf-8"))
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
        "is_streaming": STREAM_STATE["is_streaming"]
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
                    start_streaming(vpath, vdir, vzones)
                elif action == "stop_stream":
                    stop_streaming()
                elif action == "toggle_pause":
                    STREAM_STATE["is_paused"] = not STREAM_STATE["is_paused"]
                elif action == "update_zones":
                    vzones = cmd.get("zones", None)
                    vdir = cmd.get("direction", STREAM_STATE["expected_direction"])
                    if STREAM_STATE["is_streaming"]:
                        start_streaming(STREAM_STATE["current_video"], vdir, vzones)
                elif action == "select_track":
                    STREAM_STATE["selected_track_id"] = cmd.get("track_id", None)
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
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    try:
        asyncio.run(start_ws_server())
    except KeyboardInterrupt:
        STREAM_STATE["stop_event"].set()
        print("\n[Server] Shutting down Digital Twin server.")


if __name__ == "__main__":
    main()
