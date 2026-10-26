#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pedestrian Detection & Tracking Pipeline
========================================
Main processing script combining:
  - YOLOv8n Pedestrian Detection (COCO class 0: person)
  - ByteTrack Persistent Tracking
  - Trajectory & Feature Extraction
  - Context-Aware Rule Anomaly Detection
  - Annotated Video Renderer & JSON Exporters
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Import project utilities
try:
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        extract_track_ids,
        resolve_tracker,
        show_frame,
        cleanup_resources,
    )
except ImportError:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        extract_track_ids,
        resolve_tracker,
        show_frame,
        cleanup_resources,
    )

try:
    from feature_extraction import TrajectoryFeatureExtractor
    from anomaly_detection import EvacuationAnomalyEngine
except ImportError:
    from scripts.feature_extraction import TrajectoryFeatureExtractor
    from scripts.anomaly_detection import EvacuationAnomalyEngine



def load_config(config_path: Path) -> Dict[str, Any]:
    """Load system YAML configuration."""
    if not config_path.exists():
        print(f"[X] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PedestrianPipeline:
    """Core computer vision pipeline for pedestrian detection, tracking, and evacuation anomaly monitoring."""

    def __init__(self, config: Dict[str, Any], websocket_broadcaster=None):
        self.config = config
        self.websocket_broadcaster = websocket_broadcaster
        
        # Determine execution device (prefer CPU or GPU if available)
        dev_cfg = config.get("device", "auto")
        if dev_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = dev_cfg
        
        # Load YOLO model
        model_name = config.get("yolo_model", "yolov8n.pt")
        print(f"[Pipeline] Loading YOLOv8 model: {model_name} on device: {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # Tracker setup
        tracker_cfg = config.get("tracker_config", "configs/bytetrack.yaml")
        self.tracker_yaml = resolve_tracker(tracker_cfg)
        self.conf_thresh = config.get("detection_confidence", 0.08)
        # Phase 33: session-fixed imgsz (optional far_camera_profile override)
        try:
            from _common import resolve_inference_imgsz
            self.inference_imgsz = resolve_inference_imgsz(config)
        except Exception:
            self.inference_imgsz = int(config.get("inference_imgsz", 800))
        self.person_class = config.get("person_class_id", 0)

        # Selective Tiled Inference setup (Phase 21)
        tiled_cfg = config.get("tiled_inference", {})
        self.tiled_enabled = tiled_cfg.get("enabled", False)
        self.tiled_detector = None
        self.custom_byte_tracker = None
        if self.tiled_enabled:
            try:
                from scripts.tiled_inference import SelectiveTiledDetector
                from types import SimpleNamespace
                from ultralytics.trackers.byte_tracker import BYTETracker
                self.tiled_detector = SelectiveTiledDetector(
                    model=self.model,
                    tile_size=(tiled_cfg.get("tile_width", 640), tiled_cfg.get("tile_height", 540)),
                    overlap=(tiled_cfg.get("overlap_x", 0.20), tiled_cfg.get("overlap_y", 0.20)),
                    roi_rel=tuple(tiled_cfg.get("roi_corridor", [0.0, 0.15, 1.0, 0.65])),
                    conf_thresh=self.conf_thresh,
                    iou_nms_thresh=tiled_cfg.get("iou_nms_thresh", 0.50),
                    person_class_id=self.person_class,
                    mem_limit_mb=tiled_cfg.get("mem_limit_mb", 650.0),
                    enabled=True
                )
                bt_args = SimpleNamespace(
                    track_high_thresh=0.25,
                    track_low_thresh=0.08,
                    new_track_thresh=0.20,
                    track_buffer=30,
                    match_thresh=0.80,
                    fuse_score=True
                )
                self.custom_byte_tracker = BYTETracker(bt_args)
                print(f"[Pipeline] Selective Tiled Inference enabled: ROI corridor {self.tiled_detector.roi_rel}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not initialize tiled detector ({e}). Falling back to full-frame.")
                self.tiled_detector = None
                self.tiled_enabled = False

        # Initialize feature extractor & anomaly engine
        self.extractor = TrajectoryFeatureExtractor(config)
        self.anomaly_engine = EvacuationAnomalyEngine(config)

    def process_video(
        self,
        video_path: Path,
        output_video_path: Optional[Path] = None,
        export_basename: Optional[str] = None,
        display: bool = False
    ) -> Dict[str, Any]:
        """
        Processes an input CCTV/pedestrian video end-to-end.
        """
        cap = open_video(video_path)
        props = get_video_props(cap)
        fps = props["fps"] if props["fps"] > 0 else 30.0
        width, height = props["width"], props["height"]
        total_frames = props["total_frames"]

        print(f"\n=======================================================")
        print(f" Pedestrian Anomaly Detection Pipeline")
        print(f" Source: {video_path.name} ({width}x{height}, {total_frames} frames, {fps:.1f} FPS)")
        print(f" Expected Evacuation Direction: {self.config.get('expected_evacuation_direction', 'RIGHT')}")
        print(f"=======================================================\n")

        # Video writer initialization
        writer = None
        if output_video_path:
            writer = create_video_writer(output_video_path, fps, (width, height))

        # Re-initialize state per video to guarantee clean isolation
        self.extractor = TrajectoryFeatureExtractor(self.config)
        self.anomaly_engine = EvacuationAnomalyEngine(self.config)
        self.model.predictor = None

        # Setup export paths
        bname = export_basename or video_path.stem
        traj_out_path = PROJECT_ROOT / "outputs" / "trajectories" / f"{bname}_trajectories.json"
        anomaly_out_path = PROJECT_ROOT / "outputs" / "anomalies" / f"{bname}_anomalies.json"

        frame_idx = 0
        start_time = time.time()
        active_anomalies_summary: Dict[int, str] = {}
        tracked_ids_set = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run tracking: Tiled inference or full-frame baseline
            coords = []
            track_ids = []

            if self.tiled_enabled and self.tiled_detector is not None and self.custom_byte_tracker is not None:
                tracks, _ = self.tiled_detector.detect_and_track(frame, self.custom_byte_tracker)
                if len(tracks) > 0:
                    coords = tracks[:, :4]
                    track_ids = [int(t[4]) for t in tracks]
            else:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_yaml,
                    conf=self.conf_thresh,
                    imgsz=self.inference_imgsz,
                    classes=[self.person_class] if "pose" not in str(self.config.get("yolo_model", "")) else None,
                    device=self.device,
                    verbose=False
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    track_ids = extract_track_ids(boxes)
                    coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)

            current_pedestrians_state = []
            frame_anomalies = []

            if len(coords) > 0 and len(track_ids) > 0:

                for bbox, tid in zip(coords, track_ids):
                    tid = int(tid)
                    tracked_ids_set.add(tid)

                    # Extract movement features
                    feat = self.extractor.update_pedestrian(
                        track_id=tid,
                        bbox=tuple(bbox), # type: ignore
                        frame_idx=frame_idx,
                        fps=fps
                    )

                    # Evaluate context-aware anomaly rules with fall detection
                    feat = self.anomaly_engine.evaluate_pedestrian(
                        feat,
                        frame_img=frame,
                        frame_h=height,
                        frame_w=width
                    )

                    if feat["anomaly"]:
                        active_anomalies_summary[tid] = feat["anomaly_type"]
                        frame_anomalies.append(feat)

                    current_pedestrians_state.append(feat)

            # Post-frame persistence for active falls temporarily lost by ByteTrack
            raw_candidates = []
            if hasattr(self.anomaly_engine, "fall_detector") and self.anomaly_engine.fall_detector:
                has_active_fall = any(s.get("status") in ("FALL_CONFIRMED", "FALL_SUSPECTED") for s in self.anomaly_engine.fall_detector.fall_state.values())
                if has_active_fall:
                    res_raw = self.model.predict(frame, conf=0.04, classes=[self.person_class], verbose=False)
                    if res_raw and len(res_raw) > 0 and res_raw[0].boxes is not None and len(res_raw[0].boxes) > 0:
                        raw_candidates = [list(c) for c in res_raw[0].boxes.xyxy.cpu().numpy()]

            observed_tids = [p["track_id"] for p in current_pedestrians_state]
            coasting_feats = self.anomaly_engine.process_frame_persistence(
                frame_idx=frame_idx,
                observed_track_ids=observed_tids,
                raw_candidate_boxes=raw_candidates,
                frame_img=frame,
                frame_h=height,
                frame_w=width,
                fps=fps
            )
            for c_feat in coasting_feats:
                current_pedestrians_state.append(c_feat)
                if c_feat["anomaly"]:
                    active_anomalies_summary[c_feat["track_id"]] = c_feat["anomaly_type"]
                    frame_anomalies.append(c_feat)

            # Draw annotations on the CCTV video frame
            annotated_frame = self._render_annotations(frame, current_pedestrians_state, frame_idx, total_frames, fps)

            # Broadcast live state to 2D Digital Twin (if websocket active)
            if self.websocket_broadcaster:
                self.websocket_broadcaster.broadcast_frame_state(
                    frame_idx=frame_idx,
                    timestamp=round(frame_idx / fps, 2),
                    pedestrians=current_pedestrians_state,
                    video_width=width,
                    video_height=height
                )

            # Write output frame
            if writer:
                writer.write(annotated_frame)

            # Display frame if requested
            if display:
                if not show_frame(annotated_frame, window_name="Evacuation Monitor - CCTV Feed"):
                    break

        elapsed = time.time() - start_time
        processing_fps = frame_idx / max(elapsed, 0.001)

        cleanup_resources(cap, writer)

        # Export trajectories and anomaly logs
        self.extractor.export_trajectories(traj_out_path)
        self.anomaly_engine.export_anomaly_log(anomaly_out_path)

        summary = {
            "video": video_path.name,
            "total_frames": frame_idx,
            "processing_time_sec": round(elapsed, 2),
            "processing_fps": round(processing_fps, 1),
            "unique_pedestrians_tracked": len(tracked_ids_set),
            "total_anomaly_instances": len(self.anomaly_engine.anomaly_events),
            "trajectory_export": str(traj_out_path),
            "anomaly_export": str(anomaly_out_path),
        }

        print(f"\n[Pipeline Complete]")
        print(f"  Processed {frame_idx} frames in {elapsed:.2f}s ({processing_fps:.1f} FPS)")
        print(f"  Tracked Pedestrians: {len(tracked_ids_set)}")
        print(f"  Anomalies Detected: {len(self.anomaly_engine.anomaly_events)}")

        return summary

    def _render_annotations(
        self,
        frame: np.ndarray,
        pedestrians: List[Dict[str, Any]],
        frame_idx: int,
        total_frames: int,
        fps: float
    ) -> np.ndarray:
        """Render zones, bounding boxes, labels, trajectories, and alerts on CCTV frame."""
        annotated = frame.copy()

        # 1. Draw Polygonal Zones
        zones = self.config.get("zones", {})
        zone_colors = {
            "restricted_zones": (0, 0, 220),   # Red
            "evacuation_routes": (0, 180, 0),   # Green
            "safe_zones": (220, 180, 0),       # Cyan
            "exit_zones": (0, 255, 128),       # Light Green
        }
        for zgroup, color in zone_colors.items():
            for zinfo in zones.get(zgroup, []):
                pts = np.array(zinfo.get("polygon", []), dtype=np.int32)
                if len(pts) >= 3:
                    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
                    # Semi-transparent overlay
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [pts], color=color)
                    cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                    # Label
                    lbl_pos = (pts[0][0], max(20, pts[0][1] - 5))
                    cv2.putText(annotated, zinfo.get("name", zgroup), lbl_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 2. Draw Pedestrians & Trajectories
        active_anomalies = []
        for ped in pedestrians:
            tid = ped["track_id"]
            x1, y1, x2, y2 = [int(v) for v in ped["bbox"]]
            is_anomaly = ped["anomaly"]
            anomaly_type = ped["anomaly_type"]

            is_coasting = (ped.get("fall_tracking_state") == "LOST_COASTING" or not ped.get("is_live_observation", True))

            # Color coding: Red for critical anomaly/fall, Amber for warning/suspected fall, Cyan for normal
            if is_coasting:
                box_color = (0, 140, 255)  # Coasting Amber-Orange
                thickness = 2
            elif anomaly_type == "FALL_DETECTED":
                box_color = (50, 50, 235)  # Critical Red
                thickness = 2
            elif anomaly_type == "FALL_SUSPECTED":
                box_color = (0, 165, 255)  # Warning Amber
                thickness = 2
            elif is_anomaly:
                box_color = (50, 50, 235)  # Red for other anomalies
                thickness = 2
            else:
                box_color = (235, 175, 50)  # Cyan for normal
                thickness = 1

            # Bounding box
            if is_coasting:
                dash_len = 6
                for dx in range(x1, x2, dash_len * 2):
                    cv2.line(annotated, (dx, y1), (min(dx + dash_len, x2), y1), box_color, thickness)
                    cv2.line(annotated, (dx, y2), (min(dx + dash_len, x2), y2), box_color, thickness)
                for dy in range(y1, y2, dash_len * 2):
                    cv2.line(annotated, (x1, dy), (x1, min(dy + dash_len, y2)), box_color, thickness)
                    cv2.line(annotated, (x2, dy), (x2, min(dy + dash_len, y2)), box_color, thickness)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Foot marker dot
            foot_pt = (int(ped["foot"][0]), int(ped["foot"][1]))
            cv2.circle(annotated, foot_pt, 3, box_color, -1, cv2.LINE_AA)

            # Clean compact label text
            if is_coasting:
                lbl = f"#{tid} · FALL (COASTING)"
                active_anomalies.append((tid, "FALL_COASTING"))
            elif is_anomaly:
                anom_short = {
                    "FALL_DETECTED": "FALL DETECTED",
                    "FALL_SUSPECTED": "FALL SUSPECTED",
                    "PROLONGED_STOPPING": "STOPPING",
                    "WRONG_DIRECTION": "WRONG DIR",
                    "RESTRICTED_ZONE_ENTRY": "RESTRICTED"
                }.get(anomaly_type, anomaly_type)
                lbl = f"#{tid} · {anom_short}"
                active_anomalies.append((tid, anomaly_type))
            else:
                pdir = ped.get("direction", "STATIONARY")
                lbl = f"#{tid} · {pdir}" if pdir != "STATIONARY" else f"#{tid}"

            # Label background box
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (12, 18, 30), -1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), box_color, 1)
            cv2.putText(annotated, lbl, (x1 + 3, max(th + 2, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)

            # Subtle trajectory line
            traj_pts = ped["trajectory"]
            if len(traj_pts) >= 2:
                recent_pts = traj_pts[-18:]
                pts_arr = np.array(recent_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts_arr], isClosed=False, color=box_color, thickness=1, lineType=cv2.LINE_AA)

        # 3. Overlay Header Stats Bar
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (20, 20, 20), -1)
        header_text = (f"CCTV Evacuation Monitor | Frame: {frame_idx}/{total_frames} | "
                       f"Pedestrians: {len(pedestrians)} | Active Alerts: {len(active_anomalies)}")
        cv2.putText(annotated, header_text, (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated


def main():
    parser = argparse.ArgumentParser(description="Context-Aware Pedestrian Anomaly Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--zone", type=str, default=None, help="Path to custom JSON polygon zones file")
    parser.add_argument("--output", type=str, default=None, help="Path to output processed video")
    parser.add_argument("--display", action="store_true", help="Display video output during processing")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.zone:
        zone_file = Path(args.zone)
        if zone_file.exists():
            with open(zone_file, "r", encoding="utf-8") as f:
                custom_zones = json.load(f)
            cfg["zones"] = custom_zones
            print(f"[Pipeline] Loaded custom zones from: {zone_file}")

    source_path = Path(args.source)
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "videos" / f"{source_path.stem}_processed.mp4"

    pipeline = PedestrianPipeline(cfg)
    pipeline.process_video(source_path, output_video_path=out_path, display=args.display)


if __name__ == "__main__":
    main()
