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
        
        # Load person detector (configurable; default CrowdHuman YOLO11n fine-tune)
        try:
            from _common import resolve_yolo_model_path, resolve_inference_imgsz
            model_path = resolve_yolo_model_path(config, require_exists=True)
            self.inference_imgsz = resolve_inference_imgsz(config)
        except FileNotFoundError:
            raise
        except Exception:
            model_path = str(config.get("yolo_model", "weights/aegis_yolo11n_crowdhuman.pt"))
            self.inference_imgsz = int(config.get("inference_imgsz", 800))
        print(f"[Pipeline] Loading detector: {model_path} on device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Tracker setup — reuse project crowd/fall YAML via resolve_tracker (no silent default swap)
        tracker_cfg = config.get("tracker_config", "configs/bytetrack_crowd.yaml")
        self.tracker_yaml = resolve_tracker(tracker_cfg)
        self.conf_thresh = config.get("detection_confidence", 0.08)
        self.person_class = config.get("person_class_id", 0)
        self.overlay_cfg = config.get("overlay") or {}

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
                    classes=None if "pose" in str(self.config.get("yolo_model", "")).lower() else [self.person_class],
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

                    # Evaluate context-aware anomaly rules (restricted / wrong-way / stop)
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

        # 1. Main evacuation corridor only
        ov = getattr(self, "overlay_cfg", {}) or {}
        corr = self.config.get("evacuation_corridor") or {}
        corr_color = tuple(int(c) for c in (corr.get("color") or [40, 200, 80]))
        poly = np.array(corr.get("polygon") or [], dtype=np.int32)
        if len(poly) >= 3:
            cv2.polylines(annotated, [poly], isClosed=True, color=corr_color, thickness=1, lineType=cv2.LINE_AA)
        flow = np.array(corr.get("flow_line") or [], dtype=np.int32)
        if len(flow) >= 2:
            for i in range(len(flow) - 1):
                cv2.arrowedLine(annotated, tuple(flow[i]), tuple(flow[i + 1]), corr_color, 2, tipLength=0.04)

        col_normal = tuple(int(c) for c in (ov.get("color_normal_bgr") or [255, 160, 40]))
        col_anomaly = tuple(int(c) for c in (ov.get("color_anomaly_bgr") or [40, 40, 230]))

        # 2. Draw Pedestrians & Trajectories
        active_anomalies = []
        for ped in pedestrians:
            tid = ped["track_id"]
            x1, y1, x2, y2 = [int(v) for v in ped["bbox"]]
            is_anomaly = bool(ped.get("anomaly")) and ped.get("anomaly_type") in (
                "WRONG_DIRECTION", "PROLONGED_STOPPING"
            )
            anomaly_type = ped.get("anomaly_type", "NORMAL")

            t_normal = int(ov.get("box_thickness_normal", 1))
            t_alert = int(ov.get("box_thickness_alert", 2))
            font_scale = float(ov.get("label_font_scale", 0.32))
            compact = bool(ov.get("compact_labels", True))
            show_dir = bool(ov.get("show_direction_in_label", False))
            show_labels = bool(ov.get("show_track_labels", True))

            box_color = col_anomaly if is_anomaly else col_normal
            thickness = t_alert if is_anomaly else t_normal

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            foot_pt = (int(ped["foot"][0]), int(ped["foot"][1]))
            cv2.circle(annotated, foot_pt, int(ov.get("foot_marker_radius", 2)), box_color, -1, cv2.LINE_AA)

            if is_anomaly:
                anom_short = {
                    "PROLONGED_STOPPING": "STOP",
                    "WRONG_DIRECTION": "DIR",
                }.get(anomaly_type, anomaly_type)
                lbl = f"#{tid} {anom_short}"
                active_anomalies.append((tid, anomaly_type))
            else:
                pdir = ped.get("direction", "STATIONARY")
                if compact and not show_dir:
                    lbl = f"#{tid}"
                else:
                    lbl = f"#{tid} {pdir}" if show_dir and pdir not in ("STATIONARY", "UNKNOWN") else f"#{tid}"

            if show_labels or is_anomaly:
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (12, 18, 30), -1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), box_color, 1)
                cv2.putText(annotated, lbl, (x1 + 3, max(th + 2, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 245, 255), 1, cv2.LINE_AA)

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
