#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evacuation Context Anomaly Engine
=================================
Explainable rule-based anomaly detection for evacuation monitoring:
  Rule 1: WRONG_DIRECTION — Movement opposing configured evacuation flow
  Rule 2: PROLONGED_STOPPING — Pedestrian stationary for longer than allowed threshold
  Rule 3: RESTRICTED_ZONE_ENTRY — Foot position located inside a designated restricted polygon
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


try:
    from fall_detection import FallDetector
except ImportError:
    from scripts.fall_detection import FallDetector


class EvacuationAnomalyEngine:
    """Rule-based anomaly detection engine tailored for evacuation scenarios with dedicated fall detection."""

    OPPOSITE_DIRECTIONS = {
        "RIGHT": "LEFT",
        "LEFT": "RIGHT",
        "UP": "DOWN",
        "DOWN": "UP",
    }

    def __init__(self, config: Dict[str, Any], fall_detector: Optional[FallDetector] = None):
        self.config = config
        self.expected_dir = config.get("expected_evacuation_direction", "RIGHT").upper()
        self.opposite_dir = self.OPPOSITE_DIRECTIONS.get(self.expected_dir, "LEFT")
        self.stopping_thresh_frames = config.get("prolonged_stopping_frames", 30)

        # Fall detector integration
        if fall_detector is not None:
            self.fall_detector = fall_detector
        else:
            self.fall_detector = FallDetector(config)

        # Event log storage
        self.anomaly_events: List[Dict[str, Any]] = []
        # Per-track anomaly onset frame: {track_id: start_frame}
        self.track_anomaly_start: Dict[int, int] = {}

    def reset(self):
        """Reset logged events and active anomaly state."""
        self.anomaly_events.clear()
        self.track_anomaly_start.clear()
        if self.fall_detector:
            self.fall_detector.reset()

    def evaluate_pedestrian(
        self,
        feature: Dict[str, Any],
        frame_img: Optional[Any] = None,
        frame_h: int = 1080,
        frame_w: int = 1920
    ) -> Dict[str, Any]:
        """
        Evaluates rules for a single pedestrian's feature dict.
        Modifies feature in-place and returns updated dict with anomaly status, type, severity, and duration.
        """
        track_id = feature["track_id"]
        frame = feature["frame"]
        timestamp = feature["timestamp"]
        direction = feature["direction"]
        stopping_frames = feature["stopping_duration_frames"]
        zone_type = feature["zone_type"]
        bbox = feature["bbox"]
        speed_px_sec = feature.get("speed_px_sec", 0.0)

        anomalies_detected: List[str] = []

        # Rule 0: Fallen Pedestrian Detection
        fall_res = {"status": "NORMAL", "confidence": 0.0, "reasons": []}
        if self.fall_detector is not None:
            fall_res = self.fall_detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=tuple(bbox),
                frame_idx=frame,
                frame_h=frame_h,
                frame_w=frame_w,
                frame_img=frame_img,
                speed_px_sec=speed_px_sec
            )
            if fall_res["status"] in ("FALL_DETECTED", "FALL_CONFIRMED"):
                anomalies_detected.append("FALL_DETECTED")
            elif fall_res["status"] == "FALL_SUSPECTED":
                anomalies_detected.append("FALL_SUSPECTED")

        # Rule 1: Wrong Direction Check
        # Suppress WRONG_DIRECTION when the track has insufficient history (< 8 frames)
        # or unreliable movement (speed exceeding unphysical threshold > 350 px/sec)
        history_len = len(feature.get("trajectory", [])) if "trajectory" in feature else feature.get("frame", 10)
        min_dir_hist = int(self.config.get("trajectory_min_history_for_direction", 8))
        is_reliable_movement = (
            history_len >= min_dir_hist
            and 5.0 <= speed_px_sec <= 350.0
        )
        if direction == self.opposite_dir and is_reliable_movement:
            anomalies_detected.append("WRONG_DIRECTION")

        # Rule 2: Prolonged Stopping Check (Suppressed if fall detected so fall takes precedence)
        if stopping_frames >= self.stopping_thresh_frames:
            anomalies_detected.append("PROLONGED_STOPPING")

        # Rule 3: Restricted Zone Entry Check
        if zone_type == "RESTRICTED_ZONE":
            anomalies_detected.append("RESTRICTED_ZONE_ENTRY")

        is_anomalous = len(anomalies_detected) > 0
        
        # Priority order for primary anomaly:
        # 1. FALL_DETECTED (Medical / immediate life safety emergency)
        # 2. RESTRICTED_ZONE_ENTRY (Perimeter breach)
        # 3. FALL_SUSPECTED (Potential incident under confirmation)
        # 4. PROLONGED_STOPPING (Choke point congestion)
        # 5. WRONG_DIRECTION (Evacuation counterflow)
        if not is_anomalous:
            primary_anomaly = "NORMAL"
        elif "FALL_DETECTED" in anomalies_detected:
            primary_anomaly = "FALL_DETECTED"
        elif "RESTRICTED_ZONE_ENTRY" in anomalies_detected:
            primary_anomaly = "RESTRICTED_ZONE_ENTRY"
        elif "FALL_SUSPECTED" in anomalies_detected:
            primary_anomaly = "FALL_SUSPECTED"
        elif "PROLONGED_STOPPING" in anomalies_detected:
            primary_anomaly = "PROLONGED_STOPPING"
        else:
            primary_anomaly = anomalies_detected[0]

        # Defensible Rule-Based Severity Classification:
        # - CRITICAL: FALL_DETECTED, Restricted zone intrusion, severe blockage (>=3s), or multi-rule combinations
        # - WARNING: FALL_SUSPECTED, Wrong-direction flow, or initial stopping (1s - 3s)
        # - NORMAL: Compliant movement
        if not is_anomalous:
            severity = "NORMAL"
            self.track_anomaly_start.pop(track_id, None)
            duration_sec = 0.0
        else:
            if track_id not in self.track_anomaly_start:
                self.track_anomaly_start[track_id] = frame
            onset_frame = self.track_anomaly_start[track_id]
            duration_sec = max(0.1, round((frame - onset_frame + 1) / max(self.config.get("fps", 30.0), 1.0), 1))

            if "FALL_DETECTED" in anomalies_detected or "RESTRICTED_ZONE_ENTRY" in anomalies_detected:
                severity = "CRITICAL"
            elif stopping_frames >= 90:
                severity = "CRITICAL"
            elif len(anomalies_detected) > 1 and "FALL_SUSPECTED" not in anomalies_detected:
                severity = "CRITICAL"
            else:
                severity = "WARNING"

        feature["anomaly"] = is_anomalous
        feature["anomaly_type"] = primary_anomaly
        feature["all_anomalies"] = anomalies_detected
        feature["severity"] = severity
        feature["duration_sec"] = duration_sec
        feature["fall_status"] = fall_res["status"]
        feature["fall_confidence"] = fall_res["confidence"]
        feature["fall_reasons"] = fall_res["reasons"]
        feature["fall_event_id"] = fall_res.get("event_id")
        feature["fall_tracking_state"] = fall_res.get("tracking_state", "TRACKED")
        feature["is_live_observation"] = fall_res.get("is_live_observation", True)

        if is_anomalous:
            event = {
                "track_id": track_id,
                "frame": frame,
                "timestamp": timestamp,
                "anomaly_type": primary_anomaly,
                "all_anomalies": anomalies_detected,
                "severity": severity,
                "duration_sec": duration_sec,
                "x": feature["foot"][0],
                "y": feature["foot"][1],
                "speed": feature["speed_px_sec"],
                "direction": direction,
                "zone": zone_type,
                "fall_confidence": fall_res["confidence"],
                "fall_event_id": fall_res.get("event_id"),
                "fall_tracking_state": fall_res.get("tracking_state", "TRACKED"),
                "is_live_observation": fall_res.get("is_live_observation", True)
            }
            self.anomaly_events.append(event)

        return feature

    def process_frame_persistence(
        self,
        frame_idx: int,
        observed_track_ids: List[int],
        raw_candidate_boxes: Optional[List[List[float]]] = None,
        frame_img: Optional[Any] = None,
        frame_h: int = 1080,
        frame_w: int = 1920,
        fps: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Processes post-frame persistence for active fall tracks temporarily unobserved by ByteTrack.
        Returns a list of feature dictionaries for coasting or spatially re-associated tracks.
        """
        if not self.fall_detector or not hasattr(self.fall_detector, "process_frame_persistence"):
            return []

        persist_outputs = self.fall_detector.process_frame_persistence(
            frame_idx=frame_idx,
            observed_track_ids=observed_track_ids,
            raw_candidate_boxes=raw_candidate_boxes,
            frame_img=frame_img,
            frame_h=frame_h,
            frame_w=frame_w
        )

        persisted_features = []
        for p_out in persist_outputs:
            tid = p_out["track_id"]
            st = p_out["status"]
            tr_state = p_out.get("tracking_state", "LOST_COASTING")
            is_live = p_out.get("is_live_observation", False)
            bbox = p_out.get("last_known_bbox") or p_out.get("bbox", [0, 0, 10, 10])

            # Manage anomaly onset and duration
            if tid not in self.track_anomaly_start:
                self.track_anomaly_start[tid] = frame_idx
            onset_frame = self.track_anomaly_start[tid]
            duration_sec = max(0.1, round((frame_idx - onset_frame + 1) / max(fps, 1.0), 1))

            primary_anomaly = "FALL_DETECTED" if st == "FALL_CONFIRMED" else "FALL_SUSPECTED"
            severity = "CRITICAL" if primary_anomaly == "FALL_DETECTED" else "WARNING"

            feat = {
                "track_id": tid,
                "frame": frame_idx,
                "timestamp": round(frame_idx / max(fps, 1.0), 2),
                "bbox": [float(v) for v in bbox],
                "foot": [float((bbox[0] + bbox[2]) / 2.0), float(bbox[3])],
                "speed_px_sec": 0.0,
                "direction": "STATIONARY",
                "zone_type": "NORMAL_ZONE",
                "stopping_duration_frames": p_out.get("coasting_frames", 0),
                "anomaly": True,
                "anomaly_type": primary_anomaly,
                "all_anomalies": [primary_anomaly],
                "severity": severity,
                "duration_sec": duration_sec,
                "fall_status": st,
                "fall_confidence": p_out.get("confidence", 0.60),
                "fall_reasons": p_out.get("reasons", []),
                "fall_event_id": p_out.get("event_id"),
                "fall_tracking_state": tr_state,
                "is_live_observation": is_live,
                "coasting_frames": p_out.get("coasting_frames", 0),
                "max_coasting_frames": p_out.get("max_coasting_frames", 60)
            }

            event = {
                "track_id": tid,
                "frame": frame_idx,
                "timestamp": feat["timestamp"],
                "anomaly_type": primary_anomaly,
                "all_anomalies": [primary_anomaly],
                "severity": severity,
                "duration_sec": duration_sec,
                "x": feat["foot"][0],
                "y": feat["foot"][1],
                "speed": 0.0,
                "direction": "STATIONARY",
                "zone": "NORMAL_ZONE",
                "fall_confidence": feat["fall_confidence"],
                "fall_event_id": feat["fall_event_id"],
                "fall_tracking_state": tr_state,
                "is_live_observation": is_live
            }
            self.anomaly_events.append(event)
            persisted_features.append(feat)

        return persisted_features

    def export_anomaly_log(self, output_path: Path):
        """Export logged anomaly events to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.anomaly_events, f, indent=2, ensure_ascii=False)
        print(f"[AnomalyEngine] Logged {len(self.anomaly_events)} anomaly instances -> {output_path}")

