#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallen Pedestrian Detection Module — AEGIS Evacuation Vision Console
====================================================================
Dedicated track-aware temporal fall-detection engine (Phase 22):
  State Machine:
    NORMAL -> FALL_SUSPECTED -> FALL_CONFIRMED -> RECOVERY_OR_LOST

Key Features:
1. Explicit 4-State Temporal FSM:
   - NORMAL: Upright posture, normal locomotion.
   - FALL_SUSPECTED: Transient posture anomaly (rapid centroid descent, AR expansion, torso tilt).
   - FALL_CONFIRMED: Sustained collapse confirmed via multi-frame geometry or pose keypoint verticality.
   - RECOVERY_OR_LOST: Distinguishes confirmed standing recovery from temporary occlusion/track-loss.
2. Persistent Fall Tracking (Grace Period / Coasting):
   - When a prone person is temporarily unobserved due to low detector contrast against background,
     the fall state machine maintains the incident for a configurable coasting interval (e.g., 60 frames / 2.0s).
   - Flags uncertainty explicitly (`tracking_state: LOST_COASTING`, `is_live_observation: False`).
   - NEVER fabricates live detections or claims unobserved location is known.
3. Abrupt Posture Spatial Re-Association:
   - Connects low-confidence horizontal candidate detections near the collapse location back to the
     active fall track even when standard Kalman bounding-box IoU drops below association threshold.
4. Single Continuous Event Management:
   - Assigns a unique persistent `event_id` to prevent one fall from generating repeated independent alerts.
5. Standing Recovery Verification:
   - Confirms true recovery only when sustained upright posture is observed for multiple consecutive frames.
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class FallDetector:
    """
    Track-aware Hierarchical Temporal Fall Detection Engine.
    """

    # Explicit State Constants
    STATE_NORMAL = "NORMAL"
    STATE_SUSPECTED = "FALL_SUSPECTED"
    STATE_CONFIRMED = "FALL_CONFIRMED"
    STATE_RECOVERED = "RECOVERED"
    STATE_LOST_TIMEOUT = "LOST_TIMEOUT"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pose_model_path: Optional[str] = "weights/yolov8n-pose.pt"
    ):
        self.config = config or {}
        self.pose_model = None

        model_path = self.config.get("fall_pose_model", pose_model_path)
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                proj_root = Path(__file__).resolve().parent.parent
                if (proj_root / model_path).exists():
                    p = proj_root / model_path
                elif (proj_root / "weights" / Path(model_path).name).exists():
                    p = proj_root / "weights" / Path(model_path).name

            if p.exists():
                try:
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"[FallDetector] Loading YOLOv8n-pose model from: {p} on {dev}")
                    self.pose_model = YOLO(str(p))
                    self.pose_model.to(dev)
                except Exception as e:
                    print(f"[FallDetector] Warning: Could not load pose model ({e}). Operating in trajectory/geometry mode.")
                    self.pose_model = None
            else:
                print(f"[FallDetector] Pose model not found at {p}. Operating in trajectory/geometry mode.")

        # Per-track historical state
        self.track_history = defaultdict(lambda: deque(maxlen=45))  # [(frame, bbox, cy, ar, h)]
        self.standing_height = {}  # {track_id: baseline_height}
        self.fall_state = defaultdict(lambda: {
            "status": self.STATE_NORMAL,
            "event_id": None,
            "consecutive_fall_frames": 0,
            "consecutive_upright_frames": 0,
            "consecutive_pose_horizontal_frames": 0,
            "first_fall_frame": None,
            "last_observed_frame": None,
            "coasting_frames": 0,
            "last_known_bbox": None,
            "tracking_state": "TRACKED",  # "TRACKED", "LOST_COASTING", "RECOVERED", "TIMEOUT"
            "is_live_observation": True,
            "fall_score": 0.0,
            "reason": "",
            "has_dynamic_drop": False,
            "initial_standing": False,
            "reassign_cooldown": 0,  # suppress confirm after track jump / reassignment
        })

        # Configurable thresholds
        self.ar_suspect_threshold = float(self.config.get("fall_ar_suspect_threshold", 0.72))
        self.ar_confirm_threshold = float(self.config.get("fall_ar_confirm_threshold", 0.90))
        self.height_drop_ratio = float(self.config.get("fall_height_drop_ratio", 0.62))
        self.centroid_drop_threshold = float(self.config.get("fall_centroid_drop_threshold", 0.04))
        self.confirm_frames = int(self.config.get("fall_confirm_frames", 6))
        self.max_ground_speed = float(self.config.get("fall_max_ground_speed", 55.0))  # px/sec
        self.coast_frames_max = int(self.config.get("fall_coast_frames_max", 60))  # ~2.0s persistence
        self.spatial_reassoc_dist = float(self.config.get("fall_spatial_reassoc_dist", 160.0))  # px
        self.recovery_confirm_frames = int(self.config.get("fall_recovery_confirm_frames", 12))
        # Phase 28 reliability: harden CONFIRM vs SUSPECT separation
        self.min_history_for_confirm = int(self.config.get("fall_min_history_for_confirm", 8))
        self.require_dynamic_drop_for_confirm = bool(
            self.config.get("fall_confirm_require_dynamic_drop", True)
        )
        self.pose_confirm_frames = int(self.config.get("fall_pose_confirm_frames", 4))
        self.max_confirm_ground_speed = float(
            self.config.get("fall_max_confirm_ground_speed", self.max_ground_speed)
        )
        self.post_reassign_cooldown = int(self.config.get("fall_post_reassign_cooldown", 8))

        self.next_event_id = 1
        self.active_fall_events: Dict[int, Dict[str, Any]] = {}  # event_id -> metadata

    def reset(self):
        """Reset internal tracking state on video loop or stream change."""
        self.track_history.clear()
        self.standing_height.clear()
        self.fall_state.clear()
        self.active_fall_events.clear()
        self.next_event_id = 1

    def _verify_pose(self, crop: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Runs targeted YOLOv8n-pose inference on a candidate bounding-box crop.
        Returns:
            (is_torso_horizontal, is_torso_upright, torso_confidence)
        """
        if self.pose_model is None or crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
            return False, False, 0.0

        try:
            res_p = self.pose_model(crop, conf=0.08, verbose=False)
            if res_p and len(res_p) > 0 and res_p[0].keypoints is not None and len(res_p[0].keypoints.xy) > 0:
                kpts = res_p[0].keypoints.xy[0].cpu().numpy()
                if len(kpts) >= 17:
                    l_sh, r_sh = kpts[5], kpts[6]
                    l_hip, r_hip = kpts[11], kpts[12]
                    sh_valid = (l_sh[1] > 0 or r_sh[1] > 0)
                    hip_valid = (l_hip[1] > 0 or r_hip[1] > 0)

                    if sh_valid and hip_valid:
                        sh_y = np.mean([pt[1] for pt in [l_sh, r_sh] if pt[1] > 0])
                        hip_y = np.mean([pt[1] for pt in [l_hip, r_hip] if pt[1] > 0])
                        sh_x = np.mean([pt[0] for pt in [l_sh, r_sh] if pt[0] > 0])
                        hip_x = np.mean([pt[0] for pt in [l_hip, r_hip] if pt[0] > 0])

                        dy = abs(hip_y - sh_y)
                        dx = abs(hip_x - sh_x)
                        torso_len = np.sqrt(dx * dx + dy * dy) + 1e-5

                        # Sanity check: Foreshortened / degenerate pose keypoints (<20% crop height)
                        crop_h = crop.shape[0]
                        if torso_len < 0.20 * crop_h:
                            return False, False, 0.0

                        verticality = dy / torso_len

                        if verticality < 0.48:
                            # Clearly horizontal torso
                            return True, False, float(1.0 - verticality)
                        elif verticality > 0.72:
                            # Clearly upright torso
                            return False, True, float(verticality)
        except Exception:
            pass

        return False, False, 0.0

    def evaluate_pedestrian(
        self,
        track_id: int,
        bbox: tuple,
        frame_idx: int,
        frame_h: int,
        frame_w: int,
        frame_img: Optional[np.ndarray] = None,
        speed_px_sec: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates a single observed pedestrian detection.
        Updates posture baselines, temporal trajectory evidence, and transitions FSM.
        """
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        ar = bw / bh
        cy = (y1 + y2) / 2.0

        hist = self.track_history[track_id]
        cur_state = self.fall_state[track_id]
        cur_state["last_observed_frame"] = frame_idx
        cur_state["last_known_bbox"] = list(bbox)
        cur_state["tracking_state"] = "TRACKED"
        cur_state["is_live_observation"] = True
        cur_state["coasting_frames"] = 0

        # 1. Update standing height baseline when person is upright (AR < 0.60)
        min_standing_bh = max(20.0, 0.06 * float(frame_h))
        if ar < 0.60 and bh >= min_standing_bh:
            cur_state["initial_standing"] = True
            if track_id not in self.standing_height:
                self.standing_height[track_id] = bh
            else:
                self.standing_height[track_id] = 0.95 * self.standing_height[track_id] + 0.05 * bh

        standing_h = self.standing_height.get(track_id, bh)
        h_ratio = bh / max(standing_h, 1.0)

        # 2. Centroid vertical drop rate over recent frames
        dynamic_drop = False
        cy_drop_amount = 0.0
        is_tracker_jump = False
        if len(hist) >= 1:
            last_frame = hist[-1][0] if isinstance(hist[-1][0], (int, float)) else frame_idx
            # hist stores (frame, bbox, cy, ar, bh) — use center x from bbox
            last_bbox = hist[-1][1]
            last_cx = (last_bbox[0] + last_bbox[2]) / 2.0
            last_cy = hist[-1][2]
            frame_gap = int(frame_idx - last_frame) if last_frame is not None else 1
            dx = abs((x1 + x2) / 2.0 - last_cx)
            dy = abs(cy - last_cy)
            # Track reassignment / occlusion recovery: large spatial jump or multi-frame gap
            if (
                dx > 120.0
                or float(np.hypot(dx, dy)) > 200.0
                or speed_px_sec > 300.0
                or frame_gap > 10
            ):
                is_tracker_jump = True
                cur_state["has_dynamic_drop"] = False
                cur_state["consecutive_fall_frames"] = 0
                cur_state["consecutive_pose_horizontal_frames"] = 0
                cur_state["first_fall_frame"] = None
                cur_state["reassign_cooldown"] = self.post_reassign_cooldown
                # Do not carry stale fall CONFIRM across a reassigned identity
                if cur_state["status"] in (self.STATE_SUSPECTED,):
                    cur_state["status"] = self.STATE_NORMAL

        if cur_state.get("reassign_cooldown", 0) > 0:
            cur_state["reassign_cooldown"] = max(0, int(cur_state["reassign_cooldown"]) - 1)

        if not is_tracker_jump and len(hist) >= 3:
            recent_idx = max(0, len(hist) - 5)
            recent_cy = hist[recent_idx][2]
            cy_drop_amount = (cy - recent_cy) / float(frame_h)
            if cy_drop_amount > self.centroid_drop_threshold:
                dynamic_drop = True
                cur_state["has_dynamic_drop"] = True

        # 3. Geometric Posture Analysis
        near_floor = (y2 >= 0.40 * float(frame_h))
        is_horizontal_bbox = (ar >= self.ar_confirm_threshold)
        is_compressed = (ar >= self.ar_suspect_threshold and h_ratio < self.height_drop_ratio)

        # 4. Targeted Pose Verification
        pose_torso_horizontal = False
        pose_upright = False
        pose_confidence = 0.0

        is_candidate = (is_horizontal_bbox or is_compressed or (dynamic_drop and ar >= 0.55) or ar >= 0.75)
        if is_candidate and self.pose_model is not None and frame_img is not None:
            pad_x = int(bw * 0.30)
            pad_y = int(bh * 0.30)
            crop_x1 = max(0, int(x1 - pad_x))
            crop_y1 = max(0, int(y1 - pad_y))
            crop_x2 = min(frame_w, int(x2 + pad_x))
            crop_y2 = min(frame_h, int(y2 + pad_y))

            if crop_x2 > crop_x1 + 10 and crop_y2 > crop_y1 + 10:
                crop = frame_img[crop_y1:crop_y2, crop_x1:crop_x2]
                pose_torso_horizontal, pose_upright, pose_confidence = self._verify_pose(crop)

        # 5. Multi-Evidence Integration & Disambiguation
        is_fall_indicator = False
        reasons = []

        if pose_upright:
            # Torso verified upright: sitting on bench/chair, crouching, bending, standing.
            # Upright posture conclusively suppresses fall indicators.
            is_fall_indicator = False
        elif pose_torso_horizontal:
            is_fall_indicator = True
            reasons.append(f"PoseTorsoHorizontal(v={1.0 - pose_confidence:.2f})")
        elif dynamic_drop and (ar >= 0.55 or is_compressed):
            is_fall_indicator = True
            reasons.append(f"CentroidDrop={cy_drop_amount:.2f}")
        elif is_horizontal_bbox and (cur_state.get("has_dynamic_drop", False) or near_floor):
            is_fall_indicator = True
            reasons.append(f"AR={ar:.2f}>={self.ar_confirm_threshold}")
        elif is_compressed and (cur_state.get("has_dynamic_drop", False) or near_floor):
            is_fall_indicator = True
            reasons.append(f"HeightCompressed(h_ratio={h_ratio:.2f})")

        # Ground speed gate: Prone/fallen person cannot run
        if is_fall_indicator and speed_px_sec > self.max_ground_speed * 1.5:
            is_fall_indicator = False

        # 6. State Machine Transitions
        prev_status = cur_state["status"]

        if pose_torso_horizontal:
            cur_state["consecutive_pose_horizontal_frames"] += 1
        else:
            cur_state["consecutive_pose_horizontal_frames"] = 0

        if is_fall_indicator:
            cur_state["consecutive_fall_frames"] += 1
            cur_state["consecutive_upright_frames"] = 0
            if cur_state["first_fall_frame"] is None:
                cur_state["first_fall_frame"] = frame_idx
        else:
            # Recovery / normal posture evidence
            cur_state["consecutive_upright_frames"] += 1
            # Decrement fall evidence gracefully rather than immediately resetting
            cur_state["consecutive_fall_frames"] = max(0, cur_state["consecutive_fall_frames"] - 1)
            if cur_state["consecutive_upright_frames"] >= 8:
                cur_state["has_dynamic_drop"] = False

        fall_count = cur_state["consecutive_fall_frames"]
        upright_count = cur_state["consecutive_upright_frames"]
        pose_h_count = cur_state["consecutive_pose_horizontal_frames"]
        history_len = len(hist) + 1  # including current observation about to be appended
        in_cooldown = cur_state.get("reassign_cooldown", 0) > 0
        ground_speed_ok = speed_px_sec <= self.max_confirm_ground_speed
        has_drop = bool(cur_state.get("has_dynamic_drop", False) or dynamic_drop)

        # Determine next state
        if prev_status == self.STATE_CONFIRMED:
            # Check for confirmed recovery: person must stand upright for sustained frames
            if upright_count >= self.recovery_confirm_frames and ar < 0.60 and h_ratio > 0.80:
                next_status = self.STATE_RECOVERED
                cur_state["tracking_state"] = "RECOVERED"
            else:
                # Remains confirmed as long as prone or not conclusively recovered
                next_status = self.STATE_CONFIRMED
        else:
            # Phase 28: CONFIRM requires stronger evidence than SUSPECT.
            # - Never confirm during track-reassignment cooldown
            # - Prefer observed collapse (dynamic drop) over static horizontal boxes (sitting/bending)
            # - Pose-only confirm needs sustained horizontal torso + near floor + low speed
            # - Geometry-only confirm needs dynamic drop + horizontal + sustained frames
            # Pose path: horizontal torso is strong evidence — do NOT require dynamic drop.
            # Track ID switches often wipe has_dynamic_drop; pose still confirms a real prone body.
            pose_confirm = (
                pose_torso_horizontal
                and pose_h_count >= self.pose_confirm_frames
                and near_floor
                and ground_speed_ok
                and ar >= self.ar_suspect_threshold
                and not pose_upright
            )
            collapse_confirm = (
                has_drop
                and is_horizontal_bbox
                and fall_count >= max(2, self.confirm_frames // 2)
                and ground_speed_ok
                and not pose_upright
            )
            # Geometry-only sustained prone still requires observed collapse (blocks sitting/bending FP)
            sustained_prone_confirm = (
                has_drop
                and is_horizontal_bbox
                and fall_count >= self.confirm_frames
                and near_floor
                and not pose_upright
                and cur_state.get("initial_standing", False)
                and ground_speed_ok
            )
            # Pose-confirmed prone may appear on a short reassigned track; allow shorter history.
            history_ok = history_len >= (
                max(3, self.min_history_for_confirm // 2) if pose_confirm else self.min_history_for_confirm
            )
            is_confirmed = (
                (not in_cooldown)
                and (not is_tracker_jump)
                and history_ok
                and (pose_confirm or collapse_confirm or sustained_prone_confirm)
            )
            if is_confirmed:
                next_status = self.STATE_CONFIRMED
                if cur_state["event_id"] is None:
                    cur_state["event_id"] = self.next_event_id
                    self.next_event_id += 1
                    self.active_fall_events[cur_state["event_id"]] = {
                        "track_id": track_id,
                        "start_frame": frame_idx,
                        "first_evidence_frame": cur_state["first_fall_frame"] or frame_idx,
                        "status": "ACTIVE"
                    }
            elif fall_count >= 1 or pose_torso_horizontal:
                # SUSPECTED: early warning — may fire without full confirmation evidence
                next_status = self.STATE_SUSPECTED
            else:
                next_status = self.STATE_NORMAL

        confidence = 0.0
        if next_status == self.STATE_CONFIRMED:
            confidence = min(0.98, 0.75 + 0.02 * fall_count + (0.15 if pose_torso_horizontal else 0.0))
        elif next_status == self.STATE_SUSPECTED:
            confidence = min(0.70, 0.45 + 0.05 * fall_count)

        cur_state["status"] = next_status
        cur_state["fall_score"] = confidence
        cur_state["reason"] = "+".join(reasons)

        hist.append((frame_idx, bbox, cy, ar, bh))

        return {
            "track_id": track_id,
            "status": next_status,
            "event_id": cur_state["event_id"],
            "tracking_state": cur_state["tracking_state"],
            "is_live_observation": True,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "consecutive_fall_frames": fall_count,
            "aspect_ratio": round(ar, 2),
            "height_ratio": round(h_ratio, 2),
            "torso_horizontal": pose_torso_horizontal,
            "centroid_drop": round(cy_drop_amount, 2),
            "bbox": list(bbox)
        }

    def process_frame_persistence(
        self,
        frame_idx: int,
        observed_track_ids: List[int],
        raw_candidate_boxes: Optional[List[List[float]]] = None,
        frame_img: Optional[np.ndarray] = None,
        frame_h: int = 1080,
        frame_w: int = 1920
    ) -> List[Dict[str, Any]]:
        """
        Post-frame hook ensuring persistence for active falls:
        1. Attempts spatial re-association for active fall tracks missing from the current tracker output.
        2. Maintains COASTING state without fabricating live detections if candidate is temporarily obscured.
        3. Enforces timeout to close permanently lost falls.
        Returns:
            List of diagnostic telemetry records for persisting / coasting fall tracks.
        """
        coasting_outputs = []
        raw_candidates = raw_candidate_boxes or []
        used_raw_indices = set()

        for tid, state in list(self.fall_state.items()):
            # Only manage tracks that were in FALL_CONFIRMED or FALL_SUSPECTED
            if state["status"] not in (self.STATE_CONFIRMED, self.STATE_SUSPECTED):
                continue

            if tid in observed_track_ids:
                # Already observed in this frame
                continue

            # Track was NOT observed in this frame.
            last_bbox = state.get("last_known_bbox")
            if not last_bbox:
                continue

            # 1. Spatial Re-Association with untracked low-confidence horizontal candidates
            reassoc_idx = -1
            best_dist = 9999.0

            lcx = (last_bbox[0] + last_bbox[2]) / 2.0
            lcy = (last_bbox[1] + last_bbox[3]) / 2.0

            for r_idx, r_box in enumerate(raw_candidates):
                if r_idx in used_raw_indices:
                    continue
                rcx = (r_box[0] + r_box[2]) / 2.0
                rcy = (r_box[1] + r_box[3]) / 2.0
                dist = np.sqrt((rcx - lcx) ** 2 + (rcy - lcy) ** 2)

                rw = r_box[2] - r_box[0]
                rh = r_box[3] - r_box[1]
                rar = rw / max(1.0, rh)

                # Low-confidence horizontal box in close spatial vicinity
                if dist <= self.spatial_reassoc_dist and rar >= 0.70 and dist < best_dist:
                    best_dist = dist
                    reassoc_idx = r_idx

            if reassoc_idx >= 0:
                # Re-associated!
                used_raw_indices.add(reassoc_idx)
                matched_box = raw_candidates[reassoc_idx]
                eval_res = self.evaluate_pedestrian(
                    track_id=tid,
                    bbox=tuple(matched_box),
                    frame_idx=frame_idx,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    frame_img=frame_img
                )
                eval_res["tracking_state"] = "SPATIAL_REASSOCIATED"
                eval_res["reassoc_dist_px"] = round(best_dist, 1)
                coasting_outputs.append(eval_res)
            else:
                # No physical detection available -> Maintain Coasting Grace Period
                state["coasting_frames"] += 1
                c_frames = state["coasting_frames"]

                if c_frames <= self.coast_frames_max:
                    state["tracking_state"] = "LOST_COASTING"
                    state["is_live_observation"] = False
                    coasting_outputs.append({
                        "track_id": tid,
                        "status": state["status"],  # Remains FALL_CONFIRMED or SUSPECTED
                        "event_id": state["event_id"],
                        "tracking_state": "LOST_COASTING",
                        "is_live_observation": False,
                        "coasting_frames": c_frames,
                        "max_coasting_frames": self.coast_frames_max,
                        "confidence": max(0.50, state["fall_score"] * 0.95),
                        "reasons": [f"CoastingPersistence({c_frames}/{self.coast_frames_max})"],
                        "last_known_bbox": state["last_known_bbox"],
                        "bbox": state["last_known_bbox"]  # Telemetry anchor, flagged non-live
                    })
                else:
                    # Timeout reached
                    state["status"] = self.STATE_LOST_TIMEOUT
                    state["tracking_state"] = "TIMEOUT"
                    if state["event_id"] in self.active_fall_events:
                        self.active_fall_events[state["event_id"]]["status"] = "CLOSED_TIMEOUT"

        return coasting_outputs
