#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extraction — trajectories, 8-way direction, evacuation corridor
=======================================================================
Only one spatial construct: the main evacuation corridor (+ expected flow).
Obsolete zone types (restricted / exit / monitoring / safe) are rejected.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from shapely.geometry import Point, Polygon, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from direction_estimation import (
        DirectionEstimator,
        normalize_direction_label,
        direction_unit_vector,
        angle_to_compass8,
        angular_deviation_deg,
        vector_angular_deviation_deg,
        movement_angle_deg,
        CARDINAL_8,
        COMPASS_VECTORS,
    )
except ImportError:
    from scripts.direction_estimation import (
        DirectionEstimator,
        normalize_direction_label,
        direction_unit_vector,
        angle_to_compass8,
        angular_deviation_deg,
        vector_angular_deviation_deg,
        movement_angle_deg,
        CARDINAL_8,
        COMPASS_VECTORS,
    )

OBSOLETE_ZONE_KEYS = {
    "restricted_zones", "exit_zones", "monitoring_zones", "safe_zones",
    "entry_zones", "emergency_exit_zones",
}


def apply_homography(pt: Tuple[float, float], H: Optional[np.ndarray]) -> Tuple[float, float]:
    """Optional ground-plane transform. Returns input if H is None."""
    if H is None:
        return pt
    x, y = pt
    v = np.array([x, y, 1.0], dtype=np.float64)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return pt
    return float(w[0] / w[2]), float(w[1] / w[2])


class EvacuationCorridor:
    """Single main evacuation corridor + expected flow direction."""

    def __init__(self, config: Dict[str, Any]):
        self.polygon_pts: List[List[float]] = []
        self.poly = None
        self.flow_line: List[List[float]] = []
        self.expected_direction: Optional[str] = None
        self.name = "Main Evacuation Corridor"
        self.corridor_id = "corridor_main"
        self.homography: Optional[np.ndarray] = None
        self._load(config)

    def _load(self, config: Dict[str, Any]):
        zones = config.get("zones") or {}
        corr = config.get("evacuation_corridor")
        if corr is None and isinstance(zones, dict):
            # Prefer new key; fall back to first evacuation_routes entry
            if zones.get("evacuation_corridor"):
                corr = zones["evacuation_corridor"]
                if isinstance(corr, list):
                    corr = corr[0] if corr else None
            elif zones.get("evacuation_routes"):
                corr = (zones.get("evacuation_routes") or [None])[0]

        if not corr:
            # Global expected direction still usable without geometry
            ed = config.get("expected_evacuation_direction")
            self.expected_direction = normalize_direction_label(ed) if ed else None
            return

        self.name = corr.get("name", self.name)
        self.corridor_id = corr.get("id", self.corridor_id)
        self.polygon_pts = list(corr.get("polygon") or [])
        self.flow_line = list(corr.get("flow_line") or [])
        ed = corr.get("expected_direction") or config.get("expected_evacuation_direction")
        self.expected_direction = normalize_direction_label(ed) if ed else None

        H = corr.get("homography") or config.get("homography")
        if H is not None:
            try:
                self.homography = np.asarray(H, dtype=np.float64).reshape(3, 3)
            except Exception:
                self.homography = None

        if len(self.polygon_pts) >= 3:
            if SHAPELY_AVAILABLE:
                self.poly = Polygon(self.polygon_pts)
                if not self.poly.is_valid:
                    self.poly = self.poly.buffer(0)
            elif cv2 is not None:
                self.poly = np.array(self.polygon_pts, dtype=np.float32)

        # Derive flow line from polygon mid-line if missing
        if not self.flow_line and len(self.polygon_pts) >= 4:
            xs = [p[0] for p in self.polygon_pts]
            ys = [p[1] for p in self.polygon_pts]
            mid_y = (min(ys) + max(ys)) / 2.0
            self.flow_line = [[min(xs), mid_y], [max(xs), mid_y]]

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        zones = (config.get("zones") or {}) if isinstance(config.get("zones"), dict) else {}
        for key in OBSOLETE_ZONE_KEYS:
            if zones.get(key):
                return False, f"Obsolete zone category removed: {key}"
        corr = config.get("evacuation_corridor") or zones.get("evacuation_corridor")
        if isinstance(corr, list):
            corr = corr[0] if corr else None
        if corr:
            pts = corr.get("polygon") or []
            if pts and len(pts) < 3:
                return False, "Evacuation corridor polygon needs ≥3 vertices"
            if pts and SHAPELY_AVAILABLE:
                poly = Polygon(pts)
                if not poly.is_valid or poly.area <= 0:
                    return False, "Invalid evacuation corridor geometry"
        return True, "ok"

    def contains(self, x: float, y: float) -> bool:
        if self.poly is None:
            return True  # no geometry → treat all points as in corridor for flow rules
        if SHAPELY_AVAILABLE:
            return bool(self.poly.contains(Point(x, y)) or self.poly.touches(Point(x, y)))
        if cv2 is not None:
            return cv2.pointPolygonTest(self.poly, (float(x), float(y)), False) >= 0
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.corridor_id,
            "name": self.name,
            "polygon": self.polygon_pts,
            "flow_line": self.flow_line,
            "expected_direction": self.expected_direction,
        }


# Backward-compatible alias used by older imports / save_zones validation
class ZoneManager:
    """Compatibility shim — only evacuation corridor is accepted."""

    def __init__(self, zones_config: Dict[str, Any]):
        cfg = {"zones": zones_config} if zones_config else {}
        # Lift corridor from zones.evacuation_routes if present
        if zones_config and zones_config.get("evacuation_routes") and not zones_config.get("evacuation_corridor"):
            cfg["evacuation_corridor"] = zones_config["evacuation_routes"][0]
        self.corridor = EvacuationCorridor(cfg)
        self.raw_configs = zones_config or {}

    @staticmethod
    def validate_zone_dict(zones_config: Dict[str, Any]) -> Tuple[bool, str]:
        return EvacuationCorridor.validate_config({"zones": zones_config or {}})

    def get_zone_at_point(self, x: float, y: float) -> Tuple[str, str, Dict[str, Any]]:
        if self.corridor.contains(x, y):
            return (
                "EVACUATION_CORRIDOR",
                self.corridor.name,
                {
                    "id": self.corridor.corridor_id,
                    "expected_direction": self.corridor.expected_direction,
                },
            )
        return "OUTSIDE_CORRIDOR", "Outside Corridor", {}


class TrajectoryFeatureExtractor:
    """Foot-point trajectories + DirectionEstimator (image Y-down convention)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_history = max(10, min(40, int(config.get("trajectory_history_max", 24))))
        self.dir_estimator = DirectionEstimator(config)
        self.stopping_counters: Dict[int, int] = defaultdict(int)
        self.stop_start_timestamps: Dict[int, float] = {}
        self.moving_streaks: Dict[int, int] = defaultdict(int)
        self.histories = self.dir_estimator.raw_hist  # shared for twin trails / tests
        self.corridor = EvacuationCorridor(config)
        self.zone_manager = ZoneManager(config.get("zones") or {})
        if config.get("evacuation_corridor") or (config.get("zones") or {}).get("evacuation_routes"):
            self.zone_manager.corridor = self.corridor
        self.export_records: List[Dict[str, Any]] = []
        self.track_obs_count: Dict[int, int] = defaultdict(int)
        self.track_stationary_count: Dict[int, int] = defaultdict(int)
        self.track_start_pos: Dict[int, Tuple[float, float]] = {}
        self.track_last_pos: Dict[int, Tuple[float, float]] = {}

    def reset(self):
        self.dir_estimator.reset()
        self.stopping_counters.clear()
        self.stop_start_timestamps.clear()
        self.moving_streaks.clear()
        self.export_records.clear()
        self.track_obs_count.clear()
        self.track_stationary_count.clear()
        self.track_start_pos.clear()
        self.track_last_pos.clear()

    def expire_stale_tracks(self, frame_idx: int):
        self.dir_estimator.expire_stale(frame_idx)

    def update_pedestrian(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        fps: float = 30.0,
        conf: Optional[float] = None,
        is_detection: bool = True,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = bbox
        raw_fx = float((x1 + x2) / 2.0)
        raw_fy = float(y2)
        center_x = float((x1 + x2) / 2.0)
        center_y = float((y1 + y2) / 2.0)
        timestamp = float(frame_idx / max(fps, 1.0))

        ground_x, ground_y = apply_homography((raw_fx, raw_fy), self.corridor.homography)
        dir_info = self.dir_estimator.update(
            track_id, (ground_x, ground_y), frame_idx, timestamp, conf=conf, is_detection=is_detection
        )

        direction = dir_info["direction"]
        # Increment stopping counter for STATIONARY *and* for UNKNOWN when the
        # pedestrian has barely moved (displacement < min threshold). This ensures
        # prolonged-stopping is correctly detected for slow-moving / uncertain tracks
        # that never produce a clean STATIONARY label.
        displacement_px = float(dir_info.get("displacement_px", 0.0))
        min_disp = float(self.dir_estimator.min_disp_px)
        is_stopped = (direction == "STATIONARY") or (
            direction == "UNKNOWN" and displacement_px < min_disp
        )
        if is_stopped:
            self.stopping_counters[track_id] += 1
            self.moving_streaks[track_id] = 0
            if track_id not in self.stop_start_timestamps:
                self.stop_start_timestamps[track_id] = timestamp
            stop_elapsed_sec = max(0.0, timestamp - self.stop_start_timestamps[track_id])
            stopping_sec = max(stop_elapsed_sec, self.stopping_counters[track_id] / max(fps, 1.0))
            stopping_duration = max(self.stopping_counters[track_id], int(round(stopping_sec * fps)))
        else:
            self.moving_streaks[track_id] += 1
            # Hysteresis: only reset stopping accumulation if moving genuinely for >= 5 consecutive frames
            if self.moving_streaks[track_id] >= 5:
                self.stopping_counters[track_id] = 0
                self.stop_start_timestamps.pop(track_id, None)
                stopping_duration = 0
                stopping_sec = 0.0
            else:
                # Retain accumulated stopping duration during brief 1-4 frame jitter
                stop_elapsed_sec = max(0.0, timestamp - self.stop_start_timestamps.get(track_id, timestamp))
                stopping_sec = max(stop_elapsed_sec, self.stopping_counters[track_id] / max(fps, 1.0))
                stopping_duration = max(self.stopping_counters[track_id], int(round(stopping_sec * fps)))

        in_corridor = self.corridor.contains(raw_fx, raw_fy)
        zone_type = "EVACUATION_CORRIDOR" if in_corridor else "OUTSIDE_CORRIDOR"
        zone_name = self.corridor.name if in_corridor else "Outside Corridor"
        history = list(self.dir_estimator.raw_hist[track_id])
        ema = self.dir_estimator.ema.get(track_id, (raw_fx, raw_fy))

        self.track_obs_count[track_id] += 1
        if track_id not in self.track_start_pos:
            self.track_start_pos[track_id] = (raw_fx, raw_fy)
        self.track_last_pos[track_id] = (raw_fx, raw_fy)

        if dir_info.get("movement_status") == "STATIONARY":
            self.track_stationary_count[track_id] += 1

        total_obs = self.track_obs_count[track_id]
        stat_obs = self.track_stationary_count[track_id]
        stat_ratio = stat_obs / max(1, total_obs)
        start_x, start_y = self.track_start_pos[track_id]
        net_disp_px = math.hypot(raw_fx - start_x, raw_fy - start_y)

        features = {
            "track_id": int(track_id),
            "frame": int(frame_idx),
            "timestamp": round(timestamp, 2),
            "bbox": [round(float(v), 1) for v in bbox],
            "center": [round(center_x, 1), round(center_y, 1)],
            "foot": [round(raw_fx, 1), round(raw_fy, 1)],
            "foot_smoothed": [round(ema[0], 1), round(ema[1], 1)],
            "speed_px_frame": round(float(dir_info["speed_px_sec"]) / max(fps, 1.0), 2),
            "speed_px_sec": round(float(dir_info["speed_px_sec"]), 2),
            "instant_speed_px_sec": round(float(dir_info["speed_px_sec"]), 2),
            "direction": direction,
            "direction_angle_deg": dir_info.get("direction_angle_deg"),
            "direction_vector": dir_info.get("direction_vector", [0.0, 0.0]),
            "direction_confidence": dir_info.get("direction_confidence", 0.0),
            "direction_reliable": bool(dir_info.get("direction_reliable", False)),
            "stopping_duration_frames": stopping_duration,
            "stopping_duration_sec": round(stopping_duration / max(fps, 1.0), 2),
            "in_corridor": in_corridor,
            "zone_type": zone_type,
            "zone_name": zone_name,
            "zone_id": self.corridor.corridor_id,
            # Admin live expected direction is applied by anomaly engine — do not hardcode corridor
            "zone_expected_direction": None,
            "trajectory": [
                [round(pt[0], 1), round(pt[1], 1), int(pt[2]), round(pt[3], 2)]
                for pt in history
            ],
            "movement_status": dir_info.get("movement_status", "UNKNOWN"),
            "observation_count": dir_info.get("observation_count", 0),
            "displacement_px": dir_info.get("displacement_px", 0.0),
            "total_observations": total_obs,
            "stationary_observations": stat_obs,
            "stationary_ratio": round(stat_ratio, 3),
            "net_displacement_px": round(net_disp_px, 2),
            "start_foot": (round(start_x, 1), round(start_y, 1)),
            "long_direction_vector": dir_info.get("long_direction_vector", dir_info.get("direction_vector", [0.0, 0.0])),
            "curvature_deg": dir_info.get("curvature_deg", 0.0),
            "is_curved_trajectory": bool(dir_info.get("is_curved_trajectory", False)),
        }

        self.export_records.append({
            "track_id": int(track_id),
            "frame": int(frame_idx),
            "timestamp": round(timestamp, 2),
            "x": round(raw_fx, 1),
            "y": round(raw_fy, 1),
            "direction": direction,
            "in_corridor": in_corridor,
        })
        if frame_idx % 30 == 0:
            self.expire_stale_tracks(frame_idx)
        return features

    def export_trajectories(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.export_records, f, indent=2, ensure_ascii=False)
        print(f"[Extractor] Trajectory data exported ({len(self.export_records)} records) -> {output_path}")
