#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extraction Module — Trajectory and Movement Characteristics
====================================================================
Computes:
  - Trajectory history (bottom-center foot position)
  - Relative / Pixel speed (px/frame and px/sec)
  - Dominant movement direction (LEFT, RIGHT, UP, DOWN, STATIONARY)
  - Prolonged stopping duration (consecutive low-speed frames)
  - Spatial zone membership (RESTRICTED_ZONE, EVACUATION_ROUTE, SAFE_ZONE, EXIT_ZONE, NORMAL_ZONE)
  - Export functionality to outputs/trajectories/
"""

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class ZoneManager:
    """Manages polygonal zones for spatial analysis."""

    def __init__(self, zones_config: Dict[str, Any]):
        self.polygons: Dict[str, List[Tuple[str, Any]]] = {
            "RESTRICTED_ZONE": [],
            "EVACUATION_ROUTE": [],
            "SAFE_ZONE": [],
            "EXIT_ZONE": [],
        }
        self.raw_configs = zones_config
        self._build_polygons(zones_config)

    def _build_polygons(self, zones_config: Dict[str, Any]):
        mapping = {
            "restricted_zones": "RESTRICTED_ZONE",
            "evacuation_routes": "EVACUATION_ROUTE",
            "safe_zones": "SAFE_ZONE",
            "exit_zones": "EXIT_ZONE",
        }
        for cfg_key, zone_type in mapping.items():
            zone_list = zones_config.get(cfg_key, [])
            for zone_info in zone_list:
                pts = zone_info.get("polygon", [])
                name = zone_info.get("name", zone_type)
                if len(pts) >= 3:
                    if SHAPELY_AVAILABLE:
                        poly = Polygon(pts)
                    else:
                        poly = np.array(pts, dtype=np.int32)
                    self.polygons[zone_type].append((name, poly))

        # Also support legacy zone_selector format: {"zones": [{"zone_name": ..., "polygon": ...}]}
        if "zones" in zones_config and not any(k in zones_config for k in mapping.keys()):
            for z in zones_config.get("zones", []):
                pts = z.get("polygon", [])
                name = z.get("zone_name", "Restricted Zone")
                if len(pts) >= 3:
                    if SHAPELY_AVAILABLE:
                        poly = Polygon(pts)
                    else:
                        poly = np.array(pts, dtype=np.int32)
                    self.polygons["RESTRICTED_ZONE"].append((name, poly))

    def get_zone_at_point(self, x: float, y: float) -> Tuple[str, str]:
        """Returns (zone_type, zone_name) for a given foot coordinate (x, y)."""
        # Prioritize restricted zones first, then exits, safe zones, and evacuation routes
        priority_order = ["RESTRICTED_ZONE", "EXIT_ZONE", "SAFE_ZONE", "EVACUATION_ROUTE"]
        
        for zone_type in priority_order:
            for name, poly in self.polygons[zone_type]:
                if SHAPELY_AVAILABLE:
                    if poly.contains(Point(x, y)):
                        return zone_type, name
                else:
                    res = cv2.pointPolygonTest(poly, (float(x), float(y)), False) # type: ignore
                    if res >= 0:
                        return zone_type, name
        return "NORMAL_ZONE", "Normal Area"


class TrajectoryFeatureExtractor:
    """Extracts movement features and trajectories for tracked pedestrians."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_history = min(20, max(10, int(config.get("trajectory_history_max", 20))))
        self.max_displacement_px = float(config.get("trajectory_max_displacement_px", 65.0))
        self.max_frame_gap = int(config.get("trajectory_max_frame_gap", 10))
        self.min_direction_history = int(config.get("trajectory_min_history_for_direction", 8))
        self.stationary_thresh = float(config.get("stationary_speed_threshold", 2.5))
        self.direction_window = int(config.get("direction_window_frames", 10))
        
        # Per-pedestrian trajectory history: {track_id: deque([(x, y, frame, timestamp), ...])}
        self.histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        # Stopping counter per pedestrian: {track_id: int}
        self.stopping_counters: Dict[int, int] = defaultdict(int)
        
        # Zone manager
        self.zone_manager = ZoneManager(config.get("zones", {}))
        
        # Export records
        self.export_records: List[Dict[str, Any]] = []

    def reset(self):
        """Reset internal trajectory histories, counters, and export records."""
        self.histories.clear()
        self.stopping_counters.clear()
        self.export_records.clear()

    def update_pedestrian(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Update trajectory for a single pedestrian detection box [x1, y1, x2, y2].
        Returns computed feature dictionary.
        """
        x1, y1, x2, y2 = bbox
        foot_x = float((x1 + x2) / 2.0)
        foot_y = float(y2)
        center_x = float((x1 + x2) / 2.0)
        center_y = float((y1 + y2) / 2.0)
        timestamp = float(frame_idx / max(fps, 1.0))

        history = self.histories[track_id]

        # Break and reset trail if this track has a large frame gap or implausible jump
        if len(history) > 0:
            prev_x, prev_y, prev_frame, _ = history[-1]
            frame_gap = frame_idx - prev_frame
            displacement = float(np.hypot(foot_x - prev_x, foot_y - prev_y))
            if frame_gap > self.max_frame_gap or displacement > self.max_displacement_px:
                # The track was lost and re-acquired elsewhere or reassigned.
                # Clear stale history so disconnected points are never joined across the map.
                history.clear()
                self.stopping_counters[track_id] = 0

        history.append((foot_x, foot_y, frame_idx, timestamp))

        # Calculate instantaneous speed (px/frame and px/sec)
        if len(history) >= 2:
            prev_x, prev_y, prev_frame, _ = history[-2]
            dist = float(np.hypot(foot_x - prev_x, foot_y - prev_y))
            frame_diff = max(1, frame_idx - prev_frame)
            px_speed_per_frame = dist / frame_diff
            px_speed_per_sec = px_speed_per_frame * fps
        else:
            px_speed_per_frame = 0.0
            px_speed_per_sec = 0.0

        # Calculate rolling window smoothed speed (reduces detector bbox jitter)
        win = min(len(history), 8)
        if win >= 2:
            start_x, start_y, start_frame, _ = history[-win]
            win_dist = float(np.hypot(foot_x - start_x, foot_y - start_y))
            win_frames = max(1, frame_idx - start_frame)
            smoothed_speed_per_frame = win_dist / win_frames
            smoothed_speed_per_sec = smoothed_speed_per_frame * fps
        else:
            smoothed_speed_per_frame = px_speed_per_frame
            smoothed_speed_per_sec = px_speed_per_sec

        # Calculate movement direction over window from valid consecutive observations
        direction = self._compute_direction(history)

        # Update stopping duration:
        # A pedestrian is stationary if smoothed speed is below stationary threshold
        # AND direction is STATIONARY (or net displacement over window is negligible).
        # Actively moving pedestrians traversing in a direction reset the counter.
        is_stationary = (smoothed_speed_per_frame < self.stationary_thresh) and (direction == "STATIONARY")
        if is_stationary:
            self.stopping_counters[track_id] += 1
        else:
            self.stopping_counters[track_id] = 0
        stopping_duration = self.stopping_counters[track_id]

        # Determine spatial zone
        zone_type, zone_name = self.zone_manager.get_zone_at_point(foot_x, foot_y)

        features = {
            "track_id": int(track_id),
            "frame": int(frame_idx),
            "timestamp": round(timestamp, 2),
            "bbox": [round(float(v), 1) for v in bbox],
            "center": [round(center_x, 1), round(center_y, 1)],
            "foot": [round(foot_x, 1), round(foot_y, 1)],
            "speed_px_frame": round(px_speed_per_frame, 2),
            "speed_px_sec": round(smoothed_speed_per_sec, 2),
            "instant_speed_px_sec": round(px_speed_per_sec, 2),
            "direction": direction,
            "stopping_duration_frames": stopping_duration,
            "stopping_duration_sec": round(stopping_duration / max(fps, 1.0), 2),
            "zone_type": zone_type,
            "zone_name": zone_name,
            "trajectory": [
                [round(pt[0], 1), round(pt[1], 1), int(pt[2]), round(pt[3], 2)]
                for pt in list(history)
            ]
        }

        # Save record for export
        self.export_records.append({
            "track_id": int(track_id),
            "frame": int(frame_idx),
            "timestamp": round(timestamp, 2),
            "x": round(foot_x, 1),
            "y": round(foot_y, 1),
            "bbox": [round(float(v), 1) for v in bbox],
            "speed": round(px_speed_per_sec, 2),
            "direction": direction,
            "stopping_frames": stopping_duration,
            "zone": zone_type
        })

        return features

    def _compute_direction(self, history: deque) -> str:
        """Computes cardinal direction over recent history window from consecutive valid observations."""
        if len(history) < self.min_direction_history:
            return "STATIONARY"

        # Compare current point against N frames back
        window = min(len(history), self.direction_window)
        recent_pts = list(history)[-window:]

        # Verify valid continuous observations without large gaps
        for i in range(1, len(recent_pts)):
            if (recent_pts[i][2] - recent_pts[i-1][2]) > 3:
                return "STATIONARY"

        start_x, start_y, _, _ = recent_pts[0]
        curr_x, curr_y, _, _ = recent_pts[-1]

        dx = curr_x - start_x
        dy = curr_y - start_y
        dist = float(np.hypot(dx, dy))

        if dist < (self.stationary_thresh * window * 0.5):
            return "STATIONARY"

        if abs(dx) >= abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"

    def export_trajectories(self, output_path: Path):
        """Export accumulated trajectory records to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.export_records, f, indent=2, ensure_ascii=False)
        print(f"[Extractor] Trajectory data exported ({len(self.export_records)} records) -> {output_path}")
