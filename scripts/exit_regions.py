#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exit Region Management
======================
Manages configurable exit doorway / egress bottleneck regions.
When a pedestrian approaches a configured exit region and orients orthogonally
towards the exit boundary, exit-aware classification permits the egress turn
rather than misclassifying the boundary maneuver as counter-flow WRONG_DIRECTION.
Turns away from the exit boundary remain strictly evaluated as anomalies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from direction_estimation import angular_deviation_deg
except ImportError:
    from scripts.direction_estimation import angular_deviation_deg


class ExitRegionManager:
    """
    Manages configurable exit doorway / egress bottleneck regions.
    Supports camera-specific exit polygons and approach trajectory verification.
    """
    def __init__(self, exit_regions_cfg: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None):
        self.exit_regions: List[Dict[str, Any]] = []
        self.camera_map: Dict[str, List[Dict[str, Any]]] = {}
        self.enable_approach_check: bool = True

        raw_list: List[Dict[str, Any]] = []
        if isinstance(exit_regions_cfg, dict):
            if "enable_approach_check" in exit_regions_cfg:
                self.enable_approach_check = bool(exit_regions_cfg["enable_approach_check"])
            # Dict mapping camera_id -> list of regions or single region dict
            for cam_key, val in exit_regions_cfg.items():
                if isinstance(val, list):
                    for item in val:
                        item_copy = dict(item)
                        if "camera_id" not in item_copy:
                            item_copy["camera_id"] = cam_key
                        raw_list.append(item_copy)
                elif isinstance(val, dict):
                    item_copy = dict(val)
                    if "camera_id" not in item_copy:
                        item_copy["camera_id"] = cam_key
                    raw_list.append(item_copy)
        elif isinstance(exit_regions_cfg, list):
            raw_list = list(exit_regions_cfg)

        for r in raw_list:
            pts = r.get("polygon", [])
            if len(pts) >= 3:
                poly = None
                if SHAPELY_AVAILABLE:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                elif cv2 is not None:
                    poly = np.array(pts, dtype=np.float32)

                cx = float(sum(p[0] for p in pts) / len(pts))
                cy = float(sum(p[1] for p in pts) / len(pts))

                cam_id = r.get("camera_id")
                entry = {
                    "id": r.get("id", "exit_01"),
                    "name": r.get("name", "Exit"),
                    "camera_id": str(cam_id) if cam_id is not None else None,
                    "polygon": pts,
                    "poly": poly,
                    "centroid": (cx, cy),
                    "allowed_exit_directions": [str(d).upper() for d in r.get("allowed_exit_directions", [])]
                }
                self.exit_regions.append(entry)

    def _get_applicable_regions(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.exit_regions:
            return []
        if camera_id is None:
            return self.exit_regions
        cam_str = str(camera_id)
        # Match camera_id or global regions (camera_id is None)
        matched = [r for r in self.exit_regions if r["camera_id"] is None or r["camera_id"] == cam_str]
        return matched if matched else self.exit_regions

    def is_in_exit_region(
        self, x: float, y: float, camera_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        regions = self._get_applicable_regions(camera_id)
        pt = Point(x, y) if SHAPELY_AVAILABLE else None
        for reg in regions:
            poly = reg["poly"]
            if poly is None:
                continue
            if SHAPELY_AVAILABLE and pt is not None:
                if poly.contains(pt) or poly.touches(pt):
                    return reg
            elif cv2 is not None:
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    return reg
        return None

    def is_exit_turn_permitted(
        self,
        x: float,
        y: float,
        actual_dir: str,
        expected_dir: str,
        movement_vector: Optional[Tuple[float, float]] = None,
        camera_id: Optional[str] = None,
    ) -> bool:
        """
        Verifies if an egress turn is legitimate:
        1. Pedestrian must be inside or touching an exit region.
        2. If allowed_exit_directions are configured, actual_dir must match.
        3. If movement_vector is provided, heading must point towards exit centroid/boundary.
        4. Direction deviation from expected flow must be orthogonal (<= 105 deg), not reverse.
        """
        reg = self.is_in_exit_region(x, y, camera_id=camera_id)
        if not reg:
            return False

        allowed = reg.get("allowed_exit_directions")
        if allowed:
            return actual_dir in allowed

        # Check angular deviation: allow orthogonal egress (up to 105 deg)
        dev = angular_deviation_deg(actual_dir, expected_dir)
        if dev is None or dev > 105.0:
            return False

        # Approach verification: ensure movement vector points towards exit centroid
        if self.enable_approach_check and movement_vector is not None:
            dx, dy = float(movement_vector[0]), float(movement_vector[1])
            mag = math.hypot(dx, dy)
            if mag > 1e-3:
                cx, cy = reg["centroid"]
                to_cx = cx - x
                to_cy = cy - y
                to_dist = math.hypot(to_cx, to_cy)
                if to_dist > 5.0:
                    # Dot product between velocity vector and vector toward centroid
                    dot = (dx * to_cx + dy * to_cy) / (mag * to_dist)
                    # If dot < -0.2, pedestrian is moving clearly away from the exit
                    if dot < -0.2:
                        return False

        return True
