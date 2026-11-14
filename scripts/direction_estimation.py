#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direction estimation & 8-sector circular classification
=======================================================
Image coordinates: +X right, +Y down.
dx = current_x - previous_x, dy = current_y - previous_y.
Heading: atan2(-dy, dx) → 0°=RIGHT, 90°=UP (image Y-down).

Admin selector (ONLY): LEFT, RIGHT, UP, DOWN.
Internal detection: all eight movement sectors.

Circular sector order (clockwise from RIGHT):
  RIGHT → DOWN_RIGHT → DOWN → DOWN_LEFT → LEFT → UP_LEFT → UP → UP_RIGHT

Classification (exact 3-of-8 vs selected cardinal):
  circular_sector_distance ∈ {0, 1} → NORMAL
  circular_sector_distance ∈ {2, 3, 4} → ANOMALY (WRONG_DIRECTION after sustain)
  insufficient / stationary / unreliable → UNKNOWN (never red)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# Bumped whenever live classification semantics change — logged at server startup
CLASSIFIER_VERSION = "CARDINAL4_SECTOR8_V4_2026-11-08"

# Admin-selectable evacuation directions only
ADMIN_CARDINALS: Tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN")
ADMIN_CARDINAL_SET = frozenset(ADMIN_CARDINALS)

# Circular 8-way ring (clockwise from RIGHT) — matches product spec
DIRECTIONS_8: Tuple[str, ...] = (
    "RIGHT",
    "DOWN_RIGHT",
    "DOWN",
    "DOWN_LEFT",
    "LEFT",
    "UP_LEFT",
    "UP",
    "UP_RIGHT",
)
CARDINAL_8 = DIRECTIONS_8  # backward-compat alias

# Legacy compass / hyphen aliases → product labels
DIR_ALIASES = {
    "RIGHT": "RIGHT",
    "LEFT": "LEFT",
    "UP": "UP",
    "DOWN": "DOWN",
    "UP_RIGHT": "UP_RIGHT",
    "UP-RIGHT": "UP_RIGHT",
    "UPRIGHT": "UP_RIGHT",
    "UP_LEFT": "UP_LEFT",
    "UP-LEFT": "UP_LEFT",
    "UPLEFT": "UP_LEFT",
    "DOWN_RIGHT": "DOWN_RIGHT",
    "DOWN-RIGHT": "DOWN_RIGHT",
    "DOWNRIGHT": "DOWN_RIGHT",
    "DOWN_LEFT": "DOWN_LEFT",
    "DOWN-LEFT": "DOWN_LEFT",
    "DOWNLEFT": "DOWN_LEFT",
    # Compass names
    "EAST": "RIGHT",
    "WEST": "LEFT",
    "NORTH": "UP",
    "SOUTH": "DOWN",
    "NORTHEAST": "UP_RIGHT",
    "NORTH-EAST": "UP_RIGHT",
    "NORTH_EAST": "UP_RIGHT",
    "NE": "UP_RIGHT",
    "NORTHWEST": "UP_LEFT",
    "NORTH-WEST": "UP_LEFT",
    "NORTH_WEST": "UP_LEFT",
    "NW": "UP_LEFT",
    "SOUTHEAST": "DOWN_RIGHT",
    "SOUTH-EAST": "DOWN_RIGHT",
    "SOUTH_EAST": "DOWN_RIGHT",
    "SE": "DOWN_RIGHT",
    "SOUTHWEST": "DOWN_LEFT",
    "SOUTH-WEST": "DOWN_LEFT",
    "SOUTH_WEST": "DOWN_LEFT",
    "SW": "DOWN_LEFT",
}

# Image-space unit vectors (+Y down) keyed by product labels
DIRECTION_VECTORS = {
    "RIGHT": (1.0, 0.0),
    "UP_RIGHT": (math.sqrt(0.5), -math.sqrt(0.5)),
    "UP": (0.0, -1.0),
    "UP_LEFT": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "LEFT": (-1.0, 0.0),
    "DOWN_LEFT": (-math.sqrt(0.5), math.sqrt(0.5)),
    "DOWN": (0.0, 1.0),
    "DOWN_RIGHT": (math.sqrt(0.5), math.sqrt(0.5)),
}
COMPASS_VECTORS = DIRECTION_VECTORS  # backward-compat

_DIR_INDEX = {name: i for i, name in enumerate(DIRECTIONS_8)}


def normalize_direction_label(label: str) -> str:
    """Map any alias to a product 8-way label, or UNKNOWN/STATIONARY."""
    if label is None:
        return "UNKNOWN"
    u = str(label).upper().strip().replace(" ", "_").replace("-", "_")
    if u in ("UNKNOWN", "STATIONARY", ""):
        return u if u else "UNKNOWN"
    for key in (u, u.replace("_", "-"), u.replace("_", "")):
        if key in DIR_ALIASES:
            return DIR_ALIASES[key]
        if key in DIRECTION_VECTORS:
            return key
    compact = u.replace("_", "")
    if compact in DIR_ALIASES:
        return DIR_ALIASES[compact]
    return "UNKNOWN"


def coerce_admin_direction(label: str) -> str:
    """Map admin/UI/config input to one of LEFT/RIGHT/UP/DOWN only."""
    lab = normalize_direction_label(label)
    if lab in ADMIN_CARDINAL_SET:
        return lab
    # Legacy compass already normalized; reject diagonals as expected flow
    return "LEFT"


def direction_unit_vector(label: str) -> Optional[Tuple[float, float]]:
    lab = normalize_direction_label(label)
    return DIRECTION_VECTORS.get(lab)


def direction_sector_index(label: str) -> Optional[int]:
    lab = normalize_direction_label(label)
    return _DIR_INDEX.get(lab)


def circular_sector_distance(a: str, b: str) -> Optional[int]:
    """Min circular steps between two 8-way labels (0..4)."""
    ia = direction_sector_index(a)
    ib = direction_sector_index(b)
    if ia is None or ib is None:
        return None
    raw = abs(ia - ib) % 8
    return min(raw, 8 - raw)


def allowed_directions(expected: str) -> Tuple[str, ...]:
    """Selected direction + two immediate circular neighbours."""
    exp = normalize_direction_label(expected)
    idx = _DIR_INDEX.get(exp)
    if idx is None:
        return tuple()
    return (
        DIRECTIONS_8[(idx - 1) % 8],
        DIRECTIONS_8[idx],
        DIRECTIONS_8[(idx + 1) % 8],
    )


def movement_angle_deg(dx: float, dy: float) -> float:
    """0°=RIGHT, 90°=UP (negate dy for image Y-down)."""
    ang = math.degrees(math.atan2(-dy, dx))
    if ang < 0:
        ang += 360.0
    return ang


def angle_to_direction8(dx: float, dy: float) -> str:
    """Quantize movement vector into one of eight product sectors (CW ring)."""
    # movement_angle_deg is CCW from RIGHT; invert for clockwise DIRECTIONS_8
    ang_ccw = movement_angle_deg(dx, dy)
    ang_cw = (360.0 - ang_ccw) % 360.0
    idx = int((ang_cw + 22.5) // 45.0) % 8
    return DIRECTIONS_8[idx]


def angle_to_compass8(dx: float, dy: float) -> str:
    """Alias kept for older imports/tests."""
    return angle_to_direction8(dx, dy)


def vector_to_direction8(dx: float, dy: float) -> str:
    return angle_to_direction8(dx, dy)


def vector_angular_deviation_deg(dx: float, dy: float, expected: str) -> Optional[float]:
    ve = direction_unit_vector(expected)
    if ve is None:
        return None
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        return None
    ux, uy = dx / mag, dy / mag
    dot = max(-1.0, min(1.0, ux * ve[0] + uy * ve[1]))
    return float(math.degrees(math.acos(dot)))


def angular_deviation_deg(actual: str, expected: str) -> Optional[float]:
    """Continuous degrees between two labels (diagnostics / hysteresis)."""
    va = direction_unit_vector(actual)
    ve = direction_unit_vector(expected)
    if va is None or ve is None:
        return None
    dot = max(-1.0, min(1.0, va[0] * ve[0] + va[1] * ve[1]))
    return float(math.degrees(math.acos(dot)))


def classify_direction_sectors(
    actual: str,
    expected: str,
) -> Dict[str, Any]:
    """
    Shared backend classification: circular sector distance.
    distance 0 or 1 → NORMAL; 2..4 → ANOMALY; else UNKNOWN.
    """
    exp = coerce_admin_direction(expected)
    act = normalize_direction_label(actual)
    out = {
        "expected_direction": exp,
        "actual_direction": act,
        "sector_distance": None,
        "allowed_directions": list(allowed_directions(exp)),
        "classification": "UNKNOWN",
        "reason": "",
    }
    if exp not in ADMIN_CARDINAL_SET:
        out["reason"] = f"Invalid expected direction: {expected!r}"
        return out
    if act in ("UNKNOWN", "STATIONARY") or act not in _DIR_INDEX:
        out["reason"] = f"Actual direction not classifiable: {actual!r}"
        return out
    dist = circular_sector_distance(act, exp)
    out["sector_distance"] = dist
    if dist is None:
        out["reason"] = "Could not compute sector distance"
        return out
    if dist <= 1:
        out["classification"] = "NORMAL"
        out["reason"] = (
            f"Moving {act}: sector Δ={dist} from expected {exp} "
            f"(allowed: {', '.join(out['allowed_directions'])})"
        )
    else:
        out["classification"] = "ANOMALY"
        out["reason"] = (
            f"Moving {act}: sector Δ={dist} from expected {exp} "
            f"(allowed: {', '.join(out['allowed_directions'])})"
        )
    return out


class DirectionEstimator:
    """Per-track foot trajectory → movement vector + 8-way product label."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.max_history = max(8, min(48, int(cfg.get("trajectory_history_max", 28))))
        self.min_obs = int(cfg.get("direction_min_observations", cfg.get("trajectory_min_history_for_direction", 6)))
        if "direction_min_fast_observations" in cfg:
            self.min_fast_obs = int(cfg["direction_min_fast_observations"])
        elif "trajectory_min_history_for_direction" in cfg:
            self.min_fast_obs = self.min_obs
        else:
            self.min_fast_obs = min(4, self.min_obs)
        self.fast_disp_px = float(cfg.get("direction_fast_displacement_px", 9.0))
        self.window = int(cfg.get("direction_window_frames", 12))
        self.ema_alpha = float(cfg.get("direction_ema_alpha", 0.60))
        self.min_disp_px = float(cfg.get("direction_min_displacement_px", 8.0))
        self.min_speed_px_s = float(cfg.get("direction_min_speed_px_sec", 5.0))
        self.max_frame_gap = int(cfg.get("trajectory_max_frame_gap", 10))
        self.occlusion_keep = int(cfg.get("trajectory_occlusion_keep_frames", 15))
        self.max_jump_px = float(cfg.get("trajectory_max_displacement_px", 65.0))
        self.hysteresis_deg = float(cfg.get("direction_hysteresis_deg", 28.0))
        # max_step_px: per-step displacement limit on smoothed foot positions.
        # Default raised to 90px so fast-walking pedestrians are not permanently
        # classified UNKNOWN by the per-step sanity check in _estimate().
        self.max_step_px = float(cfg.get("direction_max_step_px", 90.0))
        # max_window_gap: maximum allowed frame gap WITHIN the direction estimation
        # window before the gap check fires. Set to max_frame_gap so a track that
        # was briefly lost (1-3 frames) still contributes to direction estimation
        # once it has enough post-gap observations.
        self.max_window_gap = int(cfg.get("direction_max_window_gap", self.max_frame_gap))

        self.raw_hist: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.ema: Dict[int, Tuple[float, float]] = {}
        self.last_label: Dict[int, str] = {}
        self.last_seen: Dict[int, int] = {}

    def reset(self):
        self.raw_hist.clear()
        self.ema.clear()
        self.last_label.clear()
        self.last_seen.clear()

    def expire_stale(self, frame_idx: int):
        stale = [t for t, f in self.last_seen.items() if frame_idx - f > self.occlusion_keep * 3]
        for t in stale:
            self.raw_hist.pop(t, None)
            self.ema.pop(t, None)
            self.last_label.pop(t, None)
            self.last_seen.pop(t, None)

    def update(
        self,
        track_id: int,
        foot_xy: Tuple[float, float],
        frame_idx: int,
        timestamp_sec: float,
    ) -> Dict[str, Any]:
        raw_x, raw_y = float(foot_xy[0]), float(foot_xy[1])

        if track_id in self.ema:
            ex, ey = self.ema[track_id]
            sx = self.ema_alpha * raw_x + (1.0 - self.ema_alpha) * ex
            sy = self.ema_alpha * raw_y + (1.0 - self.ema_alpha) * ey
        else:
            sx, sy = raw_x, raw_y
        self.ema[track_id] = (sx, sy)

        hist = self.raw_hist[track_id]
        if hist:
            px, py, pf, _ = hist[-1]
            gap = frame_idx - pf
            jump = math.hypot(sx - px, sy - py)
            if jump > self.max_jump_px and gap <= self.max_frame_gap:
                hist.clear()
                self.last_label.pop(track_id, None)
                self.ema[track_id] = (raw_x, raw_y)
                sx, sy = raw_x, raw_y
            elif gap > self.max_frame_gap:
                if gap > self.occlusion_keep:
                    hist.clear()
                    self.last_label.pop(track_id, None)
                    self.ema[track_id] = (raw_x, raw_y)
                    sx, sy = raw_x, raw_y
                else:
                    self.ema[track_id] = (raw_x, raw_y)
                    sx, sy = raw_x, raw_y

        hist.append((sx, sy, frame_idx, float(timestamp_sec)))
        self.last_seen[track_id] = frame_idx
        return self._estimate(track_id, hist)

    def _estimate(self, track_id: int, hist: Deque) -> Dict[str, Any]:
        out = {
            "direction": "UNKNOWN",
            "direction_angle_deg": None,
            "direction_vector": [0.0, 0.0],
            "direction_confidence": 0.0,
            "direction_reliable": False,
            "movement_status": "UNKNOWN",
            "speed_px_sec": 0.0,
            "displacement_px": 0.0,
            "observation_count": len(hist),
        }
        if len(hist) < self.min_fast_obs:
            return out

        window = min(len(hist), self.window)
        recent = list(hist)[-window:]

        # Frame-gap check: if ANY gap within the window exceeds max_window_gap,
        # the trajectory is discontinuous and direction cannot be reliably estimated.
        # We use max_window_gap (defaults to max_frame_gap=10) instead of hard-coded 5,
        # so a briefly-lost track (1-3 frames) does not permanently suppress direction.
        for i in range(1, len(recent)):
            if recent[i][2] - recent[i - 1][2] > self.max_window_gap:
                return out

        # Per-step displacement check: guard against teleporting smoothed positions.
        # Uses max_step_px (default 90px) so fast-walking pedestrians are not
        # permanently suppressed by this sanity check.
        for i in range(1, len(recent)):
            sdx = recent[i][0] - recent[i - 1][0]
            sdy = recent[i][1] - recent[i - 1][1]
            if math.hypot(sdx, sdy) > self.max_step_px:
                return out

        # Net displacement over window (EMA-smoothed feet)
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        disp = math.hypot(dx, dy)

        # Directional consistency: ratio of net displacement to total path length.
        # Genuine locomotion has high consistency (>= 0.60). Bounding box jitter
        # on stationary pedestrians oscillates back and forth, giving low consistency (< 0.45).
        path_len = sum(
            math.hypot(recent[i][0] - recent[i - 1][0], recent[i][1] - recent[i - 1][1])
            for i in range(1, len(recent))
        )
        consistency = disp / max(1.0, path_len)

        # Adaptive observation window check:
        # If we have fewer than standard min_obs (e.g. 4 or 5 frames), only confirm
        # if the track shows clear, continuous directional displacement without zigzag/jitter.
        is_fast_confirmed = False
        if len(hist) < self.min_obs:
            if disp >= self.fast_disp_px and consistency >= 0.70:
                is_fast_confirmed = True
            else:
                return out

        t0, t1 = recent[0][3], recent[-1][3]
        f0, f1 = recent[0][2], recent[-1][2]
        dt = max(1e-3, float(t1 - t0))
        if abs(t1 - t0) < 1e-6:
            dt = max(1, f1 - f0) / 30.0
        speed = disp / dt

        out["displacement_px"] = round(disp, 2)
        out["speed_px_sec"] = round(speed, 2)
        out["direction_vector"] = [round(dx, 2), round(dy, 2)]
        out["consistency"] = round(consistency, 3)

        min_req_disp = min(self.min_disp_px, self.fast_disp_px) if is_fast_confirmed else self.min_disp_px
        
        # Stationary check: sub-threshold displacement, low speed, or low directional consistency
        # when displacement is moderate (preventing stationary jitter from becoming a false direction)
        is_stationary = (
            disp < min_req_disp
            or speed < self.min_speed_px_s
            or (disp < 20.0 and consistency < 0.42 and not is_fast_confirmed)
        )

        # Long-horizon check: if a track has >= 16 frames of history, genuine locomotion
        # must show displacement growth over time. Jitter / sway stays bounded (< 18px, < 15px/s).
        if not is_stationary and len(hist) >= 16 and not is_fast_confirmed:
            long_recent = list(hist)[-min(len(hist), 24):]
            long_dx = long_recent[-1][0] - long_recent[0][0]
            long_dy = long_recent[-1][1] - long_recent[0][1]
            long_disp = math.hypot(long_dx, long_dy)
            long_dt = max(0.1, float(long_recent[-1][3] - long_recent[0][3]))
            long_speed = long_disp / long_dt
            if long_disp < 18.0 or long_speed < 15.0:
                is_stationary = True

        if is_stationary:
            out["direction"] = "STATIONARY"
            out["movement_status"] = "STATIONARY"
            out["direction_confidence"] = min(1.0, disp / max(self.min_disp_px, 1.0) * 0.3)
            out["direction_reliable"] = False
            self.last_label[track_id] = "STATIONARY"
            return out

        label = angle_to_direction8(dx, dy)
        ang = movement_angle_deg(dx, dy)

        prev = self.last_label.get(track_id)
        if prev and prev not in ("STATIONARY", "UNKNOWN") and prev != label:
            prev_dev = angular_deviation_deg(label, prev)
            if prev_dev is not None and prev_dev < self.hysteresis_deg:
                label = prev

        self.last_label[track_id] = label
        conf_boost = 0.15 if is_fast_confirmed else 0.0
        conf = min(
            1.0,
            0.40
            + conf_boost
            + 0.35 * min(1.0, disp / (self.min_disp_px * 2.5))
            + 0.25 * min(1.0, len(hist) / float(self.max_history)),
        )
        out.update({
            "direction": label,
            "direction_angle_deg": round(ang, 1),
            "direction_confidence": round(conf, 3),
            "direction_reliable": conf >= 0.45 and disp >= min_req_disp,
            "movement_status": "MOVING",
        })
        return out


class DirectionClassifier:
    """Exact 3-of-8 circular sector classification vs admin cardinal expected."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        dir_cfg = cfg.get("direction_detection") or {}
        # Kept for config compat; classification uses sector distance, not degrees
        self.angular_tolerance_deg = float(
            dir_cfg.get("angular_tolerance_deg", cfg.get("angular_tolerance_deg", 45.0))
        )
        self.max_sector_distance_normal = int(dir_cfg.get("max_sector_distance_normal", 1))
        self.min_confidence = float(dir_cfg.get("min_confidence", 0.45))
        self.expected = coerce_admin_direction(
            cfg.get("expected_evacuation_direction") or "LEFT"
        )

    def set_expected_direction(self, label: str):
        self.expected = coerce_admin_direction(label)

    def classify(
        self,
        direction_info: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> Dict[str, Any]:
        expected_label = coerce_admin_direction(expected or self.expected)
        actual_raw = direction_info.get("direction", "UNKNOWN")
        dvec = direction_info.get("direction_vector") or [0.0, 0.0]
        dx, dy = float(dvec[0]), float(dvec[1])
        conf = float(direction_info.get("direction_confidence", 0.0))
        reliable = bool(direction_info.get("direction_reliable", False))

        # Prefer re-quantize from vector when reliable so estimation/classification agree
        if reliable and (abs(dx) + abs(dy) > 1e-6):
            actual = angle_to_direction8(dx, dy)
        else:
            actual = normalize_direction_label(actual_raw)

        result = {
            "expected_direction": expected_label,
            "actual_direction": actual,
            "angular_deviation_deg": None,
            "sector_distance": None,
            "allowed_directions": list(allowed_directions(expected_label)),
            "classification": "UNKNOWN",
            "reason": "",
            "direction_confidence": conf,
            "movement_vector": [round(dx, 2), round(dy, 2)],
        }

        if expected_label == "UNKNOWN" or expected_label not in _DIR_INDEX:
            result["reason"] = f"Invalid expected direction: {expected!r}"
            return result

        if actual == "STATIONARY" or (
            not reliable and normalize_direction_label(actual_raw) in ("UNKNOWN", "STATIONARY")
        ):
            result["actual_direction"] = (
                "STATIONARY" if normalize_direction_label(actual_raw) == "STATIONARY" else actual
            )
            result["reason"] = "Stationary / insufficient movement — not a direction anomaly"
            return result

        if not reliable or actual == "UNKNOWN" or actual not in _DIR_INDEX:
            result["reason"] = "Insufficient / unreliable trajectory history"
            return result

        if abs(dx) + abs(dy) <= 1e-6:
            result["reason"] = "Zero movement vector"
            return result

        sector = classify_direction_sectors(actual, expected_label)
        result["sector_distance"] = sector["sector_distance"]
        result["allowed_directions"] = sector["allowed_directions"]
        result["classification"] = sector["classification"]
        result["reason"] = sector["reason"]
        # Continuous angle kept for diagnostics panel
        ang_dev = vector_angular_deviation_deg(dx, dy, expected_label)
        result["angular_deviation_deg"] = None if ang_dev is None else round(ang_dev, 1)
        return result


__all__ = [
    "CLASSIFIER_VERSION",
    "ADMIN_CARDINALS",
    "ADMIN_CARDINAL_SET",
    "DirectionEstimator",
    "DirectionClassifier",
    "normalize_direction_label",
    "coerce_admin_direction",
    "direction_unit_vector",
    "direction_sector_index",
    "circular_sector_distance",
    "allowed_directions",
    "classify_direction_sectors",
    "angle_to_compass8",
    "angle_to_direction8",
    "vector_to_direction8",
    "angular_deviation_deg",
    "vector_angular_deviation_deg",
    "movement_angle_deg",
    "DIRECTIONS_8",
    "CARDINAL_8",
    "DIRECTION_VECTORS",
    "COMPASS_VECTORS",
    "DIR_ALIASES",
]
