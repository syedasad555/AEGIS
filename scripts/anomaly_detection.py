#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection Engine — direction mismatch + prolonged stopping
==================================================================
Responsibilities:
  - DirectionClassifier → NORMAL / ANOMALY / UNKNOWN vs admin expected direction
  - Persistence / cooldown / resolve for WRONG_DIRECTION
  - PROLONGED_STOPPING (stationary beyond threshold) — separate from direction

Colour contract (visual_state):
  normal   → BLUE   (classification NORMAL, or UNKNOWN without other anomaly)
  anomaly  → RED    (confirmed persistent WRONG_DIRECTION or PROLONGED_STOPPING)
  unknown  → BLUE   (insufficient history — never red for direction alone)

Fall detection removed.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from feature_extraction import TrajectoryFeatureExtractor
    from direction_estimation import (
        DirectionClassifier,
        normalize_direction_label,
        angular_deviation_deg,
    )
except ImportError:
    from scripts.feature_extraction import TrajectoryFeatureExtractor
    from scripts.direction_estimation import (
        DirectionClassifier,
        normalize_direction_label,
        angular_deviation_deg,
    )


def _new_event_id() -> str:
    return str(uuid.uuid4())


@dataclass
class AnomalyEvent:
    event_id: str
    event_type: str
    camera_id: str
    track_id: int
    timestamp: float
    severity: str
    evidence: Dict[str, Any]
    duration_sec: float = 0.0
    status: str = "ACTIVE"
    frame: int = 0
    zone_id: str = "corridor_main"
    zone_name: str = "Main Evacuation Corridor"
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvacuationAnomalyEngine:
    ALLOWED_TYPES = frozenset({"WRONG_DIRECTION", "PROLONGED_STOPPING"})

    def __init__(self, config: Dict[str, Any], camera_id: str = "cam_01"):
        self.config = config
        self.camera_id = camera_id
        self.feature_extractor = TrajectoryFeatureExtractor(config)
        self.fall_detector = None
        self.classifier = DirectionClassifier(config)

        stop_cfg = config.get("stopping_detection", {})
        self.stopping_threshold_sec = float(
            stop_cfg.get("threshold_sec", max(1.0, config.get("prolonged_stopping_frames", 45) / 30.0))
        )
        self.stopping_min_frames = int(
            stop_cfg.get("min_frames", config.get("prolonged_stopping_frames", 45))
        )
        self.stopping_critical_sec = float(stop_cfg.get("critical_sec", 20.0))
        self.enable_prolonged_stopping = bool(stop_cfg.get("enabled", True))

        dir_cfg = config.get("direction_detection", {})
        # Persistence: frames of consistent ANOMALY classification before RED
        self.wrong_dir_sustain_frames = int(dir_cfg.get("wrong_way_sustain_frames", dir_cfg.get("persistence_frames", 3)))
        self.recovery_frames = int(dir_cfg.get("recovery_frames", 4))
        self.wrong_dir_cooldown_sec = float(dir_cfg.get("cooldown_sec", 4.0))
        self.require_in_corridor = bool(dir_cfg.get("require_in_corridor", False))

        self.active_events: Dict[Tuple[str, int], AnomalyEvent] = {}
        self.event_history: deque = deque(maxlen=int(config.get("anomaly_history_max", 200)))
        self.anomaly_events: List[Dict[str, Any]] = []
        self.track_anomaly_start: Dict[Tuple[str, int], float] = {}
        self.wrong_dir_counters: Dict[int, int] = defaultdict(int)
        self.recovery_counters: Dict[int, int] = defaultdict(int)
        self.last_fire_time: Dict[Tuple[str, int], float] = {}
        self._ack_ids: set = set()
        self.frame_anomalies: List[Dict[str, Any]] = []
        self.diagnostic_mode = bool(config.get("direction_diagnostic_mode", False))

        # Live admin expected direction (after counters exist)
        ed = config.get("expected_evacuation_direction")
        if ed:
            self.set_expected_direction(ed)

    def set_expected_direction(self, label: str):
        """Hot-update expected flow without restarting the pipeline."""
        old = getattr(self.classifier, "expected", None)
        self.classifier.set_expected_direction(label)
        new = self.classifier.expected
        # Clear pending sustain + active WRONG_DIRECTION so live reclass is immediate
        if hasattr(self, "wrong_dir_counters"):
            self.wrong_dir_counters.clear()
        if hasattr(self, "recovery_counters"):
            self.recovery_counters.clear()
        if hasattr(self, "active_events"):
            for key in [k for k in self.active_events if k[0] == "WRONG_DIRECTION"]:
                self.active_events.pop(key, None)
        if hasattr(self, "last_fire_time"):
            for key in [k for k in list(self.last_fire_time) if k[0] == "WRONG_DIRECTION"]:
                self.last_fire_time.pop(key, None)
        print(
            f"[AnomalyEngine] EXPECTED_DIRECTION_CHANGED: "
            f"OLD_DIRECTION={old} NEW_DIRECTION={new} "
            f"ACTIVE_CLASSIFIER_DIRECTION={self.classifier.expected}",
            flush=True,
        )

    def reset(self):
        self.feature_extractor.reset()
        self.active_events.clear()
        self.event_history.clear()
        self.anomaly_events.clear()
        self.track_anomaly_start.clear()
        self.wrong_dir_counters.clear()
        self.recovery_counters.clear()
        self.last_fire_time.clear()
        self.frame_anomalies.clear()
        self._ack_ids.clear()

    def acknowledge(self, event_id: str) -> bool:
        self._ack_ids.add(event_id)
        for key, ev in list(self.active_events.items()):
            if ev.event_id == event_id:
                ev.status = "ACKNOWLEDGED"
                return True
        for ev_d in self.event_history:
            if ev_d.get("event_id") == event_id:
                ev_d["status"] = "ACKNOWLEDGED"
                return True
        return False

    def evaluate_pedestrian(
        self,
        features: Dict[str, Any],
        frame_img: Any = None,
        frame_h: int = 0,
        frame_w: int = 0,
        **_kwargs,
    ) -> Dict[str, Any]:
        feat = dict(features)
        ts = float(feat.get("timestamp", time.time()))
        frame = int(feat.get("frame", 0))
        tid = int(feat.get("track_id", -1))

        feat["direction"] = normalize_direction_label(feat.get("direction", "UNKNOWN"))
        if "stopping_duration_sec" not in feat:
            feat["stopping_duration_sec"] = round(
                int(feat.get("stopping_duration_frames", 0)) / 30.0, 2
            )

        # --- Direction classification (authoritative) ---
        expected = self.classifier.expected
        clf = self.classifier.classify(feat, expected=expected)
        feat["expected_direction"] = clf["expected_direction"]
        feat["angular_deviation_deg"] = clf["angular_deviation_deg"]
        feat["sector_distance"] = clf.get("sector_distance")
        feat["allowed_directions"] = clf.get("allowed_directions") or []
        feat["direction_classification"] = clf["classification"]  # NORMAL|ANOMALY|UNKNOWN
        feat["direction_reason"] = clf["reason"]

        all_anomalies: List[str] = []
        primary = "NORMAL"
        severity = "NORMAL"
        duration_sec = 0.0
        event_id = None
        evidence: Dict[str, Any] = {}
        # Default: unknown/normal stay blue
        visual_state = "normal"
        if clf["classification"] == "UNKNOWN":
            visual_state = "unknown"

        w = self._eval_wrong_direction(feat, clf, frame, ts)
        if w:
            all_anomalies.append("WRONG_DIRECTION")
            primary, severity, duration_sec, event_id, evidence = w
            visual_state = "anomaly"

        if self.enable_prolonged_stopping:
            s = self._eval_prolonged_stopping(feat, frame, ts)
            if s:
                all_anomalies.append("PROLONGED_STOPPING")
                if primary == "NORMAL":
                    primary, severity, duration_sec, event_id, evidence = s
                    visual_state = "anomaly"

        feat["anomaly"] = primary != "NORMAL"
        feat["anomaly_type"] = primary
        feat["all_anomalies"] = all_anomalies
        feat["severity"] = severity
        feat["duration_sec"] = duration_sec
        feat["event_id"] = event_id
        feat["evidence"] = evidence
        feat["visual_state"] = visual_state
        feat["is_live_observation"] = True
        feat["fall_status"] = "DISABLED"
        feat["fall_confidence"] = 0.0
        feat["fall_reasons"] = []
        feat["fall_event_id"] = None
        feat["fall_tracking_state"] = "TRACKED"

        # Always attach diagnostics so CCTV / twin can show why a track is blue/red
        dvec = feat.get("direction_vector") or [0.0, 0.0]
        feat["direction_diagnostics"] = {
            "track_id": tid,
            "foot": feat.get("foot"),
            "dx": dvec[0] if len(dvec) > 0 else 0.0,
            "dy": dvec[1] if len(dvec) > 1 else 0.0,
            "direction_vector": dvec,
            "estimated_direction": feat.get("direction"),
            "direction_confidence": feat.get("direction_confidence"),
            "direction_reliable": feat.get("direction_reliable"),
            "expected_direction": expected,
            "allowed_directions": feat.get("allowed_directions"),
            "sector_distance": feat.get("sector_distance"),
            "angular_deviation_deg": feat.get("angular_deviation_deg"),
            "classification": clf["classification"],
            "final_visual_state": visual_state,
            "reason": clf["reason"] if primary == "NORMAL" else (evidence.get("reason") or clf["reason"]),
            "sustain_frames": self.wrong_dir_counters.get(tid, 0),
            "sustain_required": self.wrong_dir_sustain_frames,
            "anomaly_type": primary,
            "trajectory": feat.get("trajectory"),
            "coordinate_space": "image",  # dx/dy are image-plane, not calibrated ground
        }
        return feat

    def _eval_wrong_direction(
        self, f: Dict[str, Any], clf: Dict[str, Any], frame: int, ts: float
    ) -> Optional[Tuple[str, str, float, str, Dict]]:
        tid = int(f.get("track_id", -1))
        ev_key = ("WRONG_DIRECTION", tid)

        if self.require_in_corridor and not bool(f.get("in_corridor", True)):
            self.wrong_dir_counters[tid] = 0
            self.recovery_counters[tid] = 0
            if ev_key in self.active_events:
                self._resolve_event(ev_key, ts)
            return None

        classification = clf["classification"]

        if classification == "NORMAL":
            # Clearly moving in an allowed direction: reset wrong-dir counter immediately
            self.wrong_dir_counters[tid] = 0
            if ev_key in self.active_events:
                self.recovery_counters[tid] += 1
                if self.recovery_counters[tid] >= self.recovery_frames:
                    self._resolve_event(ev_key, ts)
                    self.recovery_counters[tid] = 0
            return None

        if classification != "ANOMALY":
            # Insufficient / stationary / UNKNOWN movement.
            # Do not immediately zero out a pending wrong-direction sequence on a single jitter frame;
            # use a leaky decrement to provide 1-frame jitter tolerance while still requiring
            # true sustained ANOMALY evidence to reach RED.
            if self.wrong_dir_counters[tid] > 0:
                self.wrong_dir_counters[tid] = max(0, self.wrong_dir_counters[tid] - 1)

            if ev_key in self.active_events:
                # Active RED event: require sustained non-anomaly frames to resolve so track doesn't flicker on 1-frame jitter
                self.recovery_counters[tid] += 1
                resolve_thresh = max(2, self.recovery_frames // 2) if classification in ("UNKNOWN", "STATIONARY") else self.recovery_frames
                if self.recovery_counters[tid] >= resolve_thresh:
                    self._resolve_event(ev_key, ts)
                    self.recovery_counters[tid] = 0
                    return None
                else:
                    # Maintain active state during temporary 1-2 frame tracking gap
                    ev = self.active_events[ev_key]
                    if ev.status != "ACKNOWLEDGED":
                        ev.last_seen = ts
                        ev.duration_sec = round(ts - ev.timestamp, 2)
                        return ("WRONG_DIRECTION", "WARNING", ev.duration_sec, ev.event_id, ev.evidence)
            return None

        # classification == ANOMALY
        self.recovery_counters[tid] = 0
        self.wrong_dir_counters[tid] += 1

        evidence = {
            "reason": clf["reason"],
            "actual_direction": clf["actual_direction"],
            "expected_direction": clf["expected_direction"],
            "sector_distance": clf.get("sector_distance"),
            "allowed_directions": clf.get("allowed_directions"),
            "deviation_deg": clf["angular_deviation_deg"],
            "sustain_frames": self.wrong_dir_counters[tid],
            "direction_confidence": f.get("direction_confidence"),
            "foot": f.get("foot"),
            "speed_px_sec": f.get("speed_px_sec"),
        }

        if self.wrong_dir_counters[tid] < self.wrong_dir_sustain_frames:
            return None  # not yet confirmed → stay blue

        if ev_key in self.active_events:
            ev = self.active_events[ev_key]
            if ev.status == "ACKNOWLEDGED":
                return None
            ev.last_seen = ts
            ev.duration_sec = round(ts - ev.timestamp, 2)
            ev.evidence = evidence
            return ("WRONG_DIRECTION", "WARNING", ev.duration_sec, ev.event_id, evidence)

        last_fire = self.last_fire_time.get(ev_key, 0.0)
        if last_fire > 0 and (ts - last_fire) < self.wrong_dir_cooldown_sec:
            return None

        self.last_fire_time[ev_key] = ts
        ev = AnomalyEvent(
            event_id=_new_event_id(),
            event_type="WRONG_DIRECTION",
            camera_id=self.camera_id,
            track_id=tid,
            timestamp=ts,
            severity="WARNING",
            evidence=evidence,
            duration_sec=0.0,
            status="ACTIVE",
            frame=frame,
            zone_id=str(f.get("zone_id") or "corridor_main"),
            zone_name=str(f.get("zone_name") or "Main Evacuation Corridor"),
            last_seen=ts,
        )
        self.active_events[ev_key] = ev
        self._record_event(ev)
        return ("WRONG_DIRECTION", "WARNING", 0.0, ev.event_id, evidence)

    def _eval_prolonged_stopping(
        self, f: Dict[str, Any], frame: int, ts: float
    ) -> Optional[Tuple[str, str, float, str, Dict]]:
        stop_sec = float(f.get("stopping_duration_sec", 0.0))
        stop_frames = int(f.get("stopping_duration_frames", 0))
        meets = stop_frames >= self.stopping_min_frames and stop_sec >= self.stopping_threshold_sec
        tid = int(f.get("track_id", -1))
        ev_key = ("PROLONGED_STOPPING", tid)

        if not meets:
            if ev_key in self.active_events:
                self._resolve_event(ev_key, ts)
            return None

        severity = "CRITICAL" if stop_sec >= self.stopping_critical_sec else "WARNING"
        evidence = {
            "reason": "Stationary beyond configured threshold",
            "stopping_duration_sec": stop_sec,
            "stopping_duration_frames": stop_frames,
            "foot": f.get("foot"),
            "direction": f.get("direction"),
        }

        if ev_key in self.active_events:
            ev = self.active_events[ev_key]
            if ev.status == "ACKNOWLEDGED":
                return None
            ev.last_seen = ts
            ev.duration_sec = stop_sec
            ev.severity = severity
            ev.evidence = evidence
            return ("PROLONGED_STOPPING", severity, stop_sec, ev.event_id, evidence)

        ev = AnomalyEvent(
            event_id=_new_event_id(),
            event_type="PROLONGED_STOPPING",
            camera_id=self.camera_id,
            track_id=tid,
            timestamp=ts,
            severity=severity,
            evidence=evidence,
            duration_sec=stop_sec,
            status="ACTIVE",
            frame=frame,
            zone_id=str(f.get("zone_id") or "corridor_main"),
            zone_name=str(f.get("zone_name") or "Main Evacuation Corridor"),
            last_seen=ts,
        )
        self.active_events[ev_key] = ev
        self._record_event(ev)
        return ("PROLONGED_STOPPING", severity, stop_sec, ev.event_id, evidence)

    def _record_event(self, ev: AnomalyEvent):
        d = ev.to_dict()
        self.event_history.append(d)
        self.anomaly_events.append(d)
        self.track_anomaly_start[(ev.event_type, ev.track_id)] = ev.timestamp

    def _resolve_event(self, ev_key: Tuple[str, int], ts: float):
        ev = self.active_events.pop(ev_key, None)
        if ev is None or ev.status == "ACKNOWLEDGED":
            return
        ev.status = "RESOLVED"
        ev.last_seen = ts
        ev.duration_sec = round(ts - ev.timestamp, 2)
        self.event_history.append(ev.to_dict())

    def process_frame(
        self,
        tracks: List[Dict[str, Any]],
        frame_idx: int,
        fps: float = 30.0,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ts = timestamp if timestamp is not None else time.time()
        out: List[Dict[str, Any]] = []
        for track in tracks:
            tid = int(track.get("track_id", track.get("id", -1)))
            if tid < 0:
                continue
            bbox = track.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            bbox_t = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            features = self.feature_extractor.update_pedestrian(tid, bbox_t, frame_idx, fps)
            features["camera_id"] = self.camera_id
            if timestamp is not None:
                features["timestamp"] = ts
            feat = self.evaluate_pedestrian(features)
            if feat.get("anomaly"):
                out.append(feat)
        self.frame_anomalies = out
        return out

    def process_frame_persistence(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.active_events.values() if e.status == "ACTIVE"]

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.event_history)[-limit:]

    def export_anomaly_log(self, output_path: Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.anomaly_events, f, indent=2, ensure_ascii=False)
        print(f"[AnomalyEngine] Exported {len(self.anomaly_events)} events -> {output_path}")


__all__ = ["EvacuationAnomalyEngine", "AnomalyEvent", "angular_deviation_deg"]
