#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corridor + direction regression (aligned with 8-sector circular classifier)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_extraction import (
    EvacuationCorridor,
    TrajectoryFeatureExtractor,
    angle_to_compass8,
    angular_deviation_deg,
    normalize_direction_label,
    OBSOLETE_ZONE_KEYS,
)
from scripts.anomaly_detection import EvacuationAnomalyEngine


class TestDirectionEightWay(unittest.TestCase):
    def test_compass_bins_image_y_down(self):
        self.assertEqual(angle_to_compass8(10, 0), "RIGHT")
        self.assertEqual(angle_to_compass8(0, -10), "UP")
        self.assertEqual(angle_to_compass8(-10, 0), "LEFT")
        self.assertEqual(angle_to_compass8(0, 10), "DOWN")

    def test_y_axis_not_inverted(self):
        self.assertEqual(angle_to_compass8(0, 20), "DOWN")

    def test_legacy_normalize(self):
        self.assertEqual(normalize_direction_label("RIGHT"), "RIGHT")
        self.assertEqual(normalize_direction_label("LEFT"), "LEFT")
        self.assertEqual(normalize_direction_label("NORTH-EAST"), "UP_RIGHT")
        self.assertEqual(normalize_direction_label("EAST"), "RIGHT")

    def test_angular_deviation(self):
        self.assertEqual(angular_deviation_deg("RIGHT", "LEFT"), 180.0)
        self.assertEqual(angular_deviation_deg("EAST", "WEST"), 180.0)


class TestCorridorOnly(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "expected_evacuation_direction": "LEFT",
            "evacuation_corridor": {
                "id": "corridor_main",
                "name": "Main Corridor",
                "expected_direction": "LEFT",
                "polygon": [[0, 0], [400, 0], [400, 200], [0, 200]],
            },
            "zones": {},
        }

    def test_validate_rejects_obsolete_zones(self):
        for key in ("restricted_zones", "exit_zones", "monitoring_zones", "safe_zones"):
            bad = {"zones": {key: [{"name": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}]}}
            ok, msg = EvacuationCorridor.validate_config(bad)
            self.assertFalse(ok, key)

    def test_contains_point(self):
        c = EvacuationCorridor(self.cfg)
        self.assertTrue(c.contains(100, 100))
        self.assertFalse(c.contains(9999, 9999))

    def test_obsolete_keys_constant(self):
        self.assertIn("restricted_zones", OBSOLETE_ZONE_KEYS)


class TestTwoAnomalies(unittest.TestCase):
    def setUp(self):
        self.config = {
            "expected_evacuation_direction": "RIGHT",
            "evacuation_corridor": {
                "id": "corridor_main",
                "polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]],
            },
            "direction_detection": {
                "max_sector_distance_normal": 1,
                "wrong_way_sustain_frames": 3,
                "recovery_frames": 2,
                "cooldown_sec": 0.0,
                "require_in_corridor": False,
            },
            "stopping_detection": {"threshold_sec": 0.5, "min_frames": 15, "enabled": True},
            "zones": {},
        }
        self.engine = EvacuationAnomalyEngine(self.config)

    def _feat(self, **kw):
        base = {
            "track_id": 1,
            "frame": 10,
            "timestamp": 1.0,
            "bbox": [100, 100, 140, 200],
            "foot": [120, 200],
            "speed_px_sec": 40.0,
            "direction": "LEFT",
            "direction_vector": [-40.0, 0.0],
            "direction_reliable": True,
            "direction_confidence": 0.85,
            "stopping_duration_frames": 0,
            "stopping_duration_sec": 0.0,
            "in_corridor": True,
            "zone_type": "EVACUATION_CORRIDOR",
            "zone_id": "corridor_main",
        }
        base.update(kw)
        return base

    def test_wrong_direction_requires_sustain(self):
        for i in range(2):
            r = self.engine.evaluate_pedestrian(self._feat(timestamp=1.0 + i * 0.1, frame=10 + i))
            self.assertFalse(r["anomaly"])
        r = self.engine.evaluate_pedestrian(self._feat(timestamp=1.3, frame=13))
        self.assertTrue(r["anomaly"])
        self.assertEqual(r["anomaly_type"], "WRONG_DIRECTION")
        self.assertEqual(r["visual_state"], "anomaly")

    def test_sideways_is_anomaly_under_sector_rule(self):
        # UP vs RIGHT → sector Δ=2 → ANOMALY after sustain
        for i in range(5):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="UP",
                direction_vector=[0.0, -40.0],
                timestamp=2.0 + i * 0.1,
                frame=20 + i,
            ))
        self.assertTrue(r["anomaly"])

    def test_unknown_and_stationary_not_anomaly(self):
        r = self.engine.evaluate_pedestrian(self._feat(
            direction="UNKNOWN", direction_vector=[0, 0],
            direction_reliable=False, direction_confidence=0.1,
        ))
        self.assertFalse(r["anomaly"])
        r = self.engine.evaluate_pedestrian(self._feat(
            direction="STATIONARY", direction_vector=[0, 0],
            direction_reliable=False, direction_confidence=0.2,
            stopping_duration_frames=0, stopping_duration_sec=0,
        ))
        self.assertFalse(r["anomaly"])

    def test_prolonged_stopping(self):
        r = self.engine.evaluate_pedestrian(self._feat(
            track_id=2,
            direction="STATIONARY",
            direction_vector=[0, 0],
            direction_reliable=False,
            direction_confidence=0.2,
            stopping_duration_frames=30,
            stopping_duration_sec=1.0,
            speed_px_sec=0.1,
        ))
        self.assertTrue(r["anomaly"])
        self.assertEqual(r["anomaly_type"], "PROLONGED_STOPPING")

    def test_no_restricted_or_fall(self):
        self.assertIsNone(self.engine.fall_detector)
        r = self.engine.evaluate_pedestrian(self._feat(
            direction="RIGHT", direction_vector=[40, 0],
        ))
        self.assertNotIn(r["anomaly_type"], (
            "RESTRICTED_ZONE_ENTRY", "FALL_DETECTED", "FALL_SUSPECTED"
        ))

    def test_resolve_clears_active(self):
        for i in range(3):
            self.engine.evaluate_pedestrian(self._feat(timestamp=1.0 + i * 0.1, frame=10 + i))
        self.assertTrue(any(e.status == "ACTIVE" for e in self.engine.active_events.values()))
        # Resume correct direction for recovery_frames
        for i in range(4):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="RIGHT",
                direction_vector=[40.0, 0.0],
                timestamp=5.0 + i * 0.1,
                frame=50 + i,
            ))
        self.assertFalse(r["anomaly"])
        self.assertFalse(any(
            e.event_type == "WRONG_DIRECTION" and e.status == "ACTIVE"
            for e in self.engine.active_events.values()
        ))


class TestTrajectoryDirectionIntegration(unittest.TestCase):
    def test_extractor_eight_way_and_stationary(self):
        cfg = {
            "trajectory_min_history_for_direction": 8,
            "direction_min_observations": 8,
            "direction_window_frames": 12,
            "direction_min_displacement_px": 8.0,
            "direction_min_speed_px_sec": 5.0,
            "evacuation_corridor": {
                "polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]],
            },
            "zones": {},
        }
        ex = TrajectoryFeatureExtractor(cfg)
        for f in range(1, 14):
            x = 100.0 + f * 15.0
            feat = ex.update_pedestrian(1, (x, 200.0, x + 40.0, 400.0), f, 30.0)
        self.assertEqual(feat["direction"], "RIGHT")

        ex2 = TrajectoryFeatureExtractor(cfg)
        for f in range(1, 14):
            feat2 = ex2.update_pedestrian(2, (200.0, 200.0, 240.0, 400.0), f, 30.0)
        self.assertEqual(feat2["direction"], "STATIONARY")

    def test_occlusion_keeps_then_breaks(self):
        cfg = {
            "direction_min_observations": 4,
            "trajectory_max_frame_gap": 10,
            "trajectory_occlusion_keep_frames": 15,
            "trajectory_max_displacement_px": 65.0,
            "evacuation_corridor": {"polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]]},
            "zones": {},
        }
        ex = TrajectoryFeatureExtractor(cfg)
        for f in range(1, 6):
            ex.update_pedestrian(3, (100.0, 100.0, 140.0, 200.0), f, 30.0)
        feat = ex.update_pedestrian(3, (900.0, 500.0, 940.0, 650.0), 40, 30.0)
        self.assertEqual(len(feat["trajectory"]), 1)


class TestColourSemanticsContract(unittest.TestCase):
    def test_visual_state_normal_vs_anomaly(self):
        cfg = {
            "expected_evacuation_direction": "RIGHT",
            "evacuation_corridor": {
                "polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]],
            },
            "direction_detection": {
                "max_sector_distance_normal": 1,
                "wrong_way_sustain_frames": 1,
                "cooldown_sec": 0.0,
                "require_in_corridor": False,
            },
            "stopping_detection": {"enabled": False},
            "zones": {},
        }
        eng = EvacuationAnomalyEngine(cfg)
        normal = eng.evaluate_pedestrian({
            "track_id": 1, "frame": 1, "timestamp": 0.0,
            "direction": "RIGHT", "direction_vector": [20, 0],
            "direction_reliable": True, "direction_confidence": 0.9,
            "in_corridor": True,
            "stopping_duration_frames": 0, "stopping_duration_sec": 0,
            "foot": [100, 100], "bbox": [80, 50, 120, 100],
        })
        self.assertEqual(normal["visual_state"], "normal")
        self.assertFalse(normal["anomaly"])

        bad = eng.evaluate_pedestrian({
            "track_id": 2, "frame": 2, "timestamp": 1.0,
            "direction": "LEFT", "direction_vector": [-40, 0],
            "direction_reliable": True, "direction_confidence": 0.9,
            "in_corridor": True,
            "stopping_duration_frames": 0, "stopping_duration_sec": 0,
            "foot": [100, 100], "bbox": [80, 50, 120, 100],
        })
        self.assertEqual(bad["visual_state"], "anomaly")
        self.assertEqual(bad["anomaly_type"], "WRONG_DIRECTION")

        unk = eng.evaluate_pedestrian({
            "track_id": 3, "frame": 3, "timestamp": 2.0,
            "direction": "UNKNOWN", "direction_vector": [0, 0],
            "direction_reliable": False, "direction_confidence": 0.1,
            "in_corridor": True,
            "stopping_duration_frames": 0, "stopping_duration_sec": 0,
            "foot": [100, 100], "bbox": [80, 50, 120, 100],
        })
        self.assertEqual(unk["visual_state"], "unknown")
        self.assertFalse(unk["anomaly"])


if __name__ == "__main__":
    unittest.main()
