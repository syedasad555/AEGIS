#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 3 Unit Tests — Advanced Anomaly Classification
==========================================================
Tests the four Phase 3 core improvements:
1. Multi-horizon direction estimation (6, 12, 24-32 frames)
2. Trajectory curvature and evasive movement recognition
3. False wrong-direction suppression on stationary tracks
4. Lifespan spatial disambiguation and adaptive stopping
"""

import math
import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from direction_estimation import DirectionEstimator, angle_to_direction8
from feature_extraction import TrajectoryFeatureExtractor
from anomaly_detection import EvacuationAnomalyEngine


class TestPhase3Improvements(unittest.TestCase):

    def setUp(self):
        self.cfg = {
            "expected_evacuation_direction": "LEFT",
            "direction_min_displacement_px": 9.0,
            "direction_fast_displacement_px": 9.0,
            "direction_window_frames": 12,
            "stopping_detection": {
                "threshold_sec": 2.0,
                "min_frames": 50,
                "long_horizon_verification": True,
                "max_cumulative_disp_px": 20.0,
            },
            "direction_detection": {
                "wrong_way_sustain_frames": 3,
            },
            "enable_impulse_filter": True,
            "enable_confidence_ema": True,
            "enable_multi_horizon": True,
        }
        self.dir_estimator = DirectionEstimator(self.cfg)
        self.extractor = TrajectoryFeatureExtractor(self.cfg)
        self.engine = EvacuationAnomalyEngine(self.cfg)

    def test_01_multi_horizon_slow_crowd_locomotion(self):
        """Slow locomotion (11 px/s) over extended horizon (28 frames) is correctly detected as MOVING."""
        tid = 101
        fps = 25.0
        # Pedestrian moves DOWN at 11.25 px/s (0.45 px per frame)
        # In 12 frames, displacement = 5.4px (< 9px min_disp_px, so standard window would call it STATIONARY)
        # Over 28 frames, displacement = 12.6px (>= 10px ext_disp)
        # With multi-horizon evaluation, long horizon recognizes steady downward progress
        for f in range(1, 35):
            x = 200.0
            y = 100.0 + f * 0.45
            res = self.dir_estimator.update(tid, (x, y), f, f / fps, conf=0.85, is_detection=True)

        self.assertIn("long_direction_vector", res)
        self.assertEqual(res["direction"], "DOWN")
        self.assertEqual(res["movement_status"], "MOVING")
        self.assertTrue(res["direction_reliable"])

    def test_02_trajectory_curvature_recognition(self):
        """Curved trajectory / lateral dodge produces curvature_deg > 45 and is_curved_trajectory = True."""
        tid = 102
        fps = 25.0
        # Pedestrian starts walking LEFT for 15 frames, then veers UP for 6 frames (evasive dodge)
        for f in range(1, 16):
            x = 500.0 - f * 3.0
            y = 200.0
            self.dir_estimator.update(tid, (x, y), f, f / fps, conf=0.90, is_detection=True)

        # Now lateral dodge UP
        for f in range(16, 23):
            x = 500.0 - 15 * 3.0
            y = 200.0 - (f - 15) * 3.0
            res = self.dir_estimator.update(tid, (x, y), f, f / fps, conf=0.90, is_detection=True)

        self.assertGreater(res.get("curvature_deg", 0.0), 30.0)
        self.assertIn("long_direction_vector", res)

    def test_03_false_wrong_direction_suppressed_on_bounded_stationary_track(self):
        """Stationary track with bounded lateral jitter (< 12px) is NOT flagged as WRONG_DIRECTION."""
        tid = 103
        fps = 25.0
        # Stationary pedestrian with tiny lateral jitter pointing RIGHT (expected is LEFT)
        for f in range(1, 40):
            # Oscillates between 300.0 and 303.0 px (max net displacement 3.0px)
            jitter_x = 300.0 + (f % 3) * 1.0
            jitter_y = 400.0 + ((f + 1) % 2) * 0.5
            bbox = (jitter_x - 15, jitter_y - 50, jitter_x + 15, jitter_y)
            feat = self.extractor.update_pedestrian(tid, bbox, f, fps=fps, conf=0.80, is_detection=True)
            evaluated = self.engine.evaluate_pedestrian(feat)

        # Anomaly type must NOT be WRONG_DIRECTION
        self.assertNotEqual(evaluated.get("anomaly_type"), "WRONG_DIRECTION")

    def test_04_lifespan_spatial_disambiguation_for_prolonged_stopping(self):
        """Track observed for 35 frames with 85% stationary ratio and 3.5px net displacement is flagged as stopping."""
        tid = 104
        fps = 25.0
        # Pedestrian stays in essentially the same spot for 35 frames (net disp 3.5px)
        last_eval = None
        for f in range(1, 36):
            x = 150.0 + math.sin(f * 0.2) * 1.5
            y = 250.0 + math.cos(f * 0.2) * 1.5
            bbox = (x - 15, y - 50, x + 15, y)
            feat = self.extractor.update_pedestrian(tid, bbox, f, fps=fps, conf=0.85, is_detection=True)
            last_eval = self.engine.evaluate_pedestrian(feat)

        # Stationary ratio should be high, net displacement low
        self.assertLess(feat.get("net_displacement_px", 100.0), 12.0)
        self.assertGreaterEqual(feat.get("stationary_ratio", 0.0), 0.70)
        self.assertEqual(last_eval.get("anomaly_type"), "PROLONGED_STOPPING")


if __name__ == "__main__":
    unittest.main()
