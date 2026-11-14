#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Phase 2 Generalisation Improvements:
1. Temporal median / impulse filtering of foot-points without directional lag.
2. Confidence-weighted trajectory smoothing.
3. Exit-aware egress verification with trajectory approach checking.
4. Short-track fast confirmation and multi-horizon slow-walking direction detection.
"""

import unittest
import math
from scripts.direction_estimation import DirectionEstimator, DirectionClassifier
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine
from scripts.exit_regions import ExitRegionManager


class TestPhase2Improvements(unittest.TestCase):

    def test_01_impulse_jitter_filtering_without_lag(self):
        """Test that single-frame impulse spikes are suppressed without delaying continuous flow."""
        est = DirectionEstimator({
            "direction_min_observations": 4,
            "direction_window_frames": 12,
            "direction_min_displacement_px": 8.0,
            "direction_fast_displacement_px": 8.0,
        })
        # Continuous steady motion to the right
        # f1: (100, 200)
        # f2: (105, 200)
        # f3: (110, 200) -> steady trend (+5px/frame)
        est.update(1, (100.0, 200.0), 1, 1/30.0)
        est.update(1, (105.0, 200.0), 2, 2/30.0)
        est.update(1, (110.0, 200.0), 3, 3/30.0)
        # Steady motion has 0 delay and follows trend
        self.assertAlmostEqual(est.ema[1][0], 110.0, delta=3.0)

        # Frame 4: An artificial single-frame jitter spike at y=235 (spike +35px)
        est.update(1, (115.0, 235.0), 4, 4/30.0)
        # The impulse spike should be suppressed by the 3-point median filter
        self.assertLess(est.ema[1][1], 215.0, "Jitter spike in Y must be suppressed by median filter")

    def test_02_confidence_weighted_smoothing(self):
        """Test that high confidence adapts faster while low confidence adapts conservatively."""
        est_high = DirectionEstimator({"direction_ema_alpha": 0.60})
        est_low = DirectionEstimator({"direction_ema_alpha": 0.60})

        # Track with high detection confidence 0.95
        est_high.update(10, (100.0, 100.0), 1, 1/30.0, conf=0.95)
        est_high.update(10, (120.0, 100.0), 2, 2/30.0, conf=0.95)

        # Track with low detection confidence 0.15 (partial occlusion)
        est_low.update(20, (100.0, 100.0), 1, 1/30.0, conf=0.15)
        est_low.update(20, (120.0, 100.0), 2, 2/30.0, conf=0.15)

        # High confidence track should adapt faster (closer to 120.0) than low confidence track
        self.assertGreater(est_high.ema[10][0], est_low.ema[20][0])

    def test_03_exit_approach_verification(self):
        """Test that egress turns towards exit boundary are permitted, while turns away are rejected."""
        exit_cfg = [{
            "id": "door_north",
            "name": "North Exit Door",
            "polygon": [[400, 50], [600, 50], [600, 150], [400, 150]],  # Door at top (y=50..150)
            "centroid": (500, 80),
            "camera_id": "cam_01",
        }]
        mgr = ExitRegionManager(exit_cfg)

        # Person inside exit region at (500, 130) moving UP towards door centroid (y=80) -> approaching exit!
        # Expected flow in corridor is LEFT, actual turn is UP (orthogonal egress)
        permitted_towards = mgr.is_exit_turn_permitted(
            x=500.0, y=130.0,
            actual_dir="UP",
            expected_dir="LEFT",
            movement_vector=(0.0, -10.0),  # moving UP (dy < 0) towards y=80
            camera_id="cam_01"
        )
        self.assertTrue(permitted_towards, "Egress heading towards exit door must be permitted")

        # Person inside exit region at (500, 130) moving DOWN away from door centroid (y=80) -> leaving exit!
        permitted_away = mgr.is_exit_turn_permitted(
            x=500.0, y=130.0,
            actual_dir="DOWN",
            expected_dir="LEFT",
            movement_vector=(0.0, 10.0),  # moving DOWN (dy > 0) away from door
            camera_id="cam_01"
        )
        self.assertFalse(permitted_away, "Turn moving away from exit door must NOT be permitted")

    def test_04_multi_horizon_slow_walking_detection(self):
        """Test that slow-walking pedestrians with steady consistency are detected over multi-horizon window."""
        est = DirectionEstimator({
            "direction_window_frames": 12,
            "direction_min_displacement_px": 9.0,
            "direction_min_speed_px_sec": 5.0,
        })

        # Pedestrian walking slowly at ~15 px/s to the right (dx=0.6px per frame at 25fps)
        # Over 25 frames, displacement reaches > 14px with 1.0 consistency!
        fps = 25.0
        for f in range(1, 26):
            x = 200.0 + f * 0.6
            out = est.update(50, (x, 300.0), f, f / fps)

        # Over 25 frames, multi-horizon should classify RIGHT rather than remaining STATIONARY
        self.assertEqual(out["direction"], "RIGHT")
        self.assertEqual(out["movement_status"], "MOVING")
        self.assertTrue(out["direction_reliable"])


if __name__ == "__main__":
    unittest.main()
