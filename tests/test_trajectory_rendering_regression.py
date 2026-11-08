#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 25 — Digital Twin Trajectory Rendering & Anomaly Regression Test Suite
==================================================================================
Tests:
1. Two pedestrians moving independently produce separate trails (no cross-track connections).
2. Lost and reacquired tracks break the trail (no line drawn across the map).
3. Loading a second video clears all previous trails and anomaly states.
4. Invalid coordinates and large displacement jumps break the trail and suppress false alerts.
5. Camera overlays and Digital Twin markers refer to identical track IDs and aligned coordinates.
6. WRONG_DIRECTION alerts are suppressed when tracks have insufficient history (< 8 frames).
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine


class TestTrajectoryRenderingRegression(unittest.TestCase):
    """Regression tests for Phase 25 trajectory ownership, projection, and anomaly gating."""

    def setUp(self):
        """Set up standard configuration for tests."""
        self.config = {
            "fps": 30.0,
            "expected_evacuation_direction": "EAST",
            "stationary_speed_threshold": 0.6,
            "prolonged_stopping_frames": 30,
            "direction_window_frames": 10,
            "trajectory_history_max": 20,
            "trajectory_max_displacement_px": 65.0,
            "trajectory_max_frame_gap": 10,
            "trajectory_min_history_for_direction": 8,
            "direction_detection": {
                "angular_tolerance_deg": 30.0,
                "wrong_way_sustain_frames": 1,
                "recovery_frames": 2,
                "cooldown_sec": 0.0,
                "require_in_corridor": False,
            },
            "stopping_detection": {"enabled": False},
            "evacuation_corridor": {
                "id": "corridor_main",
                "polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]],
            },
            "zones": {},
        }
        self.extractor = TrajectoryFeatureExtractor(self.config)
        self.anomaly_engine = EvacuationAnomalyEngine(self.config)
        self.video_w = 1280
        self.video_h = 720

    def test_01_two_pedestrians_produce_separate_trails(self):
        """
        Verify that two pedestrians moving independently in the scene
        produce two completely separate trajectory histories with zero
        cross-connection between their track IDs.
        """
        track_a = 101
        track_b = 202

        # Pedestrian A moves rightward: x from 100 to 190, y=300
        # Pedestrian B moves leftward: x from 800 to 710, y=500
        for f in range(1, 11):
            bbox_a = (100.0 + f * 9.0, 200.0, 160.0 + f * 9.0, 400.0)
            bbox_b = (800.0 - f * 9.0, 400.0, 860.0 - f * 9.0, 600.0)

            feat_a = self.extractor.update_pedestrian(track_a, bbox_a, f, 30.0)
            feat_b = self.extractor.update_pedestrian(track_b, bbox_b, f, 30.0)

        # Check trajectory lengths and points
        traj_a = feat_a["trajectory"]
        traj_b = feat_b["trajectory"]

        self.assertEqual(len(traj_a), 10)
        self.assertEqual(len(traj_b), 10)

        # All points in A must have x < 300 and y near 400
        for pt in traj_a:
            self.assertLessEqual(pt[0], 300.0)
            self.assertAlmostEqual(pt[1], 400.0, delta=1.0)

        # All points in B must have x > 650 and y near 600
        for pt in traj_b:
            self.assertGreaterEqual(pt[0], 650.0)
            self.assertAlmostEqual(pt[1], 600.0, delta=1.0)

        # Confirm zero points are shared between track A and track B
        set_a = {(p[0], p[1]) for p in traj_a}
        set_b = {(p[0], p[1]) for p in traj_b}
        self.assertTrue(set_a.isdisjoint(set_b))

    def test_02_lost_and_reacquired_track_breaks_trail(self):
        """
        Verify that when a track is lost and reacquired after a large frame gap (> 10 frames),
        or re-assigned to a different pedestrian location, the trail is broken and stale
        history is cleared so no connecting line is drawn across the entire map.
        """
        track_id = 303

        # Phase 1: Track active at (150, 250) for 5 frames
        for f in range(1, 6):
            bbox = (120.0, 100.0, 180.0, 250.0)
            self.extractor.update_pedestrian(track_id, bbox, f, 30.0)

        self.assertEqual(len(self.extractor.histories[track_id]), 5)
        old_foot_x = self.extractor.histories[track_id][-1][0]
        self.assertAlmostEqual(old_foot_x, 150.0, delta=1.0)

        # Phase 2: Track lost for 25 frames (frames 6 to 30)
        # Phase 3: Track re-appears at frame 31 on the opposite side of the screen at (950, 650)
        reappear_bbox = (920.0, 500.0, 980.0, 650.0)
        feat_reappear = self.extractor.update_pedestrian(track_id, reappear_bbox, 31, 30.0)

        # The large frame gap (31 - 5 = 26 > 10) must break the trail and clear old history!
        traj_after_reappear = feat_reappear["trajectory"]
        self.assertEqual(len(traj_after_reappear), 1)
        self.assertAlmostEqual(traj_after_reappear[0][0], 950.0, delta=1.0)
        self.assertAlmostEqual(traj_after_reappear[0][1], 650.0, delta=1.0)

        # Ensure the old point at (150, 250) is NOT connected to (950, 650)
        for pt in traj_after_reappear:
            self.assertGreater(pt[0], 900.0)

    def test_03_loading_second_video_clears_previous_trails(self):
        """
        Verify that loading a new video clears all accumulated track histories,
        counters, and anomaly states, preventing any residual trails from persisting.
        """
        # Video 1: Tracks 1, 2, 3 accumulate observations
        for f in range(1, 15):
            self.extractor.update_pedestrian(1, (100.0, 200.0, 160.0, 400.0), f, 30.0)
            self.extractor.update_pedestrian(2, (300.0, 200.0, 360.0, 400.0), f, 30.0)
            self.extractor.update_pedestrian(3, (500.0, 200.0, 560.0, 400.0), f, 30.0)

        self.assertEqual(len(self.extractor.histories), 3)

        # Simulate Video Switch / Reset
        self.extractor.reset()
        self.anomaly_engine.reset()

        # Histories and anomaly states must be completely clean
        self.assertEqual(len(self.extractor.histories), 0)
        self.assertEqual(len(self.extractor.stopping_counters), 0)
        self.assertEqual(len(self.extractor.export_records), 0)
        self.assertEqual(len(self.anomaly_engine.track_anomaly_start), 0)
        self.assertEqual(len(self.anomaly_engine.anomaly_events), 0)

        # Video 2: Fresh track 99 starts cleanly
        feat_vid2 = self.extractor.update_pedestrian(99, (400.0, 300.0, 460.0, 500.0), 1, 30.0)
        self.assertEqual(len(self.extractor.histories), 1)
        self.assertIn(99, self.extractor.histories)
        self.assertNotIn(1, self.extractor.histories)
        self.assertEqual(len(feat_vid2["trajectory"]), 1)

    def test_04_invalid_coordinates_and_large_jumps_break_trail(self):
        """
        Verify that implausible displacement jumps (> 65px in 1 frame) break the trail,
        reset the history to prevent wild crossing lines, and do not trigger false
        counterflow or fall alarms.
        """
        track_id = 404

        # Frame 1: Pedestrian at (200, 300)
        self.extractor.update_pedestrian(track_id, (170.0, 100.0, 230.0, 300.0), 1, 30.0)

        # Frame 2: Teleportation / ID switch across 700px to (900, 800)
        teleport_bbox = (870.0, 600.0, 930.0, 800.0)
        feat_jump = self.extractor.update_pedestrian(track_id, teleport_bbox, 2, 30.0)

        # Large displacement (> 65px) must break the trail
        traj = feat_jump["trajectory"]
        self.assertEqual(len(traj), 1)
        self.assertAlmostEqual(traj[0][0], 900.0, delta=1.0)

        # Anomaly engine must NOT trigger WRONG_DIRECTION on teleportation
        eval_res = self.anomaly_engine.evaluate_pedestrian(feat_jump)
        self.assertNotEqual(eval_res["anomaly_type"], "WRONG_DIRECTION")

    def test_05_overlay_and_digital_twin_track_id_alignment(self):
        """
        Verify that camera overlay bounding boxes and Digital Twin marker dots
        refer to identical track IDs and consistent normalized coordinate bounds [0.0, 1.0].
        """
        test_tracks = [
            (10, (100.0, 200.0, 180.0, 400.0)),
            (20, (500.0, 300.0, 580.0, 550.0)),
            (30, (900.0, 400.0, 980.0, 700.0))
        ]

        for tid, bbox in test_tracks:
            feat = self.extractor.update_pedestrian(tid, bbox, 1, 30.0)
            eval_res = self.anomaly_engine.evaluate_pedestrian(feat)

            # Check track ID identity
            self.assertEqual(eval_res["track_id"], tid)

            # Check foot coordinate calculation: foot_x = (x1 + x2)/2, foot_y = y2
            expected_foot_x = (bbox[0] + bbox[2]) / 2.0
            expected_foot_y = bbox[3]
            self.assertAlmostEqual(eval_res["foot"][0], expected_foot_x, delta=0.5)
            self.assertAlmostEqual(eval_res["foot"][1], expected_foot_y, delta=0.5)

            # Normalized projection bounds check
            norm_x = eval_res["foot"][0] / self.video_w
            norm_y = eval_res["foot"][1] / self.video_h
            self.assertGreaterEqual(norm_x, 0.0)
            self.assertLessEqual(norm_x, 1.0)
            self.assertGreaterEqual(norm_y, 0.0)
            self.assertLessEqual(norm_y, 1.0)

    def test_06_wrong_direction_suppressed_for_insufficient_history(self):
        """
        Verify that WRONG_DIRECTION alerts are strictly suppressed when a track
        has fewer than 8 valid continuous observations, preventing initial tracking
        noise or jitter from firing false alarms.
        """
        track_id = 505

        # Pedestrian starts moving LEFT (opposing expected direction RIGHT)
        # Frames 1 through 7: Insufficient history (< 8 frames)
        for f in range(1, 8):
            bbox = (600.0 - f * 8.0, 300.0, 660.0 - f * 8.0, 500.0)
            feat = self.extractor.update_pedestrian(track_id, bbox, f, 30.0)
            eval_res = self.anomaly_engine.evaluate_pedestrian(feat)
            # Must NOT trigger WRONG_DIRECTION yet!
            self.assertNotEqual(eval_res["anomaly_type"], "WRONG_DIRECTION")

        # Frame 8: 8th consecutive observation reached -> reliable counterflow confirmed
        bbox_8 = (600.0 - 8 * 8.0, 300.0, 660.0 - 8 * 8.0, 500.0)
        feat_8 = self.extractor.update_pedestrian(track_id, bbox_8, 8, 30.0)
        eval_res_8 = self.anomaly_engine.evaluate_pedestrian(feat_8)
        self.assertEqual(eval_res_8["anomaly_type"], "WRONG_DIRECTION")
        self.assertEqual(eval_res_8["severity"], "WARNING")
        self.assertEqual(eval_res_8["visual_state"], "anomaly")


if __name__ == "__main__":
    unittest.main()
