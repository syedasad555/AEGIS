#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 23 — Temporal Fall State Machine Regression Test Suite
==================================================================
Tests:
1. Full FSM transition lifecycle: NORMAL -> FALL_SUSPECTED -> FALL_CONFIRMED -> RECOVERED.
2. Temporary detection loss and recovery within coasting window (spatial re-association).
3. Long-term detection loss and timeout closure (LOST_TIMEOUT).
4. ADL Controls: Sitting on chair, bending, crouching, lying down without fall.
5. Repeated alert suppression and event ID immutability.
6. Spatial re-association distance gating (prevents merging unrelated nearby pedestrians).
7. Image edge pedestrians (boundary truncation does not trigger false fall alarms).
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fall_detection import FallDetector


class TestFallFSMRegression(unittest.TestCase):
    """Rigorous regression test suite for FallDetector 4-state FSM."""

    def setUp(self):
        """Initialize a fresh FallDetector for each test."""
        self.config = {
            "fall_ar_suspect_threshold": 0.72,
            "fall_ar_confirm_threshold": 0.90,
            "fall_height_drop_ratio": 0.62,
            "fall_centroid_drop_threshold": 0.04,
            "fall_confirm_frames": 4,
            "fall_max_ground_speed": 55.0,
            "fall_coast_frames_max": 60,
            "fall_spatial_reassoc_dist": 160.0,
            "fall_recovery_confirm_frames": 12,
            "fall_pose_model": None  # Pure geometry/trajectory mode for deterministic unit tests
        }
        self.detector = FallDetector(self.config, pose_model_path=None)
        self.frame_h = 1080
        self.frame_w = 1920

    def test_01_fsm_full_transition_lifecycle(self):
        """
        Verify complete 4-state transition:
        NORMAL -> FALL_SUSPECTED -> FALL_CONFIRMED -> RECOVERED.
        """
        track_id = 101

        # Phase A: Upright locomotion (NORMAL)
        # Bounding box: width 60px, height 200px (AR = 0.30), standing upright
        for f in range(1, 15):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(500.0, 400.0, 560.0, 600.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertEqual(res["status"], FallDetector.STATE_NORMAL)
            self.assertIsNone(res["event_id"])
            self.assertEqual(res["tracking_state"], "TRACKED")
            self.assertTrue(res["is_live_observation"])

        # Baseline standing height should be established (~200px)
        self.assertIn(track_id, self.detector.standing_height)
        self.assertAlmostEqual(self.detector.standing_height[track_id], 200.0, delta=5.0)

        # Phase B: Dynamic Collapse Slip (FALL_SUSPECTED)
        # Centroid drops > 0.04 over recent frames, and aspect ratio tilts to AR >= 0.55
        for f in range(15, 18):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(500.0, 450.0 + (f - 15) * 35.0, 620.0 + (f - 15) * 20.0, 640.0 + (f - 15) * 20.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        self.assertIn(res["status"], (FallDetector.STATE_SUSPECTED, FallDetector.STATE_CONFIRMED))

        # Phase C: Sustained Prone Posture (FALL_CONFIRMED)
        # Horizontal aspect ratio (width 180px, height 100px, AR = 1.80 > 0.90) near floor
        for f in range(18, 26):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(480.0, 620.0, 660.0, 720.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        self.assertEqual(res["status"], FallDetector.STATE_CONFIRMED)
        self.assertIsNotNone(res["event_id"])
        confirmed_event_id = res["event_id"]

        # Phase D: Standing Recovery (RECOVERED)
        # Person stands back up (width 60px, height 195px, AR = 0.31 < 0.60, H > 80% baseline)
        # Must sustain upright posture for at least 12 consecutive frames
        for f in range(26, 26 + 11):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(500.0, 420.0, 560.0, 615.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            # Before 12 frames, state remains confirmed to avoid premature dismissal
            self.assertEqual(res["status"], FallDetector.STATE_CONFIRMED)

        # 12th upright frame triggers confirmed recovery
        res_recovered = self.detector.evaluate_pedestrian(
            track_id=track_id,
            bbox=(500.0, 420.0, 560.0, 615.0),
            frame_idx=26 + 12,
            frame_h=self.frame_h,
            frame_w=self.frame_w
        )
        self.assertEqual(res_recovered["status"], FallDetector.STATE_RECOVERED)
        self.assertEqual(res_recovered["tracking_state"], "RECOVERED")

    def test_02_temporary_detection_loss_and_coasting_recovery(self):
        """
        Verify coasting grace period and spatial candidate re-association:
        1. Person confirms fall.
        2. Detection is lost by ByteTrack for 15 frames. State enters LOST_COASTING.
        3. Raw low-confidence candidate box in vicinity (D <= 160px) recovers track.
        4. Event ID remains identical throughout.
        """
        track_id = 202

        # 1. Establish fall
        for f in range(1, 10):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(300.0, 400.0, 360.0, 600.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        for f in range(10, 16):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(280.0, 550.0 + (f - 10) * 15.0, 460.0, 650.0 + (f - 10) * 10.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        self.assertEqual(res["status"], FallDetector.STATE_CONFIRMED)
        orig_event_id = res["event_id"]

        # 2. Simulate 15 frames of detection loss (ByteTrack drops track)
        for f in range(16, 31):
            persist_out = self.detector.process_frame_persistence(
                frame_idx=f,
                observed_track_ids=[],  # Track not observed
                raw_candidate_boxes=[],  # No candidates
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertEqual(len(persist_out), 1)
            p = persist_out[0]
            self.assertEqual(p["track_id"], track_id)
            self.assertEqual(p["tracking_state"], "LOST_COASTING")
            self.assertFalse(p["is_live_observation"])
            self.assertEqual(p["status"], FallDetector.STATE_CONFIRMED)
            self.assertEqual(p["event_id"], orig_event_id)
            self.assertEqual(p["coasting_frames"], f - 15)

        # 3. Candidate appears nearby (distance ~45px, AR = 1.60)
        nearby_raw_box = [310.0, 680.0, 470.0, 780.0]
        persist_recovered = self.detector.process_frame_persistence(
            frame_idx=31,
            observed_track_ids=[],
            raw_candidate_boxes=[nearby_raw_box],
            frame_h=self.frame_h,
            frame_w=self.frame_w
        )
        self.assertEqual(len(persist_recovered), 1)
        rec = persist_recovered[0]
        self.assertEqual(rec["track_id"], track_id)
        self.assertEqual(rec["tracking_state"], "SPATIAL_REASSOCIATED")
        self.assertTrue(rec["is_live_observation"])
        self.assertEqual(rec["event_id"], orig_event_id)
        self.assertLessEqual(rec["reassoc_dist_px"], 160.0)

    def test_03_long_term_loss_and_timeout_closure(self):
        """
        Verify that coasting tracks time out and close cleanly after 60 frames (2.0s).
        """
        track_id = 303

        # Establish fall
        for f in range(1, 10):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(600.0, 400.0, 660.0, 600.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        for f in range(10, 16):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(580.0, 580.0, 760.0, 680.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        self.assertEqual(res["status"], FallDetector.STATE_CONFIRMED)
        orig_event_id = res["event_id"]

        # Coast for 60 frames (max grace period)
        for f in range(16, 16 + 60):
            persist_out = self.detector.process_frame_persistence(
                frame_idx=f,
                observed_track_ids=[],
                raw_candidate_boxes=[],
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertEqual(len(persist_out), 1)
            self.assertEqual(persist_out[0]["tracking_state"], "LOST_COASTING")

        # Frame 61 of loss -> TIMEOUT!
        persist_timeout = self.detector.process_frame_persistence(
            frame_idx=16 + 61,
            observed_track_ids=[],
            raw_candidate_boxes=[],
            frame_h=self.frame_h,
            frame_w=self.frame_w
        )
        # Should no longer return active coasting output
        self.assertEqual(len(persist_timeout), 0)
        # Internal state should be marked LOST_TIMEOUT and event closed
        self.assertEqual(self.detector.fall_state[track_id]["status"], FallDetector.STATE_LOST_TIMEOUT)
        self.assertEqual(self.detector.fall_state[track_id]["tracking_state"], "TIMEOUT")
        self.assertEqual(self.detector.active_fall_events[orig_event_id]["status"], "CLOSED_TIMEOUT")

    def test_04_adl_sitting_on_chair(self):
        """
        Verify sitting down on a chair does NOT trigger FALL_CONFIRMED.
        """
        track_id = 404

        # Walking upright
        for f in range(1, 15):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(800.0, 400.0, 860.0, 600.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )

        # Sitting down onto chair: Bounding box widens slightly (AR = 0.65), height drops to ~130px
        # Movement is controlled (no sudden collapse: cy drop < 0.04/frame)
        for f in range(15, 40):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(800.0, 480.0, 885.0, 610.0),  # AR = 85/130 = 0.654 < 0.90
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertNotEqual(res["status"], FallDetector.STATE_CONFIRMED)
            self.assertIsNone(res["event_id"])

    def test_05_adl_bending_and_crouching(self):
        """
        Verify deep bending and crouching activities do NOT trigger false fall alarms.
        """
        track_id = 505

        # Upright baseline
        for f in range(1, 15):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(400.0, 350.0, 460.0, 560.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )

        # Controlled bend down over 5 frames (descent is controlled, ~5px per frame)
        for f in range(15, 18):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(395.0, 370.0 + (f - 15) * 5.0, 475.0, 560.0),  # Controlled bending
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertNotEqual(res["status"], FallDetector.STATE_CONFIRMED)

        # Stands back up
        for f in range(18, 30):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(400.0, 350.0, 460.0, 560.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertEqual(res["status"], FallDetector.STATE_NORMAL)

    def test_06_adl_smooth_lying_down_without_collapse(self):
        """
        Verify lying down smoothly without rapid collapse does not trigger fall confirmation.
        """
        track_id = 606

        # Upright
        for f in range(1, 15):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(200.0, 300.0, 260.0, 500.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )

        # Gradual controlled descent over 25 frames (cy drop rate < 0.015 per frame, well below 0.04 threshold)
        for f in range(15, 40):
            dy = (f - 15) * 2.5
            dh = (f - 15) * 2.0
            dw = (f - 15) * 2.0
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(200.0, 300.0 + dy, 260.0 + dw, 500.0 + dy - dh),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertFalse(self.detector.fall_state[track_id]["has_dynamic_drop"])

    def test_07_repeated_alert_suppression_and_event_id_stability(self):
        """
        Verify that a single continuous fall generates exactly ONE event_id
        that remains constant for the entire incident duration.
        """
        track_id = 707

        # Establish fall
        for f in range(1, 10):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(500.0, 400.0, 560.0, 600.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        for f in range(10, 16):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(480.0, 580.0, 660.0, 680.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )

        initial_event_id = self.detector.fall_state[track_id]["event_id"]
        self.assertIsNotNone(initial_event_id)

        # Remain fallen for 80 frames
        for f in range(16, 96):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(480.0, 580.0, 660.0, 680.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertEqual(res["event_id"], initial_event_id)
            self.assertEqual(res["status"], FallDetector.STATE_CONFIRMED)

        # Confirm only 1 event was logged in active_fall_events
        self.assertEqual(len(self.detector.active_fall_events), 1)

    def test_08_spatial_reassociation_distance_gating(self):
        """
        Verify that spatial re-association strictly respects the 160px distance gate,
        preventing merging with unrelated bystanders walking nearby.
        """
        track_id = 808

        # Establish fall at (300, 700)
        for f in range(1, 10):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(270.0, 500.0, 330.0, 700.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
        for f in range(10, 16):
            self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(250.0, 650.0, 430.0, 750.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )

        # Last known center: X = (250+430)/2 = 340, Y = (650+750)/2 = 700
        # Bystander appears at X = 600, Y = 700 (Distance = 260px > 160px)
        distant_candidate = [520.0, 650.0, 680.0, 750.0]  # AR = 160/100 = 1.60

        persist_out = self.detector.process_frame_persistence(
            frame_idx=17,
            observed_track_ids=[],
            raw_candidate_boxes=[distant_candidate],
            frame_h=self.frame_h,
            frame_w=self.frame_w
        )
        self.assertEqual(len(persist_out), 1)
        # Must NOT re-associate with distant candidate! Should remain in LOST_COASTING
        self.assertEqual(persist_out[0]["tracking_state"], "LOST_COASTING")
        self.assertFalse(persist_out[0]["is_live_observation"])

    def test_09_image_boundary_and_edge_pedestrians(self):
        """
        Verify that pedestrians entering or exiting near frame edges
        (where bounding boxes are truncated by the screen boundary)
        do not produce spurious fall confirmations.
        """
        track_id = 909

        # Truncated box at top boundary: Y = 0 to 80, width 120 (AR = 1.50)
        # But near_floor is False (Y2 = 80 << 0.40 * 1080 = 432)
        for f in range(1, 15):
            res = self.detector.evaluate_pedestrian(
                track_id=track_id,
                bbox=(500.0, 0.0, 620.0, 80.0),
                frame_idx=f,
                frame_h=self.frame_h,
                frame_w=self.frame_w
            )
            self.assertNotEqual(res["status"], FallDetector.STATE_CONFIRMED)


if __name__ == "__main__":
    unittest.main()
