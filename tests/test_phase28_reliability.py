#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 28 — Reliability Regression Tests
=============================================
1. FALL_SUSPECTED vs FALL_CONFIRMED separation under Phase 28 gates.
2. Sitting / bending must not CONFIRM without dynamic collapse.
3. Track reassignment cooldown suppresses false CONFIRM.
4. Detection config retains CPU-balanced imgsz=800 + crowd tracker (no silent tiled enable).
"""

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fall_detection import FallDetector


@unittest.skip("Fall detection removed from AEGIS — see test_zone_direction_anomalies")
class TestPhase28FallReliability(unittest.TestCase):
    def setUp(self):
        self.config = {
            "fall_ar_suspect_threshold": 0.72,
            "fall_ar_confirm_threshold": 0.90,
            "fall_height_drop_ratio": 0.62,
            "fall_centroid_drop_threshold": 0.04,
            "fall_confirm_frames": 6,
            "fall_max_ground_speed": 55.0,
            "fall_min_history_for_confirm": 8,
            "fall_confirm_require_dynamic_drop": True,
            "fall_pose_confirm_frames": 4,
            "fall_max_confirm_ground_speed": 55.0,
            "fall_post_reassign_cooldown": 8,
            "fall_coast_frames_max": 60,
            "fall_spatial_reassoc_dist": 160.0,
            "fall_recovery_confirm_frames": 12,
            "fall_pose_model": None,
        }
        self.det = FallDetector(self.config, pose_model_path=None)
        self.fh, self.fw = 1080, 1920

    def test_01_static_horizontal_sitting_does_not_confirm(self):
        """Wide seated bbox without abrupt centroid drop must not CONFIRM."""
        tid = 2801
        for f in range(1, 20):
            self.det.evaluate_pedestrian(
                tid, (700.0, 400.0, 760.0, 600.0), f, self.fh, self.fw, speed_px_sec=10.0
            )
        # Gradual sit: AR becomes high but cy change is slow (no dynamic drop)
        last = None
        for f in range(20, 50):
            last = self.det.evaluate_pedestrian(
                tid,
                (690.0, 470.0, 850.0, 590.0),  # AR≈1.33, near floor, no rapid drop
                f,
                self.fh,
                self.fw,
                speed_px_sec=8.0,
            )
        self.assertNotEqual(last["status"], FallDetector.STATE_CONFIRMED)
        self.assertIsNone(last["event_id"])

    def test_02_genuine_collapse_still_confirms(self):
        """Rapid centroid drop + prone posture must still reach FALL_CONFIRMED."""
        tid = 2802
        for f in range(1, 12):
            self.det.evaluate_pedestrian(
                tid, (500.0, 400.0, 560.0, 600.0), f, self.fh, self.fw, speed_px_sec=20.0
            )
        # Abrupt collapse
        for f in range(12, 16):
            self.det.evaluate_pedestrian(
                tid,
                (490.0, 450.0 + (f - 12) * 40.0, 640.0 + (f - 12) * 15.0, 650.0 + (f - 12) * 15.0),
                f,
                self.fh,
                self.fw,
                speed_px_sec=40.0,
            )
        last = None
        for f in range(16, 28):
            last = self.det.evaluate_pedestrian(
                tid,
                (480.0, 620.0, 670.0, 720.0),
                f,
                self.fh,
                self.fw,
                speed_px_sec=5.0,
            )
        self.assertEqual(last["status"], FallDetector.STATE_CONFIRMED)
        self.assertIsNotNone(last["event_id"])

    def test_03_track_reassignment_cooldown_blocks_instant_confirm(self):
        """Large spatial jump clears fall evidence and applies cooldown."""
        tid = 2803
        for f in range(1, 10):
            self.det.evaluate_pedestrian(
                tid, (200.0, 400.0, 260.0, 600.0), f, self.fh, self.fw, speed_px_sec=15.0
            )
        # Jump across frame (reassignment) then immediately horizontal near floor
        res = self.det.evaluate_pedestrian(
            tid,
            (1400.0, 700.0, 1600.0, 820.0),
            20,  # frame gap > 10 from f=9
            self.fh,
            self.fw,
            speed_px_sec=10.0,
        )
        self.assertNotEqual(res["status"], FallDetector.STATE_CONFIRMED)
        self.assertGreater(self.det.fall_state[tid].get("reassign_cooldown", 0), 0)

    def test_04_suspect_can_fire_without_confirm(self):
        """Early collapse indicators may SUSPECT before CONFIRM evidence matures."""
        tid = 2804
        for f in range(1, 10):
            self.det.evaluate_pedestrian(
                tid, (500.0, 400.0, 560.0, 600.0), f, self.fh, self.fw
            )
        statuses = []
        for f in range(10, 14):
            r = self.det.evaluate_pedestrian(
                tid,
                (495.0, 430.0 + (f - 10) * 30.0, 600.0 + (f - 10) * 20.0, 620.0 + (f - 10) * 10.0),
                f,
                self.fh,
                self.fw,
            )
            statuses.append(r["status"])
        self.assertTrue(
            any(s == FallDetector.STATE_SUSPECTED for s in statuses)
            or statuses[-1] in (FallDetector.STATE_SUSPECTED, FallDetector.STATE_CONFIRMED)
        )


class TestPhase28DetectionConfig(unittest.TestCase):
    def test_01_cpu_balanced_config_preserved(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(int(cfg.get("inference_imgsz", 0)), 800)
        self.assertFalse(bool(cfg.get("tiled_inference", {}).get("enabled", False)))
        self.assertIn("crowd", str(cfg.get("tracker_config", "")))
        # Fall detection removed — confirm knobs should not be required
        self.assertNotIn("fall_confirm_require_dynamic_drop", cfg)


if __name__ == "__main__":
    unittest.main()
