#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 30 — Tracking diagnosis regression tests.

Locks evidence from 4917-5 (dets without track IDs under crowd ByteTrack)
without changing the production baseline tracker YAML.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import extract_track_ids


class TestPhase30BaselineConfigFrozen(unittest.TestCase):
    def test_01_crowd_baseline_thresholds_unchanged(self):
        """Production crowd tracker must keep Phase 29 thresholds until an explicit A/B decision."""
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("tracker_config"), "configs/bytetrack_crowd.yaml")
        self.assertEqual(int(cfg.get("inference_imgsz", 0)), 800)
        self.assertAlmostEqual(float(cfg.get("detection_confidence", 0)), 0.08)
        tiled = cfg.get("tiled_inference") or {}
        self.assertFalse(bool(tiled.get("enabled", False)))

        thr = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "bytetrack_crowd.yaml").read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(float(thr["new_track_thresh"]), 0.30)
        self.assertAlmostEqual(float(thr["track_high_thresh"]), 0.35)
        self.assertAlmostEqual(float(thr["track_low_thresh"]), 0.05)

    def test_02_far_candidate_yaml_exists_but_is_not_wired(self):
        """Tuning candidate may exist for A/B; must not be the active config tracker."""
        cand = PROJECT_ROOT / "configs" / "bytetrack_crowd_far_candidate.yaml"
        self.assertTrue(cand.exists(), "Phase 30 far-scene candidate YAML missing")
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        self.assertNotEqual(cfg.get("tracker_config"), "configs/bytetrack_crowd_far_candidate.yaml")
        thr = yaml.safe_load(cand.read_text(encoding="utf-8"))
        self.assertLess(float(thr["new_track_thresh"]), 0.30)
        # High-score pool gate must also be low — Ultralytics only inits tracks from remain_inds
        self.assertLessEqual(float(thr["track_high_thresh"]), float(thr["new_track_thresh"]) + 1e-6)

    def test_03_extract_track_ids_never_fabricates_when_id_missing(self):
        class _Boxes:
            id = None

        self.assertEqual(extract_track_ids(_Boxes()), [])


class TestPhase3049175Reproduce(unittest.TestCase):
    """Reproduce Phase 29 failure: detections present, zero track IDs on 4917-5."""

    @classmethod
    def setUpClass(cls):
        cls.video = PROJECT_ROOT / "UCF_CrowdsDataset" / "4917-5_70.mov"
        cls.skip_reason = None
        if not cls.video.exists():
            cls.skip_reason = "UCF clip 4917-5_70.mov not present"

    def test_04_4917_5_dets_without_tracks_under_baseline(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

        from tools.diagnose_tracking_phase30 import diagnose_clip

        # Short window is enough to reproduce; keep runtime bounded for the suite
        res = diagnose_clip(
            relpath="UCF_CrowdsDataset/4917-5_70.mov",
            max_frames=20,
            imgsz=800,
            conf=0.08,
            tracker_rel="configs/bytetrack_crowd.yaml",
            device="cpu",
            dump_every_box=False,
        )
        self.assertEqual(res["status"], "OK", res)
        s = res["summary"]
        self.assertGreater(s["frames_with_detections"], 0)
        self.assertEqual(s["frames_with_tracks"], 0)
        self.assertEqual(s["unique_track_ids"], 0)
        self.assertEqual(s["frames_with_dets_but_zero_ids"], res["frames_processed"])
        # Evidence: almost all YOLO boxes sit below ByteTrack new_track_thresh
        self.assertGreaterEqual(s["frac_boxes_below_new_track_thresh"], 0.90)
        self.assertLess(s["conf_all_mean"], 0.30)
        self.assertEqual(res["exceptions"], [])


if __name__ == "__main__":
    unittest.main()
