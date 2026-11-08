#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 33 — far_camera_profile remains disabled; resolve_inference_imgsz session rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import resolve_inference_imgsz


class TestPhase33FarCameraProfile(unittest.TestCase):
    def test_01_production_defaults_preserved(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        # Detector is configurable; default is CrowdHuman fine-tune (rollback: yolov8n.pt)
        self.assertTrue(str(cfg.get("yolo_model", "")).endswith("aegis_yolo11n_crowdhuman.pt")
                        or cfg.get("yolo_model") == "yolov8n.pt")
        self.assertEqual(int(cfg.get("inference_imgsz", 0)), 800)
        self.assertEqual(cfg.get("tracker_config"), "configs/bytetrack_crowd.yaml")
        self.assertFalse(bool(cfg.get("tiled_inference", {}).get("enabled", False)))
        far = cfg.get("far_camera_profile") or {}
        self.assertFalse(bool(far.get("enabled", False)), "far profile must stay disabled by default")
        self.assertEqual(int(far.get("inference_imgsz", 0)), 960)

    def test_02_resolve_uses_baseline_when_disabled(self):
        cfg = {
            "inference_imgsz": 800,
            "far_camera_profile": {"enabled": False, "inference_imgsz": 960},
        }
        self.assertEqual(resolve_inference_imgsz(cfg), 800)

    def test_03_resolve_uses_far_when_enabled(self):
        cfg = {
            "inference_imgsz": 800,
            "far_camera_profile": {"enabled": True, "inference_imgsz": 960},
        }
        self.assertEqual(resolve_inference_imgsz(cfg), 960)

    def test_04_missing_far_block_falls_back(self):
        self.assertEqual(resolve_inference_imgsz({"inference_imgsz": 800}), 800)
        self.assertEqual(resolve_inference_imgsz({}), 800)


if __name__ == "__main__":
    unittest.main()
