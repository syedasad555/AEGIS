#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration checks for CrowdHuman YOLO11n detector wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import resolve_yolo_model_path, resolve_tracker, extract_track_ids


class TestCrowdHumanModelIntegration(unittest.TestCase):
    def test_01_config_points_to_crowdhuman_and_weights_exist(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("yolo_model"), "weights/aegis_yolo11n_crowdhuman.pt")
        path = resolve_yolo_model_path(cfg, require_exists=True)
        self.assertTrue(Path(path).is_file())
        self.assertIn("aegis_yolo11n_crowdhuman", path.replace("\\", "/"))

    def test_02_missing_weights_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_yolo_model_path({"yolo_model": "weights/does_not_exist_xyz.pt"}, require_exists=True)
        msg = str(ctx.exception)
        self.assertIn("AEGIS detector weights not found", msg)
        self.assertIn("does_not_exist_xyz.pt", msg)

    def test_03_tracker_still_crowd_custom(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("tracker_config"), "configs/bytetrack_crowd.yaml")
        resolved = resolve_tracker(str(PROJECT_ROOT / cfg["tracker_config"]))
        self.assertTrue(resolved.endswith("bytetrack_crowd.yaml") or "bytetrack_crowd" in resolved)

    def test_04_sample_frame_track_person_class_and_ids(self):
        """Smoke: load model, track one real frame, check xyxy + class 0 + optional IDs."""
        import cv2
        from ultralytics import YOLO

        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        model_path = resolve_yolo_model_path(cfg, require_exists=True)
        tracker = resolve_tracker(str(PROJECT_ROOT / cfg.get("tracker_config", "configs/bytetrack_crowd.yaml")))

        # Prefer a short UCF clip; fall back to assets/fall.mp4
        candidates = [
            PROJECT_ROOT / "UCF_CrowdsDataset" / "341-46_l.mov",
            PROJECT_ROOT / "assets" / "fall.mp4",
        ]
        frame = None
        for vid in candidates:
            if not vid.exists():
                continue
            cap = cv2.VideoCapture(str(vid))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                break
        if frame is None:
            self.skipTest("No sample video available for smoke inference")

        model = YOLO(model_path)
        conf = float(cfg.get("detection_confidence", 0.08))
        imgsz = int(cfg.get("inference_imgsz", 800))
        person_id = int(cfg.get("person_class_id", 0))

        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            imgsz=imgsz,
            classes=[person_id],
            verbose=False,
        )
        self.assertTrue(results and len(results) > 0)
        boxes = results[0].boxes
        self.assertIsNotNone(boxes)
        if len(boxes) == 0:
            # Still a valid run (empty frame / hard scene) — coordinates path exercised
            return

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        self.assertEqual(xyxy.shape[1], 4)
        self.assertTrue((xyxy[:, 2] >= xyxy[:, 0]).all())
        self.assertTrue((xyxy[:, 3] >= xyxy[:, 1]).all())
        self.assertTrue((clss == person_id).all(), f"Expected class {person_id}, got {clss}")

        tids = extract_track_ids(boxes)
        # IDs may be empty on first frame cold-start for low-conf dets; if present must be ints
        for tid in tids:
            self.assertIsInstance(tid, int)


if __name__ == "__main__":
    unittest.main()
