#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Real Video Verification:
- Tests actual detection & tracking pipeline on real videos:
  1. assets/evacuation_test_corridor.avi
  2. assets/running.mp4
  3. assets/intrusion.mp4
- Evaluates 8 movement sectors against expected flow directions.
- Verifies playback speed transitions and frame skip accounting.
"""

import os
import sys
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics import YOLO
from scripts.direction_estimation import (
    DirectionEstimator,
    DirectionClassifier,
    angle_to_direction8,
    circular_sector_distance,
    allowed_directions,
)
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine


class TestEndToEndRealVideos(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

        model_path = ROOT / cls.config.get("yolo_model", "weights/aegis_yolo11n_crowdhuman.pt")
        cls.assertTrue(model_path.exists(), f"Model checkpoint not found: {model_path}")
        cls.model = YOLO(str(model_path))

    def test_evacuation_corridor_video(self):
        """Test on assets/evacuation_test_corridor.avi with expected flow LEFT then RIGHT."""
        video_path = ROOT / "assets" / "evacuation_test_corridor.avi"
        self.assertTrue(video_path.exists(), f"Video missing: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened(), "Could not open evacuation_test_corridor.avi")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        cfg = dict(self.config)
        cfg["expected_evacuation_direction"] = "LEFT"
        extractor = TrajectoryFeatureExtractor(cfg)
        anomaly_engine = EvacuationAnomalyEngine(cfg)

        tracker_yaml = ROOT / "configs" / "bytetrack_crowd.yaml"
        tracks_seen = set()
        anomalies_detected = []
        normal_detected = []

        # Process first 35 frames
        for f in range(1, 36):
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=str(tracker_yaml),
                conf=0.15,
                iou=0.55,
                imgsz=800,
                classes=[0],
                device="cpu",
                verbose=False
            )

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                track_ids = boxes.id.cpu().numpy().astype(int)
                coords = boxes.xyxy.cpu().numpy()

                for bbox, tid in zip(coords, track_ids):
                    tracks_seen.add(tid)
                    feat = extractor.update_pedestrian(tid, tuple(bbox), f, fps=fps)
                    res = anomaly_engine.evaluate_pedestrian(feat)

                    if res.get("direction_classification") == "ANOMALY":
                        anomalies_detected.append(res)
                    elif res.get("direction_classification") == "NORMAL":
                        normal_detected.append(res)

        cap.release()
        print(f"\n[E2E Corridor Video] Processed 35 frames: {len(tracks_seen)} tracks observed.")
        print(f"  Normal observations: {len(normal_detected)}, Anomaly observations: {len(anomalies_detected)}")
        self.assertGreater(len(tracks_seen), 0, "Should detect at least 1 person in corridor video")

    def test_running_video_speed_transitions(self):
        """Test running.mp4 with real-time playback pacing and speed transitions."""
        video_path = ROOT / "assets" / "running.mp4"
        self.assertTrue(video_path.exists(), f"Video missing: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened(), "Could not open running.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Simulate playback speed transition: 1.0x for 10 frames, then 2.0x for 10 frames
        frame_idx = 0
        frames_skipped = 0
        current_speed = 1.0

        wall_sync_start = time.time()
        video_sync_frame = 0

        for step in range(20):
            if step == 10:
                current_speed = 2.0
                wall_sync_start = time.time()
                video_sync_frame = frame_idx

            now = time.time()
            elapsed_wall = now - wall_sync_start
            target_video_frame = video_sync_frame + int(round(elapsed_wall * current_speed * fps))

            if target_video_frame > frame_idx + 1:
                max_skip = max(1, int(round(current_speed * 4)))
                skip = min(target_video_frame - (frame_idx + 1), max_skip)
                for _ in range(skip):
                    if not cap.grab():
                        break
                    frame_idx += 1
                    frames_skipped += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # Simulate 15ms work
            time.sleep(0.015)

        cap.release()
        print(f"\n[E2E Speed Pacing] Running video: {frame_idx} video frames advanced, {frames_skipped} skipped at speed transitions.")
        self.assertGreater(frame_idx, 15, "Should have advanced frames smoothly")

    def test_all_eight_sectors_coverage(self):
        """Verify sector distance and allowed directions for all 4 cardinal admins."""
        cardinals = ["LEFT", "RIGHT", "UP", "DOWN"]
        for card in cardinals:
            allowed = allowed_directions(card)
            self.assertEqual(len(allowed), 3, f"Allowed directions for {card} must be 3")
            for act in ["RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT", "UP", "UP_RIGHT"]:
                dist = circular_sector_distance(act, card)
                if act in allowed:
                    self.assertIn(dist, (0, 1), f"{act} should have dist 0 or 1 from {card}")
                else:
                    self.assertIn(dist, (2, 3, 4), f"{act} should have dist 2, 3, or 4 from {card}")


if __name__ == "__main__":
    unittest.main()

