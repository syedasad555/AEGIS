#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-system integration — corridor flow, two anomalies, WS schema.
"""

import json
import sys
import unittest
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import open_video, get_video_props
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine


class TestFullSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.config = {
            "fps": 30.0,
            "expected_evacuation_direction": "EAST",
            "prolonged_stopping_frames": 15,
            "direction_detection": {
                "angular_tolerance_deg": 30.0,
                "wrong_way_sustain_frames": 1,
                "recovery_frames": 2,
                "cooldown_sec": 0.0,
                "require_in_corridor": False,
            },
            "stopping_detection": {
                "enabled": True,
                "threshold_sec": 0.4,
                "min_frames": 15,
                "critical_sec": 20.0,
            },
            "evacuation_corridor": {
                "id": "corridor_main",
                "name": "MAIN CORRIDOR",
                "polygon": [[50, 100], [1800, 100], [1800, 900], [50, 900]],
            },
            "zones": {},
        }
        self.extractor = TrajectoryFeatureExtractor(self.config)
        self.anomaly_engine = EvacuationAnomalyEngine(self.config)

    def test_01_video_loading_and_format_support(self):
        mp4_path = PROJECT_ROOT / "assets" / "intrusion.mp4"
        if mp4_path.exists():
            cap = open_video(mp4_path)
            self.assertTrue(cap.isOpened())
            props = get_video_props(cap)
            self.assertGreater(props["width"], 0)
            cap.release()

        avi_path = PROJECT_ROOT / "assets" / "evacuation_test_corridor.avi"
        if avi_path.exists():
            cap = open_video(avi_path)
            self.assertTrue(cap.isOpened())
            cap.release()

        invalid_path = PROJECT_ROOT / "assets" / "non_existent_file_xyz.mp4"
        with self.assertRaises(SystemExit):
            open_video(invalid_path)

    def test_02_two_rule_anomaly_generation(self):
        feat_wrong = {
            "track_id": 1,
            "frame": 10,
            "timestamp": 0.33,
            "bbox": (100.0, 300.0, 160.0, 500.0),
            "foot": (130.0, 500.0),
            "speed_px_sec": 45.0,
            "direction": "WEST",
            "direction_vector": [-40.0, 0.0],
            "direction_reliable": True,
            "direction_confidence": 0.9,
            "stopping_duration_frames": 0,
            "in_corridor": True,
            "zone_type": "EVACUATION_CORRIDOR",
        }
        res1 = self.anomaly_engine.evaluate_pedestrian(feat_wrong)
        self.assertTrue(res1["anomaly"])
        self.assertEqual(res1["anomaly_type"], "WRONG_DIRECTION")
        self.assertEqual(res1["severity"], "WARNING")
        self.assertEqual(res1["visual_state"], "anomaly")

        feat_stop = {
            "track_id": 2,
            "frame": 25,
            "timestamp": 0.83,
            "bbox": (300.0, 300.0, 360.0, 500.0),
            "foot": (330.0, 500.0),
            "speed_px_sec": 1.0,
            "direction": "STATIONARY",
            "direction_vector": [0.0, 0.0],
            "direction_reliable": False,
            "direction_confidence": 0.2,
            "stopping_duration_frames": 20,
            "stopping_duration_sec": 0.67,
            "in_corridor": True,
            "zone_type": "EVACUATION_CORRIDOR",
        }
        res2 = self.anomaly_engine.evaluate_pedestrian(feat_stop)
        self.assertTrue(res2["anomaly"])
        self.assertEqual(res2["anomaly_type"], "PROLONGED_STOPPING")
        self.assertIsNone(self.anomaly_engine.fall_detector)

    def test_03_websocket_payload_schema_consistency(self):
        sample_pedestrian = {
            "track_id": 42,
            "frame": 120,
            "timestamp": 4.0,
            "bbox": [100.0, 200.0, 180.0, 400.0],
            "foot": [140.0, 400.0],
            "speed_px_sec": 35.5,
            "direction": "EAST",
            "zone_type": "EVACUATION_CORRIDOR",
            "in_corridor": True,
            "stopping_duration_frames": 0,
            "anomaly": False,
            "anomaly_type": "NORMAL",
            "all_anomalies": [],
            "severity": "NORMAL",
            "duration_sec": 0.0,
            "visual_state": "normal",
            "fall_status": "DISABLED",
            "fall_confidence": 0.0,
            "fall_reasons": [],
            "fall_event_id": None,
            "fall_tracking_state": "TRACKED",
            "is_live_observation": True,
            "event_id": None,
        }
        sample_alert = {
            "track_id": 42,
            "type": "WRONG_DIRECTION",
            "severity": "WARNING",
            "duration_sec": 2.5,
            "timestamp": 4.0,
            "speed": 35.5,
            "direction": "WEST",
            "event_id": "evt-test",
            "status": "ACTIVE",
            "is_live_observation": True,
        }
        ws_payload = {
            "type": "frame_update",
            "frame": 120,
            "total_frames": 300,
            "timestamp": 4.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "video_name": "test_stream.mp4",
            "frame_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
            "pedestrians": [sample_pedestrian],
            "active_alerts": [sample_alert],
            "expected_direction": "EAST",
            "evacuation_corridor": self.config["evacuation_corridor"],
            "zones": {},
            "legend": {"normal": "blue", "anomaly": "red", "selected": "amber"},
            "has_pedestrians": True,
            "empty_warning": False,
        }
        serialized = json.dumps(ws_payload)
        deserialized = json.loads(serialized)
        for k in ("type", "pedestrians", "active_alerts", "expected_direction", "evacuation_corridor", "legend"):
            self.assertIn(k, deserialized)
        ped = deserialized["pedestrians"][0]
        self.assertEqual(ped["visual_state"], "normal")
        self.assertEqual(ped["fall_status"], "DISABLED")

    def test_04_digital_twin_coordinate_mapping(self):
        video_w, video_h = 1920, 1080
        self.assertAlmostEqual(960.0 / video_w, 0.50, delta=0.001)
        self.assertAlmostEqual(540.0 / video_h, 0.50, delta=0.001)
        clamped_x = max(0.0, min(1.0, -50.0 / video_w))
        clamped_y = max(0.0, min(1.0, 1200.0 / video_h))
        self.assertEqual(clamped_x, 0.0)
        self.assertEqual(clamped_y, 1.0)

    def test_05_real_video_pipeline_smoke(self):
        video_path = PROJECT_ROOT / "assets" / "intrusion.mp4"
        if not video_path.exists():
            video_path = PROJECT_ROOT / "assets" / "running.mp4"
        if not video_path.exists():
            self.skipTest("No demo video found in assets")

        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened())
        extractor = TrajectoryFeatureExtractor(self.config)
        anomaly_engine = EvacuationAnomalyEngine(self.config)
        frame_count = 0
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            feat = extractor.update_pedestrian(1, (200.0, 400.0, 280.0, 600.0), frame_count, 30.0)
            eval_res = anomaly_engine.evaluate_pedestrian(feat)
            self.assertEqual(eval_res["track_id"], 1)
            self.assertEqual(eval_res.get("fall_status"), "DISABLED")
            self.assertIn(eval_res["visual_state"], ("normal", "anomaly", "unknown"))
        cap.release()
        self.assertEqual(frame_count, 30)


if __name__ == "__main__":
    unittest.main()
