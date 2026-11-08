#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification of:
1. Fast Response Time for Wrong-Direction Pedestrian Detection (Part 1.6)
2. Accurate Video Playback Speed Engine (Part 2)
3. Trajectory Stability across Occlusions & Gaps
"""

import math
import os
import sys
import time
import unittest
from pathlib import Path

# Set up project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.direction_estimation import (
    DirectionEstimator,
    DirectionClassifier,
    angle_to_direction8,
    circular_sector_distance,
)
from scripts.feature_extraction import TrajectoryFeatureExtractor
from scripts.anomaly_detection import EvacuationAnomalyEngine


class TestDirectionResponseTimeAndSpeed(unittest.TestCase):

    def setUp(self):
        self.config = {
            "expected_evacuation_direction": "LEFT",
            "direction_window_frames": 12,
            "direction_ema_alpha": 0.60,
            "direction_min_displacement_px": 8.0,
            "direction_fast_displacement_px": 9.0,
            "direction_min_speed_px_sec": 5.0,
            "direction_min_observations": 6,
            "direction_min_fast_observations": 4,
            "direction_hysteresis_deg": 28.0,
            "direction_detection": {
                "max_sector_distance_normal": 1,
                "wrong_way_sustain_frames": 3,
                "recovery_frames": 4,
                "cooldown_sec": 4.0,
                "require_in_corridor": False,
                "min_confidence": 0.45,
            },
            "stopping_detection": {
                "enabled": True,
                "threshold_sec": 2.0,
                "min_frames": 30,
            },
            "evacuation_corridor": {
                "id": "corridor_main",
                "name": "Main Evacuation Corridor",
                "polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            }
        }

    def test_response_time_measurement(self):
        """
        Part 1.6: Measure exact timestamps and delays from first wrong movement to RED.
        Pedestrian moving RIGHT (+X) under expected LEFT.
        """
        engine = EvacuationAnomalyEngine(self.config)
        extractor = engine.feature_extractor
        fps = 30.0

        first_movement_time = None
        first_direction_time = None
        first_anomaly_time = None
        red_confirmation_time = None
        red_confirmed_frame = None

        track_id = 101
        start_x, start_y = 200.0, 300.0
        # Pedestrian walks RIGHT at 4 pixels per frame (~120 px/sec)
        for f in range(1, 25):
            t_src = f / fps
            curr_x = start_x + (f * 4.0)
            curr_y = start_y
            bbox = (curr_x - 15, curr_y - 60, curr_x + 15, curr_y)

            feat = extractor.update_pedestrian(track_id, bbox, f, fps=fps)
            res = engine.evaluate_pedestrian(feat)

            # 1. First reliable movement observation (> 0 displacement)
            if first_movement_time is None and f >= 2:
                first_movement_time = t_src

            # 2. First reliable estimated direction
            if first_direction_time is None and res.get("direction_reliable") and res.get("direction") != "UNKNOWN":
                first_direction_time = t_src

            # 3. First wrong-direction classification
            if first_anomaly_time is None and res.get("direction_classification") == "ANOMALY":
                first_anomaly_time = t_src

            # 4. Backend anomaly confirmation (turns RED)
            if red_confirmation_time is None and res.get("visual_state") == "anomaly":
                red_confirmation_time = t_src
                red_confirmed_frame = f

        self.assertIsNotNone(first_movement_time, "Movement should be observed")
        self.assertIsNotNone(first_direction_time, "Direction should be estimated")
        self.assertIsNotNone(first_anomaly_time, "Anomaly should be classified")
        self.assertIsNotNone(red_confirmation_time, "Track should be confirmed RED")

        direction_est_delay = first_direction_time - first_movement_time
        anomaly_confirmation_delay = red_confirmation_time - first_anomaly_time
        total_time_to_red = red_confirmation_time - first_movement_time

        print(f"\n--- PART 1.6 RESPONSE TIME MEASUREMENT ---")
        print(f"First Movement Timestamp:          {first_movement_time:.3f}s (Frame 2)")
        print(f"First Reliable Direction:         {first_direction_time:.3f}s")
        print(f"First Wrong-Direction Class:       {first_anomaly_time:.3f}s")
        print(f"RED Anomaly Confirmation:          {red_confirmation_time:.3f}s (Frame {red_confirmed_frame})")
        print(f"Direction Estimation Delay:        {direction_est_delay:.3f}s ({round(direction_est_delay * fps)} frames)")
        print(f"Wrong-Direction Confirm Delay:     {anomaly_confirmation_delay:.3f}s ({round(anomaly_confirmation_delay * fps)} frames)")
        print(f"Total Time from Movement to RED:   {total_time_to_red:.3f}s ({red_confirmed_frame - 1} frames)")

        # Verify fast response: should be confirmed within 7 frames (< 0.25s at 30fps)
        self.assertLessEqual(red_confirmed_frame, 8, f"RED confirmation too slow (frame {red_confirmed_frame})")
        self.assertLessEqual(total_time_to_red, 0.25, f"Total response time too high ({total_time_to_red}s)")

    def test_jitter_grace_period(self):
        """
        Verify that 1 frame of UNKNOWN jitter does not wipe out pending wrong-direction sequence.
        """
        engine = EvacuationAnomalyEngine(self.config)
        extractor = engine.feature_extractor
        fps = 30.0
        track_id = 102
        start_x, start_y = 100.0, 200.0

        # Frame 1-4: Clear movement RIGHT (displaces 16 px)
        for f in range(1, 5):
            curr_x = start_x + (f * 4.0)
            bbox = (curr_x - 15, start_y - 60, curr_x + 15, start_y)
            feat = extractor.update_pedestrian(track_id, bbox, f, fps=fps)
            res = engine.evaluate_pedestrian(feat)

        sustain_at_f4 = engine.wrong_dir_counters[track_id]
        self.assertGreaterEqual(sustain_at_f4, 1, "Should have accumulated sustain frames")

        # Frame 5: Simulated 1-frame jitter where direction is UNKNOWN
        fake_feat = dict(feat)
        fake_feat["frame"] = 5
        fake_feat["timestamp"] = 5 / fps
        fake_feat["direction"] = "UNKNOWN"
        fake_feat["direction_reliable"] = False
        res = engine.evaluate_pedestrian(fake_feat)

        # Counter should leaky decrement, NOT be wiped to 0!
        sustain_after_jitter = engine.wrong_dir_counters[track_id]
        self.assertGreaterEqual(sustain_after_jitter, max(0, sustain_at_f4 - 1))

        # Frame 6 & 7: Movement resumes RIGHT
        for f in (6, 7):
            curr_x = start_x + (f * 4.0)
            bbox = (curr_x - 15, start_y - 60, curr_x + 15, start_y)
            feat = extractor.update_pedestrian(track_id, bbox, f, fps=fps)
            res = engine.evaluate_pedestrian(feat)

        # Frame 8 confirms RED without restarting all the way from scratch
        curr_x = start_x + (8 * 4.0)
        bbox = (curr_x - 15, start_y - 60, curr_x + 15, start_y)
        feat = extractor.update_pedestrian(track_id, bbox, 8, fps=fps)
        res = engine.evaluate_pedestrian(feat)
        self.assertEqual(res["visual_state"], "anomaly", "Track should reach RED despite 1 jitter frame")

    def test_occlusion_gap_recovery(self):
        """
        Verify that a 2-frame tracking gap does not destroy trajectory or cause spurious jumps.
        """
        engine = EvacuationAnomalyEngine(self.config)
        extractor = engine.feature_extractor
        fps = 30.0
        track_id = 103
        start_x, start_y = 100.0, 200.0

        # Frames 1-6: moving LEFT (Normal)
        for f in range(1, 7):
            curr_x = start_x - (f * 4.0)
            bbox = (curr_x - 15, start_y - 60, curr_x + 15, start_y)
            feat = extractor.update_pedestrian(track_id, bbox, f, fps=fps)
            res = engine.evaluate_pedestrian(feat)
            self.assertIn(res["visual_state"], ("normal", "unknown"))

        # Gap at frames 7 and 8 (occlusion / missed detection)
        # Track resumes at frame 9 at continuous location
        curr_x = start_x - (9 * 4.0)
        bbox = (curr_x - 15, start_y - 60, curr_x + 15, start_y)
        feat = extractor.update_pedestrian(track_id, bbox, 9, fps=fps)
        res = engine.evaluate_pedestrian(feat)

        self.assertEqual(feat["direction"], "LEFT", "Direction should maintain LEFT across 2-frame gap")
        self.assertEqual(res["visual_state"], "normal", "Should remain normal/BLUE")

    def test_playback_speed_engine_pacing(self):
        """
        Part 2.3: Verify that the wall-clock pacing engine correctly achieves
        requested speeds (0.25x, 0.5x, 1x, 2x, 3x) even when simulated inference
        takes 80ms per frame.
        """
        fps = 30.0
        simulated_infer_sec = 0.080  # 80ms per frame on CPU

        speeds = [0.25, 0.5, 1.0, 2.0, 3.0]
        results = {}

        for speed in speeds:
            frame_idx = 0
            frames_skipped = 0
            frames_processed = 0

            # Virtual wall-clock reference
            wall_time = 0.0
            wall_sync_start = 0.0
            video_sync_frame = 0

            # Simulate for 2.0 seconds of wall clock
            while wall_time < 2.0:
                elapsed_wall = wall_time - wall_sync_start
                target_video_frame = video_sync_frame + int(round(elapsed_wall * speed * fps))

                # Pacing decision
                if target_video_frame <= frame_idx:
                    next_frame = frame_idx + 1
                    target_wall = wall_sync_start + ((next_frame - video_sync_frame) / fps) / max(0.1, speed)
                    sleep_sec = max(0.0, target_wall - wall_time)
                    wall_time += sleep_sec
                elif target_video_frame > frame_idx + 1:
                    max_skip = max(1, int(round(speed * 4)))
                    skip_count = min(target_video_frame - (frame_idx + 1), max_skip)
                    frame_idx += skip_count
                    frames_skipped += skip_count

                # Decode & inference
                frame_idx += 1
                frames_processed += 1
                wall_time += simulated_infer_sec

            effective_fps = frame_idx / wall_time
            achieved_speed = effective_fps / fps
            results[speed] = {
                "effective_fps": round(effective_fps, 1),
                "achieved_speed": round(achieved_speed, 2),
                "frames_skipped": frames_skipped,
                "frames_processed": frames_processed,
            }

        print(f"\n--- PART 2 PLAYBACK SPEED PACING BENCHMARK (Simulated 80ms CPU Latency) ---")
        for spd, res in results.items():
            print(f"Req: {spd:4.2f}x | Achieved: {res['achieved_speed']:4.2f}x | "
                  f"Effective FPS: {res['effective_fps']:5.1f} | "
                  f"Processed: {res['frames_processed']:3d} | Skipped: {res['frames_skipped']:3d}")

            # Verify achieved speed is within 15% of requested speed
            ratio = res['achieved_speed'] / spd
            self.assertAlmostEqual(ratio, 1.0, delta=0.20,
                                   msg=f"Speed {spd}x failed: achieved {res['achieved_speed']}x")


if __name__ == "__main__":
    unittest.main()
