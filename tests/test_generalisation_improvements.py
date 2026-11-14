import unittest
from scripts.anomaly_detection import EvacuationAnomalyEngine

class TestGeneralisationImprovements(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "expected_evacuation_direction": "RIGHT",
            "stopping_detection": {
                "threshold_sec": 3.0,
                "min_frames": 10,
                "enabled": True,
                "long_horizon_verification": True,
                "max_cumulative_disp_px": 20.0
            },
            "direction_detection": {
                "wrong_way_sustain_frames": 3,
                "cooldown_sec": 0.0,
                "require_in_corridor": False
            },
            "exit_regions": [
                {
                    "id": "exit_top",
                    "name": "North Exit Door",
                    "polygon": [[400, 50], [500, 50], [500, 150], [400, 150]],
                    "allowed_exit_directions": ["UP", "UP_RIGHT"]
                }
            ],
            "zones": {}
        }
        self.engine = EvacuationAnomalyEngine(self.cfg)

    def test_long_horizon_stopping_suppression_on_moving_pedestrian(self):
        feat1 = {
            "track_id": 99,
            "frame": 1,
            "timestamp": 0.0,
            "foot": [100.0, 100.0],
            "direction": "STATIONARY",
            "movement_status": "STATIONARY",
            "stopping_duration_sec": 0.0,
            "stopping_duration_frames": 0
        }
        self.engine.evaluate_pedestrian(feat1)
        
        feat2 = {
            "track_id": 99,
            "frame": 10,
            "timestamp": 3.2,
            "foot": [135.0, 100.0],
            "direction": "STATIONARY",
            "movement_status": "STATIONARY",
            "stopping_duration_sec": 3.2,
            "stopping_duration_frames": 15
        }
        res2 = self.engine.evaluate_pedestrian(feat2)
        self.assertNotEqual(res2.get("anomaly_type"), "PROLONGED_STOPPING")

    def test_long_horizon_stopping_confirms_genuinely_stationary_pedestrian(self):
        feat1 = {
            "track_id": 98,
            "frame": 1,
            "timestamp": 0.0,
            "foot": [100.0, 100.0],
            "direction": "STATIONARY",
            "movement_status": "STATIONARY",
            "stopping_duration_sec": 0.0,
            "stopping_duration_frames": 0
        }
        self.engine.evaluate_pedestrian(feat1)
        
        feat2 = {
            "track_id": 98,
            "frame": 10,
            "timestamp": 3.2,
            "foot": [102.0, 101.0],
            "direction": "STATIONARY",
            "movement_status": "STATIONARY",
            "stopping_duration_sec": 3.2,
            "stopping_duration_frames": 15
        }
        res2 = self.engine.evaluate_pedestrian(feat2)
        self.assertEqual(res2.get("anomaly_type"), "PROLONGED_STOPPING")
        self.assertEqual(res2.get("visual_state"), "anomaly")

    def test_exit_aware_turn_permits_orthogonal_egress(self):
        for i in range(5):
            feat = {
                "track_id": 97,
                "frame": i + 1,
                "timestamp": (i + 1) * 0.1,
                "foot": [450.0, 100.0],
                "direction": "UP",
                "direction_reliable": True,
                "direction_vector": [0.0, -1.0],
                "movement_status": "MOVING",
                "speed_px_sec": 25.0,
                "direction_confidence": 0.8
            }
            res = self.engine.evaluate_pedestrian(feat)
            self.assertNotEqual(res.get("anomaly_type"), "WRONG_DIRECTION")

    def test_counter_flow_outside_exit_still_triggers_wrong_direction(self):
        for i in range(5):
            feat = {
                "track_id": 96,
                "frame": i + 1,
                "timestamp": (i + 1) * 0.1,
                "foot": [200.0, 200.0],
                "direction": "LEFT",
                "direction_reliable": True,
                "direction_vector": [-1.0, 0.0],
                "movement_status": "MOVING",
                "speed_px_sec": 35.0,
                "direction_confidence": 0.9
            }
            res = self.engine.evaluate_pedestrian(feat)
        self.assertEqual(res.get("anomaly_type"), "WRONG_DIRECTION")
        self.assertEqual(res.get("visual_state"), "anomaly")

if __name__ == "__main__":
    unittest.main()
