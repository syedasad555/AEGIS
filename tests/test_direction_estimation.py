#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cardinal admin selector (4) × 8-sector movement classification (32 cases).
Admin may select only LEFT/RIGHT/UP/DOWN; backend detects all eight.
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direction_estimation import (
    ADMIN_CARDINALS,
    DIRECTIONS_8,
    DirectionEstimator,
    DirectionClassifier,
    normalize_direction_label,
    coerce_admin_direction,
    angle_to_direction8,
    angle_to_compass8,
    circular_sector_distance,
    allowed_directions,
    classify_direction_sectors,
)
from scripts.anomaly_detection import EvacuationAnomalyEngine
from scripts.feature_extraction import TrajectoryFeatureExtractor


MOVE_VECTORS = {
    "RIGHT": (40.0, 0.0),
    "UP_RIGHT": (30.0, -30.0),
    "UP": (0.0, -40.0),
    "UP_LEFT": (-30.0, -30.0),
    "LEFT": (-40.0, 0.0),
    "DOWN_LEFT": (-30.0, 30.0),
    "DOWN": (0.0, 40.0),
    "DOWN_RIGHT": (30.0, 30.0),
}

EXPECTED_NORMAL = {
    "LEFT": {"LEFT", "UP_LEFT", "DOWN_LEFT"},
    "RIGHT": {"RIGHT", "UP_RIGHT", "DOWN_RIGHT"},
    "UP": {"UP", "UP_LEFT", "UP_RIGHT"},
    "DOWN": {"DOWN", "DOWN_LEFT", "DOWN_RIGHT"},
}


def _walk(estimator, tid, start, delta, n=12, fps=30.0, frame0=1):
    x, y = start
    out = None
    for i in range(n):
        f = frame0 + i
        out = estimator.update(tid, (x, y), f, f / fps)
        x += delta[0]
        y += delta[1]
    return out


class TestAdminSelectorFourOnly(unittest.TestCase):
    def test_html_selectors_cardinals_only(self):
        html = (PROJECT_ROOT / "digital_twin" / "index.html").read_text(encoding="utf-8")
        for sel_id in ("initialDirectionSelect", "directionSelect"):
            m = re.search(
                rf'<select[^>]*id="{sel_id}"[^>]*>(.*?)</select>',
                html,
                re.S,
            )
            self.assertIsNotNone(m, sel_id)
            body = m.group(1)
            values = re.findall(r'option value="([^"]+)"', body)
            self.assertEqual(set(values), {"LEFT", "RIGHT", "UP", "DOWN"}, msg=sel_id)
            for banned in (
                "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
                "NORTH-EAST", "NORTH-WEST", "SOUTH-EAST", "SOUTH-WEST",
                "NORTHEAST", "NORTHWEST",
            ):
                self.assertNotIn(banned, values)

    def test_coerce_admin_rejects_diagonal_as_expected(self):
        self.assertEqual(coerce_admin_direction("LEFT"), "LEFT")
        self.assertEqual(coerce_admin_direction("RIGHT"), "RIGHT")
        self.assertEqual(coerce_admin_direction("UP"), "UP")
        self.assertEqual(coerce_admin_direction("DOWN"), "DOWN")
        self.assertEqual(coerce_admin_direction("EAST"), "RIGHT")
        self.assertEqual(coerce_admin_direction("WEST"), "LEFT")
        # Diagonals are not admin-selectable expected flow
        self.assertEqual(coerce_admin_direction("UP_LEFT"), "LEFT")
        self.assertEqual(coerce_admin_direction("DOWN_RIGHT"), "LEFT")


class TestAliasesAndYAxis(unittest.TestCase):
    def test_product_aliases(self):
        self.assertEqual(normalize_direction_label("LEFT"), "LEFT")
        self.assertEqual(normalize_direction_label("UP_LEFT"), "UP_LEFT")
        self.assertEqual(normalize_direction_label("WEST"), "LEFT")
        self.assertEqual(normalize_direction_label("NORTH-WEST"), "UP_LEFT")

    def test_image_y_down(self):
        self.assertEqual(angle_to_direction8(10, 0), "RIGHT")
        self.assertEqual(angle_to_direction8(0, -10), "UP")
        self.assertEqual(angle_to_direction8(-10, 0), "LEFT")
        self.assertEqual(angle_to_direction8(0, 10), "DOWN")
        self.assertEqual(angle_to_compass8(0, -10), "UP")

    def test_cw_ring_wraparound(self):
        # Spec order: RIGHT → DOWN_RIGHT → … → UP_RIGHT → (wrap) RIGHT
        self.assertEqual(DIRECTIONS_8[0], "RIGHT")
        self.assertEqual(DIRECTIONS_8[1], "DOWN_RIGHT")
        self.assertEqual(DIRECTIONS_8[-1], "UP_RIGHT")
        self.assertEqual(circular_sector_distance("DOWN_RIGHT", "RIGHT"), 1)
        self.assertEqual(circular_sector_distance("UP_RIGHT", "RIGHT"), 1)
        self.assertEqual(set(allowed_directions("RIGHT")), {"DOWN_RIGHT", "RIGHT", "UP_RIGHT"})
        self.assertEqual(set(allowed_directions("LEFT")), {"UP_LEFT", "LEFT", "DOWN_LEFT"})
        self.assertEqual(set(allowed_directions("UP")), {"UP_RIGHT", "UP", "UP_LEFT"})
        self.assertEqual(set(allowed_directions("DOWN")), {"DOWN_LEFT", "DOWN", "DOWN_RIGHT"})


class TestThirtyTwoCardinalMatrix(unittest.TestCase):
    """Mandatory 4 expected × 8 actual = 32 classification cases."""

    def test_all_32(self):
        clf = DirectionClassifier({
            "expected_evacuation_direction": "LEFT",
            "direction_detection": {"max_sector_distance_normal": 1},
        })
        failures = []
        for expected in ADMIN_CARDINALS:
            normal = EXPECTED_NORMAL[expected]
            self.assertEqual(len(normal), 3, msg=expected)
            for actual in DIRECTIONS_8:
                dx, dy = MOVE_VECTORS[actual]
                r = clf.classify({
                    "direction": actual,
                    "direction_vector": [dx, dy],
                    "direction_reliable": True,
                    "direction_confidence": 0.9,
                }, expected)
                want = "NORMAL" if actual in normal else "ANOMALY"
                if r["classification"] != want:
                    failures.append(
                        f"exp={expected} act={actual}: got {r['classification']} "
                        f"want {want} sectorΔ={r.get('sector_distance')}"
                    )
                s = classify_direction_sectors(actual, expected)
                if s["classification"] != want:
                    failures.append(
                        f"sectors() exp={expected} act={actual}: {s['classification']} != {want}"
                    )
        self.assertEqual(failures, [], msg="\n".join(failures[:20]))
        self.assertEqual(len(ADMIN_CARDINALS) * len(DIRECTIONS_8), 32)

    def test_example_tables(self):
        clf = DirectionClassifier({"expected_evacuation_direction": "LEFT"})
        for expected, normal in EXPECTED_NORMAL.items():
            for actual in DIRECTIONS_8:
                dx, dy = MOVE_VECTORS[actual]
                r = clf.classify({
                    "direction": actual,
                    "direction_vector": [dx, dy],
                    "direction_reliable": True,
                    "direction_confidence": 0.9,
                }, expected)
                if actual in normal:
                    self.assertEqual(r["classification"], "NORMAL", msg=f"{expected}/{actual}")
                else:
                    self.assertEqual(r["classification"], "ANOMALY", msg=f"{expected}/{actual}")


class TestEightWayEstimation(unittest.TestCase):
    def setUp(self):
        self.est_cfg = {
            "direction_min_observations": 8,
            "direction_window_frames": 12,
            "direction_min_displacement_px": 10.0,
            "direction_min_speed_px_sec": 5.0,
        }

    def test_all_eight(self):
        starts = {
            "RIGHT": ((100, 200), (12, 0)),
            "LEFT": ((400, 200), (-12, 0)),
            "UP": ((200, 400), (0, -12)),
            "DOWN": ((200, 100), (0, 12)),
            "UP_RIGHT": ((100, 300), (10, -10)),
            "UP_LEFT": ((300, 300), (-10, -10)),
            "DOWN_RIGHT": ((100, 100), (10, 10)),
            "DOWN_LEFT": ((300, 100), (-10, 10)),
        }
        for i, (label, (start, delta)) in enumerate(starts.items()):
            est = DirectionEstimator(self.est_cfg)
            out = _walk(est, 100 + i, start, delta, n=12)
            self.assertEqual(out["direction"], label, msg=label)
            self.assertTrue(out["direction_reliable"], msg=label)

    def test_stationary(self):
        est = DirectionEstimator(self.est_cfg)
        out = _walk(est, 1, (200, 200), (0.2, 0.1), n=12)
        self.assertEqual(out["direction"], "STATIONARY")
        self.assertFalse(out["direction_reliable"])

    def test_jitter_not_direction(self):
        est = DirectionEstimator(self.est_cfg)
        x, y = 200.0, 200.0
        out = None
        for i in range(1, 15):
            jx = ((-1) ** i) * 1.5
            jy = ((-1) ** (i + 1)) * 1.2
            out = est.update(2, (x + jx, y + jy), i, i / 30.0)
        self.assertIn(out["direction"], ("STATIONARY", "UNKNOWN"))

    def test_insufficient_history(self):
        est = DirectionEstimator(self.est_cfg)
        out = est.update(3, (100, 100), 1, 0.0)
        self.assertEqual(out["direction"], "UNKNOWN")

    def test_id_reassignment_jump(self):
        est = DirectionEstimator(self.est_cfg)
        _walk(est, 4, (100, 100), (10, 0), n=10)
        out = est.update(4, (900, 600), 11, 11 / 30.0)
        self.assertEqual(out["observation_count"], 1)
        self.assertEqual(out["direction"], "UNKNOWN")


class TestClassifierEdgeCases(unittest.TestCase):
    def setUp(self):
        self.clf = DirectionClassifier({
            "expected_evacuation_direction": "LEFT",
            "direction_detection": {"max_sector_distance_normal": 1},
        })

    def _info(self, direction, dx, dy, reliable=True, conf=0.8):
        return {
            "direction": direction,
            "direction_vector": [dx, dy],
            "direction_reliable": reliable,
            "direction_confidence": conf,
        }

    def test_stationary_unknown(self):
        r = self.clf.classify(self._info("STATIONARY", 0, 0, reliable=False, conf=0.2), "LEFT")
        self.assertEqual(r["classification"], "UNKNOWN")

    def test_unreliable_unknown(self):
        r = self.clf.classify(self._info("LEFT", -5, 0, reliable=False, conf=0.2), "LEFT")
        self.assertEqual(r["classification"], "UNKNOWN")

    def test_sector_distance_in_result(self):
        r = self.clf.classify(self._info("LEFT", -40, 0), "LEFT")
        self.assertEqual(r["sector_distance"], 0)
        self.assertEqual(r["classification"], "NORMAL")
        r2 = self.clf.classify(self._info("UP", 0, -40), "LEFT")
        self.assertEqual(r2["sector_distance"], 2)
        self.assertEqual(r2["classification"], "ANOMALY")


class TestAnomalyPersistenceAndColours(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "expected_evacuation_direction": "LEFT",
            "direction_detection": {
                "max_sector_distance_normal": 1,
                "wrong_way_sustain_frames": 3,
                "recovery_frames": 2,
                "cooldown_sec": 0.0,
                "require_in_corridor": False,
            },
            "stopping_detection": {"enabled": False},
            "evacuation_corridor": {
                "polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]],
            },
            "zones": {},
        }
        self.engine = EvacuationAnomalyEngine(self.cfg)

    def _feat(self, **kw):
        base = {
            "track_id": 1,
            "frame": 10,
            "timestamp": 1.0,
            "bbox": [100, 100, 140, 200],
            "foot": [120, 200],
            "direction": "RIGHT",
            "direction_vector": [40.0, 0.0],
            "direction_reliable": True,
            "direction_confidence": 0.8,
            "stopping_duration_frames": 0,
            "stopping_duration_sec": 0.0,
            "in_corridor": True,
            "trajectory": [[120, 200, 10, 1.0]],
        }
        base.update(kw)
        return base

    def test_persistence_before_red(self):
        for i in range(2):
            r = self.engine.evaluate_pedestrian(self._feat(timestamp=1.0 + i * 0.1, frame=10 + i))
            self.assertFalse(r["anomaly"])
            self.assertIn(r["visual_state"], ("normal", "unknown"))
        r = self.engine.evaluate_pedestrian(self._feat(timestamp=1.3, frame=13))
        self.assertTrue(r["anomaly"])
        self.assertEqual(r["visual_state"], "anomaly")
        self.assertEqual(r["anomaly_type"], "WRONG_DIRECTION")

    def test_matching_and_adjacent_blue(self):
        r = self.engine.evaluate_pedestrian(self._feat(
            direction="LEFT", direction_vector=[-40, 0],
        ))
        self.assertEqual(r["direction_classification"], "NORMAL")
        self.assertEqual(r["visual_state"], "normal")
        r2 = self.engine.evaluate_pedestrian(self._feat(
            track_id=2, direction="UP_LEFT", direction_vector=[-30, -30],
        ))
        self.assertEqual(r2["direction_classification"], "NORMAL")
        self.assertEqual(r2["sector_distance"], 1)

    def test_unknown_not_red(self):
        r = self.engine.evaluate_pedestrian(self._feat(
            direction="UNKNOWN", direction_vector=[0, 0],
            direction_reliable=False, direction_confidence=0.1,
        ))
        self.assertFalse(r["anomaly"])
        self.assertNotEqual(r["visual_state"], "anomaly")

    def test_active_anomaly_not_sticky_on_unknown(self):
        for i in range(3):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="RIGHT", direction_vector=[40, 0],
                timestamp=1.0 + i * 0.1, frame=10 + i,
            ))
        self.assertEqual(r["visual_state"], "anomaly")
        for i in range(4):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="UNKNOWN", direction_vector=[0, 0],
                direction_reliable=False, direction_confidence=0.1,
                timestamp=2.0 + i * 0.1, frame=20 + i,
            ))
        self.assertNotEqual(r["visual_state"], "anomaly")

    def test_live_expected_direction_change(self):
        for i in range(3):
            self.engine.evaluate_pedestrian(self._feat(timestamp=1.0 + i, frame=10 + i))
        self.assertTrue(any(e.status == "ACTIVE" for e in self.engine.active_events.values()))
        self.engine.set_expected_direction("RIGHT")
        for i in range(4):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="RIGHT", direction_vector=[40, 0],
                timestamp=10.0 + i, frame=50 + i,
            ))
        self.assertEqual(r["direction_classification"], "NORMAL")
        self.assertFalse(any(
            e.event_type == "WRONG_DIRECTION" and e.status == "ACTIVE"
            for e in self.engine.active_events.values()
        ))

    def test_cctv_twin_same_visual_state(self):
        for i in range(3):
            r = self.engine.evaluate_pedestrian(self._feat(
                direction="UP", direction_vector=[0, -40],
                timestamp=1.0 + i, frame=10 + i,
            ))
        self.assertEqual(r["direction_classification"], "ANOMALY")
        self.assertEqual(r["visual_state"], "anomaly")
        self.assertEqual(
            r["direction_diagnostics"]["final_visual_state"],
            r["visual_state"],
        )

    def test_extractor_pipeline_left_walk(self):
        ex = TrajectoryFeatureExtractor({
            "direction_min_observations": 8,
            "direction_window_frames": 12,
            "direction_min_displacement_px": 10.0,
            "direction_min_speed_px_sec": 5.0,
            "evacuation_corridor": {"polygon": [[0, 0], [2000, 0], [2000, 1000], [0, 1000]]},
            "zones": {},
        })
        eng = EvacuationAnomalyEngine({
            **self.cfg,
            "direction_detection": {
                **self.cfg["direction_detection"],
                "wrong_way_sustain_frames": 1,
            },
        })
        feat = None
        for f in range(1, 14):
            x = 400.0 - f * 12.0
            feat = ex.update_pedestrian(7, (x, 200.0, x + 40.0, 400.0), f, 30.0)
            feat = eng.evaluate_pedestrian(feat)
        self.assertEqual(feat["direction"], "LEFT")
        self.assertEqual(feat["direction_classification"], "NORMAL")
        self.assertEqual(feat["visual_state"], "normal")


class TestOcclusionAndReuse(unittest.TestCase):
    def test_track_id_reuse_clears_state(self):
        est = DirectionEstimator({
            "direction_min_observations": 8,
            "direction_window_frames": 12,
            "direction_min_displacement_px": 10.0,
            "direction_min_speed_px_sec": 5.0,
            "trajectory_max_displacement_px": 65.0,
        })
        _walk(est, 5, (100, 100), (10, 0), n=12)
        out = est.update(5, (800, 500), 13, 13 / 30.0)
        self.assertEqual(out["direction"], "UNKNOWN")
        self.assertEqual(out["observation_count"], 1)


if __name__ == "__main__":
    unittest.main()
