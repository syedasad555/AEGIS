# AEGIS Phase 28 — Reliability Fix Report

**Date:** 2026-10-24  
**Objectives:** (A) pedestrian detection/tracking reliability on CPU, (B) fewer false FALL_CONFIRMED alerts without killing genuine-fall event recall.  
**UI:** unchanged.

---

## 1. Baseline (before Phase 28 fall hardening)

### Detection (Phase 27, identical GT frames, IoU≥0.50, conf=0.08)

| Config | Precision | Recall | F1 | ~FPS (CPU) |
|---|---:|---:|---:|---:|
| imgsz **800** + crowd | 0.663 | 0.566 | **0.611** | ~2.7 |
| imgsz **960** + crowd | 0.716 | 0.620 | 0.664 | ~2.1 |
| tiled @800 | 0.512 | 0.627 | 0.564 | ~1.2 |

### Fall (prior persistent-fall benchmark, `phase23_persistent_fall_results.json`)

| Metric | Value |
|---|---:|
| Event recall | 66.67% (2/3) |
| ADL false alarms / min | 0.00 |
| Frame precision | 100% |
| Frame recall | 40.74% |

### Unavailable video

- **`341-46_l.mov`:** not present in the repository. Detection work reused Phase 27 annotated frames (corridor / running / intrusion / MOT17 / MOT20).

---

## 2. Task A — Detection / tracking decision

**Kept:** `inference_imgsz: 800`, `tracker_config: configs/bytetrack_crowd.yaml`, `tiled_inference.enabled: false`.

**Why:** Phase 27 + Phase 28 re-score on the same 15 GT frames show imgsz 960 raises F1 (+0.05) but drops live CPU throughput; tiling raises false positives and collapses FPS. Crowd tracker remains better on consecutive dense frames for ID retention (Phase 27 MOT20 compare).

**Phase 28 re-measurement (same GT):**

| imgsz | P | R | F1 | ~pipeline FPS |
|---:|---:|---:|---:|---:|
| 800 | 0.663 | 0.566 | 0.611 | ~1.0–2.7* |
| 960 | 0.716 | 0.620 | 0.664 | ~2.1 |

\*FPS varies with machine load; relative ranking unchanged.

**Detector vs tracker:** raw YOLO counts and ByteTrack ID counts remain reported separately; IDs are never invented. Untracked remainder is detector→association gap, not fabricated tracks.

**ID switches / fragmentation:** not measured on corridor/running/intrusion (no person-ID ground truth). Requires MOT-style ID annotations (available only on MOT17/MOT20).

---

## 3. Task B — Fall reliability changes

### Root causes of false CONFIRMs (code + diagnosis)

1. Geometry-only path could CONFIRM from sustained wide boxes near the floor **without** an observed collapse (sitting / bending look-alikes).
2. Track jumps / ID reassignments could carry or re-trigger fall evidence incorrectly.
3. Over-hardening (first Phase 28 pass) blocked pose-confirmed prone people on **short reassigned tracks** after ByteTrack ID switches during the fall (seen on `indoor_fall.mp4`).

### Fixes in `scripts/fall_detection.py` + `configs/config.yaml`

| Gate | Behavior |
|---|---|
| **FALL_SUSPECTED** | Early warning: centroid drop / AR / pose hints (unchanged role). |
| **FALL_CONFIRMED (geometry)** | Requires **dynamic centroid drop** + horizontal box + sustained frames + low ground speed. |
| **FALL_CONFIRMED (pose)** | Sustained horizontal torso (≥3 frames), near floor, AR≥suspect — **does not** require dynamic drop (survives ID switch). |
| **Reassignment cooldown** | Large spatial jump or frame gap clears fall counters and suppresses CONFIRM briefly. |
| **Min history** | Geometry needs ≥8 observations; pose confirm may use half after ID switch. |

### Final fall benchmark (executed)

Command:

```bash
python -c "from pathlib import Path; import yaml; from tools.evaluate_temporal_fall import run_temporal_fall_benchmark; ..."
# or via tools/evaluate_temporal_fall.py after config imgsz/tracker wiring
```

| Metric | Baseline (P23) | Final (P28) |
|---|---:|---:|
| Event recall | 66.67% (2/3) | **66.67% (2/3)** |
| Missed events | 1 | 1 (`assets/fall.mp4`) |
| ADL FA / min | 0.00 | **0.00** |
| Frame precision | 100% | **100%** |
| Frame recall | 40.74% | 38.89% |
| Mean detection delay | 0.24 s | 0.34 s |

**Per clip (final):**

| Clip | Type | Event TP/FP/FN | Delay |
|---|---|---|---:|
| indoor_fall.mp4 | POSITIVE | 1 / 0 / 0 | 0.45 s |
| le2i_coffee_fall.mp4 | POSITIVE | 1 / 0 / 0 | 0.24 s |
| fall.mp4 | POSITIVE | 0 / 0 / 1 | — |
| indoor_normal_adl.mp4 | NEGATIVE | 0 / 0 / 0 | — |
| le2i_coffee_adl.mp4 | NEGATIVE | 0 / 0 / 0 | — |

### Non-fall demo FA scan (first 90 frames, CONFIRM events)

| Video | Confirmed fall events | Suspected frames |
|---|---:|---:|
| corridor.avi | **0** | 12 |
| running.mp4 | **0** | 0 |
| intrusion.mp4 | **0** | 0 |
| 341-46_l.mov | MISSING | — |

Corridor may still show **FALL_SUSPECTED** (WARNING) during ambiguous posture; CONFIRM/CRITICAL stayed at zero on these windows.

---

## 4. Task C — Integration & tests

```bash
python tests/run_all_tests.py
```

**Result:** **25/25 passed** (includes new `tests/test_phase28_reliability.py`).

Architecture preserved: CCTV/Digital Twin sync and video-switch trail clearing unchanged (Phase 25/26 behavior). No UI redesign.

---

## 5. Commands executed

```bash
# Unit / regression
python tests/run_all_tests.py
python -m unittest tests.test_fall_fsm_regression tests.test_phase28_reliability

# Detection + fall + FA scan bundle
python tools/evaluate_phase28_reliability.py

# Fall-only re-benchmark after pose-path rebalance
python -c "from tools.evaluate_temporal_fall import run_temporal_fall_benchmark; ..."
```

Artifacts:

- `outputs/reports/phase28_reliability_results.json`
- `outputs/reports/phase28_fall_benchmark.json`
- `PHASE28_RELIABILITY_REPORT.md`
- `tests/test_phase28_reliability.py`
- `tools/evaluate_phase28_reliability.py`

---

## 6. Files changed

| File | Change |
|---|---|
| `scripts/fall_detection.py` | Phase 28 CONFIRM gates, reassignment cooldown, pose vs geometry paths |
| `configs/config.yaml` | Fall reliability knobs; keep imgsz=800 + crowd tracker |
| `tools/evaluate_temporal_fall.py` | Prefer config imgsz/tracker |
| `tools/evaluate_phase28_reliability.py` | Compact A/B evaluation harness |
| `tests/test_phase28_reliability.py` | Sitting block, genuine collapse, cooldown, config guards |

---

## 7. Known limitations

1. **`assets/fall.mp4` still missed at event level** under crowd tracker + imgsz 800 (high-angle snow ramp, track fragmentation during slide). Extra GT for person IDs across the collapse would allow targeted association fixes.
2. **`341-46_l.mov` unavailable** — cannot report per-video metrics for it.
3. **Frame-level fall recall remains modest** (~39%): system prioritizes event precision and low FA over dense frame coverage.
4. **ID switch / fragmentation metrics** need MOT-style ID GT (not available for demo MP4s).
5. Short ADL duration (~0.26 min) → FA/min is exact on that set but statistically limited; longer negative footage would strengthen the FA estimate.
6. Not 100% detection or fall recall — not claimed.

---

## 8. Recommended demo configuration

```yaml
inference_imgsz: 800
detection_confidence: 0.08
tracker_config: configs/bytetrack_crowd.yaml
tiled_inference.enabled: false
fall_confirm_require_dynamic_drop: true
fall_pose_confirm_frames: 3
fall_min_history_for_confirm: 8
fall_post_reassign_cooldown: 8
```

**Evidence of success:** event recall matched prior baseline (2/3), ADL and demo-window CONFIRM false alarms stayed at **0**, unit tests prove sitting does not CONFIRM while genuine collapse still does, and regression suite is **25/25 green**.
