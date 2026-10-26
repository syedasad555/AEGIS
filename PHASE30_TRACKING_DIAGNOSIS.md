# AEGIS Phase 30 — Tracking Reliability Diagnosis

**Date:** 2026-10-24  
**Baseline (frozen):** YOLOv8n, `imgsz=800`, `conf=0.08`, `configs/bytetrack_crowd.yaml`, tiled **OFF**  
**UI / model / global tracker / imgsz:** unchanged  
**Phase 29 reports:** preserved  

---

## Metric honesty

| Claimed? | Metric |
|---|---|
| No | Precision, recall, IDF1, HOTA, MOTA |
| Yes (observed) | Per-frame det/track counts, unique ByteTrack IDs, lifetimes, conf vs thresholds, proc FPS |

**`unique_track_ids` ≠ person count.** Short-lived IDs and re-initializations inflate this number without MOT GT.

---

## 1. Reproduced failure: `4917-5_70.mov`

| Observation | Value |
|---|---|
| Frames processed | 60 (consecutive) |
| Mean raw detections | **2.717** |
| Mean tracked | **0.0** |
| Unique IDs | **0** |
| Frames with dets but zero IDs | **60 / 60** |
| `boxes.id` tensor | **None every frame** |
| Exceptions / warnings | none |
| Mean det conf | **0.130** (min 0.081, max 0.304) |
| Boxes with conf &lt; `new_track_thresh` (0.30) | **162 / 163 (99.4%)** |
| Typical box size | ~7×15 px (area ≈ 70–110) on 720×404 |
| `predict` vs `track` boxes (frames 1–3) | Same xyxy / conf — coordinate scale OK |

Frame examples (baseline): every early frame has person-class boxes with conf ≈0.08–0.16 and `track_id=None`.

Machine-readable: `outputs/reports/phase30_4917_5_diag.json`, `outputs/reports/phase30_per_frame_4917_5.json`.

---

## 2. Tracker pipeline trace (code + Ultralytics)

### AEGIS path (tiled OFF)

1. `model.track(..., persist=True, tracker=bytetrack_crowd, conf=0.08, imgsz=800, classes=[0])`  
   — `tools/validate_ucf_phase29.py`, `scripts/pedestrian_pipeline.py`, `server.py`
2. `boxes = results[0].boxes` → `n_det = len(boxes)`
3. `extract_track_ids(boxes)` (`_common.py`) returns `[]` when `boxes.id is None` — **never fabricates IDs**
4. Downstream features / anomalies run **only** when track IDs exist (`if tids:` / `len(track_ids) > 0`)
5. Per-clip fresh `YOLO()` in Phase 29/30 tools → tracker state not reset mid-clip; no unexpected recreate on these runs

### Ultralytics ByteTrack gate (root mechanism)

From installed `ultralytics/trackers/byte_tracker.py`:

- High-score pool: `scores >= track_high_thresh`
- Low-score pool: `track_low_thresh < scores < track_high_thresh` — **association to existing tracks only**
- **New tracks** are initialized only from unmatched **high-score** detections, then filtered by `new_track_thresh`

Crowd baseline YAML:

| Knob | Value |
|---|---:|
| `track_high_thresh` | **0.35** |
| `new_track_thresh` | **0.30** |
| `track_low_thresh` | 0.05 |

YOLO `conf=0.08` admits far/small boxes into Results, but with mean conf ≈0.13 they never enter the high-score init path → `boxes.id` stays `None` → AEGIS correctly reports zero tracks.

**Not a bbox-format bug, not a frame-order bug, not a mid-clip reset.** Evidence points to **threshold policy vs far-scene confidence**, plus tiny boxes.

---

## 3. ID fragmentation: `879-38_l.mov` (baseline)

| Metric | Observed |
|---|---:|
| Mean detections / tracks | 32.07 / 32.07 |
| Unique ByteTrack IDs (60 frames) | **85** |
| Median / mean track lifetime (frames) | **14 / 22.6** |
| Lifetime hist | 1f:1, 2f:5, 3–5:9, 6–15:30, 16–30:15, ≥31:25 |
| Sum `ids_new` events | 100 (includes frame-1 cold start) |
| Sum `ids_lost` events | 64 |
| Abrupt center jumps (&gt;15% diagonal) | 1 event |

Interpretation: ~32 concurrent tracks with **85 IDs over 60 frames** ≈ ID churn / fragmentation, **not** 85 distinct people. Most IDs live ≥6 frames; only 6 IDs are 1–2 frames long. No MOT GT → ID switches not labeled.

---

## 4. Four-clip baseline comparison (same sampling)

| Clip | Decode | Mean det | Mean trk | Unique IDs | Life med | 1f / 2f tracks | ~Proc FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| `4917-5_70` | OK | 2.72 | **0.00** | **0** | 0 | 0 / 0 | 5.2 |
| `879-38_l` | OK | 32.07 | 32.07 | 85 | 14 | 1 / 5 | 5.9 |
| `341-46_l` | OK | 5.53 | 5.53 | 35 | 7 | 2 / 4 | 7.0 |
| `913-37_l` | OK | 10.95 | 10.95 | 30 | 15 | 2 / 4 | 6.4 |

All four playable. Full JSON: `outputs/reports/phase30_tracking_diagnosis.json` (+ `phase30_tracking_diagnosis_summary.json`).

---

## 5. Evidence-backed decision: **no silent baseline change**

### Concrete code bug?

**None found** that explains 4917-5 zero tracks. Untracked boxes are dropped by design after `extract_track_ids` returns empty.

### Controlled tuning candidates (not applied to `config.yaml`)

File: `configs/bytetrack_crowd_far_candidate.yaml` (**not wired**).

| Experiment | 4917-5 mean trk / uniq | 879 uniq / life med / 1f | Notes |
|---|---|---|---|
| Baseline crowd (high=0.35, new=0.30) | 0.00 / 0 | 85 / 14 / 1 | Frozen production |
| Interim A/B high=0.25, new=0.12 | **0.00 / 0** | 115 / 8 / 10 | Proves lowering `new_track_thresh` alone is insufficient while high-pool gate stays high |
| Candidate high=0.10, new=0.10 | **0.067 / 1** | **133 / 6 / 32** | Tiny far-scene gain; dense scenes **worse** ID churn (`341` uniq 35→110, median life 7→1) |

**Conclusion:** Tracker-threshold relaxation is the wrong global fix. Far outdoor needs higher effective detection confidence / resolution (imgsz / ROI / model — deferred), while dense scenes need the current crowd thresholds (or tighter) to limit proliferation.

### Files changed this phase

| File | Role |
|---|---|
| `tools/diagnose_tracking_phase30.py` | Per-frame diagnosis runner (+ optional `--tracker` A/B) |
| `configs/bytetrack_crowd_far_candidate.yaml` | Documented A/B candidate only |
| `tests/test_phase30_tracking_diagnosis.py` | Reproduces 4917-5 failure; freezes baseline thresholds |
| `PHASE30_TRACKING_DIAGNOSIS.md` | This report |
| `outputs/reports/phase30_*.json` | Machine-readable diagnostics |

**Not changed:** `configs/config.yaml` tracker/imgsz/conf, UI, YOLO weights, Phase 29 reports, dataset files.

---

## 6. Validation

```
python tests/run_all_tests.py
→ 29/29 Passed (100.0%) in ~13s
```

Includes Phase 30 tests: baseline thresholds frozen, far-candidate not wired, `extract_track_ids` never fabricates IDs, and live reproduce of `4917-5` dets-without-tracks (20 frames).

---

## 7. Remaining limitations

1. No pedestrian / ID ground truth in `UCF_CrowdsDataset` → no IDF1/HOTA/MOTA.  
2. Far-scene detection confidence stays low at imgsz 800; tracker cannot invent reliable IDs from sub-threshold boxes.  
3. Dense-scene unique-ID inflation is observed churn, not a counted person census.  
4. Next measured experiments (separate phase): imgsz 960 A/B on `4917-*` only; optional ROI; do **not** globally lower `track_high_thresh` without dense-scene regression checks.
