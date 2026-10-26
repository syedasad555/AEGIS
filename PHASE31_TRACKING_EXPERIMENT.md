# AEGIS Phase 31 — Confidence-Aware Tracking Experiment

**Date:** 2026-10-24  
**Production config:** unchanged (`configs/config.yaml`, `configs/bytetrack_crowd.yaml`)  
**Isolated artefacts:** `outputs/experiments/phase31/trackers/`, `outputs/reports/phase31_*`  
**Ultralytics:** 8.4.173  
**Frozen constants:** YOLOv8n, `imgsz=800`, detector `conf=0.08`, tiled OFF, 60 consecutive frames / clip  

No precision / recall / IDF1 / HOTA / MOTA claimed (no pedestrian GT).

---

## 1. Controlled baseline (same clips as Phase 30)

| Clip | Mean det | Mean trk | Unique IDs | Life med | Short-lived (1–2f) / uniq | Conf mean | ~FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4917-5_70` | 2.72 | **0.00** | **0** | 0 | — | 0.130 | 3.5 |
| `879-38_l` | 32.07 | 32.07 | 85 | **14** | 7.1% | 0.647 | 4.3 |
| `341-46_l` | 5.53 | 5.53 | 35 | 7 | 17.1% | 0.308 | 4.7 |
| `913-37_l` | 10.95 | 10.95 | 30 | 15 | 20.0% | 0.589 | 4.6 |

**`4917-5` confidence distribution (all boxes, 60 frames):**

| Bin | Count |
|---|---:|
| [0.08, 0.10) | 40 |
| [0.10, 0.15) | 88 |
| [0.15, 0.20) | 22 |
| [0.20, 0.25) | 8 |
| [0.25, 0.30) | 4 |
| [0.30, 0.35) | 1 |
| ≥ 0.35 | **0** |

Pool split under baseline (`high=0.35`, `low=0.05`): **high=0**, **low=163**. Every detection is low-pool only → cannot initialize tracks.

Process RSS (approx.): start ~205 MB → end ~273 MB (experiment process; clip peaks ~360 MB).

---

## 2. Threshold semantics (installed ByteTrack)

Source: `ultralytics/trackers/byte_tracker.py` (`_split_detections`, `_init_new_tracks`).

| Knob | Role |
|---|---|
| Detector `conf` (0.08) | Filters boxes **before** ByteTrack |
| `track_high_thresh` | `score >= high` → **high pool**: 1st association + **only** pool that can mint new IDs |
| `track_low_thresh` | `low < score < high` → **low pool**: 2nd association to **existing** tracks only |
| `new_track_thresh` | Among unmatched **high-pool** dets, require `score >= new` to activate |

**Critical:** Changing `new_track_thresh` or `track_low_thresh` does **not** move detections into the high pool. Low-confidence detections **cannot create tracks** under stock Ultralytics 8.4 ByteTrack (`low_conf_track_creation_supported: false`).

---

## 3. Isolated parameter sweep

Candidate YAMLs written only under `outputs/experiments/phase31/trackers/` (not wired into production).

| Config | high / low / new | Intent |
|---|---|---|
| `baseline_crowd` | 0.35 / 0.05 / 0.30 | Production mirror |
| `fall_like` | 0.25 / 0.05 / 0.15 | Mirror fall tracker |
| `new_only_lower` | 0.35 / 0.05 / 0.12 | Negative control (Phase 30) |
| `moderate_high` | 0.20 / 0.05 / 0.20 | Mid-conf into high pool |
| `far_aggressive` | 0.10 / 0.05 / 0.10 | Max far init (known churn risk) |

Selection policy: **not** max track count / unique IDs. Prefer far continuity + dense lifetime / low short-lived proportion / lower ID inflation.

### Dense + far headline results

| Config | 4917 trk / uniq | 879 uniq / life_med / short% / inflation | Usefulness* |
|---|---|---|---:|
| baseline_crowd | 0 / 0 | **85 / 14 / 7% / 2.65** | −2 |
| new_only_lower | 0 / 0 | 85 / 14 / 7% / 2.65 | −2 |
| fall_like | 0 / 0 | 115 / 8 / 17% / 3.62 | −4 |
| far_aggressive | **0.07 / 1** | 133 / 6 / **30%** / 4.31 | −6 |
| moderate_high | 0 / 0 | 121 / 7 / **22%** / 3.85 | −12 |

\*Heuristic continuity score (not accuracy). Full tables: `outputs/reports/phase31_tracking_experiment_summary.json`.

`341-46_l` under aggressive settings collapses (uniq 35→110, life med 7→1, short-lived 68%) — clear false/unstable ID proliferation.

FPS across configs stayed ~3.5–6.5 (CPU); no meaningful speed win from looser thresholds.

---

## 4. Manual review (overlays)

Frames in `outputs/reports/phase31_review/`.

### Far clip `4917-5` (f=15)

- **baseline:** tiny sidewalk pedestrian labeled `NO_ID 0.15` (red). Other far people often undetected.
- **far_aggressive:** same frame still `NO_ID 0.15` — threshold drop did not create a stable ID on this visible person. Aggregate metrics show only ~1 fleeting ID over 60 frames (life med 4) — not useful continuity.

### Dense clip `879-38_l` (f=40)

- **baseline:** green IDs with moderate numbers; identities look coherent through overlap; low-conf occluded boxes still keep IDs via association.
- **far_aggressive:** many **very large ID integers** (e.g. 300–600+) in the same scene type → ID churn / re-minting, not preserved identities across the clip. Matches higher unique-ID count and short-lived proportion.

**Unstable / false-track risk:** Looser high/new thresholds increase short-lived IDs on `341` and `879` without fixing far outdoor. Do **not** treat higher unique ID counts as more people.

---

## 5. Decision

### Recommend production change?

**No.** Keep `configs/bytetrack_crowd.yaml` as the production tracker.

Reasons:

1. No candidate creates useful far-scene tracks while preserving dense continuity.  
2. Stock ByteTrack cannot promote low-pool dets into new IDs — the far failure is **detector confidence / scale**, not a missing low-thresh knob.  
3. Aggressive high/new settings worsen ID churn (observed), especially on `341-46_l` and `879-38_l`.

### Architecture-level next steps (not done here)

1. **Raise far-person detector confidence** — imgsz 960 A/B or ROI crop on outdoor cameras (measured, scene-gated).  
2. **Scene-conditioned tracker profiles** (crowd vs far) via camera metadata — never a silent global swap.  
3. **Custom tracker** only if product requires low-score initialization (fork / non-Ultralytics ByteTrack) — out of current architecture scope.

---

## 6. Deliverables & tests

| Artefact | Path |
|---|---|
| Report | `PHASE31_TRACKING_EXPERIMENT.md` |
| Full JSON | `outputs/reports/phase31_tracking_experiment.json` |
| Summary JSON | `outputs/reports/phase31_tracking_experiment_summary.json` |
| Review overlays | `outputs/reports/phase31_review/` |
| Isolated tracker YAMLs | `outputs/experiments/phase31/trackers/` |
| Runner | `tools/experiment_tracking_phase31.py` |

```
python tests/run_all_tests.py
→ 29/29 Passed (100.0%)
```

**Unchanged:** production model, production tracker YAML, UI, Phase 29/30 reports, dataset files.
