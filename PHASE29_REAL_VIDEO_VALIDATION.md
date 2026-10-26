# AEGIS Phase 29 — Real UCF Video Validation

**Date:** 2026-10-24  
**Baseline (unchanged vs Phase 27/28 CPU recommendation):** YOLOv8n, `imgsz=800`, `conf=0.08`, `configs/bytetrack_crowd.yaml`, **tiled OFF**, device CPU  
**Tool:** `tools/validate_ucf_phase29.py`  
**Results JSON:** `outputs/reports/phase29_ucf_validation.json`  
**UI / model:** not redesigned or retuned in this phase

---

## 1. Scope and honesty constraints

| Claim | Status |
|---|---|
| Pipeline runs on real UCF crowded clips | Yes (observed) |
| Pedestrian bbox precision / recall / F1 on UCF | **Not claimed** — no bbox GT in folder |
| Track ID accuracy / IDF1 | **Not claimed** — no ID GT |
| Fall event recall on UCF | **Not claimed** — no fall labels; CONFIRM count is observed only |
| Processing FPS vs source FPS | Reported separately; never conflated |

Crashes, decode failures, and empty track runs are reported as-is.

---

## 2. Run configuration

| Parameter | Value |
|---|---|
| Max frames / clip | **60** (consecutive from start; sparse sampling avoided so ByteTrack stays warm) |
| Clips | 7 curated playable crowded / requested videos |
| Fall detector | YOLOv8n-pose (loaded per clip as in live pipeline) |

---

## 3. Per-clip observed results

| Video | WxH | Src FPS | Frames | Mean raw dets | Mean tracked | Unique IDs | Det−track gap | FALL_CONF events | FALL_SUSP frames | Other anomaly frames | Latency ms | ~Proc FPS | Crashes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4917-3_70.mov` | 720×404 | 25 | 60 | 1.00 | 0.42 | 1 | 0.58 | **0** | 0 | 25 | 272 | 3.7 | none |
| `4917-5_70.mov` | 720×404 | 25 | 60 | 2.72 | **0.00** | **0** | 2.72 | **0** | 0 | 0 | 138 | 7.2 | none |
| `879-38_l.mov` | 480×360 | 25 | 60 | **32.07** | **32.07** | **85** | 0.00 | **0** | 15 | 473 | 256 | 3.9 | none |
| `821-10_l.mov` | 480×360 | 25 | 60 | 5.78 | 5.78 | 17 | 0.00 | **0** | 26 | 48 | 294 | 3.4 | none |
| `341-46_l.mov` | 480×360 | 25 | 60 | 5.53 | 5.53 | 35 | 0.00 | **0** | 0 | 173 | 137 | 7.3 | none |
| `913-37_l.mov` | 480×360 | 25 | 60 | 10.95 | 10.95 | 30 | 0.00 | **0** | 8 | 199 | 342 | 2.9 | none |
| `RF1-12373_70.mov` | 720×480 | 30 | 60 | 1.05 | 0.88 | 2 | 0.17 | **0** | 8 | 12 | 200 | 5.0 | none |

All seven clips: **status OK**, no pipeline crashes.

---

## 4. Interpretation (baseline only)

### What worked

- Decode + YOLO track + anomaly engine ran end-to-end on all selected UCF clips (including previously missing `341-46_l` / `913-37_l`).
- Dense SVQ3 crowds (`879-38_l`) produce high raw and tracked counts with **zero** det−track gap on the sampled window.
- **No FALL_CONFIRMED events** on these crowd clips under Phase 28 hardening — consistent with “no labeled falls” and reduced false CONFIRM intent (not proof of zero false alarms without GT).
- Processing throughput roughly **~3–7 FPS** on this CPU for 60-frame windows (varies with crowd density / pose cost). **Source FPS remains 25–30** — wall-clock real-time is not claimed.

### Stress / failure modes observed (honest)

1. **Far / small pedestrians on 720×404 outdoor (`4917-*`):** very low mean detections (1–3); `4917-5` had detections but **zero assigned track IDs** in the first 60 frames → association gap, not fabricated IDs.  
2. **ID churn in dense scenes:** `879-38_l` shows mean ~32 tracks but **85 unique IDs** over 60 frames → fragmentation / switches likely; cannot quantify without MOT GT.  
3. **FALL_SUSPECTED** appears on several clips (e.g. 26 frames on `821-10_l`) without CONFIRM — expected under Phase 28 SUSPECT vs CONFIRM split; not scored as FA without labels.  
4. **`other_anomaly_frames`** can be high (crowd motion / wrong-direction style rules) — counts are raw frame×pedestrian flags from the engine, not validated events.  
5. Four dataset clips remain **unreadable** (see audit); excluded from this run.

### Digital Twin sync check

Foot positions are normalized by **that clip’s** W×H (`norm_xy = foot / [W,H]`). Validation stores samples from first / mid / last processed frames only (script fix: do not dump every frame). Mapping is geometry-consistent per video; cross-video Twin scale still depends on UI canvas mapping (unchanged this phase).

---

## 5. Regression suite

```
python tests/run_all_tests.py
→ 25/25 Passed (100.0%)
```

Machine-readable: `outputs/reports/phase23_regression_results.json` / `phase25_regression_results.json`.

---

## 6. Decisions locked until next phase

| Item | Decision |
|---|---|
| imgsz | Keep **800** |
| Tracker | Keep **bytetrack_crowd** |
| Tiling | Stay **OFF** |
| UI redesign | **Deferred** until after this baseline |
| Model retune / conf change | **Deferred** |
| UCF GT scoring | Requires external annotations — out of scope |

### Suggested next work (not done here)

- Optional re-encode of 4 unreadable `.mov` files.  
- If detection on far outdoor crowds is prioritized: evaluate imgsz 960 or ROI crop **with** a measured A/B, not blind UI changes.  
- Reduce ID fragmentation only with MOT-style labels or controlled consecutive-frame ID metrics.

---

## 7. Artefacts

| File | Role |
|---|---|
| `PHASE29_UCF_DATASET_AUDIT.md` | Folder / codec / GT audit |
| `PHASE29_REAL_VIDEO_VALIDATION.md` | This report |
| `outputs/reports/phase29_ucf_inventory.json` | Full inventory |
| `outputs/reports/phase29_ucf_validation.json` | Observed per-clip metrics |
| `tools/audit_ucf_dataset.py` | Audit tool |
| `tools/validate_ucf_phase29.py` | Baseline runner |
| `configs/config.yaml` → `ucf_dataset_root` | Configurable dataset path |
