# AEGIS Phase 32 — Far-Pedestrian Detector Experiment

**Date:** 2026-10-24  
**Production:** unchanged (`yolov8n.pt`, `inference_imgsz: 800`, `conf: 0.08`, `bytetrack_crowd.yaml`, UI untouched)  
**Isolated runner:** `tools/experiment_detector_phase32.py`  
**Results:** `outputs/reports/phase32_detector_experiment.json` (+ `_summary.json`)  
**Overlays (raw detector, pre-track):** `outputs/reports/phase32_review/`  

Identical frames: **60 consecutive** from `4917-5_70.mov` and `879-38_l.mov`.  
No precision/recall (no bbox GT). Track counts are **not** accuracy.

---

## 1. Detector vs tracker (method)

| Stage | What is measured |
|---|---|
| `model.predict` | Raw person boxes + confidences (before ByteTrack) |
| `model.track` + crowd YAML | How many boxes get IDs; unique IDs (not person census) |
| High-score eligibility | Count of raw dets with `conf ≥ track_high_thresh (0.35)` |

Green overlay boxes = conf ≥ 0.35; red = below (still raw detections).

---

## 2. YOLOv8n imgsz sweep — `4917-5` (far outdoor)

| imgsz | Mean raw dets | Conf mean / max | Dets ≥0.35 (60f total) | Mean tracked | Unique IDs | Pred ~FPS | Mem after MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **800** (prod) | 2.72 | 0.130 / 0.304 | **0** | **0.00** | **0** | 4.7 | ~438 |
| **960** | 7.05 | 0.166 / 0.401 | **7** | **0.70** | **2** | 6.9 | ~462 |
| **1280** | 12.62 | 0.199 / 0.553 | **69** | **2.85** | **6** | 3.7 | ~480 |

### Manual review (raw f=15)

- **800:** One red box (~0.15) on a real far sidewalk pedestrian; many other people missed.
- **960:** ~5 red boxes on **real** far pedestrians (conf ~0.13–0.30); none green this frame — matches sparse ≥0.35 count.
- **1280:** More real pedestrians; **two green** boxes (~0.37–0.39) eligible for new tracks; also at least one **false** box on a vehicle (~0.25).

**Confidence shift:** At 800, **zero** boxes reach 0.35. At 960 a thin tail appears; at 1280 a usable high-score tail exists but with more low-conf clutter / FP risk.

---

## 3. YOLOv8n imgsz sweep — `879-38_l` (dense)

| imgsz | Mean raw dets | Conf mean | Dets ≥0.35 (total) | Mean tracked | Unique IDs | Pred ~FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 49.5 | 0.488 | 1845 | 32.1 | 85 | 8.7 |
| 960 | 51.9 | 0.526 | 2061 | 35.7 | 84 | 5.9 |
| 1280 | 60.1 | 0.497 | 2207 | 38.2 | 98 | 3.1 |

Dense scene already has many high-score dets at 800. Raising imgsz adds boxes and some track concurrency; unique IDs do not explode like Phase 31 threshold loosening. **Speed cost is real** (~8.7 → 3.1 pred FPS).

Note: raw `predict` box counts exceed Phase 29/31 `track()` box counts on this clip — tracker path and predict path are reported separately on purpose.

---

## 4. Isolated ROI crop (`4917-5` only — not production)

ROI = full width × top **55%** height (`[0,0,720,222]`). Boxes mapped with `+ (x0,y0)` and clamped.

| imgsz | Full mean dets | ROI mean dets | Full ≥0.35 | ROI ≥0.35 | ROI conf mean | ROI-only ~FPS | Dup IoU≥0.5 events |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 | 2.72 | 5.17 | 0 | 3 | 0.159 | 15.5 | 143 |
| 960 | 7.05 | 10.23 | 7 | **85** | 0.216 | 12.5 | 415 |

**Documented risks**

- **Coordinate mapping:** crop-local → full-frame via offset; OK for this static ROI.
- **Boundary:** boxes touching ROI bottom edge flagged; people straddling the cut can be truncated.
- **Duplicates:** if ROI **and** full-frame both run, IoU≥0.5 overlaps are common (143–415 events) → must NMS/merge before tracking.
- **Overhead:** extra pass ≈ full+ROI latency; ROI-as-replacement misses lower-frame pedestrians.

**Not integrated** into server / pipeline.

---

## 5. YOLO11s comparison (identical frames, conf=0.08)

Weights: `yolo11s.pt` (Ultralytics v8.4.0 asset download). Device CPU.

| Clip | imgsz | Mean dets | ≥0.35 | Mean tracked | Unique IDs | Pred ~FPS |
|---|---:|---:|---:|---:|---:|---:|
| 4917-5 | 800 | 4.17 | 3 | **0.0** | **0** | 4.4 |
| 4917-5 | 960 | 6.70 | 0 | **0.0** | **0** | 3.4 |
| 879-38 | 800 | 33.7 | 668 | 16.6 | 38 | 3.6 |
| 879-38 | 960 | 39.1 | 1055 | 21.6 | 58 | 2.5 |

**Not automatically better:** On far outdoor, YOLO11s still yields **zero tracks** under production ByteTrack. On dense, it produces **fewer** concurrent tracks than YOLOv8n at the same imgsz and is slower. No production model swap.

---

## 6. Recommendation (smallest practical improvement)

| Option | Far verified pedestrians | High-score / tracks | Speed / mem | Verdict |
|---|---|---|---|---|
| Keep imgsz **800** | Sparse, all low-conf | 0 ≥0.35, 0 tracks | Best global FPS | **Keep as production default** |
| Scene-gated imgsz **960** | More real people (manual) | Some ≥0.35; sparse tracks appear | Mild cost; mem +~25 MB | **Smallest useful optional upgrade for far cameras only** |
| imgsz **1280** | Most boxes; some green | Clear track init | ~3–4 FPS; more FPs | Optional max-quality / offline only |
| ROI crop | Helps confidence locally | Strong ≥0.35 at 960 ROI | Needs merge logic | Architecture candidate — **not** ready for prod |
| YOLO11s | Mixed | No far tracks here | Slower | **Do not adopt** on this evidence |

**Decision for this phase:** Leave production at **imgsz=800 + yolov8n + bytetrack_crowd**.  

If a later phase applies a change, the **smallest evidence-backed step** is **camera/profile-gated `inference_imgsz: 960` for far outdoor feeds only**, with dense cameras staying at 800. Do not globally force 1280 or YOLO11s without further measured review.

Evidence is **sufficient** for the imgsz ranking on these two clips; it is **insufficient** to claim recall/precision gains or to ship ROI without a merge design.

---

## 7. Tests & artefacts

```
python tests/run_all_tests.py
→ 29/29 Passed (100.0%)
```

Production verified unchanged: `yolov8n.pt`, imgsz 800, conf 0.08, `bytetrack_crowd` high=0.35 / new=0.30.

| Path | Role |
|---|---|
| `PHASE32_DETECTOR_EXPERIMENT.md` | This report |
| `outputs/reports/phase32_detector_experiment.json` | Full machine-readable |
| `outputs/reports/phase32_detector_experiment_summary.json` | Slim summary |
| `outputs/reports/phase32_review/` | Raw pre-track overlays |
| `outputs/reports/phase32_review/roi/` | Isolated ROI overlays |
| `yolo11s.pt` | Downloaded for experiment only (not wired in config) |
