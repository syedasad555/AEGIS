# AEGIS — Person Detection & Tracking Improvement

**Date:** 2026-10-26  
**Order:** inspect → baseline → candidates → integrate only if validated  
**Out of scope:** crowd counting/density, CSRNet, head detectors, second detection models, ByteTrack threshold loosening  

---

## 1. Pipeline & production configuration (Phase 1)

| Setting | Verified value |
|---|---|
| Detector | `yolov8n.pt` |
| `inference_imgsz` | **800** |
| Confidence | 0.08 |
| Tracker | `configs/bytetrack_crowd.yaml` (high=0.35, new=0.30, low=0.05) |
| Tiled | **off** |
| `far_camera_profile.enabled` | **false** (optional imgsz 960) |

**Path:** `server.py` / `pedestrian_pipeline.py` → `model.track` → `extract_track_ids` → features → `EvacuationAnomalyEngine` (fall, wrong-way, prolonged stop, zones) → WebSocket / Digital Twin.

**Session imgsz:** `_common.resolve_inference_imgsz()` — fixed per stream; no auto scene classification.

**UCF:** `UCF_CrowdsDataset/` — 38 `.mov`, no bbox/ID GT (Phase 29). Selected clips decode OK.

**Prior reports reused:** Phase 29–32 (far dets without IDs; threshold tuning unsafe; imgsz 960 helps far; YOLO11s not better).

---

## 2. Baseline (Phase 2) — production unchanged

Identical settings: YOLOv8n, imgsz 800, bytetrack_crowd, conf 0.08, 60 consecutive frames, CPU.

| Clip | Group | Mean det | Mean trk | Max trk | Unique IDs | Life med | Short % | ≥0.35 dets | ~FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4917-5 | far | 2.72 | **0.00** | 0 | 0 | 0 | — | **0** | 4.8 |
| 4917-3 | far | 1.00 | 0.42 | 1 | 1 | 25 | 0% | 17 | 9.7 |
| RF1-12373 | far | 1.05 | 0.88 | 2 | 2 | 26 | 50% | 32 | 8.0 |
| 879-38_l | dense | 32.1 | 32.1 | 37 | 85 | 14 | 7% | 1727 | 6.8 |
| 913-37_l | dense | 11.0 | 11.0 | 14 | 30 | 15 | 20% | 555 | 7.1 |
| 341-46_l | medium | 5.53 | 5.53 | 11 | 35 | 7 | 17% | 136 | 6.8 |
| 821-10_l | medium | 5.78 | 5.78 | 8 | 17 | 18 | 6% | 223 | 7.4 |

**No P/R/IDF1** — unique IDs ≠ person count. Baseline FPS on this CPU is about **5–10** for production settings (measured).

Overlays: `outputs/reports/detection_tracking_review/*_prod_v8n_800_*.jpg`

---

## 3. Candidate comparison (Phase 3)

Tracker held fixed. Detector/imgsz varied only.

### YOLOv8n imgsz **960**

| Clip | Δ mean trk | Δ ≥0.35 | Unique IDs | Short % | ~FPS |
|---|---:|---:|---:|---:|---:|
| 4917-5 | **0 → 0.70** | 0 → 6 | 0 → 2 | 50% | 7.6 |
| 4917-3 | 0.42 → 0.88 | 17 → 4* | 1 | 0% | 6.5 |
| RF1-12373 | 0.88 → 0.98 | 32 → 43 | 2 | 50% | 4.4 |
| 879-38_l | 32 → 36 | ↑ | 85 → 84 | **5%** (better) | 4.1 |
| 913-37_l | 11 → 11 | ↑ | 30 → 33 | 21% | 4.9 |

\*High-score count can vary with box set; track continuity on 4917-3 improved (life med 25→53).

Dense short-lived proportion did **not** explode; FPS remains ~4–5 on dense/medium.

### YOLO11s imgsz 800 (weights already on disk)

| Clip | Mean trk @800 (y11s) | vs prod |
|---|---:|---|
| 4917-5 | **0.0** | no far tracks |
| 4917-3 | **0.0** | worse |
| RF1 | **0.0** | worse |
| 879-38_l | 16.6 | fewer concurrent tracks |
| FPS | ~3.1–4.1 | slower |

**Do not adopt YOLO11s** as production detector on this evidence.

### Fine-tuning

CrowdHuman / YOLOv8n fine-tune **not started** — requires labelled data and a separate training plan after this benchmark. Not justified before measuring pretrained/imgsz options.

### Manual review notes

- Far `4917-5` at 800: low-conf `NO_ID` boxes on tiny pedestrians.  
- Far at 960: sparse tracks appear (still incomplete coverage of all distant people).  
- Dense at 960: more boxes; ID count similar; slightly longer median life on 879.  
- YOLO11s far overlays: detections without usable track init under crowd ByteTrack.

---

## 4. Integration decision (Phase 4)

| Action | Status |
|---|---|
| Change global production imgsz/model/tracker | **No** |
| Add second detector / density model | **No** |
| Optional far profile | **Already present** — `far_camera_profile` in `config.yaml`, **enabled: false** |
| Recommendation | Operators may set `far_camera_profile.enabled: true` for **known far outdoor cameras/sessions** only (imgsz 960). Dense cameras stay at 800. |

No further production code changes required beyond the existing Phase 33 flag + `resolve_inference_imgsz` wiring (already in `_common.py`, `server.py`, `pedestrian_pipeline.py`).

### Rollback

```yaml
yolo_model: "yolov8n.pt"
inference_imgsz: 800
tracker_config: "configs/bytetrack_crowd.yaml"
tiled_inference.enabled: false
far_camera_profile.enabled: false
```

Restart `python server.py`. Original `yolov8n.pt` / pose weights were never overwritten.

---

## 5. Tests (Phase 5)

```
python tests/run_all_tests.py
→ 33/33 Passed (100.0%)
```

Focused tests: `tests/test_phase33_far_camera_profile.py` (profile disabled by default; resolve 800 vs 960).

Downstream fall / wrong-way / stop / intrusion / twin paths unchanged (same bbox + track_id interfaces).

---

## 6. Reproduce

```bash
python tools/benchmark_detection_tracking.py --max-frames 60
# outputs/reports/detection_tracking_benchmark.json
# outputs/reports/detection_tracking_benchmark_summary.json
# outputs/reports/detection_tracking_review/
```

---

## 7. Files touched in this effort

| File | Why |
|---|---|
| `tools/benchmark_detection_tracking.py` | Reproducible baseline + candidate bench |
| `DETECTION_TRACKING_IMPROVEMENT_REPORT.md` | This deliverable |
| `outputs/reports/detection_tracking_*.json` | Machine-readable results |
| `outputs/reports/detection_tracking_review/` | Annotated frames |

**Already present (not re-changed for integration):** `configs/config.yaml` (`far_camera_profile`), `_common.resolve_inference_imgsz`, `server.py` / `pedestrian_pipeline.py` session resolve.

---

## 8. Remaining limitations

1. No UCF bbox/ID GT → cannot claim detection/tracking accuracy metrics.  
2. imgsz 960 helps far scenes modestly; many distant people still missed.  
3. Global 960 would slow dense cameras (~4 FPS) — keep profile opt-in.  
4. Fine-tuned pedestrian YOLO and ROI-merge remain future work with labelled data / design.
