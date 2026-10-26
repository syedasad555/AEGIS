# AEGIS Phase 27 — Person Detection Recall and Tracking Optimization

**Date:** 2026-10-24  
**Device:** CPU (PyTorch 2.14.1+cpu), YOLOv8n  
**IoU match threshold for P/R/F1:** **0.50** (greedy one-to-one)  
**Detection confidence (all configs):** **0.08**  
**UI:** unchanged

---

## 1. Objective

Reduce missed pedestrians in crowded CCTV footage while remaining responsive on a CPU-only laptop. Compare configurations on **identical frames**, score against **ground truth**, and separate **detector** counts from **tracker** (valid ByteTrack ID) counts. Track IDs were never invented.

---

## 2. Ground truth

### 2.1 Annotated set (15 frames)

| Source | Frames | GT origin |
|---|---|---|
| `corridor` (evacuation_test_corridor.avi) | 40, 70, 100 | Manual visual verification (YOLO-assist + NMS/geometry filter) |
| `running` (running.mp4) | 80, 110, 140 | Manual visual verification |
| `intrusion` (intrusion.mp4) | 80, 130, 180 | Manual visual verification; shop-window mannequin/reflection band removed after review |
| **MOT17-02** (crowded outdoor / occlusions) | 30, 60, 90 | **Official MOTChallenge GT** |
| **MOT20-01** (dense indoor crowd) | 30, 60, 90 | **Official MOTChallenge GT** |

**Artifacts:**
- Frames: `data/phase27_gt/frames/`
- Overlays: `data/phase27_gt/overlays/`
- GT JSON: `data/phase27_gt/ground_truth.json`

### 2.2 Size bins (by bbox height)

| Bin | Definition |
|---|---|
| small | height &lt; 60 px |
| medium | 60 ≤ height &lt; 160 px |
| large | height ≥ 160 px |

### 2.3 GT limitations (honest)

- Corridor / running / intrusion GT is **assist-aided and visually verified**, not exhaustively pixel-perfect for every heavy occlusion.
- MOT17 / MOT20 GT is **authoritative** for crowded recall claims.
- Higher raw box counts alone are **not** treated as better performance.

---

## 3. Reproducible commands

```bash
# Rebuild GT candidates + merge MOT official boxes (if regenerating)
python tools/build_phase27_manual_gt.py

# Fair same-frame evaluation: 640/800/960 × fall/crowd (+ optional tiling)
python tools/evaluate_phase27_recall.py --include-tiled --conf 0.08

# Machine-readable results
# outputs/reports/phase27_recall_results.json

# Existing regression suite
python tests/run_all_tests.py
```

---

## 4. Fair baseline results (identical 15 frames)

Detector metrics are **identical** for fall vs crowd at the same `imgsz` (tracker does not change YOLO boxes). Tracker columns are **mean boxes that received valid ByteTrack IDs** (single-frame cold start per annotated frame — conservative for ID assignment).

| imgsz | Tracker | Tiled | Precision | Recall | F1 | Mean det | Mean tracked | Untracked~ | Pipeline ms | ~FPS |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 | fall | no | 0.697 | 0.506 | 0.586 | 20.9 | 12.0 | 6.0 | 348 | 2.9 |
| 640 | crowd | no | 0.697 | 0.506 | 0.586 | 20.9 | 9.9 | 8.9 | 218 | 4.6 |
| **800** | fall | no | 0.663 | **0.566** | **0.611** | 24.5 | 14.1 | 7.6 | 367 | 2.7 |
| **800** | crowd | no | 0.663 | **0.566** | **0.611** | 24.5 | 11.7 | 7.6* | 375 | 2.7 |
| **960** | fall | no | **0.716** | **0.620** | **0.664** | 24.9 | 14.0 | — | ~490 | **2.0** |
| **960** | crowd | no | **0.716** | **0.620** | **0.664** | 24.9 | 12.3 | — | ~470 | **2.1** |
| 800 | crowd | **yes** | 0.512 | 0.627 | 0.564 | 35.1 | 11.7 | — | ~810 | **1.2** |
| 640 | crowd | **yes** | 0.515 | 0.592 | 0.551 | 33.0 | 9.9 | — | ~830 | **1.2** |

\*Untracked estimates from JSON; tiling raises recall slightly but collapses precision and FPS.

### 4.1 Size-stratified **recall** (detection vs GT)

| imgsz | Small R | Medium R | Large R |
|---:|---:|---:|---:|
| 640 | 0.357 | 0.389 | 0.711 |
| 800 | 0.429 | 0.459 | 0.755 |
| 960 | **0.500** | **0.533** | **0.774** |

Small / distant people remain the hardest failure mode. Raising `imgsz` helps all bins; gains are largest for small/medium.

---

## 5. Tracking (consecutive crowded video)

Single-frame cold starts under-count crowd tracker (higher `new_track_thresh`). On **consecutive** MOT20-01 frames **40–59**, `imgsz=800`, `conf=0.08`:

| Tracker | Det P | Det R | Det F1 | Mean det | Mean tracked | Mean untracked~ |
|---|---:|---:|---:|---:|---:|---:|
| bytetrack_fall | 0.804 | 0.461 | 0.586 | 30.3 | 14.3 | 14.8 |
| **bytetrack_crowd** | 0.804 | 0.461 | 0.586 | 30.3 | **16.9** | **12.1** |

**Conclusion:** For live streaming, **crowd** assigns more IDs and leaves fewer detections untracked. Detection P/R unchanged (same YOLO). Ablating `new_track_thresh` (0.30 → 0.22 → 0.15) on this mid-sequence window did **not** change mean tracked counts (tracks already established); no threshold change applied without clearer gain.

**Rule retained:** only boxes with real ByteTrack IDs count as tracked; untracked diagnostic overlays remain optional (`show_untracked_detections: false`).

---

## 6. Small-person / tiling investigation

| Change | Effect on recall | Effect on precision | Latency | Verdict |
|---|---|---|---|---|
| imgsz 640 → 800 | +6.0 pp overall; small +7.1 pp | −3.3 pp | Still ~2.7 FPS | **Keep for CPU default** |
| imgsz 800 → 960 | +5.3 pp overall; small +7.1 pp | +5.3 pp | ~2.0 FPS | Optional if user accepts slower live stream |
| Selective tiled (ROI) | Slight recall↑ | **Large precision↓** (~0.51) | ~1.2 FPS | **Do not enable globally** |
| conf=0.08 | Preserved (Phase 26) | Accepts more weak dets | — | Keep |
| Motion blur / occlusion / high angle | Primary miss causes for small people | — | — | Documented limitation |

---

## 7. Recommended CPU configuration

Already aligned in `configs/config.yaml` after Phase 26/27 evidence:

```yaml
yolo_model: "yolov8n.pt"
detection_confidence: 0.08
inference_imgsz: 800          # best CPU trade-off; use 960 for max recall if FPS OK
tracker_config: "configs/bytetrack_crowd.yaml"
tiled_inference.enabled: false
show_untracked_detections: false
```

**Why not 960 by default?** F1 is higher (0.664 vs 0.611) but live processing drops to ~2 FPS on this CPU — often below source video rate.  
**Why not tiling?** Recall gains are offset by many false positives and ~1.2 FPS.

---

## 8. Application validation

```bash
python tests/run_all_tests.py
```

**Result (executed):** **20/20 passed** (1.39s).

Manual expectations (architecture unchanged):
- Video switch clears trails / twin state (Phase 25 behavior preserved).
- CCTV overlays, track IDs, alerts, and Digital Twin share the same pedestrian payload.
- Processing FPS in UI is pipeline throughput, not source FPS (Phase 26).

---

## 9. Known limitations

1. **Not 100% detection** — even at imgsz=960, overall recall ≈ 0.62 on this 15-frame mix; small-person recall ≈ 0.50.
2. Manual GT for asset clips is verified but not MOT-grade exhaustive.
3. Dense MOT20 still leaves a large det→track gap (~12 untracked boxes/frame at imgsz=800); many GT people are never detected by YOLOv8n at all (det recall ~0.46 on MOT20 window).
4. CPU-only; CUDA would change FPS recommendations.
5. Tiled inference remains available for offline experiments only.

---

## 10. Changed / added files

| Path | Role |
|---|---|
| `data/phase27_gt/ground_truth.json` | Annotated + official GT |
| `data/phase27_gt/frames/` | Eval frames |
| `data/phase27_gt/overlays/` | Visual verification overlays |
| `tools/build_phase27_manual_gt.py` | GT bootstrap |
| `tools/evaluate_phase27_recall.py` | Fair P/R/F1 + latency harness |
| `outputs/reports/phase27_recall_results.json` | Full numeric results |
| `PHASE27_DETECTION_RECALL_REPORT.md` | This report |
| `configs/config.yaml` | Remains imgsz=800 + crowd tracker (recommended) |

No UI redesign. No fabricated detections, IDs, or unverified recall claims.
