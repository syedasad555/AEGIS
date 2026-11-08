# AEGIS Phase 33 — Optional Far-Camera Resolution Profile

**Date:** 2026-10-26  
**Production default:** unchanged — YOLOv8n, `imgsz=800`, `bytetrack_crowd`, tiled off  
**Profile:** `far_camera_profile` in `configs/config.yaml` — **`enabled: false`**

## Mechanism

- `resolve_inference_imgsz(config)` in `_common.py` returns 800 unless `far_camera_profile.enabled` is true, then 960.
- Resolved once per streamer/pipeline session (not every frame).
- No automatic scene classification.

## Evidence

See `DETECTION_TRACKING_IMPROVEMENT_REPORT.md` and `outputs/reports/detection_tracking_benchmark_summary.json`.

On far clips (esp. `4917-5`), imgsz 960 yields high-score dets and sparse tracks where 800 had zero tracks. Dense clips stay usable (~4 FPS, short-lived proportion not worse).

## Decision

- Keep profile **disabled** in production config.
- Operators may enable it for selected far outdoor cameras only.
- Do not set global `inference_imgsz: 960`.

## Rollback

Set `far_camera_profile.enabled: false` and restart the server.
