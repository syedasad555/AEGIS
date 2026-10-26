# AEGIS Phase 29 — UCF Crowds Dataset Audit

**Date:** 2026-10-24  
**Dataset path:** `UCF_CrowdsDataset/` (config key `ucf_dataset_root`)  
**Inventory JSON:** `outputs/reports/phase29_ucf_inventory.json`  
**Tool:** `tools/audit_ucf_dataset.py` (read-only; originals not modified)

---

## 1. Folder structure

| Finding | Detail |
|---|---|
| Layout | **Flat** directory of QuickTime `.mov` clips — no nested class / train / test folders |
| Video files | **38** |
| Non-video files | **0** |
| Image-sequence dirs | **0** |
| Annotation-like files | **0** (no `.txt` / `.csv` / `.json` / `.xml` / `.mat` beside videos) |

---

## 2. Ground truth

**Pedestrian bounding-box / track ID GT is not present in this folder.**

UCF Crowds–style packages often ship density maps or count labels elsewhere. This checkout contains **media only**. Filename tokens such as “anomaly” must **not** be treated as bbox GT.

Consequence for metrics: Phase 29 validation records **observed** detection / track / fall / FPS counts only. **Precision / recall / F1 are not claimed** on these clips.

---

## 3. Decode audit (OpenCV)

| Metric | Value |
|---:|
| Opened + first frame readable | **34 / 38** |
| Unplayable (open OK, read fail) | **4 / 38** |

### Unplayable clips (codec / backend)

| File | Observed issue |
|---|---|
| `001-0436.mov` | FourCC `????`; first frame read failed (MSMF/zlib-class failure on this Windows OpenCV build) |
| `001-0438.mov` | Same |
| `rush_01.mov` | Same |
| `sfw20110024_l.mov` | Same |

No conversion was applied in Phase 29 (audit-only). Optional later step: re-encode these four to H.264 for tooling compatibility.

### Playable codecs / resolutions (observed)

| Codec (FourCC) | Present |
|---|---|
| `h264` | Yes (majority of playable set) |
| `svq3` | Yes (several `*_l.mov` clips) |

| Resolution samples | Notes |
|---|---|
| 480×360 | Common for `*_l.mov` |
| 720×404 | e.g. `4917-*_70.mov` |
| 720×480 | e.g. RF / some outdoor |
| 720×576 | Several `*_70.mov` |

Source FPS typically **15–30** (metadata); `frame_count` is reliable on h264, sometimes `-1` / incomplete on failed opens.

---

## 4. Previously requested clips

| Clip | Status in this repo |
|---|---|
| `341-46_l.mov` | **Found** under `UCF_CrowdsDataset/341-46_l.mov` (playable, 480×360, h264/svq3 family per inventory) |
| `913-37_l.mov` | **Found** under `UCF_CrowdsDataset/913-37_l.mov` (playable) |

They were missing from `assets/` in earlier phases; UCF folder is the correct location.

---

## 5. Crowded / representative candidates for validation

Selected for baseline run (configurable list in `tools/validate_ucf_phase29.py`):

1. `4917-3_70.mov` — long dense outdoor crowd  
2. `4917-5_70.mov` — long dense outdoor crowd  
3. `879-38_l.mov` — dense SVQ3-era crowd  
4. `821-10_l.mov` — medium-long crowd  
5. `341-46_l.mov` — previously requested  
6. `913-37_l.mov` — previously missing from assets  
7. `RF1-12373_70.mov` — RF / indoor-ish h264  

Full per-file table: `outputs/reports/phase29_ucf_inventory.json`.

---

## 6. Audit conclusions

1. Dataset is usable for **smoke / qualitative real-crowd baseline** on 34/38 clips.  
2. **Do not** score detection recall against this folder without external GT.  
3. Keep dataset root configurable (`ucf_dataset_root`); do not hardcode single filenames into the server.  
4. Four unreadable clips should be excluded from automated pipelines until re-encoded.  
5. Proceed to observed-metrics validation (see `PHASE29_REAL_VIDEO_VALIDATION.md`) **before** any UI redesign or model retune.
