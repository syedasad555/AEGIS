# Benchmark Datasets Documentation — Phase 19 Generalization

## 1. Multiple Object Tracking Benchmark (MOT17)
- **Sequence:** `MOT17-02-FRCNN`
- **Environment:** Outdoor pedestrian mall / public walkway
- **Characteristics:** Moving camera viewpoints, medium pedestrian density, natural crossing paths, severe pedestrian-to-pedestrian and static object occlusions.
- **Resolution:** 1920 × 1080 @ 30 FPS
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Citation:** Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). *MOT16: A Benchmark for Multi-Object Tracking.* arXiv:1603.00831.

---

## 2. MOT20 Dense Crowd Benchmark (MOT20)
- **Sequence:** `MOT20-01`
- **Environment:** High-density indoor pedestrian corridor / transit shopping concourse
- **Characteristics:** Extreme crowd density (up to dozens of overlapping individuals per frame), frequent occlusions, distant pedestrians, small bounding box sizes.
- **Resolution:** 1920 × 1080 @ 25 FPS
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Citation:** Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K., & Leal-Taixé, L. (2020). *MOT20: A benchmark for multi-object tracking in crowded scenes.* arXiv:2003.09003.

---

## 3. Fall & Activity of Daily Living (ADL) Benchmark
- **Sequences:**
  - `indoor_fall.mp4`: Indoor living space fall event (Positive).
  - `indoor_normal_adl.mp4`: Normal ADL sequence containing walking, sitting down on a chair, and standing back up (Negative).
  - `le2i_coffee_fall.mp4`: Le2i Coffee room fall event (Positive).
  - `le2i_coffee_adl.mp4`: Le2i Coffee room bending and reaching activity (Negative).
  - `assets/fall.mp4`: Outdoor incline icy ramp fall (Positive).
- **Source & Provenance:** Le2i Fall Detection Dataset (Laboratoire Electronique, Informatique et Image, Université de Bourgogne) and OpenMMLab Fall Benchmark repository (EltonCCL).
- **License:** Research and Non-Commercial Academic Use License.
- **Citation:** Charfi, I., Miteran, J., Dubois, J., Atri, M., & Tourki, R. (2013). *Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and AdaBoost-based classification.* Journal of Electronic Imaging, 22(4), 041106.

---

## 4. Separation & Data Integrity
All evaluation sequences were kept strictly isolated from model training or tuning. No parameters were retrofitted to specific test frame indices. Evaluation measures true generalization performance across unseen video perspectives.
