# AEGIS UCF Evaluation Reproducibility Specification

## 1. Environment & Setup

- **Operating System:** Windows 11
- **Python Version:** 3.10+ (C:\Users\syeda\python.exe)
- **PyTorch:** 2.14.1+cpu
- **Ultralytics:** 8.3
- **OpenCV:** 5.0.0
- **Model Checkpoint:** `weights/aegis_yolo11n_crowdhuman.pt`
- **Tracker Configuration:** `configs/bytetrack_crowd.yaml`

## 2. Dataset Partition Specification

- **Total Videos:** 18 (17 playable, 1 corrupted `rush_01.mov`)
- **Training Partition (8 videos):** `2018-6_70.mov`, `2082-3_70.mov`, `341-46_l.mov`, `620-72_l.mov`, `637-147_l.mov`, `7399-1_70.mov`, `9019-13_l.mov`, `913-35_l.mov`
- **Validation Partition (4 videos):** `3687-18_70.mov`, `821-10_l.mov`, `9-19_l.mov`, `913-37_l.mov`
- **Held-Out Test Partition (5 videos):** `688-10_l.mov`, `861-13_70.mov`, `879-38_l.mov`, `906-17_l.mov`, `913-36_l.mov`

## 3. Reproduction Command

```bash
python evaluation/run_ucf_complete_pipeline.py
```
