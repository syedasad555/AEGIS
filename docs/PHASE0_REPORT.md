# Phase 0 Report — Repository Inspection & Baseline Analysis

**Project Title:** Context-Aware Pedestrian Anomaly Detection with a 2D Digital Twin for Evacuation Monitoring  
**Baseline Repository:** Crowd Abnormal Behavior Detection  
**Date:** October 5, 2026  

---

## 1. Existing Architecture & Project Structure

The baseline repository provides a computer vision foundation for crowd and pedestrian analysis using YOLOv8 detection and ByteTrack tracking.

### Repository Layout:
- `assets/`: Contains 3 primary test videos (`fall.mp4`, `intrusion.mp4`, `running.mp4`).
- `configs/`: Configurations for ByteTrack (`bytetrack.yaml`, `bytetrack_fall.yaml`) and sample polygon zones (`forbidden_zone.example.json`).
- `scripts/`: Detection scripts for baseline tasks:
  - `fall_detection.py` (Pose estimation & trajectory drop)
  - `running_detection.py` (Speed computation & running alerts)
  - `intrusion_detection.py` (Polygon intrusion detection)
- `tools/`: Utility scripts for environment checks (`check_env.py`), polygon zone annotation (`zone_selector.py`), performance measurement (`measure_fps.py`, `measure_pipeline.py`), and frame viewing.
- `_common.py`: Shared core module providing video I/O (`open_video`, `create_video_writer`), track ID extraction (`extract_track_ids`), tracker path resolution (`resolve_tracker`), and cleanup logic.
- `outputs/`: Target directory for storing processing results, video exports, trajectories, and anomaly logs.

---

## 2. Reusable Components

The following components are fully functional and will be reused:
1. **YOLO Detection Pipeline (`ultralytics.YOLO`)**: Pre-trained YOLOv8n object detector for COCO class 0 (`person`).
2. **ByteTrack Multi-Object Tracking**: Built-in Ultralytics tracking integration (`model.track(..., tracker=..., persist=True)`).
3. **Video I/O & Display (`_common.py`)**: `open_video`, `get_video_props`, `create_video_writer`, and frame display loop.
4. **Track ID Extraction (`_common.py:extract_track_ids`)**: Safe extraction of integer IDs across PyTorch/NumPy representations.
5. **Interactive Zone Annotation Tool (`tools/zone_selector.py`)**: OpenCV GUI tool for marking custom polygonal evacuation/restricted zones and exporting JSON coordinates.
6. **Geometry & Intrusion Check (`shapely` / `cv2.pointPolygonTest`)**: Point-in-polygon verification for spatial zone logic.

---

## 3. Components Requiring Modification

1. **Class Filtering**: Ensure hard filtering strictly for COCO Class 0 (`person`) in all pipelines.
2. **Config Centralization**: Transition from scattered script constants to a unified YAML configuration file (`configs/config.yaml`).
3. **Zone Model**: Expand single forbidden zone logic into multi-zone spatial model (Evacuation Route, Restricted Zone, Safe Zone, Exit Zone).
4. **Anomaly Engine**: Replace individual event scripts with a unified rule-based evacuation anomaly engine checking:
   - Wrong-direction movement against expected evacuation direction.
   - Prolonged stopping (speed below threshold for extended duration).
   - Restricted zone entry.

---

## 4. Components to Be Added

1. **Environment Setup & Verification (`requirements.txt`)**: Clean dependency specification targeting CPU execution (YOLOv8n + ByteTrack + OpenCV + Shapely + Web/WebSockets).
2. **Unified Pedestrian Tracking Pipeline (`scripts/pedestrian_pipeline.py`)**: Standardized pipeline outputting pedestrian IDs, bottom-center foot coordinates, bounding boxes, and frame metrics.
3. **Trajectory & Movement Feature Extractor (`scripts/feature_extraction.py`)**: Position history tracking, relative pixel speed, movement direction vector, and export to `outputs/trajectories/`.
4. **Evacuation Context Anomaly Engine (`scripts/anomaly_detection.py`)**: Modular rule evaluation engine saving structured JSON anomaly logs to `outputs/anomalies/`.
5. **Unified Configuration (`configs/config.yaml`)**: Centralized config for detection thresholds, tracker settings, evacuation parameters, and zone coordinates.
6. **2D Digital Twin Web Application (`digital_twin/`)**: Lightweight HTML/CSS/JavaScript Canvas front-end rendering virtual map, pedestrian positions, trajectories, evacuation routes, and alert dashboard.
7. **Pipeline-to-Twin Communication & Dashboard**: Local WebSocket server / HTTP streamer broadcasting live frame states to the 2D Digital Twin and split CCTV/Digital Twin interface.
8. **Automated Baseline & Evaluation Tests**: Test runner executing the 3 test videos and writing metrics to `outputs/reports/`.

---

## 5. Required Dependencies

- **Python Version**: Python 3.10+ (compatible with standard laptop CPUs).
- **Core ML & Vision**: `ultralytics>=8.0.0`, `torch`, `torchvision`, `opencv-python>=4.8.0`, `numpy`.
- **Spatial Geometry**: `shapely>=2.0.0`.
- **System & Visualization**: `psutil`, `matplotlib`, `pyyaml`.
- **Web / Communication**: `websockets` or `flask`/`aiohttp` for live streaming to the Digital Twin dashboard.

---

## 6. Baseline Execution Method

The existing baseline scripts are executed as follows:
```bash
# Environment check
python tools/check_env.py

# Zone selection (interactive)
python tools/zone_selector.py --source assets/intrusion.mp4 --output configs/forbidden_zone.json

# Intrusion detection baseline
python scripts/intrusion_detection.py --source assets/intrusion.mp4 --zone configs/forbidden_zone.example.json

# Running detection baseline
python scripts/running_detection.py --source assets/running.mp4

# Fall detection baseline
python scripts/fall_detection.py --source assets/fall.mp4
```
