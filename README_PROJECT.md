# Context-Aware Pedestrian Anomaly Detection with a 2D Digital Twin for Evacuation Monitoring

A research prototype for computer-vision-based evacuation monitoring. This system integrates real-time YOLOv8 pedestrian detection, ByteTrack multi-object tracking, dynamic trajectory feature extraction, explainable evacuation rule analysis, and a 2D Digital Twin dashboard.

---

## 1. Problem Statement & Objective

During emergency evacuations (e.g. building fires, crowd panics, security alarms), surveillance monitoring requires rapid identification of unsafe pedestrian behavior. Standard object detectors only identify where people are located, but fail to analyze movement context such as:
- Pedestrians collapsing or falling to the ground (Fallen Pedestrian Detection).
- Pedestrians walking against the designated evacuation direction (Wrong-Direction Movement).
- Pedestrians stopping or blocking bottleneck corridors (Prolonged Stopping).
- Pedestrians entering dangerous or blocked corridors (Restricted-Zone Entry).

**System Objective:** Process pedestrian CCTV feeds, track unique pedestrian identities, extract trajectories, evaluate movement against evacuation context and fall detection rules, and mirror live pedestrian states into a 2D Digital Twin map with dynamic alerts.

---

## 2. Architecture & Pipeline

```
+------------------+     +-------------------+     +-------------------------+
|  CCTV Video /    | --> |  YOLOv8n Person   | --> |  ByteTrack Multi-Object |
|  Pedestrian Feed |     |  Detection (c=0)  |     |  Tracking (Track IDs)   |
+------------------+     +-------------------+     +-------------------------+
                                                                |
                                                                v
+------------------+     +-------------------+     +-------------------------+
|  2D Digital Twin | <-- | Evacuation Rule & | <-- | Trajectory Extraction & |
|  Dashboard UI    |     | Fall Engine       |     | Targeted Pose (YOLO-Pose|
+------------------+     +-------------------+     +-------------------------+
```

---

## 3. Baseline vs Our Contribution

| Component | Baseline Repository | Our Contribution |
| :--- | :--- | :--- |
| **Object Detection** | Standard YOLOv8 detection | Low-threshold detection ($c=0.08$) + class 0 (`person`) filtering |
| **Tracking** | ByteTrack ID assignment | Persistent ID trajectory management with deformation-resilient association |
| **Feature Extraction** | Basic speed or posture hints | Foot tracking, speed, direction vectors, aspect ratio, standing height baseline |
| **Fall Detection** | None (unintegrated prototype) | Two-Tier Temporal Geometry + Targeted Crop YOLOv8-Pose Disambiguation |
| **Context Analysis** | Isolated event detectors | Priority Evacuation Rules (`FALL_DETECTED`, `RESTRICTED_ZONE_ENTRY`, `FALL_SUSPECTED`, etc.) |
| **Digital Twin** | None (Raw OpenCV windows) | Interactive HTML5 Canvas 2D Digital Twin Map & Synchronized Inspection Console |
| **Communication** | Local file outputs only | Real-time WebSocket streaming & synced web playback |

---

## 4. Technology Stack

- **Core & Inference:** Python 3.10+, PyTorch (CPU compatible), Ultralytics YOLOv8n
- **Multi-Object Tracking:** ByteTrack
- **Computer Vision & Geometry:** OpenCV (`opencv-python`), Shapely (`shapely`), NumPy
- **Server & Networking:** Python `http.server`, `websockets`
- **2D Digital Twin Frontend:** HTML5, CSS3 (Modern Dark Theme), JavaScript Canvas 2D API

---

## 5. Installation & Setup

1. **Clone / Open Repository:**
   ```bash
   cd crowd-abnormal-behavior-detection-main
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Environment:**
   ```bash
   python tools/check_env.py
   ```

---

## 6. How to Run

### A. Run 2D Digital Twin Web Dashboard Server
```bash
python server.py
```
Open your browser at: **`http://localhost:8080/`**

### B. Run Pedestrian Anomaly Pipeline
```bash
# Process intrusion video scenario
python scripts/pedestrian_pipeline.py --source assets/intrusion.mp4

# Process running/evacuation video scenario
python scripts/pedestrian_pipeline.py --source assets/running.mp4

# Process fall video scenario
python scripts/pedestrian_pipeline.py --source assets/fall.mp4

# Process with custom polygon zones configuration
python scripts/pedestrian_pipeline.py --source assets/intrusion.mp4 --zone configs/forbidden_zone.example.json
```

### C. Run Full Evaluation Benchmark
```bash
python tools/run_eval.py
```

### D. Interactive Polygonal Zone Annotator
```bash
python tools/zone_selector.py --source assets/intrusion.mp4 --output configs/forbidden_zone.json
```

---

## 7. Configuration (`configs/config.yaml`)

Centralized configuration file for detection parameters, evacuation direction, movement thresholds, and polygon zone boundaries:

```yaml
yolo_model: "yolov8n.pt"
detection_confidence: 0.35
person_class_id: 0
expected_evacuation_direction: "RIGHT"  # RIGHT, LEFT, UP, DOWN
stationary_speed_threshold: 0.6         # px/frame (~18 px/sec at 30 FPS, smoothed over 8 frames)
prolonged_stopping_frames: 30           # frames (1.0 sec threshold at 30 FPS)
zones:
  restricted_zones: [...]
  evacuation_routes: [...]
  safe_zones: [...]
  exit_zones: [...]
```

---

## 8. Evacuation Anomaly Rules & Severity Engine

1. **Rule 1 — Wrong Direction (`WRONG_DIRECTION`)**:
   Triggers when a pedestrian's computed movement direction opposes the configured expected evacuation flow direction (e.g. moving `LEFT` when expected direction is `RIGHT`). Evaluated as `WARNING`.
2. **Rule 2 — Prolonged Stopping (`PROLONGED_STOPPING`)**:
   Triggers when a pedestrian's smoothed relative speed remains below `stationary_speed_threshold` and movement is stationary for >= `prolonged_stopping_frames` (30 frames). Classified as `WARNING` for durations 1.0s–3.0s, and escalates to `CRITICAL` for durations >= 3.0s.
3. **Rule 3 — Restricted Zone Entry (`RESTRICTED_ZONE_ENTRY`)**:
   Triggers immediately when a pedestrian's bottom-center foot coordinate enters any designated `RESTRICTED_ZONE` polygon. Classified as `CRITICAL`.

*Multi-Rule Trigger:* Any pedestrian concurrently violating more than one rule is escalated to `CRITICAL`. All active alerts track onset frames to report verified duration in seconds.

---

## 9. AEGIS Evacuation Vision Console & Interactive Features

The console features a modern, professional computer-vision monitoring interface engineered for 1366×768 and 1920×1080 display viewports:
- **Header Bar:** System telemetry, current scenario identifier, live evacuation flow direction toggle (`RIGHT ➔`, `LEFT ⬅`, `UP ⬆`, `DOWN ⬇`), Stream Pause/Resume, Zones & Rules config modal, and Change Video navigation.
- **4 Real-Time KPI Cards:** Pedestrians Detected, Active Tracks, Active Anomalies, Processing FPS.
- **Side-by-Side Monitoring Workspace:**
  - **Left Viewport (CCTV Vision Sensor):**
    - Live video stream annotated with thin 1-2px bounding boxes (Cyan for normal flow, Red for active anomalies, Amber for warnings).
    - Compact pedestrian tags (`#ID · STOPPING` / `#ID · WRONG_DIR`) positioned to avoid obscuring bodies.
    - Subtle recent trajectories (18-point history trail) and low-opacity zone fills.
    - Active inspection focus brackets when a track is selected.
  - **Right Viewport (Architectural Corridor Digital Twin):**
    - Realistic corridor architectural floor plan with structural outer walls, west access portal, east emergency fire doors, central walkway corridors, dynamic flow chevrons, and restricted floor zones.
    - Synchronized ByteTrack pedestrian dots and labels with 1-to-1 parity to CCTV view.
    - Bottom map legend and camera-to-floor-plan perspective mapping footnote.
- **Incident Alert Console & Bidirectional Track Inspection:**
  - Structured live alert table reporting Severity (`CRITICAL` / `WARNING`), Pedestrian Track ID, Anomaly Type, Duration (sec), and Timestamp.
  - Clicking any alert row or pedestrian dot triggers bidirectional focus highlighting across both CCTV and Digital Twin viewports.
  - Clear inspection button to return to global scene monitoring.

---

## 10. Output Organization

```
outputs/
├── baseline/      # Phase 2 baseline output videos
├── videos/        # Processed CCTV videos with overlaid analytics
├── trajectories/  # JSON trajectory records (ID, frame, x, y, speed, direction)
├── anomalies/     # JSON anomaly event logs
└── reports/       # Evaluation metrics (EVALUATION_REPORT.md, metrics.json)
```

---

## 11. Coordinate Transformation & System Limitations

- **Camera-to-Map Coordinate Mapping:**
  The current 2D Digital Twin projects normalized camera bounding-box coordinates $(x_{\text{norm}}, y_{\text{norm}})$ onto the architectural floor plan using linear perspective scaling:
  $$X_{\text{map}} = X_0 + x_{\text{norm}} \cdot W_{\text{corridor}}$$
  $$Y_{\text{map}} = Y_0 + y_{\text{norm}} \cdot H_{\text{corridor}}$$
- **Monocular Foreshortening Limitation:**
  In angled monocular CCTV viewpoints, distant pedestrians appear closer together along the vertical axis due to perspective compression. For sub-meter metric positioning, planar homography calibration ($H \in \mathbb{R}^{3 \times 3}$) or camera intrinsic/extrinsic parameterization is recommended when ground-truth floor fiducials are available.
- **Speed Reporting:** Relative pixel speed is reported in `px/sec` and normalized `px/frame` rather than metric meters per second ($m/s$).
- **ByteTrack ID Persistence:** Tracking relies on visual bounding box overlap (IoU) and Kalman filter state estimation. Full long-duration occlusions or re-entries may trigger a new track ID.

