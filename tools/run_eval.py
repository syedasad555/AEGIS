#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation & Benchmarking Script
=================================
Runs evaluation across all test videos (intrusion.mp4, running.mp4, fall.mp4),
analyzes tracking statistics, rule anomaly detection metrics, and CPU throughput,
and generates structured outputs under outputs/reports/.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _common import get_video_props, open_video
from scripts.pedestrian_pipeline import PedestrianPipeline, load_config


def run_evaluation() -> Dict[str, Any]:
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    pipeline = PedestrianPipeline(cfg)

    test_videos = [
        ("intrusion.mp4", PROJECT_ROOT / "assets" / "intrusion.mp4"),
        ("running.mp4", PROJECT_ROOT / "assets" / "running.mp4"),
        ("fall.mp4", PROJECT_ROOT / "assets" / "fall.mp4"),
    ]

    results_summary = []
    total_frames_all = 0
    total_pedestrians_all = 0
    total_anomalies_all = 0

    eval_start = time.time()

    for name, path in test_videos:
        if not path.exists():
            print(f"[Warning] Test video not found: {path}")
            continue

        out_path = PROJECT_ROOT / "outputs" / "videos" / f"{path.stem}_processed.mp4"
        res = pipeline.process_video(path, output_video_path=out_path, display=False)

        # Read back anomaly log for rule breakdown
        anomaly_file = PROJECT_ROOT / "outputs" / "anomalies" / f"{path.stem}_anomalies.json"
        rule_breakdown = {
            "WRONG_DIRECTION": 0,
            "PROLONGED_STOPPING": 0,
            "RESTRICTED_ZONE_ENTRY": 0,
            "FALL_DETECTED": 0,
            "FALL_SUSPECTED": 0
        }
        if anomaly_file.exists():
            with open(anomaly_file, "r", encoding="utf-8") as f:
                anom_data = json.load(f)
                for item in anom_data:
                    atype = item.get("anomaly_type", "OTHER")
                    if atype in rule_breakdown:
                        rule_breakdown[atype] += 1
                    else:
                        rule_breakdown[atype] = rule_breakdown.get(atype, 0) + 1

        res["rule_breakdown"] = rule_breakdown
        results_summary.append(res)

        total_frames_all += res["total_frames"]
        total_pedestrians_all += res["unique_pedestrians_tracked"]
        total_anomalies_all += res["total_anomaly_instances"]

    total_eval_time = time.time() - eval_start
    overall_fps = total_frames_all / max(total_eval_time, 0.001)

    eval_report = {
        "evaluation_mode": "Rule-based demonstration evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_videos": len(results_summary),
        "total_frames_processed": total_frames_all,
        "total_time_seconds": round(total_eval_time, 2),
        "overall_processing_fps": round(overall_fps, 1),
        "total_unique_pedestrians": total_pedestrians_all,
        "total_anomaly_events": total_anomalies_all,
        "per_video_results": results_summary
    }

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "outputs" / "reports" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Metrics saved -> {metrics_path}")

    # Generate Markdown Report
    report_md_path = PROJECT_ROOT / "outputs" / "reports" / "EVALUATION_REPORT.md"
    generate_markdown_report(eval_report, report_md_path)

    return eval_report


def generate_markdown_report(data: Dict[str, Any], output_path: Path):
    md = f"""# System Evaluation Report — Evacuation Anomaly Detection

**Evaluation Method:** {data['evaluation_mode']}  
**Date & Time:** {data['timestamp']}  
**CPU Processing Speed:** {data['overall_processing_fps']} FPS ({data['total_frames_processed']} frames in {data['total_time_seconds']}s)  

---

## 1. Overall Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Total Test Videos Evaluated** | `{data['total_test_videos']}` |
| **Total Video Frames Processed** | `{data['total_frames_processed']}` |
| **Unique Pedestrian Track IDs** | `{data['total_unique_pedestrians']}` |
| **Total Anomaly Instances Detected** | `{data['total_anomaly_events']}` |
| **Average Processing Speed (CPU)** | `{data['overall_processing_fps']} FPS` |

---

## 2. Per-Video Benchmark Results

"""
    for v in data["per_video_results"]:
        breakdown = v["rule_breakdown"]
        md += f"""### Scenario Video: `{v['video']}`
- **Frames:** {v['total_frames']}
- **Processing Time:** {v['processing_time_sec']}s ({v['processing_fps']} FPS)
- **Tracked Pedestrians:** {v['unique_pedestrians_tracked']} IDs
- **Total Anomaly Instances:** {v['total_anomaly_instances']}
- **Rule Detections Breakdown:**
  - `FALL_DETECTED`: {breakdown.get('FALL_DETECTED', 0)}
  - `FALL_SUSPECTED`: {breakdown.get('FALL_SUSPECTED', 0)}
  - `WRONG_DIRECTION`: {breakdown.get('WRONG_DIRECTION', 0)}
  - `PROLONGED_STOPPING`: {breakdown.get('PROLONGED_STOPPING', 0)}
  - `RESTRICTED_ZONE_ENTRY`: {breakdown.get('RESTRICTED_ZONE_ENTRY', 0)}

---
"""

    md += """## 3. Methodological Note

As specified in the research guidelines, this evaluation reports empirical system performance on the baseline pedestrian test dataset under rule-based evacuation context constraints. No artificial accuracy metrics are claimed.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Eval] Markdown report generated -> {output_path}")


if __name__ == "__main__":
    run_evaluation()
