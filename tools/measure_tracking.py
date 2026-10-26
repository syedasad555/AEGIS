#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""Measure ByteTrack tracking metrics.

Track metrics: ID Switch count, average trajectory length, trajectory completeness.

Usage:
    python tools/measure_tracking.py --video <video_path> --weights <weights_path>
"""

import argparse
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Measure ByteTrack tracking metrics")
    parser.add_argument("--video", type=str, required=True,
                        help="输入视频路径")
    parser.add_argument("--weights", type=str, required=True,
                        help="YOLO 权重路径 (.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="检测置信度阈值 (默认: 0.5)")
    args = parser.parse_args()

    VIDEO = args.video
    WEIGHTS = args.weights
    CONF = args.conf

    model = YOLO(WEIGHTS)
    track_frames = defaultdict(int)
    id_switch = 0

    results = model.track(
        source=VIDEO, persist=True,
        tracker="bytetrack.yaml", classes=[0],
        conf=CONF, stream=True, verbose=False,
    )

    prev_ids = set()
    total_frames = 0

    for r in results:
        total_frames += 1
        if r.boxes.id is not None:
            cur_ids = set(r.boxes.id.int().tolist())
            for tid in cur_ids:
                track_frames[tid] += 1
            new_ids = cur_ids - prev_ids
            lost_ids = prev_ids - cur_ids
            id_switch += min(len(new_ids), len(lost_ids))
            prev_ids = cur_ids
        else:
            prev_ids = set()

    avg_len = np.mean(list(track_frames.values())) if track_frames else 0.0
    complete_count = sum(1 for v in track_frames.values() if v > 0.8 * total_frames)
    completeness = complete_count / len(track_frames) * 100 if track_frames else 0.0
    unique_tracks = len(track_frames)

    print(f"跟踪配置: {WEIGHTS}")
    print(f"测试视频: {VIDEO}")
    print(f"总帧数: {total_frames}")
    print(f"唯一 track ID 数: {unique_tracks}")
    print()
    print("=== 跟踪指标 ===")
    print(f"ID Switch 次数: {id_switch}")
    print(f"平均轨迹长度 (帧): {avg_len:.1f}")
    print(f"轨迹完整性 (>80% 帧): {completeness:.1f}% ({complete_count}/{unique_tracks})")


if __name__ == "__main__":
    main()
