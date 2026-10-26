#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""Measure pure detection FPS (no tracking overhead).

Usage:
    python tools/measure_fps.py --video <video_path> --weights <weights_path>
"""

import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Measure pure detection FPS")
    parser.add_argument("--video", type=str, required=True,
                        help="输入视频路径")
    parser.add_argument("--weights", type=str, required=True,
                        help="YOLO 权重路径 (.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="检测置信度阈值 (默认: 0.5)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="测试最大帧数 (默认: 300)")
    args = parser.parse_args()

    VIDEO = args.video
    WEIGHTS = args.weights
    CONF = args.conf
    MAX_FRAMES = args.max_frames

    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(VIDEO)

    # Warmup — discard first few frames
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False, conf=CONF, classes=[0])

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    times = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        model(frame, verbose=False, conf=CONF, classes=[0])
        times.append(time.time() - t0)
        frame_count += 1
        if frame_count >= MAX_FRAMES:
            break

    cap.release()

    avg_ms = np.mean(times) * 1000
    fps = 1.0 / np.mean(times)

    print(f"测速配置: {WEIGHTS}")
    print(f"测试视频: {VIDEO}")
    print(f"处理帧数: {len(times)}")
    print(f"图片尺寸: {img_w}x{img_h}")
    print()
    print("=== 检测 FPS ===")
    print(f"平均耗时: {avg_ms:.1f} ms/帧")
    print(f"检测 FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
