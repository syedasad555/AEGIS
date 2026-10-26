#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""Measure per-stage latency for pipeline analysis.

Stages:
  1. 视频解码与预处理
  2. YOLO检测推理
  3. 检测后处理 (NMS)
  4. ByteTrack跟踪匹配
  5. 特征提取
  6. 异常规则判别
  7. 可视化渲染

Usage:
    python tools/measure_pipeline.py --video <video_path> --weights <weights_path>
"""

import argparse
import time
import numpy as np
import cv2
from collections import defaultdict
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Measure per-stage pipeline latency")
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
    WARMUP = 5

    model = YOLO(WEIGHTS)

    # ── Phase A: model.track() speed dict ────────────────────────────
    print("Phase A: Detection + Tracking pipeline ...")

    times_a = defaultdict(list)
    frame_idx = 0

    results_gen = model.track(
        source=VIDEO, persist=True,
        tracker="bytetrack.yaml", classes=[0],
        conf=CONF, stream=True, verbose=False,
    )

    for r in results_gen:
        if frame_idx < WARMUP:
            frame_idx += 1
            continue

        if r.speed:
            times_a['preprocess'].append(r.speed['preprocess'])
            times_a['inference'].append(r.speed['inference'])
            times_a['postprocess'].append(r.speed['postprocess'])

        frame_idx += 1
        if frame_idx >= MAX_FRAMES:
            break

    preprocess_ms = np.mean(times_a['preprocess'])
    inference_ms = np.mean(times_a['inference'])
    postprocess_ms = np.mean(times_a['postprocess'])

    # ── Phase B: model.predict() to isolate tracking overhead ─────────
    print("Phase B: Detection-only pipeline (no tracking) ...")

    times_b = defaultdict(list)
    frame_idx = 0

    results_gen_b = model.predict(
        source=VIDEO, classes=[0],
        conf=CONF, stream=True, verbose=False,
    )

    for r in results_gen_b:
        if frame_idx < WARMUP:
            frame_idx += 1
            continue

        if r.speed:
            times_b['preprocess'].append(r.speed['preprocess'])
            times_b['inference'].append(r.speed['inference'])
            times_b['postprocess'].append(r.speed['postprocess'])

        frame_idx += 1
        if frame_idx >= MAX_FRAMES:
            break

    # ── Phase C: Benchmark feature extraction + rule decision ─────────
    print("Phase C: Feature extraction + Rule decision benchmark ...")

    # Get a sample detection result for realistic feature counts
    cap = cv2.VideoCapture(VIDEO)
    ret, frame = cap.read()
    cap.release()
    sample_results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 classes=[0], conf=CONF, verbose=False)

    if sample_results[0].boxes is not None and len(sample_results[0].boxes) > 0:
        sample_boxes = sample_results[0].boxes.xyxy.cpu().numpy()
        sample_ids = (sample_results[0].boxes.id.int().cpu().numpy()
                      if sample_results[0].boxes.id is not None
                      else np.arange(len(sample_boxes)))
        n_boxes = len(sample_boxes)
    else:
        n_boxes = 8
        np.random.seed(42)
        sample_boxes = np.random.rand(n_boxes, 4) * np.array([1920, 1080, 1920, 1080])
        sample_ids = np.arange(n_boxes)

    feat_times = []
    for _ in range(500):
        t0 = time.time()
        xyxy = sample_boxes.copy()
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        _ = w * h  # benchmark: measure vectorized compute
        aspects = w / (h + 1e-6)
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        feat_times.append((time.time() - t0) * 1000)

    feature_ms = np.mean(feat_times)

    # Rule decision benchmark
    rule_times = []
    traj = {i: [(cx[i], cy[i], 0)] for i in range(n_boxes)}
    for _ in range(500):
        t0 = time.time()
        for i in range(n_boxes):
            aspect_i = aspects[i]
            cy_i = cy[i]
            if aspect_i > 3.0 or aspect_i < 0.3:
                pass
            if cy_i > 0.85 * 1080:
                pass
            if i in traj and len(traj[i]) >= 2:
                prev_cx, prev_cy, _ = traj[i][-2]
                dist = np.sqrt((cx[i] - prev_cx)**2 + (cy[i] - prev_cy)**2)
                if dist > 80:
                    pass
        rule_times.append((time.time() - t0) * 1000)

    rule_ms = np.mean(rule_times)

    # ── Phase D: Visualization benchmark ──────────────────────────────
    print("Phase D: Visualization benchmark ...")

    vis_times = []
    for _ in range(200):
        frame_copy = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        t0 = time.time()

        for i in range(n_boxes):
            x1, y1, x2, y2 = sample_boxes[i].astype(int)
            color = (0, 255, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{sample_ids[i]}"
            cv2.putText(frame_copy, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        pts = np.array([[200, 750], [800, 600], [1400, 700], [1600, 950],
                        [600, 1000], [100, 900]], np.int32)
        cv2.polylines(frame_copy, [pts], True, (0, 0, 255), 2)

        cv2.rectangle(frame_copy, (0, 0), (1920, 35), (0, 0, 200), -1)
        cv2.putText(frame_copy, "ALERT: Intrusion | Running: 2 | Fall: 1",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        vis_times.append((time.time() - t0) * 1000)

    visualization_ms = np.mean(vis_times)

    # ── Video decode benchmark ───────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO)
    decode_times = []
    for _ in range(200):
        t0 = time.time()
        ret, _ = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, _ = cap.read()
        decode_times.append((time.time() - t0) * 1000)
    cap.release()
    decode_ms = np.mean(decode_times)

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n视频解码:       {decode_ms:.2f} ms")
    print(f"预处理:         {preprocess_ms:.2f} ms")
    print(f"YOLO 推理:      {inference_ms:.2f} ms")
    print(f"NMS 后处理:     {postprocess_ms:.2f} ms")
    print(f"特征提取:       {feature_ms:.4f} ms")
    print(f"规则判别:       {rule_ms:.4f} ms")
    print(f"可视化渲染:     {visualization_ms:.2f} ms")
    total = (decode_ms + preprocess_ms + inference_ms + postprocess_ms +
             feature_ms + rule_ms + visualization_ms)
    print("───────────────────────")
    print(f"总计:           {total:.2f} ms ({1000/total:.1f} FPS)")


if __name__ == "__main__":
    main()
