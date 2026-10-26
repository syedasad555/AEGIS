#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
奔跑检测脚本 — YOLO 检测 + ByteTrack 跟踪 + 速度计算
修复版：检测层区域化阈值 + 新生ID缓冲期 + ID切换证据继承 + 轨迹清理
- 用 model.track() 实现检测+跟踪，仅追踪 person (class=0)
- 计算每个 track_id 的中心点移动速度（像素/秒）
- 连续 3 帧速度超阈值 → 触发奔跑告警，30 帧冷却
- 仅对奔跑者绘制红框 + "running X.XX" 标签
- 所有人绘制运动轨迹线（奔跑者红色粗线，普通人灰色细线）

Usage:
    python scripts/running_detection.py --source <video_path> [--output <output_path>]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        resolve_tracker,
        show_frame,
        extract_event_keyframes,
        cleanup_resources,
    )
except ImportError:
    # Fallback when _common is not on sys.path (e.g. running script directly)
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        resolve_tracker,
        show_frame,
        extract_event_keyframes,
        cleanup_resources,
    )

# ============================================================
# 配置参数
# ============================================================
DEFAULT_SOURCE = None  # 用户必须通过 --source 指定输入视频
DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "running_demo.mp4")
DEFAULT_LOG = str(PROJECT_ROOT / "outputs" / "running_events.json")
DEFAULT_MODEL = "weights/yolov8s.pt"

# ---- 检测阈值（Fix 1：区域化双阈值）----
DET_CONF_MAIN = 0.55                       # 中央区域严格阈值（防柱子误检）
DET_CONF_EDGE = 0.35                       # 边缘区域宽松阈值（救截断真人）
EDGE_MARGIN = 80                           # 边缘判定像素边界
IOU_THRESHOLD = 0.5
PERSON_CLASS = 0                           # COCO person

# ---- ByteTrack（Fix 3：调优参数由 configs/bytetrack.yaml 提供）----
TRACKER_CONFIG = str(PROJECT_ROOT / "configs" / "bytetrack.yaml")

# ---- 轨迹 & 置信度参数 ----
TRAJ_MAXLEN = 60                           # 最多保留的轨迹点（增大以支持证据继承）
RUN_CONF_WINDOW = 10                       # 时序窗口（帧）太短不稳，太长延迟

# ---- 置信度归一化参数 ----
SPEED_THRESH_NORM = 80.0                   # 像素/秒，达此速度 = speed_score 1.0
DISPL_THRESH_NORM = 150.0                  # 像素，窗口累计位移达此值 = displ_score 1.0
RUN_CONF_TRIGGER = 0.50                    # 显示阈值（<0.5 不显示 running，0.4 多检 / 0.6 少检）

# ---- 告警日志冷却 ----
ALERT_COOLDOWN_FRAMES = 30                 # 同一 ID 两次日志记录的最小间隔

# ---- 新生 ID 缓冲期（Fix 2）----
ID_DISPLAY_MIN_FRAMES = 5                  # 至少存活 N 帧才渲染

# ---- ID 切换继承（Fix 4）----
SWITCH_DIST_THRESH = 80                    # 像素距离阈值
SWITCH_TIME_THRESH = 30                    # 帧数阈值

# ---- 轨迹清理（Fix 5）----
TRAJ_TIMEOUT_FRAMES = 15                   # ID 消失后轨迹保留帧数

# ---- 轨迹绘制 ----
TRAJ_LINE_NORMAL_COLOR = (180, 180, 180)   # 普通人：灰色细线
TRAJ_LINE_RUNNING_COLOR = (0, 0, 255)      # 奔跑者：红色粗线
TRAJ_LINE_NORMAL_THICK = 1
TRAJ_LINE_RUNNING_THICK = 3

# ---- 绘制 ----
BOX_COLOR = (0, 0, 255)                    # BGR 红色
BOX_THICKNESS = 3
LABEL_PREFIX = "running"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)               # 白色文字

# ---- 关键帧自动抽取 ----
KEYFRAME_EXTRACT_ENABLED = True
KEYFRAME_RANGE = 10
KEYFRAME_STEP = 2
KEYFRAME_OUTPUT_DIR = "outputs/highlights"

# ============================================================
# 全局状态
# ============================================================
traj_history = defaultdict(lambda: deque(maxlen=TRAJ_MAXLEN))  # track_id → [(cx,cy,frame_idx), ...]
events_log = []                             # 告警事件列表
run_cooldown = {}                           # {tid: 上次告警日志帧号}（防重复记录）

# Fix 2: 新生 ID 缓冲
id_birth_frame = {}                         # {track_id: 首次出现的 frame_idx}
id_alive_frames = defaultdict(int)          # {track_id: 存活帧数}

# Fix 4: ID 切换证据继承
last_centers = {}                           # {tid: (cx, cy, frame_idx)}

# 当前帧置信度缓存
current_confidences = {}                    # {tid: running_confidence}


# ============================================================
# Fix 1: 区域化检测过滤
# ============================================================
def filter_detections(xyxy, confs, frame_w, frame_h):
    """两段阈值过滤：中央区域严格（DET_CONF_MAIN），边缘区域宽松（DET_CONF_EDGE）。
    返回保留的索引列表。"""
    keep = []
    for i in range(len(confs)):
        x1, y1, x2, y2 = xyxy[i]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        is_edge = (cx < EDGE_MARGIN or cx > frame_w - EDGE_MARGIN or
                   cy < EDGE_MARGIN or cy > frame_h - EDGE_MARGIN)
        threshold = DET_CONF_EDGE if is_edge else DET_CONF_MAIN
        if confs[i] >= threshold:
            keep.append(i)
    return keep


# ============================================================
# Fix 2: 新生 ID 渲染判定
# ============================================================
def should_render(tid):
    """ID 存活帧数 >= ID_DISPLAY_MIN_FRAMES 才允许渲染。"""
    return id_alive_frames.get(tid, 0) >= ID_DISPLAY_MIN_FRAMES


# ============================================================
# Fix 4: ID 切换证据继承
# ============================================================
def inherit_evidence_on_id_switch(new_tid, new_center, frame_idx, current_ids):
    """新 ID 首次出现时，查找附近刚消失的旧 ID 并继承其轨迹历史。
    置信度通过轨迹自然继承，无需单独处理。"""
    for old_tid, (ox, oy, of) in list(last_centers.items()):
        if old_tid in current_ids:
            continue
        if frame_idx - of > SWITCH_TIME_THRESH:
            continue
        dist = np.sqrt((new_center[0] - ox) ** 2 + (new_center[1] - oy) ** 2)
        if dist < SWITCH_DIST_THRESH:
            # 继承轨迹历史（置信度通过 traj_history 自然继承）
            if old_tid in traj_history:
                traj_history[new_tid] = deque(traj_history[old_tid], maxlen=TRAJ_MAXLEN)
            # 继承存活帧数（确保新 ID 立即能渲染）
            if old_tid in id_alive_frames:
                id_alive_frames[new_tid] = id_alive_frames[old_tid]
            # 继承冷却状态
            if old_tid in run_cooldown:
                run_cooldown[new_tid] = run_cooldown[old_tid]
            # 清理旧 ID
            del last_centers[old_tid]
            traj_history.pop(old_tid, None)
            id_alive_frames.pop(old_tid, None)
            run_cooldown.pop(old_tid, None)
            id_birth_frame.pop(old_tid, None)
            current_confidences.pop(old_tid, None)
            print(f"  [ID switch] {old_tid} -> {new_tid}  距离:{dist:.0f}px  证据继承")
            return True
    return False


# ============================================================
# Fix 5: 轨迹清理
# ============================================================
def cleanup_dead_tracks(current_ids, frame_idx):
    """清除超过 TRAJ_TIMEOUT_FRAMES 帧未出现的 ID 的所有状态。"""
    dead_ids = []
    for tid in list(traj_history.keys()):
        if tid not in current_ids:
            last_seen = last_centers.get(tid, (0, 0, frame_idx))[2]
            if frame_idx - last_seen > TRAJ_TIMEOUT_FRAMES:
                dead_ids.append(tid)

    for tid in dead_ids:
        traj_history.pop(tid, None)
        last_centers.pop(tid, None)
        id_alive_frames.pop(tid, None)
        id_birth_frame.pop(tid, None)
        run_cooldown.pop(tid, None)
        current_confidences.pop(tid, None)

    return len(dead_ids)


# ============================================================
# 连续置信度计算（替代旧版二值速度判别）
# ============================================================
def compute_running_confidence(track_id, fps=30):
    """
    输入：track_id，从 traj_history 读取 (cx, cy, frame_idx) 轨迹
    输出：0.00 ~ 1.00 的连续置信度
    证据 1（0.5 权重）：平均速度
    证据 2（0.3 权重）：累计位移
    证据 3（0.2 权重）：方向稳定性
    """
    traj = list(traj_history[track_id])
    if len(traj) < RUN_CONF_WINDOW:
        return 0.0

    recent = traj[-RUN_CONF_WINDOW:]

    # 证据 1：平均速度
    speeds = []
    for i in range(1, len(recent)):
        x1, y1, f1 = recent[i - 1][0], recent[i - 1][1], recent[i - 1][2]
        x2, y2, f2 = recent[i][0], recent[i][1], recent[i][2]
        dt = max((f2 - f1) / fps, 1e-6)
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speeds.append(dist / dt)
    avg_speed = np.mean(speeds) if speeds else 0.0
    speed_score = min(1.0, avg_speed / SPEED_THRESH_NORM)

    # 证据 2：累计位移
    total_displ = sum(s * (1.0 / fps) for s in speeds)
    displ_score = min(1.0, total_displ / DISPL_THRESH_NORM)

    # 证据 3：方向稳定性
    angles = []
    for i in range(1, len(recent)):
        dx = recent[i][0] - recent[i - 1][0]
        dy = recent[i][1] - recent[i - 1][1]
        if dx * dx + dy * dy > 4:  # 位移 > 2px 才计入
            angles.append(np.arctan2(dy, dx))
    if len(angles) >= 3:
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        direc_score = (sin_mean ** 2 + cos_mean ** 2) ** 0.5
    else:
        direc_score = 0.0

    # 加权融合
    conf = 0.5 * speed_score + 0.3 * displ_score + 0.2 * direc_score
    return round(min(1.0, conf), 2)


# ============================================================
# 轨迹绘制（Fix 5：双重过滤）
# ============================================================
def draw_trajectory_lines(frame, current_ids):
    """绘制所有人的运动轨迹线。奔跑者红色粗线，普通人灰色细线。
    仅绘制当前活跃且通过 should_render 的 ID。"""
    for tid, traj in traj_history.items():
        if tid not in current_ids:
            continue
        if not should_render(tid):
            continue
        if len(traj) < 2:
            continue
        points = [(int(t[0]), int(t[1])) for t in traj]
        is_running = current_confidences.get(tid, 0.0) >= RUN_CONF_TRIGGER
        color = TRAJ_LINE_RUNNING_COLOR if is_running else TRAJ_LINE_NORMAL_COLOR
        thick = TRAJ_LINE_RUNNING_THICK if is_running else TRAJ_LINE_NORMAL_THICK
        cv2.circle(frame, points[-1], radius=4, color=color, thickness=-1)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, thickness=thick, lineType=cv2.LINE_AA)


# ============================================================
# 绘制奔跑框（升级版：颜色渐变 + ID:X running 0.XX 标签）
# ============================================================
def draw_running_box(frame, tid, bbox, confidence):
    """奔跑者：颜色渐变（橙 0.5-0.7 → 红 0.7+），标签 'ID:X running 0.XX'"""
    if not should_render(tid):
        return
    x1, y1, x2, y2 = map(int, bbox)

    # 颜色渐变（BGR）：橙(0,128,255) → 红(0,0,255)
    if confidence < RUN_CONF_TRIGGER:
        return
    elif confidence < 0.7:
        color = (0, 128, 255)   # 橙色
    else:
        color = (0, 0, 255)     # 红色

    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"ID:{tid} running {confidence:.2f}"
    font_scale = 0.55
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, font_thick)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - baseline - 4),
                FONT, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


# ============================================================
# 普通行人框（青色，保留 ID 跟踪可视化）
# ============================================================
def draw_normal_box(frame, tid, bbox):
    """普通行人：青色框 + 'ID:X' 标签"""
    if not should_render(tid):
        return
    x1, y1, x2, y2 = map(int, bbox)
    color = (255, 255, 0)  # 青色 BGR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    label = f"ID:{tid}"
    font_scale = 0.45
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, font_thick)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                FONT, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)


# 关键帧抽取通过 extract_event_keyframes 实现（已从 _common 导入）


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="奔跑检测脚本 — YOLO + ByteTrack + 速度计算")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="输入视频路径 (必须指定)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"输出视频路径 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG),
                        help=f"JSON 事件日志路径 (默认: {DEFAULT_LOG})")
    parser.add_argument("--conf-trigger", type=float, default=RUN_CONF_TRIGGER,
                        help=f"Running 显示/告警阈值 (默认: {RUN_CONF_TRIGGER}, 低→多检 高→少检)")
    parser.add_argument("--speed-norm", type=float, default=SPEED_THRESH_NORM,
                        help=f"速度归一化参考值 px/s (默认: {SPEED_THRESH_NORM})")
    parser.add_argument("--displ-norm", type=float, default=DISPL_THRESH_NORM,
                        help=f"位移归一化参考值 px (默认: {DISPL_THRESH_NORM})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"YOLO 模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--tracker", type=str, default=TRACKER_CONFIG,
                        help=f"ByteTrack 配置文件路径 (默认: {TRACKER_CONFIG})")
    parser.add_argument("--extract-keyframes", action="store_true", default=KEYFRAME_EXTRACT_ENABLED,
                        help="检测到奔跑后自动抽取关键帧 (默认: 开启)")
    parser.add_argument("--no-extract-keyframes", action="store_true",
                        help="禁用关键帧自动抽取")
    parser.add_argument("--keyframe-dir", type=str, default=KEYFRAME_OUTPUT_DIR,
                        help=f"关键帧输出目录 (默认: {KEYFRAME_OUTPUT_DIR})")
    args = parser.parse_args()

    if args.source is None:
        print("[X] 错误: 必须通过 --source 指定输入视频路径")
        print("    用法: python scripts/running_detection.py --source <video_path>")
        sys.exit(1)

    VIDEO_INPUT = Path(args.source)
    VIDEO_OUTPUT = Path(args.output)
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH = Path(args.log)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not VIDEO_INPUT.exists():
        print(f"[X] 未找到输入视频: {VIDEO_INPUT}")
        sys.exit(1)

    # Fix 3: 检查 tracker 配置文件是否存在，fallback 到内置
    tracker_yaml = resolve_tracker(args.tracker)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda")
    print(f"[i] 推理设备: {device.upper()}" + (" (FP16)" if half else ""))
    print(f"[i] 检测阈值: 中央{DET_CONF_MAIN} / 边缘{DET_CONF_EDGE} (边界{EDGE_MARGIN}px)")
    print(f"[i] 新生ID显示门槛: {ID_DISPLAY_MIN_FRAMES} 帧  |  ID切换继承: {SWITCH_DIST_THRESH}px/{SWITCH_TIME_THRESH}帧")
    print(f"[i] Running 置信度: 触发>{args.conf_trigger}  |  速度归一化{args.speed_norm}px/s  |  位移归一化{args.displ_norm}px")
    print(f"[i] 轨迹超时清理: {TRAJ_TIMEOUT_FRAMES} 帧  |  告警冷却: {ALERT_COOLDOWN_FRAMES} 帧")

    print(f"[i] 加载检测模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[X] 模型加载失败: {e}")
        sys.exit(1)

    cap = open_video(str(VIDEO_INPUT))
    props = get_video_props(cap)
    fps, frame_w, frame_h, total_frames = (props["fps"], props["width"],
                                            props["height"], props["total_frames"])
    print(f"[i] 视频信息: {frame_w}x{frame_h}, {fps:.1f} FPS, {total_frames} 帧")

    out = create_video_writer(str(VIDEO_OUTPUT), fps, (frame_w, frame_h))

    frame_idx = 0
    alert_frames = []
    conf_trigger = args.conf_trigger

    print("\n[>] 开始逐帧推理 (检测+跟踪+置信度计算)...")
    print("    (按 Q 可提前终止)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix 1: model.track() 用宽松阈值，后续 filter_detections 做区域化过滤
        results = model.track(
            frame, conf=DET_CONF_EDGE, iou=IOU_THRESHOLD,
            device=device, half=half, tracker=tracker_yaml,
            persist=True, verbose=False, classes=[PERSON_CLASS],
        )

        active_ids = set()

        for result in results:
            if result.boxes is None:
                continue

            boxes_data = result.boxes.data
            if boxes_data is None or len(boxes_data) == 0:
                continue

            # 提取 track IDs
            if result.boxes.id is not None:
                try:
                    ids_arr = result.boxes.id.cpu().numpy().astype(int)
                except Exception:
                    ids_arr = np.asarray(result.boxes.id).astype(int)
            else:
                ids_arr = list(range(len(boxes_data)))

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            # Fix 1: 区域化检测过滤
            keep_indices = filter_detections(xyxy, confs, frame_w, frame_h)

            for i in keep_indices:
                tid = int(ids_arr[i])
                active_ids.add(tid)

                x1, y1, x2, y2 = xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox = (x1, y1, x2, y2)

                # Fix 2: 更新 ID 存活状态
                if tid not in id_birth_frame:
                    id_birth_frame[tid] = frame_idx
                    # Fix 4: 新 ID 首次出现时尝试继承证据
                    inherit_evidence_on_id_switch(tid, (cx, cy), frame_idx, active_ids)
                id_alive_frames[tid] += 1

                # Fix 4: 更新最近位置
                last_centers[tid] = (cx, cy, frame_idx)

                # 记录轨迹 (cx, cy, frame_idx)
                traj_history[tid].append((cx, cy, frame_idx))

                # ---- 计算 running 置信度 ----
                if should_render(tid):
                    run_conf = compute_running_confidence(tid, fps=fps)
                    current_confidences[tid] = run_conf

                    if run_conf >= conf_trigger:
                        # 奔跑者：红/橙色框 + 标签
                        draw_running_box(frame, tid, bbox, run_conf)

                        # 告警日志（带冷却防重复）
                        last_log = run_cooldown.get(tid, -ALERT_COOLDOWN_FRAMES)
                        if frame_idx - last_log >= ALERT_COOLDOWN_FRAMES:
                            alert_frames.append(frame_idx)
                            events_log.append({
                                "frame_id": frame_idx,
                                "track_id": tid,
                                "confidence": run_conf,
                                "bbox": [round(float(x1), 1), round(float(y1), 1),
                                         round(float(x2), 1), round(float(y2), 1)],
                            })
                            run_cooldown[tid] = frame_idx
                            print(f"  [ALERT] 帧 #{frame_idx}  ID:{tid}  running {run_conf:.2f}")
                    else:
                        # 普通行人：青色框
                        draw_normal_box(frame, tid, bbox)

        # Fix 5: 轨迹清理
        cleanup_dead_tracks(active_ids, frame_idx)

        # Fix 5: 绘制所有人的轨迹线
        draw_trajectory_lines(frame, active_ids)

        out.write(frame)
        frame_idx += 1

        # ---- 进度 & 统计 ----
        if frame_idx % max(1, int(fps)) == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            # 统计置信度分布
            confs_list = [v for k, v in current_confidences.items()
                          if k in active_ids and should_render(k)]
            print(f"    进度: {frame_idx}/{total_frames} ({pct:.1f}%)  "
                  f"| 告警: {len(events_log)}  "
                  f"| conf>={conf_trigger}:{len([c for c in confs_list if c >= conf_trigger])}/{len(confs_list)}", end="\r")

        if frame_idx % (max(1, int(fps)) * 5) == 0:
            # 每 5 秒输出详细统计
            confs_list = [v for k, v in current_confidences.items()
                          if k in active_ids and should_render(k)]
            high = sum(1 for c in confs_list if c >= 0.7)
            mid = sum(1 for c in confs_list if 0.5 <= c < 0.7)
            low = sum(1 for c in confs_list if c < 0.5)
            print(f"\n  [STATS] frame={frame_idx} active={len(active_ids)}, "
                  f"conf 分布: >=0.7:{high}  0.5-0.7:{mid}  <0.5:{low}")

        if not show_frame(frame, "Running Detection (press Q to quit)", "q"):
            print("\n[!] 用户提前终止。")
            break

    cleanup_resources(cap, out)

    # ---- 保存 JSON 事件日志 ----
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return super().default(obj)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(events_log, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # ---- 汇总输出 ----
    print("\n\n[OK] 处理完成！")
    print(f"    总帧数: {frame_idx}")
    print(f"    总告警数: {len(events_log)}")
    print(f"    输出视频: {VIDEO_OUTPUT}")
    print(f"    事件日志: {LOG_PATH}")
    if events_log:
        print(f"    告警帧号: {[e['frame_id'] for e in events_log]}")
        # 置信度汇总
        all_confs = [e['confidence'] for e in events_log]
        print(f"    告警置信度范围: {min(all_confs):.2f} ~ {max(all_confs):.2f}  均值: {sum(all_confs)/len(all_confs):.2f}")

        # ---- 自动抽取关键帧 ----
        do_extract = args.extract_keyframes and not args.no_extract_keyframes
        if do_extract:
            extract_event_keyframes(
                video_path=VIDEO_OUTPUT,
                event_frames=alert_frames,
                output_dir=args.keyframe_dir,
                range_frames=KEYFRAME_RANGE,
                step=KEYFRAME_STEP,
                label="running",
            )
    else:
        print("\n[!] 未检测到奔跑事件。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
