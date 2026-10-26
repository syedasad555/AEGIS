#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
基于 YOLOv8 + ByteTrack 的禁入区域入侵检测系统
===============================================
功能:
  - YOLOv8 检测行人 (class=0)
  - ByteTrack 多目标跟踪，分配唯一 ID
  - 自定义多边形禁入区域
  - 行人运动轨迹绘制 (红色渐变线条)
  - 入侵判定: 脚底中心点或轨迹进入禁入区域 → 红框 + INTRUSION 标签 (轨迹线段与区域边界相交也触发)
  - 双层颜色编码: 正常行人绿色框 / 入侵者红色加粗框
  - 禁入区域动态着色: 无人时紫色+黄填充 / 入侵时红色+橙红填充
  - 顶部告警横幅 (红色背景 + 闪烁警示灯)
  - 时序平滑: 3帧确认 + 15帧保持
  - JSON 事件日志记录
  - 支持视频文件 / 摄像头实时输入
  - 按 ESC 键退出

依赖: ultralytics, opencv-python, shapely, numpy

Usage:
    # 步骤 1：标定禁入区域
    python tools/zone_selector.py --source <video_path> --output configs/forbidden_zone.json
    # 步骤 2：运行检测
    python scripts/intrusion_detection.py --source <video_path> --zone configs/forbidden_zone.json

    # 或使用脚本内硬编码的示例区域（不做区域标定）
    python scripts/intrusion_detection.py --source <video_path>
"""

import argparse
import cv2
import json
import numpy as np
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO

try:
    from _common import PROJECT_ROOT
except ImportError:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from _common import PROJECT_ROOT

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("⚠️  警告: shapely 未安装，将使用 cv2.pointPolygonTest 替代。")
    print("    建议安装: pip install shapely")

# ============================================================
# 配置参数
# ============================================================
MODEL_PATH = "yolov8n.pt"          # YOLOv8 模型权重路径 (n/s/m/l/x)
DETECTION_CONF = 0.35              # 检测置信度阈值 (只保留高于此值的检测)
PERSON_CLASS_ID = 0                # 行人类别 ID (COCO 数据集中 person=0)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                    # 推理设备: "cuda" / "cpu"

# --- 跟踪参数 ---
TRACKER_CONFIG = "bytetrack.yaml"  # ByteTrack 配置文件 (使用 ultralytics 内置)
TRACK_PERSIST = True               # 是否在帧间持久化跟踪 (视频必须为 True)

# --- 轨迹参数 ---
TRAJECTORY_MAX_LENGTH = 50         # 轨迹历史最大长度 (帧数)
TRAJECTORY_LINE_WIDTH = 2          # 轨迹线宽
TRAJECTORY_COLOR = (0, 0, 255)     # 轨迹颜色 BGR (红色)

# --- 入侵判定鲁棒性参数 ---
INTRUSION_CONFIRM_FRAMES = 3   # 连续 N 帧在区域内才算入侵
INTRUSION_HANGOVER_FRAMES = 15 # 触发后保持 N 帧（避免闪烁）

# 全局入侵状态 (key=tid)
_intrusion_history: Dict[int, int] = {}
_intrusion_hangover: Dict[int, int] = {}
_intrusion_was_triggered: set = set()  # 已记录过JSON日志的ID


def update_intrusion_state(tid: int, in_zone: bool) -> bool:
    """
    更新入侵状态，返回是否标记为入侵者。

    采用"确认+保持"双窗口策略：
    - 须连续 INTRUSION_CONFIRM_FRAMES 帧在区域内才触发
    - 触发后保持 INTRUSION_HANGOVER_FRAMES 帧再解除
    """
    history = _intrusion_history
    hangover = _intrusion_hangover

    if in_zone:
        history[tid] = history.get(tid, 0) + 1
        if history[tid] >= INTRUSION_CONFIRM_FRAMES:
            hangover[tid] = INTRUSION_HANGOVER_FRAMES
            return True
    else:
        history[tid] = 0

    # hangover 期内仍然标记为入侵者
    if hangover.get(tid, 0) > 0:
        hangover[tid] -= 1
        return True

    return False

# --- 颜色体系：双层颜色编码 ---
COLOR_NORMAL_BOX = (0, 255, 0)        # 绿色（正常行人）
COLOR_INTRUSION_BOX = (0, 0, 255)     # 红色（入侵者）
COLOR_ZONE_NORMAL = (255, 0, 255)     # 紫色（区域无人）
COLOR_ZONE_TRIGGERED = (0, 0, 255)    # 红色（区域被入侵）
COLOR_ZONE_FILL_NORMAL = (0, 255, 255)    # 黄色填充
COLOR_ZONE_FILL_TRIGGERED = (0, 100, 255) # 橙红填充

# --- 显示参数 ---
BOX_THICKNESS_NORMAL = 2           # 正常检测框线宽
BOX_THICKNESS_INTRUSION = 3        # 入侵检测框线宽
ZONE_THICKNESS_NORMAL = 2          # 正常区域边框线宽
ZONE_THICKNESS_TRIGGERED = 4       # 触发时区域边框线宽
ZONE_FILL_ALPHA = 0.3              # 禁入区域填充透明度 (0~1)
FONT_SCALE = 0.6                   # 字体大小
FONT_THICKNESS = 2                 # 字体粗细
BANNER_HEIGHT = 60                 # 告警横幅高度

# --- 输入/输出参数 ---
# 默认值：用户必须通过 --source 指定；若不指定则使用 0（摄像头）
DEFAULT_VIDEO_SOURCE = None
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "intrusion_demo.mp4")
DEFAULT_ZONE_CONFIG = str(PROJECT_ROOT / "configs" / "forbidden_zone.example.json")
OUTPUT_FPS = 30                    # 输出视频帧率
OUTPUT_SIZE = None                 # 输出视频尺寸 (w, h)，None 则使用原始尺寸

# --- 运行参数 ---
DISPLAY_WINDOW_NAME = "Intrusion Detection - YOLOv8 + ByteTrack"
DISPLAY_SCALE = 1.0                # 显示缩放比例 (用于大分辨率视频)
PAUSE_ON_ALARM = False             # 是否在首次入侵时暂停
SHOW_FPS = True                    # 是否在画面显示 FPS


# ============================================================================
# [禁入区域定义] — 示例多边形顶点（仅在未指定 --zone 时使用）
# ============================================================================
# 注意: 坐标为 (x, y) 像素坐标，原点在左上角
# 请根据你的视频实际分辨率调整以下坐标，或使用 tools/zone_selector.py 标定
# 示例: 画面中央偏下的一块矩形区域 (适用于 1280x720)
FORBIDDEN_ZONE_POINTS = [
    (400, 300),
    (880, 300),
    (880, 600),
    (400, 600),
]

# 更多示例区域 (取消注释使用):
# 三角形区域:
# FORBIDDEN_ZONE_POINTS = [(640, 100), (200, 600), (1080, 600)]

# 不规则四边形:
# FORBIDDEN_ZONE_POINTS = [(300, 200), (900, 250), (1000, 500), (200, 550)]

# 画面右半部分:
# FORBIDDEN_ZONE_POINTS = [(960, 0), (1920, 0), (1920, 1080), (960, 1080)]


# ============================================================================
# 入侵检测系统类
# ============================================================================

class IntrusionDetectionSystem:
    """
    禁入区域入侵检测系统

    整合 YOLOv8 检测、ByteTrack 跟踪、轨迹记录、禁区判断与可视化。
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化检测系统

        Args:
            config: 可选的配置字典，覆盖默认参数
        """
        # ----- 加载配置 -----
        self._init_config(config)

        # ----- 加载 YOLOv8 模型 -----
        print(f"[初始化] 加载 YOLOv8 模型: {self.cfg['model_path']}")
        self.model = YOLO(self.cfg["model_path"])
        self.model.to(self.cfg["device"])
        print(f"[初始化] 模型加载完成，设备: {self.cfg['device']}")

        # ----- 构建禁入区域 Polygon -----
        raw_zones = self.cfg["forbidden_zone_points"]
        self.zones: List[dict] = self._init_zone_polygons(raw_zones)

        print(f"[初始化] 禁入区域数: {len(self.zones)}")
        for z in self.zones:
            print(f"  - {z['name']}: {len(z['points'])} vertices")

        # ----- 轨迹历史存储: {track_id: deque of (cx, cy)} -----
        self.trajectories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # ----- 帧计数器 -----
        self.frame_count = 0

        # ----- 视频输出 -----
        self.video_writer = None

        print(f"[初始化] 轨迹最大长度: {self.cfg['trajectory_max_length']} 帧")
        print(f"[初始化] 检测置信度阈值: {self.cfg['detection_conf']}")
        print("[初始化] 系统就绪 ✅")

    # ------------------------------------------------------------------
    # 配置管理
    # ------------------------------------------------------------------

    def _init_config(self, user_config: Optional[dict]) -> None:
        """合并用户配置与默认配置"""
        defaults = {
            "model_path": MODEL_PATH,
            "detection_conf": DETECTION_CONF,
            "person_class_id": PERSON_CLASS_ID,
            "device": DEVICE,
            "tracker_config": TRACKER_CONFIG,
            "track_persist": TRACK_PERSIST,
            "trajectory_max_length": TRAJECTORY_MAX_LENGTH,
            "trajectory_line_width": TRAJECTORY_LINE_WIDTH,
            "trajectory_color": TRAJECTORY_COLOR,
            "normal_box_color": COLOR_NORMAL_BOX,
            "intrusion_box_color": COLOR_INTRUSION_BOX,
            "zone_normal_color": COLOR_ZONE_NORMAL,
            "zone_triggered_color": COLOR_ZONE_TRIGGERED,
            "zone_fill_normal": COLOR_ZONE_FILL_NORMAL,
            "zone_fill_triggered": COLOR_ZONE_FILL_TRIGGERED,
            "box_thickness_normal": BOX_THICKNESS_NORMAL,
            "box_thickness_intrusion": BOX_THICKNESS_INTRUSION,
            "zone_thickness_normal": ZONE_THICKNESS_NORMAL,
            "zone_thickness_triggered": ZONE_THICKNESS_TRIGGERED,
            "font_scale": FONT_SCALE,
            "font_thickness": FONT_THICKNESS,
            "banner_height": BANNER_HEIGHT,
            "zone_fill_alpha": ZONE_FILL_ALPHA,
            "video_source": DEFAULT_VIDEO_SOURCE,
            "output_video_path": DEFAULT_OUTPUT_PATH,
            "output_fps": OUTPUT_FPS,
            "output_size": OUTPUT_SIZE,
            "display_window_name": DISPLAY_WINDOW_NAME,
            "display_scale": DISPLAY_SCALE,
            "pause_on_alarm": PAUSE_ON_ALARM,
            "show_fps": SHOW_FPS,
            "forbidden_zone_points": FORBIDDEN_ZONE_POINTS,
        }
        if user_config:
            defaults.update(user_config)
        self.cfg = defaults

    # ------------------------------------------------------------------
    # 区域管理
    # ------------------------------------------------------------------

    def _init_zone_polygons(self, raw_zones) -> List[dict]:
        """
        初始化所有禁入区域的多边形数据。

        支持输入:
          - [{"name": "z1", "points": [(x,y),...]}, ...]  (load_zone_config 格式)
          - [(x,y), ...]  (旧版硬编码坐标)

        返回:
          [{"name": str, "points": list, "np": ndarray, "shapely": Polygon|None}, ...]
        """
        zones = []

        # Normalize to list of dicts
        if raw_zones and isinstance(raw_zones[0], dict):
            zone_dicts = raw_zones
        else:
            # Legacy: single polygon as list of tuples
            zone_dicts = [{"name": "zone_1", "points": raw_zones}]

        for zd in zone_dicts:
            pts = zd["points"]
            zone = {
                "name": zd.get("name", "zone"),
                "points": pts,
                "np": np.array(pts, dtype=np.int32),
            }
            if SHAPELY_AVAILABLE and len(pts) >= 3:
                zone["shapely"] = Polygon(pts)
            else:
                zone["shapely"] = None
            zones.append(zone)

        return zones

    def _point_in_zone(self, point: Tuple[float, float], zone: dict) -> bool:
        """判断点是否在某个禁入区域内"""
        if zone["shapely"] is not None:
            return zone["shapely"].contains(Point(point))
        return cv2.pointPolygonTest(zone["np"], point, False) >= 0

    # ------------------------------------------------------------------
    # 区域判断
    # ------------------------------------------------------------------

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2):
        """Check if line segments p1p2 and q1q2 intersect (handles collinear overlap)."""
        def _orient(a, b, c):
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        def _on_segment(a, b, c):
            return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

        o1 = _orient(p1, p2, q1)
        o2 = _orient(p1, p2, q2)
        o3 = _orient(q1, q2, p1)
        o4 = _orient(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and _on_segment(p1, p2, q1):
            return True
        if o2 == 0 and _on_segment(p1, p2, q2):
            return True
        if o3 == 0 and _on_segment(q1, q2, p1):
            return True
        if o4 == 0 and _on_segment(q1, q2, p2):
            return True
        return False

    def _trajectory_crosses_zone(self, track_id: int, zone: dict, current_point: Tuple[float, float]) -> bool:
        """Check if the segment from last stored position to current_point crosses into the zone."""
        traj = self.get_trajectory(track_id)
        if len(traj) == 0:
            return False

        p1 = traj[-1]  # last known position
        p2 = current_point

        if zone["shapely"] is not None:
            from shapely.geometry import LineString
            return LineString([p1, p2]).intersects(zone["shapely"])

        pts = zone["points"]
        n = len(pts)
        for i in range(n):
            if self._segments_intersect(p1, p2, pts[i], pts[(i + 1) % n]):
                return True
        return False

    def is_inside_any_zone(self, point: Tuple[float, float]) -> Tuple[bool, Optional[str]]:
        """
        判断给定点是否在任意禁入区域内。

        Args:
            point: (x, y) 像素坐标

        Returns:
            (is_inside, zone_name) — zone_name 为命中的第一个区域名，未命中则为 None
        """
        for zone in self.zones:
            if self._point_in_zone(point, zone):
                return True, zone["name"]
        return False, None

    # ------------------------------------------------------------------
    # 轨迹管理
    # ------------------------------------------------------------------

    def update_trajectory(self, track_id: int, center: Tuple[int, int]) -> None:
        """
        更新指定 ID 的轨迹历史

        Args:
            track_id: 跟踪 ID
            center: 当前帧的中心点 (cx, cy)
        """
        max_len = self.cfg["trajectory_max_length"]
        traj = self.trajectories[track_id]
        traj.append(center)
        # 保持轨迹长度不超过上限
        if len(traj) > max_len:
            self.trajectories[track_id] = traj[-max_len:]

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """获取指定 ID 的轨迹点列表"""
        return self.trajectories.get(track_id, [])

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def draw_forbidden_zone(self, frame: np.ndarray, triggered_zones: set) -> np.ndarray:
        """
        绘制所有禁入区域，每个区域根据是否被触发动态着色。

        Args:
            frame: 输入帧 (会被原地修改)
            triggered_zones: 当前被触发的区域名称集合

        Returns:
            绘制后的帧
        """
        for zone in self.zones:
            is_triggered = zone["name"] in triggered_zones

            if is_triggered:
                fill_color = self.cfg["zone_fill_triggered"]
                edge_color = self.cfg["zone_triggered_color"]
                edge_thickness = self.cfg["zone_thickness_triggered"]
            else:
                fill_color = self.cfg["zone_fill_normal"]
                edge_color = self.cfg["zone_normal_color"]
                edge_thickness = self.cfg["zone_thickness_normal"]

            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone["np"]], fill_color)
            alpha = self.cfg["zone_fill_alpha"]
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.polylines(
                frame, [zone["np"]], isClosed=True,
                color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA,
            )

        return frame

    def draw_trajectory(
        self, frame: np.ndarray, track_id: int
    ) -> np.ndarray:
        """
        绘制指定 ID 的运动轨迹 (红色折线)

        Args:
            frame: 当前帧
            track_id: 跟踪 ID

        Returns:
            绘制后的帧
        """
        points = self.get_trajectory(track_id)
        if len(points) < 2:
            return frame

        pts_array = np.array(points, dtype=np.int32)
        color = self.cfg["trajectory_color"]
        thickness = self.cfg["trajectory_line_width"]

        # 逐段绘制，实现渐变色效果（轨迹越旧越暗）
        n = len(pts_array)
        for i in range(1, n):
            # 越靠近当前帧越亮
            fade_factor = i / max(n - 1, 1)
            faded_color = tuple(
                int(c * (0.3 + 0.7 * fade_factor)) for c in color
            )
            cv2.line(
                frame,
                tuple(pts_array[i - 1]),
                tuple(pts_array[i]),
                faded_color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        return frame

    def draw_detection_box(
        self,
        frame: np.ndarray,
        box_xyxy: Tuple[int, int, int, int],
        track_id: int,
        is_intruder: bool,
    ) -> np.ndarray:
        """
        绘制检测框和 ID 标签。入侵者使用红色加粗框 + INTRUSION 标签。

        Args:
            frame: 当前帧
            box_xyxy: (x1, y1, x2, y2) 边界框
            track_id: 跟踪 ID
            is_intruder: 是否入侵者

        Returns:
            绘制后的帧
        """
        x1, y1, x2, y2 = box_xyxy

        if is_intruder:
            color = self.cfg["intrusion_box_color"]
            thickness = self.cfg["box_thickness_intrusion"]
            label = f"ID:{track_id} INTRUSION"
        else:
            color = self.cfg["normal_box_color"]
            thickness = self.cfg["box_thickness_normal"]
            label = f"ID:{track_id}"

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # 标签文字 + 实心背景条
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, self.cfg["font_scale"], self.cfg["font_thickness"])
        label_y = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, label_y), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 5),
            font, self.cfg["font_scale"], (255, 255, 255),
            self.cfg["font_thickness"], lineType=cv2.LINE_AA,
        )

        return frame

    def draw_alert_banner(
        self, frame: np.ndarray, intruder_ids: set, frame_w: int, frame_idx: int
    ) -> np.ndarray:
        """
        画面顶部告警横幅，仅当 intruder_ids 非空时绘制。

        Args:
            frame: 当前帧
            intruder_ids: 当前入侵的 track ID 集合
            frame_w: 帧宽度
            frame_idx: 帧序号 (用于闪烁控制)

        Returns:
            绘制后的帧
        """
        if not intruder_ids:
            return frame

        banner_h = self.cfg["banner_height"]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        ids_str = ", ".join([f"ID:{tid}" for tid in sorted(intruder_ids)])
        text = f"[!] INTRUSION ALERT - {ids_str}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        text_x = (frame_w - tw) // 2
        text_y = (banner_h + th) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)

        # 左右闪烁红点 (每 10 帧切换)
        if (frame_idx // 10) % 2 == 0:
            cv2.circle(frame, (30, banner_h // 2), 15, (0, 0, 255), -1)
            cv2.circle(frame, (frame_w - 30, banner_h // 2), 15, (0, 0, 255), -1)

        return frame

    def draw_info_panel(
        self, frame: np.ndarray, intruder_ids: List[int], fps: float
    ) -> np.ndarray:
        """
        在画面左上角绘制信息面板 (入侵 ID 列表 + FPS)

        Args:
            frame: 当前帧
            intruder_ids: 当前入侵的 track ID 列表
            fps: 当前帧率

        Returns:
            绘制后的帧
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.cfg["font_scale"]
        thickness = self.cfg["font_thickness"]

        # 面板背景
        panel_x, panel_y = 10, 10
        line_height = 24
        num_lines = 2 + (len(intruder_ids) if intruder_ids else 1)
        panel_h = num_lines * line_height + 16
        panel_w = 320

        # 半透明背景
        sub_img = frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        if sub_img.size > 0:  # 防止越界
            white_bg = np.ones_like(sub_img, dtype=np.uint8) * 40
            cv2.addWeighted(sub_img, 0.3, white_bg, 0.7, 0, sub_img)
            frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = sub_img

        # 标题
        cv2.putText(
            frame,
            "--- Intrusion Status ---",
            (panel_x + 8, panel_y + 20),
            font,
            scale,
            (200, 200, 200),
            thickness,
            lineType=cv2.LINE_AA,
        )

        y_offset = panel_y + 44

        # 入侵 ID 列表
        if intruder_ids:
            ids_str = ", ".join(str(i) for i in sorted(intruder_ids))
            cv2.putText(
                frame,
                f"[!] Intruders: [{ids_str}]",
                (panel_x + 8, y_offset),
                font,
                scale,
                (0, 0, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "[+] Clear: No Intruders",
                (panel_x + 8, y_offset),
                font,
                scale,
                (0, 255, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )

        # FPS 显示
        if self.cfg["show_fps"]:
            y_offset += line_height
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (panel_x + 8, y_offset),
                font,
                scale,
                (200, 200, 200),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    # 核心处理逻辑
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, results
    ) -> Tuple[np.ndarray, List[int]]:
        """
        处理单帧: 提取跟踪结果、判断入侵、绘制可视化

        Args:
            frame: 原始帧 (numpy array, BGR)
            results: ultralytics Results 对象 (来自 model.track 的单帧结果)

        Returns:
            (annotated_frame, intruder_ids) 元组
        """
        current_intruders: set = set()       # intruding track IDs
        triggered_zones: set = set()          # triggered zone names
        frame_h, frame_w = frame.shape[:2]

        # 检查是否有检测结果
        boxes_obj = results.boxes
        if boxes_obj is None or not hasattr(boxes_obj, "id") or boxes_obj.id is None:
            self.draw_forbidden_zone(frame, triggered_zones)
            return frame, []

        # 提取跟踪数据
        track_ids = boxes_obj.id.cpu().numpy().astype(int)
        boxes_xyxy = boxes_obj.xyxy.cpu().numpy().astype(int)
        classes = boxes_obj.cls.cpu().numpy().astype(int)
        confs = boxes_obj.conf.cpu().numpy()

        person_class = self.cfg["person_class_id"]
        person_data = []

        # --- 第一阶段: 判定入侵状态 ---
        for i in range(len(track_ids)):
            cls = classes[i]
            conf = confs[i]
            tid = track_ids[i]
            x1, y1, x2, y2 = boxes_xyxy[i]

            if cls != person_class or conf < self.cfg["detection_conf"]:
                continue

            # 脚底中心点判定
            foot_x = (x1 + x2) / 2
            foot_y = float(y2)
            in_zone_raw, hit_zone = self.is_inside_any_zone((foot_x, foot_y))

            # 轨迹穿入判定: 上一帧位置到当前帧位置的线段与区域边界相交也视为入侵
            if not in_zone_raw:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                for zone in self.zones:
                    if self._trajectory_crosses_zone(tid, zone, (cx, cy)):
                        in_zone_raw = True
                        hit_zone = zone["name"]
                        break

            # 鲁棒性判定
            is_intruding = update_intrusion_state(tid, in_zone_raw)

            # JSON 日志: 首次触发时记录
            if is_intruding and tid not in _intrusion_was_triggered:
                _intrusion_was_triggered.add(tid)
                event = {
                    "frame": self.frame_count,
                    "event": "intrusion",
                    "tid": int(tid),
                    "zone": hit_zone or "unknown",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "trigger_frames": INTRUSION_CONFIRM_FRAMES,
                }
                print(f"[EVENT] {json.dumps(event, ensure_ascii=False)}")

            if is_intruding:
                current_intruders.add(tid)
                if hit_zone:
                    triggered_zones.add(hit_zone)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            person_data.append((tid, x1, y1, x2, y2, cx, cy, is_intruding))

        # --- 第二阶段: 绘制 ---
        # 1. 所有禁入区 (每个区域独立着色)
        self.draw_forbidden_zone(frame, triggered_zones)

        # 2. 行人轨迹 + 检测框
        for tid, x1, y1, x2, y2, cx, cy, is_intruding in person_data:
            self.update_trajectory(tid, (cx, cy))
            self.draw_trajectory(frame, tid)
            self.draw_detection_box(frame, (x1, y1, x2, y2), tid, is_intruding)

        # 3. 告警横幅
        self.draw_alert_banner(frame, current_intruders, frame_w, self.frame_count)

        return frame, list(current_intruders)

    # ------------------------------------------------------------------
    # 运行主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        运行入侵检测主循环

        处理视频流、逐帧推理、显示结果。
        """
        source = self.cfg["video_source"]
        if source is None:
            print("[X] 错误: 必须指定 --source 输入视频路径或摄像头 (0)")
            return
        print(f"[运行] 视频源: {source}")

        # 执行跟踪推理 (stream=True 逐帧返回生成器)
        results_gen = self.model.track(
            source=source,
            stream=True,
            persist=self.cfg["track_persist"],
            conf=self.cfg["detection_conf"],
            classes=[self.cfg["person_class_id"]],  # 只检测行人
            tracker=self.cfg["tracker_config"],
            device=self.cfg["device"],
            verbose=False,  # 不打印每帧日志
        )

        # 视频写入器初始化标志
        writer_initialized = False
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        print("[运行] 开始处理视频流...")
        print("[运行] 按 ESC 键退出")
        print("-" * 50)

        try:
            for result in results_gen:
                self.frame_count += 1

                # 获取原始帧
                frame = result.orig_img.copy()

                # 如果需要保存视频，初始化 writer
                if self.cfg["output_video_path"] and not writer_initialized:
                    h, w = frame.shape[:2]
                    if self.cfg["output_size"]:
                        w, h = self.cfg["output_size"]
                    fourcc = getattr(cv2, 'VideoWriter_fourcc')(*"mp4v")
                    self.video_writer = cv2.VideoWriter(
                        self.cfg["output_video_path"],
                        fourcc,
                        self.cfg["output_fps"],
                        (w, h),
                    )
                    if not self.video_writer.isOpened():
                        print(f"⚠️  无法创建输出视频: {self.cfg['output_video_path']}")
                        self.video_writer = None
                    writer_initialized = True

                # 处理当前帧
                annotated_frame, intruder_ids = self.process_frame(frame, result)

                # 绘制信息面板
                annotated_frame = self.draw_info_panel(
                    annotated_frame, intruder_ids, current_fps
                )

                # 缩放显示
                display_scale = self.cfg["display_scale"]
                if display_scale != 1.0:
                    h, w = annotated_frame.shape[:2]
                    new_w, new_h = int(w * display_scale), int(h * display_scale)
                    display_frame = cv2.resize(annotated_frame, (new_w, new_h))
                else:
                    display_frame = annotated_frame

                # 显示画面
                cv2.imshow(self.cfg["display_window_name"], display_frame)

                # 保存视频
                if self.video_writer is not None:
                    self.video_writer.write(annotated_frame)

                # FPS 计算
                fps_frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 键
                    print("\n[退出] 用户按 ESC 键，正在退出...")
                    break
                elif key == ord(" "):  # 空格键暂停/继续
                    print("⏸️  暂停，按任意键继续...")
                    cv2.waitKey(0)
                elif key == ord("s"):  # S 键截图
                    screenshot_path = f"screenshot_{self.frame_count:06d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 截图已保存: {screenshot_path}")

        except KeyboardInterrupt:
            print("\n[退出] 用户中断...")
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """释放所有资源"""
        print("[清理] 正在释放资源...")
        cv2.destroyAllWindows()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[清理] 视频已保存至: {self.cfg['output_video_path']}")
        print(f"[清理] 共处理 {self.frame_count} 帧")
        print("[清理] 程序退出 ✅")


# ============================================================================
# 工具函数
# ============================================================================

def load_zone_config(json_path: str) -> List[dict]:
    """
    从 JSON 文件加载禁入区域配置。

    支持三种格式:
      1. 多区域格式 (zone_selector.py 输出):
         {"zones": [{"zone_name": "...", "polygon": [[x,y],...]}, ...]}
      2. 单区域格式:
         {"zone_name": "...", "polygon": [[x,y], ...], ...}
      3. 纯列表格式:
         [[x1, y1], [x2, y2], ...]

    Args:
        json_path: JSON 配置文件路径

    Returns:
        [{"name": str, "points": [(x1,y1),...]}, ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    zones = []

    if isinstance(data, dict) and "zones" in data:
        # Multi-zone format
        for z in data["zones"]:
            zones.append({
                "name": z.get("zone_name", "zone"),
                "points": [tuple(p) for p in z["polygon"]],
            })
        print(f"[Zone] Loaded {len(zones)} zones from {json_path}")
        for z in zones:
            print(f"  - {z['name']}: {len(z['points'])} vertices")
    elif isinstance(data, dict) and "polygon" in data:
        # Single-zone format (backward compat)
        zones.append({
            "name": data.get("zone_name", "zone_1"),
            "points": [tuple(p) for p in data["polygon"]],
        })
        print(f"[Zone] Loaded single zone '{zones[0]['name']}' "
              f"({len(zones[0]['points'])} vertices) from {json_path}")
    elif isinstance(data, list):
        # Raw polygon list
        zones.append({"name": "zone_1", "points": [tuple(p) for p in data]})
        print(f"[Zone] Loaded raw polygon ({len(zones[0]['points'])} vertices) from {json_path}")
    else:
        raise ValueError(f"Unrecognized zone config format in {json_path}")

    return zones


def main():
    """程序主入口"""
    parser = argparse.ArgumentParser(
        description="Intrusion Detection - YOLOv8 + ByteTrack"
    )
    parser.add_argument(
        "--source", type=str, default=DEFAULT_VIDEO_SOURCE,
        help="输入视频路径 或 0 使用摄像头 (必须指定)"
    )
    parser.add_argument(
        "--zone", type=str, default=None,
        help="禁入区域 JSON 配置文件路径 (由 tools/zone_selector.py 生成)。"
             "若不指定，使用脚本内示例多边形。"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"输出视频路径，或 'none' 禁用保存 (默认: {DEFAULT_OUTPUT_PATH})"
    )
    args = parser.parse_args()

    if args.source is None:
        print("[X] 错误: 必须通过 --source 指定输入视频路径")
        print("    用法: python scripts/intrusion_detection.py --source <video_path> [--zone configs/forbidden_zone.json]")
        sys.exit(1)

    # Load zone config
    if args.zone:
        zone_points = load_zone_config(args.zone)
    else:
        print("[i] 未指定 --zone，使用脚本内示例多边形 (仅用于演示，请用 zone_selector.py 标定实际区域)")
        zone_points = FORBIDDEN_ZONE_POINTS

    # Override defaults with CLI args
    config_overrides = {
        "video_source": args.source,
        "forbidden_zone_points": zone_points,
        "output_video_path": None if args.output.lower() == "none" else args.output,
    }

    print("=" * 55)
    print("   Intrusion Detection - YOLOv8 + ByteTrack")
    print("=" * 55)
    print(f"   Source: {args.source}")
    print(f"   Zone:   {args.zone or 'hardcoded (FORBIDDEN_ZONE_POINTS)'}")
    print(f"   Output: {config_overrides['output_video_path'] or 'disabled'}")
    print()

    system = IntrusionDetectionSystem(config_overrides)

    print()
    print("Controls:")
    print("   ESC  - Quit")
    print("   SPACE - Pause/Resume")
    print("   S    - Screenshot")
    print("-" * 55)
    print()

    system.run()


if __name__ == "__main__":
    import sys
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
