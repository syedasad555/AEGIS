#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
Interactive forbidden zone selector tool — supports multiple zones.

Usage:
    python tools/zone_selector.py --source <video_path> --output <json_path>

Controls:
    Left-click  : add vertex to current zone
    Right-click : undo last vertex of current zone
    n           : create a new zone
    Tab / ]     : next zone
    [           : previous zone
    d           : delete current zone
    c           : clear current zone vertices
    s           : save all zones to JSON
    q           : quit without saving
    1-9         : seek to 10%-90% of video
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path

# Per-zone color palette (BGR)
ZONE_COLORS = [
    (0, 100, 255),   # orange-red
    (255, 100, 0),   # sky-blue
    (0, 255, 100),   # lime-green
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (100, 255, 0),   # spring-green
    (255, 0, 100),   # violet
]


class ZoneSelector:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.zones = []          # [{"name": str, "points": [(x,y),...]}, ...]
        self.active_idx = 0      # which zone is being edited
        self.frame = None
        self.original_frame = None
        self.window_name = "Zone Selector - Multi-Zone"

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {self.width}x{self.height}, {self.total_frames} frames, {self.fps:.1f} FPS")
        self._add_zone()
        self.seek_frame(0)

    def _add_zone(self):
        name = f"zone_{len(self.zones) + 1}"
        self.zones.append({"name": name, "points": []})
        self.active_idx = len(self.zones) - 1
        print(f"[New] Created '{name}' (total: {len(self.zones)})")

    def _active_zone(self):
        if not self.zones:
            self._add_zone()
        return self.zones[self.active_idx]

    def _zone_color(self, idx):
        return ZONE_COLORS[idx % len(ZONE_COLORS)]

    def seek_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame.copy()
            self.frame = frame.copy()
            pct = 100 * frame_idx / max(self.total_frames - 1, 1)
            print(f"[Seek] frame {frame_idx} ({pct:.0f}%)")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            zone = self._active_zone()
            zone["points"].append((x, y))
            print(f"[Add] {zone['name']} point {len(zone['points'])}: ({x}, {y})")
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            zone = self._active_zone()
            if zone["points"]:
                removed = zone["points"].pop()
                print(f"[Undo] {zone['name']} removed: {removed}")
                self.redraw()

    def redraw(self):
        self.frame = self.original_frame.copy()
        h, w = self.frame.shape[:2]

        # --- Top info banners ---
        cv2.rectangle(self.frame, (0, 0), (w, 72), (50, 50, 50), -1)

        if self.zones:
            active = self._active_zone()
            info1 = (f"[{self.active_idx + 1}/{len(self.zones)}] Editing: {active['name']} "
                     f"| Vertices: {len(active['points'])}")
        else:
            info1 = "No zones | Press 'n' to create one"
        cv2.putText(self.frame, info1, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        info2 = "L-click:add | R-click:undo | n:new | Tab:next | d:delete | c:clear | s:save | q:quit | 1-9:seek"
        cv2.putText(self.frame, info2, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        info3 = f"Total zones: {len(self.zones)}"
        cv2.putText(self.frame, info3, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if not self.zones:
            cv2.imshow(self.window_name, self.frame)
            return

        # --- Draw all zones ---
        for zi, zone in enumerate(self.zones):
            pts = zone["points"]
            if len(pts) == 0:
                continue

            color = self._zone_color(zi)
            is_active = (zi == self.active_idx)
            edge_thick = 3 if is_active else 2

            # Vertices
            for i, pt in enumerate(pts):
                cv2.circle(self.frame, pt, 5, color, -1)
                cv2.circle(self.frame, pt, 5, (255, 255, 255), 1)
                label = f"{zone['name']}-{i + 1}"
                cv2.putText(self.frame, label, (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Edges
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(self.frame, pts[i], pts[i + 1], color, edge_thick)

            # Closed polygon + fill
            if len(pts) >= 3:
                cv2.line(self.frame, pts[-1], pts[0], color, edge_thick, cv2.LINE_AA)
                overlay = self.frame.copy()
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], color)
                cv2.addWeighted(overlay, 0.2, self.frame, 0.8, 0, self.frame)

        cv2.imshow(self.window_name, self.frame)

    def save(self):
        valid_zones = [z for z in self.zones if len(z["points"]) >= 3]
        if not valid_zones:
            print("[Error] No valid zones (need >= 3 vertices each)")
            return False

        data = {
            "zones": [
                {
                    "zone_name": z["name"],
                    "polygon": [[int(x), int(y)] for x, y in z["points"]],
                    "description": f"Zone '{z['name']}' — {len(z['points'])} vertices"
                }
                for z in valid_zones
            ],
            "video_resolution": [self.width, self.height],
        }

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        total_pts = sum(len(z["points"]) for z in valid_zones)
        print(f"[Save] {len(valid_zones)} zones ({total_pts} vertices) -> {self.output_path}")
        for z in valid_zones:
            print(f"  {z['name']}: {z['points']}")
        return True

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1280, self.width), min(720, self.height))
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.redraw()

        print("\n=== Controls ===")
        print("L-click: add | R-click: undo | n: new zone | Tab: next | [: prev")
        print("d: delete zone | c: clear zone | s: save | q: quit | 1-9: seek\n")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("[Quit] Exiting without saving")
                break
            elif key == ord('s'):
                if self.save():
                    print("[Save] Saved successfully!")
                    break
                else:
                    print("[Save] Save failed — need at least one zone with >=3 vertices")
            elif key == ord('n'):
                self._add_zone()
                self.redraw()
            elif key in (ord('\t'), ord(']')):  # Tab or ]
                if self.zones:
                    self.active_idx = (self.active_idx + 1) % len(self.zones)
                    print(f"[Switch] Editing: {self._active_zone()['name']}")
                    self.redraw()
            elif key == ord('['):
                if self.zones:
                    self.active_idx = (self.active_idx - 1) % len(self.zones)
                    print(f"[Switch] Editing: {self._active_zone()['name']}")
                    self.redraw()
            elif key == ord('d'):
                if len(self.zones) > 1:
                    name = self.zones[self.active_idx]["name"]
                    del self.zones[self.active_idx]
                    if self.active_idx >= len(self.zones):
                        self.active_idx = len(self.zones) - 1
                    print(f"[Delete] Removed '{name}' ({len(self.zones)} remaining)")
                    self.redraw()
                else:
                    print("[Delete] Cannot delete last zone — clear it with 'c' instead")
            elif key == ord('c'):
                zone = self._active_zone()
                zone["points"] = []
                print(f"[Clear] Cleared all vertices from {zone['name']}")
                self.redraw()
            elif key in range(ord('1'), ord('9') + 1):
                pct = (key - ord('0')) / 10
                seek_to = int(self.total_frames * pct)
                self.seek_frame(seek_to)
                self.redraw()

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive forbidden zone selector")
    parser.add_argument("--source", type=str, required=True,
                        help="输入视频路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSON 配置路径 (用于 intrusion_detection.py)")
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"[X] 视频不存在: {args.source}")
        return

    selector = ZoneSelector(args.source, args.output)
    selector.run()


if __name__ == "__main__":
    main()
