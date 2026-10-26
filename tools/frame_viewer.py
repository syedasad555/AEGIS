#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""Frame-level video viewer for ground truth annotation.

Controls:
  Space       — Pause / Resume
  A / Left    — Step backward 1 frame
  D / Right   — Step forward 1 frame
  S           — Jump back 30 frames
  W           — Jump forward 30 frames
  Q / Esc     — Quit

Usage:
  python tools/frame_viewer.py <video_path>
"""

import sys
import cv2


def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    paused = False
    current_frame = 0

    win_name = f"Frame Viewer — {video_path}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}  FPS: {fps:.1f}  Duration: {total_frames/fps:.1f}s")
    print()
    print("Controls:")
    print("  [Space] Pause/Play   [A/←] Back 1   [D/→] Forward 1")
    print("  [S] Jump -30         [W] Jump +30    [Q/Esc] Quit")
    print()

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {current_frame}")
            break

        # Overlay info
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(frame, f"Frame: {current_frame}/{total_frames}  [{status}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):  # Space
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'} at frame {current_frame}")
        elif key == ord('d') or key == 83:  # D or Right arrow
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('a') or key == 81:  # A or Left arrow
            current_frame = max(0, current_frame - 1)
        elif key == ord('w'):
            current_frame = min(total_frames - 1, current_frame + 30)
        elif key == ord('s'):
            current_frame = max(0, current_frame - 30)
        elif not paused:
            current_frame += 1

        if current_frame >= total_frames:
            print("End of video reached.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/frame_viewer.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
