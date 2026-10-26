#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Selective Tiled Inference Engine
======================================
Implements high-resolution sliced inference for detecting distant, small,
and heavily occluded pedestrians while safeguarding host CPU and RAM resources.

Key Features:
1. Full-frame inference preserved as primary baseline and fallback.
2. Configurable Region of Interest (ROI) to tile only high-value scene regions
   (e.g., upper corridor containing distant pedestrians) without wasting CPU on sky or foreground.
3. Configurable tile size (width, height) and overlap ratio.
4. Coordinate translation: Maps tile bounding boxes back to global frame space.
5. Cross-tile duplicate suppression via cv2.dnn.NMSBoxes.
6. Clean integration with ByteTrack: feeds consolidated detections into BYTETracker
   without generating surrogate sequential IDs.
7. Memory guardian: sequential tile processing and dynamic RSS memory checks
   falling back to full-frame inference if host RAM budget is exceeded.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


class SimpleDetections:
    """
    Lightweight container compatible with Ultralytics BYTETracker.update().
    Exposes conf, cls, xywh attributes and supports boolean mask indexing.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(self.xyxy) == 0:
            self.xywh = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
        else:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            xc = self.xyxy[:, 0] + w / 2.0
            yc = self.xyxy[:, 1] + h / 2.0
            self.xywh = np.stack([xc, yc, w, h], axis=1)
            self.conf = np.asarray(conf, dtype=np.float32)
            self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: Any) -> "SimpleDetections":
        res = SimpleDetections([], [], [])
        res.xyxy = self.xyxy[mask]
        res.xywh = self.xywh[mask]
        res.conf = self.conf[mask]
        res.cls = self.cls[mask]
        return res


class SelectiveTiledDetector:
    """
    Selective Sliced / Tiled Inference Detector for YOLOv8.
    """

    def __init__(
        self,
        model: YOLO,
        tile_size: Tuple[int, int] = (640, 540),
        overlap: Tuple[float, float] = (0.20, 0.20),
        roi_rel: Optional[Tuple[float, float, float, float]] = (0.0, 0.15, 1.0, 0.65),
        conf_thresh: float = 0.25,
        iou_nms_thresh: float = 0.50,
        person_class_id: int = 0,
        mem_limit_mb: float = 650.0,
        enabled: bool = True
    ):
        """
        Args:
            model: Pre-loaded YOLO instance (avoids duplicate model instances in RAM).
            tile_size: (tile_w, tile_h) resolution for sliced inference.
            overlap: (overlap_x, overlap_y) fraction between adjacent tiles.
            roi_rel: (x1_rel, y1_rel, x2_rel, y2_rel) normalized ROI bounding box.
                     If None, tiles the whole frame.
            conf_thresh: Confidence threshold for person detections.
            iou_nms_thresh: IoU threshold for cross-tile NMS box deduplication.
            person_class_id: Class ID for pedestrians (default 0 for COCO person).
            mem_limit_mb: Max process RSS memory in MB before falling back to full-frame only.
            enabled: Master toggle for selective tiled inference.
        """
        self.model = model
        self.tile_w, self.tile_h = tile_size
        self.overlap_x, self.overlap_y = overlap
        self.roi_rel = roi_rel
        self.conf_thresh = conf_thresh
        self.iou_nms_thresh = iou_nms_thresh
        self.person_class_id = person_class_id
        self.mem_limit_mb = mem_limit_mb
        self.enabled = enabled

        self.process = psutil.Process(os.getpid())
        self.last_fallback_reason: Optional[str] = None
        self.stats = {
            "full_frame_inferences": 0,
            "tiled_inferences": 0,
            "total_tiles_processed": 0,
            "cross_tile_duplicates_suppressed": 0,
            "memory_fallbacks": 0
        }

    def compute_tile_crops(self, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
        """
        Computes absolute bounding coordinates [x1, y1, x2, y2] for tiles covering the configured ROI.
        """
        if self.roi_rel is not None:
            rx1, ry1, rx2, ry2 = self.roi_rel
            roi_x1 = max(0, int(rx1 * frame_w))
            roi_y1 = max(0, int(ry1 * frame_h))
            roi_x2 = min(frame_w, int(rx2 * frame_w))
            roi_y2 = min(frame_h, int(ry2 * frame_h))
        else:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, frame_w, frame_h

        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        if roi_w <= 0 or roi_h <= 0:
            return []

        step_x = max(1, int(self.tile_w * (1.0 - self.overlap_x)))
        step_y = max(1, int(self.tile_h * (1.0 - self.overlap_y)))

        x_coords = []
        x = roi_x1
        while x < roi_x2:
            x2 = min(roi_x2, x + self.tile_w)
            x1 = max(roi_x1, x2 - self.tile_w)
            x_coords.append((x1, x2))
            if x2 >= roi_x2:
                break
            x += step_x

        y_coords = []
        y = roi_y1
        while y < roi_y2:
            y2 = min(roi_y2, y + self.tile_h)
            y1 = max(roi_y1, y2 - self.tile_h)
            y_coords.append((y1, y2))
            if y2 >= roi_y2:
                break
            y += step_y

        crops = []
        for y1, y2 in y_coords:
            for x1, x2 in x_coords:
                crops.append((x1, y1, x2, y2))

        return crops

    def _check_memory_safety(self) -> bool:
        """
        Verifies that process RSS memory and available system RAM remain safe.
        Returns False if memory budget is exceeded.
        """
        try:
            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            if rss_mb > self.mem_limit_mb:
                self.last_fallback_reason = f"Process RSS {rss_mb:.1f}MB exceeds limit {self.mem_limit_mb:.1f}MB"
                self.stats["memory_fallbacks"] += 1
                return False

            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            if avail_mb < 200.0:
                self.last_fallback_reason = f"System available RAM {avail_mb:.1f}MB critically low (<200MB)"
                self.stats["memory_fallbacks"] += 1
                return False
        except Exception:
            pass
        return True

    def detect(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        run_full_frame: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Performs selective tiled pedestrian inference on the input frame.

        Args:
            frame: Input BGR frame image.
            imgsz: Resolution for inference (640 or 800).
            run_full_frame: Whether to execute full-frame pass as base.

        Returns:
            Tuple of:
            - boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2]
            - scores: np.ndarray of shape (N,)
            - class_ids: np.ndarray of shape (N,)
            - meta: dictionary with diagnostic timings and tile count
        """
        h, w = frame.shape[:2]
        t0 = time.time()

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_classes: List[int] = []

        is_pose = "pose" in str(getattr(self.model, "model_name", "")) or hasattr(self.model, "kpt_shape")

        # 1. Base Full-Frame Pass
        t_ff_start = time.time()
        ff_count = 0
        if run_full_frame:
            res_ff = self.model.predict(
                frame,
                imgsz=imgsz,
                conf=self.conf_thresh,
                classes=[self.person_class_id] if not is_pose else None,
                verbose=False
            )
            if res_ff and len(res_ff) > 0 and res_ff[0].boxes is not None and len(res_ff[0].boxes) > 0:
                b_xyxy = res_ff[0].boxes.xyxy.cpu().numpy()
                b_conf = res_ff[0].boxes.conf.cpu().numpy()
                b_cls = res_ff[0].boxes.cls.cpu().numpy()

                for bx, sc, cl in zip(b_xyxy, b_conf, b_cls):
                    if int(cl) == self.person_class_id or is_pose:
                        all_boxes.append([float(x) for x in bx])
                        all_scores.append(float(sc))
                        all_classes.append(int(cl))
                        ff_count += 1
        t_ff = time.time() - t_ff_start
        self.stats["full_frame_inferences"] += 1

        # 2. Selective Tiled Pass (if enabled and memory permits)
        t_tile_start = time.time()
        tiles_processed = 0
        tile_detections_raw = 0

        can_tile = self.enabled and self._check_memory_safety()

        if can_tile:
            tile_crops = self.compute_tile_crops(w, h)
            for x1, y1, x2, y2 in tile_crops:
                # Sequential tile extraction: extract crop, run inference, discard crop
                tile_crop = frame[y1:y2, x1:x2]
                if tile_crop.shape[0] < 10 or tile_crop.shape[1] < 10:
                    continue

                res_tile = self.model.predict(
                    tile_crop,
                    imgsz=self.tile_w,
                    conf=self.conf_thresh,
                    classes=[self.person_class_id] if not is_pose else None,
                    verbose=False
                )
                tiles_processed += 1

                if res_tile and len(res_tile) > 0 and res_tile[0].boxes is not None and len(res_tile[0].boxes) > 0:
                    t_xyxy = res_tile[0].boxes.xyxy.cpu().numpy()
                    t_conf = res_tile[0].boxes.conf.cpu().numpy()
                    t_cls = res_tile[0].boxes.cls.cpu().numpy()

                    for bx, sc, cl in zip(t_xyxy, t_conf, t_cls):
                        if int(cl) == self.person_class_id or is_pose:
                            # Map coordinates back to full-frame space
                            gx1 = float(bx[0] + x1)
                            gy1 = float(bx[1] + y1)
                            gx2 = float(bx[2] + x1)
                            gy2 = float(bx[3] + y1)
                            all_boxes.append([gx1, gy1, gx2, gy2])
                            all_scores.append(float(sc))
                            all_classes.append(int(cl))
                            tile_detections_raw += 1

            self.stats["tiled_inferences"] += 1
            self.stats["total_tiles_processed"] += tiles_processed
        t_tile = time.time() - t_tile_start

        # 3. Cross-Tile and Full-Frame Duplicate Suppression (NMS)
        t_nms_start = time.time()
        total_raw_boxes = len(all_boxes)

        if total_raw_boxes == 0:
            final_boxes = np.empty((0, 4), dtype=np.float32)
            final_scores = np.empty((0,), dtype=np.float32)
            final_classes = np.empty((0,), dtype=np.int32)
        else:
            # Convert [x1, y1, x2, y2] to cv2 format [x, y, w, h]
            cv2_boxes = []
            for b in all_boxes:
                bx = b[0]
                by = b[1]
                bw = max(1.0, b[2] - b[0])
                bh = max(1.0, b[3] - b[1])
                cv2_boxes.append([bx, by, bw, bh])

            indices = cv2.dnn.NMSBoxes(
                bboxes=cv2_boxes,
                scores=all_scores,
                score_threshold=self.conf_thresh,
                nms_threshold=self.iou_nms_thresh
            )

            if len(indices) > 0:
                keep = np.asarray(indices).flatten()
                final_boxes = np.asarray([all_boxes[i] for i in keep], dtype=np.float32)
                final_scores = np.asarray([all_scores[i] for i in keep], dtype=np.float32)
                final_classes = np.asarray([all_classes[i] for i in keep], dtype=np.int32)
            else:
                final_boxes = np.empty((0, 4), dtype=np.float32)
                final_scores = np.empty((0,), dtype=np.float32)
                final_classes = np.empty((0,), dtype=np.int32)

        t_nms = time.time() - t_nms_start
        t_total = time.time() - t0

        suppressed = total_raw_boxes - len(final_boxes)
        self.stats["cross_tile_duplicates_suppressed"] += max(0, suppressed)

        meta = {
            "total_latency_ms": round(t_total * 1000, 1),
            "full_frame_latency_ms": round(t_ff * 1000, 1),
            "tiled_latency_ms": round(t_tile * 1000, 1),
            "nms_latency_ms": round(t_nms * 1000, 1),
            "full_frame_detections": ff_count,
            "tile_detections_raw": tile_detections_raw,
            "tiles_processed": tiles_processed,
            "suppressed_duplicates": max(0, suppressed),
            "final_detections": len(final_boxes),
            "tiled_enabled": can_tile,
            "fallback_reason": self.last_fallback_reason if not can_tile else None
        }

        return final_boxes, final_scores, final_classes, meta

    def detect_and_track(
        self,
        frame: np.ndarray,
        tracker: BYTETracker,
        imgsz: int = 640
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs selective tiled detection and feeds the consolidated detections
        directly into the provided BYTETracker.

        Returns:
            Tuple of:
            - tracks: np.ndarray of shape (N, 8) with rows:
                      [x1, y1, x2, y2, track_id, score, cls, idx]
            - meta: diagnostic metadata
        """
        boxes, scores, classes, meta = self.detect(frame, imgsz=imgsz)
        det_container = SimpleDetections(boxes, scores, classes)
        t_track_start = time.time()
        tracks = tracker.update(det_container)
        meta["track_latency_ms"] = round((time.time() - t_track_start) * 1000, 1)

        # Integrity check: Ensure tracker outputs real numpy array
        if tracks is None or len(tracks) == 0:
            tracks = np.empty((0, 8), dtype=np.float32)

        return tracks, meta
