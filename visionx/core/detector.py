"""
VisionX Sliced Grid Player Detector
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import os
from typing import Optional
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from visionx.config import (
    PERSON_CLASS_ID,
    OVERHEAD_ASSIST_CLASSES,
    DEFAULT_WEIGHTS,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_OVERLAP,
    DEFAULT_TILE_IMGSZ,
    DEFAULT_CONF,
    DEFAULT_IOU,
    MIN_PLAYER_DIM,
    MAX_PLAYER_DIM,
)
from visionx.core.grid import slice_frame_grid
from visionx.core.pitch import get_pitch_polygon, is_inside_pitch


def load_model(weights: str = DEFAULT_WEIGHTS) -> YOLO:
    """Loads YOLO weights on available GPU/CPU hardware."""
    if not os.path.exists(weights):
        alt = "yolov8m.pt" if weights == "yolov8s.pt" else "yolov8s.pt"
        if os.path.exists(alt):
            weights = alt
    return YOLO(weights)


def detect_persons_grid(
    model: YOLO,
    image: np.ndarray,
    n_cols: int = DEFAULT_GRID_COLS,
    n_rows: int = DEFAULT_GRID_ROWS,
    overlap: float = DEFAULT_GRID_OVERLAP,
    tile_imgsz: int = DEFAULT_TILE_IMGSZ,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    topdown_assist: bool = True,
    pitch_poly: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sliced Grid Inference for tiny aerial players:
    1. Slices the full frame into a grid of overlapping tiles.
    2. Runs batched YOLO inference over tiles so tiny players are magnified into YOLO's receptive field.
    3. Remaps coordinates back to the main frame: x_main = x_tile + x_offset, y_main = y_tile + y_offset.
    4. Filters detections inside pitch_poly and validates physical scale (8-65px).
    5. Applies global NMS across all tiles to merge overlaps at seams.
    """
    H, W = image.shape[:2]
    if pitch_poly is None:
        pitch_poly = get_pitch_polygon(image)

    tiles, offsets = slice_frame_grid(image, n_cols=n_cols, n_rows=n_rows, overlap=overlap)

    target_classes = [PERSON_CLASS_ID]
    if topdown_assist:
        target_classes.extend(OVERHEAD_ASSIST_CLASSES)

    # Batched inference over all tiles on GPU
    results = model.predict(
        source=tiles,
        classes=target_classes,
        conf=conf,
        iou=0.30,
        imgsz=tile_imgsz,
        batch=len(tiles),
        verbose=False,
    )

    raw_boxes = []
    raw_scores = []
    raw_cls = []

    for i, res in enumerate(results):
        ox, oy, tw, th = offsets[i]
        if res.boxes is not None and len(res.boxes) > 0:
            b_data = res.boxes.data.cpu().numpy()
            for b in b_data:
                x1, y1, x2, y2, sc, cls_id = b
                gx1 = x1 + ox
                gy1 = y1 + oy
                gx2 = x2 + ox
                gy2 = y2 + oy
                w = gx2 - gx1
                h = gy2 - gy1
                cx = (gx1 + gx2) / 2.0
                cy = (gy1 + gy2) / 2.0

                # Strict pitch boundary check
                if not is_inside_pitch(pitch_poly, cx, cy):
                    continue

                # Realistic physical scale check for 4K aerial players
                if MIN_PLAYER_DIM <= w <= MAX_PLAYER_DIM and MIN_PLAYER_DIM <= h <= MAX_PLAYER_DIM:
                    raw_boxes.append([gx1, gy1, gx2, gy2])
                    raw_scores.append(float(sc))
                    raw_cls.append(0)

    if not raw_boxes:
        return np.empty((0, 6))

    # Global NMS across shifted tiles to eliminate seam duplicates
    t_boxes = torch.tensor(raw_boxes, dtype=torch.float32)
    t_scores = torch.tensor(raw_scores, dtype=torch.float32)
    keep = nms(t_boxes, t_scores, iou_threshold=iou)

    final_boxes = []
    for idx in keep.cpu().numpy():
        b = raw_boxes[idx]
        final_boxes.append([b[0], b[1], b[2], b[3], raw_scores[idx], raw_cls[idx]])

    return np.array(final_boxes)


def detect_persons(
    model: YOLO,
    image: np.ndarray,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = 1280,
    augment: bool = False,
    topdown_assist: bool = True,
    pitch_poly: Optional[np.ndarray] = None,
    use_grid: bool = True,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_overlap: float = DEFAULT_GRID_OVERLAP,
    tile_imgsz: int = DEFAULT_TILE_IMGSZ,
) -> np.ndarray:
    """
    Main detection interface:
    By default, activates Sliced Grid Inference for tiny aerial players.
    """
    if use_grid:
        return detect_persons_grid(
            model=model,
            image=image,
            n_cols=grid_cols,
            n_rows=grid_rows,
            overlap=grid_overlap,
            tile_imgsz=tile_imgsz,
            conf=conf,
            iou=iou,
            topdown_assist=topdown_assist,
            pitch_poly=pitch_poly,
        )

    # Standard full-frame fallback
    H, W = image.shape[:2]
    if pitch_poly is None:
        pitch_poly = get_pitch_polygon(image)

    target_classes = [PERSON_CLASS_ID]
    if topdown_assist:
        target_classes.extend(OVERHEAD_ASSIST_CLASSES)

    results = model.predict(
        source=image,
        classes=target_classes,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        augment=augment,
        verbose=False,
    )

    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        raw_boxes = results[0].boxes.data.cpu().numpy()
        for b in raw_boxes:
            x1, y1, x2, y2, sc, cls_id = b
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not is_inside_pitch(pitch_poly, cx, cy):
                continue
            w, h = x2 - x1, y2 - y1
            if MIN_PLAYER_DIM <= w <= MAX_PLAYER_DIM and MIN_PLAYER_DIM <= h <= MAX_PLAYER_DIM:
                boxes.append([x1, y1, x2, y2, float(sc), 0])

    return np.array(boxes) if boxes else np.empty((0, 6))
