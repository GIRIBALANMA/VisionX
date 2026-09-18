"""
VisionX Geometry & Spatial Coordinate Helpers
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def to_foot_point(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    """
    Computes centroid ground contact point for top-down camera views.
    In overhead drone footage, the center of the bounding box represents
    the player's ground contact location on the pitch.
    """
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def to_foot_points(boxes: np.ndarray) -> List[Dict[str, Any]]:
    """
    Converts bounding boxes array [N, 6] to structured detection dictionaries:
    [{'bbox': [x1, y1, x2, y2], 'foot_point': [fx, fy], 'confidence': conf}, ...]
    """
    points = []
    for box in boxes:
        x1, y1, x2, y2, conf = box[:5]
        ground_pt = to_foot_point(x1, y1, x2, y2)
        points.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "foot_point": ground_pt,
            "confidence": float(conf),
        })
    return points
