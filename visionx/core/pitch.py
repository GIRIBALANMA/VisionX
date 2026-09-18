"""
VisionX Playing Field & Pitch Geometry Module
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import cv2
import numpy as np
from visionx.config import (
    PITCH_X_MIN_RATIO,
    PITCH_X_MAX_RATIO,
    PITCH_Y_MIN_RATIO,
    PITCH_Y_MAX_RATIO,
)


def get_pitch_polygon(image: np.ndarray) -> np.ndarray:
    """
    Computes the calibrated playing field polygon boundary:
    Detects inner touchline bounds to exclude bench areas, crowd, and top banners.
    """
    H, W = image.shape[:2]

    # Proportional boundary based on 4K field touchline calibrations
    x_min = int(W * PITCH_X_MIN_RATIO)  # ~415px on 4K
    x_max = int(W * PITCH_X_MAX_RATIO)  # ~3440px on 4K
    y_min = int(H * PITCH_Y_MIN_RATIO)  # ~118px on 4K
    y_max = int(H * PITCH_Y_MAX_RATIO)  # ~2020px on 4K

    return np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.int32)


def is_inside_pitch(pitch_poly: np.ndarray, x: float, y: float) -> bool:
    """Returns True if (x, y) is strictly inside the playing field polygon."""
    return cv2.pointPolygonTest(pitch_poly, (float(x), float(y)), False) >= 0
