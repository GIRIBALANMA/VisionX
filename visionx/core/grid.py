"""
VisionX Sliced Grid Partitioning Engine
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

from typing import List, Tuple
import numpy as np
from visionx.config import DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, DEFAULT_GRID_OVERLAP


def slice_frame_grid(
    image: np.ndarray,
    n_cols: int = DEFAULT_GRID_COLS,
    n_rows: int = DEFAULT_GRID_ROWS,
    overlap: float = DEFAULT_GRID_OVERLAP,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Slices a high-resolution image into an overlapping regular grid of tiles.
    Returns:
        tiles: list of cropped image patches
        offsets: list of (x_offset, y_offset, tile_w, tile_h)
    """
    H, W = image.shape[:2]
    tile_w = int(W / (n_cols - (n_cols - 1) * overlap))
    tile_h = int(H / (n_rows - (n_rows - 1) * overlap))
    step_x = int(tile_w * (1 - overlap))
    step_y = int(tile_h * (1 - overlap))

    tiles = []
    offsets = []
    for r in range(n_rows):
        for c in range(n_cols):
            x1 = min(c * step_x, W - tile_w)
            y1 = min(r * step_y, H - tile_h)
            x2 = min(x1 + tile_w, W)
            y2 = min(y1 + tile_h, H)
            tiles.append(image[y1:y2, x1:x2])
            offsets.append((x1, y1, x2 - x1, y2 - y1))

    return tiles, offsets
