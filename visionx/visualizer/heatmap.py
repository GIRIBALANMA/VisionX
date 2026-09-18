"""
VisionX Gaussian Movement Heatmap Engine
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from visionx.visualizer.plot import save_heatmap_figure

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soccer",
    ["#0a0a2a", "#1a237e", "#1565c0", "#00bcd4", "#4caf50", "#cddc39", "#ffeb3b", "#ff5722"],
)


def build_accumulator(points: List[List[float]], H: int, W: int, sigma: float = 35.0) -> np.ndarray:
    """
    Builds a smooth heatmap density field from ground contact foot points
    using a 2D Gaussian filter.
    """
    acc = np.zeros((H, W), dtype=np.float32)
    for pt in points:
        x, y = pt[:2]
        xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        acc[yi, xi] += 1.0
    return gaussian_filter(acc, sigma=sigma)


def overlay_heatmap_on_image(
    base_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.65,
    cmap=HEATMAP_CMAP,
) -> np.ndarray:
    """Blends a colored density map on top of a background image."""
    norm_heat = heatmap / (heatmap.max() + 1e-8)
    coloured = (cmap(norm_heat)[:, :, :3] * 255).astype(np.uint8)
    coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    mask = (norm_heat > 0.02).astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = (
        base_img.astype(np.float32) * (1.0 - alpha * mask_3ch)
        + coloured_bgr.astype(np.float32) * (alpha * mask_3ch)
    ).clip(0, 255).astype(np.uint8)
    return result


def generate_heatmaps(
    tracks_json: Union[str, Path],
    out_dir: Union[str, Path] = "./outputs/heatmaps",
    background_image: Optional[Union[str, Path]] = None,
    sigma: float = 35.0,
    top_n_players: int = 6,
    alpha: float = 0.65,
) -> Dict[str, str]:
    """
    Processes a tracks.json file and outputs:
      - outputs/heatmaps/overall_heatmap.png
      - outputs/heatmaps/summary_heatmaps.png
      - outputs/heatmaps/player_<id>_heatmap.png (for top N tracked players)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(tracks_json) as f:
        data = json.load(f)

    meta = data["meta"]
    W, H = meta["resolution"]
    tracks = data["tracks"]

    # Background loading or synthetic pitch fallback
    if background_image and os.path.exists(background_image):
        bg = cv2.imread(str(background_image))
        bg = cv2.resize(bg, (W, H))
    else:
        bg = np.full((H, W, 3), 30, dtype=np.uint8)
        bg[:, :, 1] = 80
        lc = (200, 220, 200)
        t = 3
        cv2.rectangle(bg, (int(W * 0.05), int(H * 0.05)), (int(W * 0.95), int(H * 0.95)), lc, t)
        cv2.line(bg, (int(W * 0.5), int(H * 0.05)), (int(W * 0.5), int(H * 0.95)), lc, t)
        cv2.ellipse(bg, (int(W * 0.5), int(H * 0.5)), (int(W * 0.12), int(H * 0.20)), 0, 0, 360, lc, t)

    # 1. Overall Heatmap
    all_points = []
    for tid_str, detections in tracks.items():
        for det in detections:
            all_points.append(det["foot_point"])

    overall_acc = build_accumulator(all_points, H, W, sigma=sigma)
    overall_path = out_dir / "overall_heatmap.png"
    save_heatmap_figure(
        overall_acc, bg,
        f"Overall Player Presence Heatmap ({len(tracks)} tracks, {len(all_points)} pts)",
        str(overall_path),
        cmap=HEATMAP_CMAP,
        overlay_func=overlay_heatmap_on_image,
        alpha=alpha,
    )

    # 2. Per-Player Heatmaps for Top N Most Active
    sorted_tracks = sorted(tracks.items(), key=lambda x: len(x[1]), reverse=True)
    top_tracks = sorted_tracks[:top_n_players]

    player_paths = {}
    for tid_str, detections in top_tracks:
        pts = [d["foot_point"] for d in detections]
        if len(pts) < 10:
            continue
        p_acc = build_accumulator(pts, H, W, sigma=sigma)
        p_path = out_dir / f"player_{tid_str}_heatmap.png"
        save_heatmap_figure(
            p_acc, bg,
            f"Player {tid_str} Movement ({len(pts)} pts)",
            str(p_path),
            cmap=HEATMAP_CMAP,
            overlay_func=overlay_heatmap_on_image,
            alpha=alpha,
        )
        player_paths[tid_str] = str(p_path)

    # 3. Multi-Panel Summary Grid
    if top_tracks:
        n_show = min(6, len(top_tracks))
        ncols = 3
        nrows = int(np.ceil(n_show / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = np.array(axes).reshape(-1)

        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < n_show:
                tid_str, detections = top_tracks[idx]
                pts = [d["foot_point"] for d in detections]
                p_acc = build_accumulator(pts, H, W, sigma=sigma)
                overlay = overlay_heatmap_on_image(bg, p_acc, alpha=alpha)
                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Track {tid_str} ({len(pts)} pts)", fontsize=11, fontweight="bold")
            ax.axis("off")

        fig.tight_layout()
        summary_path = out_dir / "summary_heatmaps.png"
        fig.savefig(str(summary_path), dpi=120, bbox_inches="tight")
        plt.close(fig)

    return {
        "overall": str(overall_path),
        "summary": str(out_dir / "summary_heatmaps.png"),
        "players": player_paths,
    }
