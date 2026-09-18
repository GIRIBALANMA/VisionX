"""
VisionX Matplotlib Plotting & Chart Generation
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np


def plot_track_count(per_frame: Dict[str, List[Any]], out_path: str):
    """Plots active track count per frame over time as a validation chart."""
    frames = sorted(int(k) for k in per_frame.keys())
    counts = [len(per_frame[str(f)]) for f in frames]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frames, counts, color="#00e5ff", linewidth=1.8)
    ax.fill_between(frames, counts, alpha=0.15, color="#00e5ff")
    ax.set_xlabel("Frame Number", fontsize=11, labelpad=8)
    ax.set_ylabel("Active Tracks", fontsize=11, labelpad=8)
    ax.set_title("Active Player Tracks Per Frame (ByteTrack + Sliced Grid Inference)", fontsize=13, pad=12)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0, top=max(counts + [25]) + 3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_heatmap_figure(
    heatmap: np.ndarray,
    background: np.ndarray,
    title: str,
    out_path: str,
    cmap,
    overlay_func,
    alpha: float = 0.65,
):
    """Renders and saves a single heatmap figure with a colorbar."""
    overlay = overlay_func(background, heatmap, alpha=alpha, cmap=cmap)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(overlay_rgb)
    norm_heat = heatmap / (heatmap.max() + 1e-8)
    ax.contourf(norm_heat, levels=8, cmap=cmap, alpha=0.0)

    norm = mcolors.Normalize(vmin=0, vmax=heatmap.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Presence Density")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
