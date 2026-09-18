"""
Stage 4 — Heatmap Generation
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)

Input:  tracks.json (output of video_track.py)
Output: 
    - outputs/heatmaps/player_<id>_heatmap.png  (individual player KDE)
    - outputs/heatmaps/overall_heatmap.png       (all players combined)
    - outputs/heatmaps/summary.png               (multi-panel overview)

Approach:
    - Use 2D Gaussian accumulator (cv2 blur) for fast, smooth heatmaps.
    - Optionally overlay on a pitch template or the first video frame.
    - Per-player heatmaps for top N most-tracked players.
    - Overall heatmap aggregates all foot points from all tracks.
"""

import argparse
import json
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soccer",
    ["#0a0a2a", "#1a237e", "#1565c0", "#00bcd4", "#4caf50", "#cddc39", "#ffeb3b", "#ff5722"],
)


def load_tracks(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def build_accumulator(points, H, W, sigma=25):
    """
    Build a smooth heatmap accumulator from a list of (x, y) foot points.
    Uses Gaussian blurring to produce a density field.
    """
    acc = np.zeros((H, W), dtype=np.float32)
    for x, y in points:
        xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        acc[yi, xi] += 1.0
    acc = gaussian_filter(acc, sigma=sigma)
    return acc


def overlay_heatmap_on_image(base_img, heatmap, alpha=0.65):
    """Overlay a coloured heatmap on a background image."""
    norm_heat = heatmap / (heatmap.max() + 1e-8)
    coloured = (HEATMAP_CMAP(norm_heat)[:, :, :3] * 255).astype(np.uint8)
    coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    # Blend only where heatmap has density > threshold
    mask = (norm_heat > 0.02).astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = (base_img.astype(np.float32) * (1 - alpha * mask_3ch) +
              coloured_bgr.astype(np.float32) * alpha * mask_3ch).clip(0, 255).astype(np.uint8)
    return result


def add_colourbar(ax, heatmap, cmap, label="Presence Density"):
    norm = mcolors.Normalize(vmin=0, vmax=heatmap.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label=label)


def save_heatmap_figure(heatmap, background, title, out_path, alpha=0.65):
    """Render and save a single heatmap figure."""
    overlay = overlay_heatmap_on_image(background, heatmap, alpha=alpha)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(overlay_rgb)
    norm_heat = heatmap / (heatmap.max() + 1e-8)
    ax.contourf(norm_heat, levels=8, cmap=HEATMAP_CMAP, alpha=0.0)  # for colourbar
    add_colourbar(ax, heatmap, HEATMAP_CMAP)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    tracks_json: str,
    out_dir: str = "./outputs/heatmaps",
    background_image: str = None,    # optional: path to first frame or pitch template
    sigma: float = 25,
    top_n_players: int = 5,          # render per-player heatmap for top N
    alpha: float = 0.65,
):
    os.makedirs(out_dir, exist_ok=True)
    data = load_tracks(tracks_json)

    meta = data["meta"]
    W, H = meta["resolution"]
    tracks = data["tracks"]

    # Background image
    if background_image and os.path.exists(background_image):
        bg = cv2.imread(background_image)
        bg = cv2.resize(bg, (W, H))
    else:
        # Plain dark green synthetic pitch
        bg = np.full((H, W, 3), 30, dtype=np.uint8)
        bg[:, :, 1] = 80  # greenish
        # Draw pitch lines
        lc = (200, 220, 200)
        t = 3
        cv2.rectangle(bg, (int(W * 0.05), int(H * 0.05)),
                      (int(W * 0.95), int(H * 0.95)), lc, t)
        cv2.line(bg, (int(W * 0.5), int(H * 0.05)),
                 (int(W * 0.5), int(H * 0.95)), lc, t)
        cv2.ellipse(bg, (int(W * 0.5), int(H * 0.5)),
                    (int(W * 0.12), int(H * 0.20)), 0, 0, 360, lc, t)

    print(f"Resolution: {W}x{H} | {len(tracks)} unique track IDs | sigma={sigma}")

    # ── Overall heatmap ─────────────────────────────────────────────────────
    all_points = []
    for tid_str, detections in tracks.items():
        for det in detections:
            all_points.append(det["foot_point"])

    print(f"Total foot points: {len(all_points)}")
    overall_acc = build_accumulator(all_points, H, W, sigma=sigma)
    overall_path = os.path.join(out_dir, "overall_heatmap.png")
    save_heatmap_figure(overall_acc, bg,
                        f"Overall Player Presence Heatmap ({len(tracks)} tracks)",
                        overall_path, alpha=alpha)
    print(f"  Overall heatmap  -> {overall_path}")

    # ── Per-player heatmaps (top N by frame count) ─────────────────────────
    sorted_tracks = sorted(tracks.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_tracks = sorted_tracks[:top_n_players]

    for tid_str, detections in top_tracks:
        pts = [d["foot_point"] for d in detections]
        acc = build_accumulator(pts, H, W, sigma=sigma)
        out_path = os.path.join(out_dir, f"player_{tid_str}_heatmap.png")
        save_heatmap_figure(
            acc, bg,
            f"Player Track {tid_str} | {len(detections)} detections",
            out_path, alpha=alpha,
        )
        print(f"  Player {tid_str} heatmap  -> {out_path} ({len(detections)} pts)")

    # ── Summary grid — top 6 players in one panel ──────────────────────────
    top6 = sorted_tracks[:6]
    n = len(top6)
    if n > 0:
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
        axes = np.array(axes).flatten()

        for idx, (tid_str, detections) in enumerate(top6):
            pts = [d["foot_point"] for d in detections]
            acc = build_accumulator(pts, H, W, sigma=sigma)
            overlay = overlay_heatmap_on_image(bg, acc, alpha=alpha)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(overlay_rgb)
            axes[idx].set_title(f"Track {tid_str} ({len(detections)} frames)",
                                fontsize=10, fontweight="bold")
            axes[idx].axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Top Player Heatmaps — VisionX PR-01", fontsize=14, fontweight="bold")
        fig.tight_layout()
        summary_path = os.path.join(out_dir, "summary_heatmaps.png")
        fig.savefig(summary_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary grid     -> {summary_path}")

    print("\nHeatmap generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 - Heatmap Generation")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json from video_track.py")
    parser.add_argument("--out", default="./outputs/heatmaps")
    parser.add_argument("--bg", default=None, help="Background image (optional)")
    parser.add_argument("--sigma", type=float, default=25, help="Gaussian kernel sigma")
    parser.add_argument("--top-n", type=int, default=5, help="Per-player heatmaps for top N")
    parser.add_argument("--alpha", type=float, default=0.65, help="Heatmap blend alpha")
    args = parser.parse_args()

    run(tracks_json=args.tracks, out_dir=args.out, background_image=args.bg,
        sigma=args.sigma, top_n_players=args.top_n, alpha=args.alpha)
