"""
VisionX CLI: Movement Heatmap Generation
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import argparse
from visionx.visualizer.heatmap import (
    generate_heatmaps,
    build_accumulator,
    overlay_heatmap_on_image,
    HEATMAP_CMAP,
)
from visionx.visualizer.plot import save_heatmap_figure


def run(
    tracks_json: str,
    out_dir: str = "./outputs/heatmaps",
    background_image: str = None,
    sigma: float = 35.0,
    top_n_players: int = 6,
    alpha: float = 0.65,
):
    results = generate_heatmaps(
        tracks_json=tracks_json,
        out_dir=out_dir,
        background_image=background_image,
        sigma=sigma,
        top_n_players=top_n_players,
        alpha=alpha,
    )
    print("Heatmap generation complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 - Heatmap Generation")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--out", default="./outputs/heatmaps", help="Output directory")
    parser.add_argument("--bg", default=None, help="Background image path (first frame)")
    parser.add_argument("--sigma", type=float, default=35.0, help="Gaussian KDE blur sigma")
    parser.add_argument("--top-n", type=int, default=6, help="Generate heatmap for top N most active players")
    parser.add_argument("--alpha", type=float, default=0.65, help="Heatmap overlay blend alpha")
    args = parser.parse_args()

    run(
        tracks_json=args.tracks,
        out_dir=args.out,
        background_image=args.bg,
        sigma=args.sigma,
        top_n_players=args.top_n,
        alpha=args.alpha,
    )
