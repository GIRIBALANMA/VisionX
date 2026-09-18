"""
VisionX CLI: Multi-Object Player Tracking (ByteTrack + Sliced Grid)
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import argparse
from visionx.config import (
    DEFAULT_WEIGHTS,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_OVERLAP,
    DEFAULT_TILE_IMGSZ,
)
from visionx.core.tracker import run_video_tracking, create_tracker, to_foot_point
from visionx.visualizer.draw import draw_tracking_frame
from visionx.visualizer.plot import plot_track_count


def run(
    video_path: str,
    out_dir: str = "./outputs/video",
    weights: str = DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    use_grid: bool = True,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_overlap: float = DEFAULT_GRID_OVERLAP,
    tile_imgsz: int = DEFAULT_TILE_IMGSZ,
    stride: int = 1,
    preview_frames: int = 0,
    max_frames: int = 0,
    topdown_assist: bool = True,
):
    result = run_video_tracking(
        video_path=video_path,
        out_dir=out_dir,
        weights=weights,
        conf=conf,
        iou=iou,
        use_grid=use_grid,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        grid_overlap=grid_overlap,
        tile_imgsz=tile_imgsz,
        stride=stride,
        preview_frames=preview_frames,
        max_frames=max_frames,
        topdown_assist=topdown_assist,
    )
    return result["json_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 - Sliced Grid ByteTrack player tracking")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="./outputs/video")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--grid-overlap", type=float, default=DEFAULT_GRID_OVERLAP)
    parser.add_argument("--tile-imgsz", type=int, default=DEFAULT_TILE_IMGSZ)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--preview-frames", type=int, default=0, help="Frames to annotate in video (0 = all processed frames)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = all)")
    parser.add_argument("--no-topdown-assist", action="store_true")
    args = parser.parse_args()

    run(
        video_path=args.video,
        out_dir=args.out,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        use_grid=not args.no_grid,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        grid_overlap=args.grid_overlap,
        tile_imgsz=args.tile_imgsz,
        stride=args.stride,
        preview_frames=args.preview_frames,
        max_frames=args.max_frames,
        topdown_assist=not args.no_topdown_assist,
    )
