"""
VisionX CLI: Single-Frame Person Detection
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import argparse
import json
import os
from pathlib import Path
import cv2

from visionx.config import (
    DEFAULT_WEIGHTS,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_OVERLAP,
    DEFAULT_TILE_IMGSZ,
)
from visionx.core.detector import load_model, detect_persons, detect_persons_grid
from visionx.core.grid import slice_frame_grid
from visionx.core.pitch import get_pitch_polygon, is_inside_pitch
from visionx.utils.geometry import to_foot_point, to_foot_points
from visionx.visualizer.draw import draw_detection_frame


def draw_and_save(image, points, out_path):
    """Renders and writes annotated image with player tags."""
    vis = draw_detection_frame(image, points, frame_num=0)
    cv2.imwrite(str(out_path), vis)


def run(
    image_path: str,
    out_dir: str = "./outputs",
    weights: str = DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = 1280,
    use_grid: bool = True,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_overlap: float = DEFAULT_GRID_OVERLAP,
    tile_imgsz: int = DEFAULT_TILE_IMGSZ,
    topdown_assist: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(weights)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    boxes = detect_persons(
        model=model,
        image=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        use_grid=use_grid,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        grid_overlap=grid_overlap,
        tile_imgsz=tile_imgsz,
        topdown_assist=topdown_assist,
    )

    points = to_foot_points(boxes)

    base = Path(image_path).stem
    out_img = out_dir / f"{base}_annotated.jpg"
    out_json = out_dir / f"{base}_detections.json"

    draw_and_save(image, points, out_img)

    result = {
        "image": image_path,
        "total_players": len(points),
        "detections": points,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Detected {len(points)} player(s) using Sliced Grid Inference.")
    print(f"Annotated image  -> {out_img}")
    print(f"Coordinates JSON -> {out_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 - Sliced Grid Person Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="./outputs", help="Output directory")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid slicing")
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS, help="Grid columns")
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_ROWS, help="Grid rows")
    parser.add_argument("--grid-overlap", type=float, default=DEFAULT_GRID_OVERLAP, help="Grid tile overlap ratio")
    parser.add_argument("--tile-imgsz", type=int, default=DEFAULT_TILE_IMGSZ, help="Tile inference resolution")
    parser.add_argument("--no-topdown-assist", action="store_true", help="Disable overhead assist classes")
    args = parser.parse_args()

    run(
        image_path=args.image,
        out_dir=args.out,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        use_grid=not args.no_grid,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        grid_overlap=args.grid_overlap,
        tile_imgsz=args.tile_imgsz,
        topdown_assist=not args.no_topdown_assist,
    )