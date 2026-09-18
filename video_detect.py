"""
VisionX CLI: Per-Frame Video Player Detection (Sliced Grid)
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import argparse
import json
import time
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
from visionx.core.detector import load_model, detect_persons
from visionx.core.pitch import get_pitch_polygon
from visionx.utils.geometry import to_foot_points
from visionx.utils.video import transcode_to_h264
from visionx.visualizer.draw import draw_detection_frame


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
    topdown_assist: bool = True,
    preview_frames: int = 0,
    max_frames: int = 0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = video_path.stem

    print(f"Video: {base} | {W}x{H} @ {fps:.1f}fps | {total_frames} frames | Sliced Grid={use_grid}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pitch_poly = get_pitch_polygon(first_frame)
    model = load_model(weights)

    preview_path = out_dir / f"{base}_preview.mp4"
    preview_h264 = out_dir / f"{base}_preview_h264.mp4"
    preview_writer = None

    all_detections = {}
    frame_num = 0
    processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_num >= max_frames:
            break

        if frame_num % stride != 0:
            frame_num += 1
            continue

        boxes = detect_persons(
            model=model,
            image=frame,
            conf=conf,
            iou=iou,
            use_grid=use_grid,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            grid_overlap=grid_overlap,
            tile_imgsz=tile_imgsz,
            topdown_assist=topdown_assist,
            pitch_poly=pitch_poly,
        )

        points = to_foot_points(boxes)
        all_detections[str(frame_num)] = points

        if preview_frames <= 0 or frame_num < preview_frames:
            annotated = draw_detection_frame(frame, points, frame_num, pitch_poly)
            if preview_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (W, H))
            preview_writer.write(annotated)

        processed += 1
        if processed % 30 == 0:
            elapsed = time.time() - t0
            infer_fps = processed / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_num}/{total_frames} | {processed} processed | "
                  f"{infer_fps:.1f} FPS | {len(points)} players")

        frame_num += 1

    cap.release()
    if preview_writer:
        preview_writer.release()

    elapsed_total = time.time() - t0

    json_path = out_dir / f"{base}_detections_raw.json"
    meta = {
        "video": str(video_path),
        "resolution": [W, H],
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed,
        "stride": stride,
        "elapsed_seconds": round(elapsed_total, 2),
        "frames": all_detections,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    if preview_path.exists():
        try:
            transcode_to_h264(preview_path, preview_h264)
            print(f"  Browser H.264 Preview -> {preview_h264}")
        except Exception as e:
            print(f"  Warning: H.264 transcoding skipped: {e}")

    print(f"\nDone. {processed} frames in {elapsed_total:.1f}s ({processed / elapsed_total:.1f} FPS)")
    print(f"  Detections JSON -> {json_path}")
    if preview_path.exists():
        print(f"  Preview video   -> {preview_path}")

    return str(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 - Sliced Grid Video Player Detection")
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
        topdown_assist=not args.no_topdown_assist,
        preview_frames=args.preview_frames,
        max_frames=args.max_frames,
    )
