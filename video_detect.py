"""
Stage 2 — Sliced Grid Per-frame Video Player Detection
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)

Input:  video file (e.g. inputs/video/Video.mp4)
Output: 
    - outputs/video/<name>_detections_raw.json (per-frame {bbox, foot_point, conf})
    - outputs/video/<name>_preview.mp4         (first N frames annotated)
    - outputs/video/<name>_preview_h264.mp4    (browser-playable H.264 preview)
"""

import argparse
import json
import os
import subprocess
import time
import cv2
import numpy as np

from person import load_model, detect_persons, get_pitch_polygon, to_foot_points


def draw_frame_detections(frame, points, frame_num, pitch_poly=None):
    vis = frame.copy()
    if pitch_poly is not None:
        cv2.polylines(vis, [pitch_poly], True, (0, 220, 255), 2)

    for i, p in enumerate(points):
        x1, y1, x2, y2 = map(int, p["bbox"])
        fx, fy = map(int, p["foot_point"])
        conf = p["confidence"]

        cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2)
        cv2.circle(vis, (fx, fy), 4, (0, 0, 255), -1)

        label = f"P{i+1} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(y1 - 4, th + 2)
        cv2.rectangle(vis, (x1 - 3, tag_y - th - 2), (x1 + tw + 4, tag_y + 2), (0, 160, 0), -1)
        cv2.putText(vis, label, (x1 + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, f"Frame {frame_num} | {len(points)} players (VisionX Sliced Grid)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def transcode_to_h264(input_path: str, output_path: str):
    """Fast transcode video to browser-playable H.264 (yuv420p)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def run(
    video_path: str,
    out_dir: str = "./outputs/video",
    weights: str = "yolov8s.pt",
    conf: float = 0.08,
    iou: float = 0.25,
    use_grid: bool = True,
    grid_cols: int = 4,
    grid_rows: int = 3,
    grid_overlap: float = 0.20,
    tile_imgsz: int = 800,
    stride: int = 1,
    topdown_assist: bool = True,
    preview_frames: int = 0,
    max_frames: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Video: {base} | {W}x{H} @ {fps:.1f}fps | {total_frames} frames | Sliced Grid={use_grid}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Computing pitch boundary from first frame...", end=" ", flush=True)
    pitch_poly = get_pitch_polygon(first_frame)
    print("done")

    model = load_model(weights)

    preview_path = os.path.join(out_dir, f"{base}_preview.mp4")
    preview_h264 = os.path.join(out_dir, f"{base}_preview_h264.mp4")
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

        # Preview video (write all frames when preview_frames <= 0)
        if preview_frames <= 0 or frame_num < preview_frames:
            annotated = draw_frame_detections(frame, points, frame_num, pitch_poly)
            if preview_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (W, H))
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

    json_path = os.path.join(out_dir, f"{base}_detections_raw.json")
    meta = {
        "video": video_path,
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

    # Transcode preview for browser streaming
    if os.path.exists(preview_path):
        try:
            transcode_to_h264(preview_path, preview_h264)
            print(f"  Browser H.264 Preview -> {preview_h264}")
        except Exception as e:
            print(f"  Warning: H.264 transcoding skipped: {e}")

    print(f"\nDone. {processed} frames in {elapsed_total:.1f}s "
          f"({processed/elapsed_total:.1f} FPS)")
    print(f"  Detections JSON -> {json_path}")
    if os.path.exists(preview_path):
        print(f"  Preview video   -> {preview_path}")

    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 - Sliced Grid Video Player Detection")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="./outputs/video")
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-overlap", type=float, default=0.20)
    parser.add_argument("--tile-imgsz", type=int, default=800)
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
