"""
VisionX Multi-Object Player Tracker (ByteTrack + Sliced Grid)
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Union
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from visionx.config import (
    DEFAULT_WEIGHTS,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_OVERLAP,
    DEFAULT_TILE_IMGSZ,
    TRACKER_HIGH_THRESH,
    TRACKER_LOW_THRESH,
    TRACKER_NEW_THRESH,
    TRACKER_BUFFER,
    TRACKER_MATCH_THRESH,
    TRACKER_FUSE_SCORE,
)
from visionx.core.detector import load_model, detect_persons
from visionx.core.pitch import get_pitch_polygon
from visionx.utils.geometry import to_foot_point
from visionx.utils.video import transcode_to_h264
from visionx.visualizer.draw import draw_tracking_frame
from visionx.visualizer.plot import plot_track_count


def create_tracker(
    high_thresh: float = TRACKER_HIGH_THRESH,
    low_thresh: float = TRACKER_LOW_THRESH,
    new_thresh: float = TRACKER_NEW_THRESH,
    buffer: int = TRACKER_BUFFER,
    match_thresh: float = TRACKER_MATCH_THRESH,
    fuse_score: bool = TRACKER_FUSE_SCORE,
) -> BYTETracker:
    """Instantiates a BYTETracker tuned for aerial top-down player tracking."""
    tracker_args = SimpleNamespace(
        track_high_thresh=high_thresh,
        track_low_thresh=low_thresh,
        new_track_thresh=new_thresh,
        track_buffer=buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
    )
    return BYTETracker(tracker_args)


def run_video_tracking(
    video_path: Union[str, Path],
    out_dir: Union[str, Path] = "./outputs/video",
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
) -> Dict[str, Any]:
    """
    Executes end-to-end multi-object tracking over drone footage:
    - Reads video frames and computes pitch boundary
    - Runs batched Sliced Grid detection
    - Updates persistent tracks with ByteTrack
    - Annotates and compiles output video + H.264 browser transcode
    - Writes trajectories JSON and active track validation plot
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = video_path.stem

    print(f"Tracking: {base} | {W}x{H} @ {fps_vid:.1f}fps | {total_frames} frames | Sliced Grid={use_grid}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame of video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pitch_poly = get_pitch_polygon(first_frame)
    model = load_model(weights)
    tracker = create_tracker()

    preview_path = out_dir / f"{base}_tracks_preview.mp4"
    preview_h264 = out_dir / f"{base}_tracks_preview_h264.mp4"
    preview_writer = None

    tracks = {}
    per_frame = {}
    processed = 0
    frame_num = 0
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

        dets = detect_persons(
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

        frame_dets = []

        if len(dets) > 0:
            b_tensor = torch.tensor(dets, dtype=torch.float32)
            b_obj = Boxes(b_tensor, orig_shape=(H, W))
            tracked_results = tracker.update(b_obj)

            if tracked_results is not None and len(tracked_results) > 0:
                for trk in tracked_results:
                    x1, y1, x2, y2 = trk[:4]
                    tid = int(trk[4])
                    score = float(trk[5])
                    foot_pt = to_foot_point(x1, y1, x2, y2)

                    det_dict = {
                        "track_id": tid,
                        "foot_point": foot_pt,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": score,
                    }
                    frame_dets.append(det_dict)

                    tid_str = str(tid)
                    if tid_str not in tracks:
                        tracks[tid_str] = []
                    tracks[tid_str].append({
                        "frame": frame_num,
                        "foot_point": foot_pt,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": score,
                    })

        per_frame[str(frame_num)] = frame_dets

        # Write to video if preview_frames <= 0 (all) or frame_num < preview_frames
        if preview_frames <= 0 or frame_num < preview_frames:
            annotated = draw_tracking_frame(frame, frame_dets, frame_num, pitch_poly)
            if preview_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(str(preview_path), fourcc, fps_vid, (W, H))
            preview_writer.write(annotated)

        processed += 1
        if processed % 30 == 0:
            elapsed = time.time() - t0
            infer_fps = processed / elapsed if elapsed > 0 else 0
            active_ids = len(frame_dets)
            print(f"  Frame {frame_num}/{total_frames} | {processed} processed | "
                  f"{infer_fps:.1f} FPS | {active_ids} active tracks")

        frame_num += 1

    cap.release()
    if preview_writer:
        preview_writer.release()

    elapsed_total = time.time() - t0

    # 1. Save Tracks JSON
    json_path = out_dir / f"{base}_tracks.json"
    output_data = {
        "meta": {
            "video": str(video_path),
            "resolution": [W, H],
            "fps": fps_vid,
            "total_frames": total_frames,
            "processed_frames": processed,
            "stride": stride,
            "elapsed_seconds": round(elapsed_total, 2),
            "unique_track_ids": len(tracks),
        },
        "tracks": tracks,
        "per_frame": per_frame,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # 2. Transcode to Web H.264
    if preview_path.exists():
        try:
            transcode_to_h264(preview_path, preview_h264)
            print(f"  Browser H.264 Preview -> {preview_h264}")
        except Exception as e:
            print(f"  Warning: H.264 transcoding skipped: {e}")

    # 3. Track Count Validation Chart
    chart_path = out_dir / f"{base}_tracks_count.png"
    plot_track_count(per_frame, str(chart_path))

    print(f"\nDone. {processed} frames | {len(tracks)} unique track IDs "
          f"| {elapsed_total:.1f}s ({processed / elapsed_total:.1f} FPS)")
    print(f"  Tracks JSON    -> {json_path}")
    if preview_path.exists():
        print(f"  Preview video  -> {preview_path}")
    print(f"  Track chart    -> {chart_path}")

    return {
        "json_path": str(json_path),
        "preview_video": str(preview_path),
        "preview_h264": str(preview_h264),
        "chart_path": str(chart_path),
        "processed_frames": processed,
        "unique_track_ids": len(tracks),
    }
