"""
Stage 3 — Sliced Grid Multi-Object Tracking (ByteTrack)
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)

Input:  video file (e.g. inputs/video/Video.mp4)
Output: 
    - outputs/video/<name>_tracks.json             (per-track foot-point trajectories)
    - outputs/video/<name>_tracks_preview.mp4      (annotated video with track IDs)
    - outputs/video/<name>_tracks_preview_h264.mp4 (browser-playable H.264 preview)
    - outputs/video/<name>_tracks_count.png        (active tracks validation plot)
"""

import argparse
import json
import os
import subprocess
import time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from person import load_model, detect_persons, get_pitch_polygon


def to_foot_point(x1, y1, x2, y2):
    """Centroid ground contact point for top-down camera views."""
    return [float((x1 + x2) / 2), float((y1 + y2) / 2)]


def draw_tracking_frame(frame, frame_results, frame_num, pitch_poly=None):
    """Draw tracked bounding boxes with persistent ID colours and field boundary."""
    vis = frame.copy()
    if pitch_poly is not None:
        cv2.polylines(vis, [pitch_poly], True, (0, 220, 255), 2)

    for det in frame_results:
        tid = det["track_id"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        fx, fy = map(int, det["foot_point"])
        conf = det["conf"]

        # Consistent colour per track_id
        seed_rng = np.random.default_rng(tid * 31337)
        colour = tuple(int(c) for c in seed_rng.integers(80, 255, 3))

        cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), colour, 2)
        cv2.circle(vis, (fx, fy), 4, (0, 0, 255), -1)
        cv2.circle(vis, (fx, fy), 6, (255, 255, 255), 1)

        label = f"T{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(y1 - 5, th + 2)
        cv2.rectangle(vis, (x1 - 3, tag_y - th - 2), (x1 + tw + 4, tag_y + 2), colour, -1)
        cv2.putText(vis, label, (x1 + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    active_ids = len(set(d["track_id"] for d in frame_results))
    cv2.putText(vis, f"Frame {frame_num} | {active_ids} active tracks (VisionX Sliced Grid)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def plot_track_count(per_frame, out_path):
    """Plot number of active track_ids per frame — validation chart."""
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
    print(f"  Track count chart -> {out_path}")


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
    preview_frames: int = 0,
    max_frames: int = 0,
    topdown_assist: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Tracking: {base} | {W}x{H} @ {fps_vid:.1f}fps | {total_frames} frames | Sliced Grid={use_grid}")

    # Read first frame to compute pitch boundary once
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame of video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Computing pitch boundary from first frame...", end=" ", flush=True)
    pitch_poly = get_pitch_polygon(first_frame)
    print("done")

    model = load_model(weights)

    # Initialize ByteTracker with association thresholds tuned for aerial view
    tracker_args = SimpleNamespace(
        track_high_thresh=0.10,
        track_low_thresh=0.04,
        new_track_thresh=0.08,
        track_buffer=60,
        match_thresh=0.85,
        fuse_score=False,
    )
    tracker = BYTETracker(tracker_args)

    # Preview writer
    preview_path = os.path.join(out_dir, f"{base}_tracks_preview.mp4")
    preview_h264 = os.path.join(out_dir, f"{base}_tracks_preview_h264.mp4")
    preview_writer = None

    tracks = {}        # {track_id_str: [{"frame", "foot_point", "bbox", "conf"}, ...]}
    per_frame = {}     # {frame_num_str: [{"track_id", "foot_point", "bbox", "conf"}, ...]}
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

        # Sliced Grid detection on current frame
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

        # Preview video (if preview_frames <= 0, write all processed frames)
        if preview_frames <= 0 or frame_num < preview_frames:
            annotated = draw_tracking_frame(frame, frame_dets, frame_num, pitch_poly)
            if preview_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(preview_path, fourcc, fps_vid, (W, H))
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

    # Save tracks JSON
    json_path = os.path.join(out_dir, f"{base}_tracks.json")
    output = {
        "meta": {
            "video": video_path,
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
        json.dump(output, f, indent=2)

    # Transcode preview for browser streaming
    if os.path.exists(preview_path):
        try:
            transcode_to_h264(preview_path, preview_h264)
            print(f"  Browser H.264 Preview -> {preview_h264}")
        except Exception as e:
            print(f"  Warning: H.264 transcoding skipped: {e}")

    # Validation chart
    chart_path = os.path.join(out_dir, f"{base}_tracks_count.png")
    plot_track_count(per_frame, chart_path)

    print(f"\nDone. {processed} frames | {len(tracks)} unique track IDs "
          f"| {elapsed_total:.1f}s ({processed/elapsed_total:.1f} FPS)")
    print(f"  Tracks JSON    -> {json_path}")
    if os.path.exists(preview_path):
        print(f"  Preview video  -> {preview_path}")
    print(f"  Track chart    -> {chart_path}")

    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 - Sliced Grid ByteTrack player tracking")
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
        preview_frames=args.preview_frames,
        max_frames=args.max_frames,
        topdown_assist=not args.no_topdown_assist,
    )
