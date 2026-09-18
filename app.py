"""
VisionX Studio - Minimalist Video Comparison & Pipeline Runner Server
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import os
import sys
import json
import time
import uuid
import glob
import subprocess
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template, abort

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs" / "video"
OUTPUTS_VIDEO_DIR = BASE_DIR / "outputs" / "video"
OUTPUTS_HEATMAPS_DIR = BASE_DIR / "outputs" / "heatmaps"
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"

# Ensure directories exist
OUTPUTS_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job tracker
jobs = {}

def get_video_metadata(video_path: Path):
    """Extract metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(res.stdout)
        stream = data.get("streams", [{}])[0]
        
        # Calculate fps
        r_fps = stream.get("r_frame_rate", "30/1")
        if "/" in r_fps:
            num, den = map(float, r_fps.split("/"))
            fps = round(num / den, 2) if den > 0 else 30.0
        else:
            fps = float(r_fps)
            
        duration = float(stream.get("duration", 0.0))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        frames = int(stream.get("nb_frames", int(duration * fps)))
        
        size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": round(duration, 2),
            "frames": frames,
            "size_mb": size_mb
        }
    except Exception as e:
        return {
            "width": 0, "height": 0, "fps": 30.0, "duration": 0.0, "frames": 0,
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "error": str(e)
        }

def transcode_to_h264(input_path: Path, output_path: Path):
    """Fast transcode video to browser-playable H.264 (yuv420p)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos", methods=["GET"])
def list_videos():
    """List all available input videos with metadata and existing outputs."""
    video_files = sorted(INPUTS_DIR.glob("*.mp4"))
    results = []
    
    for v in video_files:
        meta = get_video_metadata(v)
        base_name = v.stem
        
        # Check existing outputs
        track_json = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks.json"
        track_preview_h264 = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks_preview_h264.mp4"
        track_preview_raw = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks_preview.mp4"
        detect_preview_h264 = OUTPUTS_VIDEO_DIR / f"{base_name}_preview_h264.mp4"
        detect_preview_raw = OUTPUTS_VIDEO_DIR / f"{base_name}_preview.mp4"
        
        # Ensure H.264 previews exist if raw previews exist
        if track_preview_raw.exists() and not track_preview_h264.exists():
            try:
                transcode_to_h264(track_preview_raw, track_preview_h264)
            except Exception:
                pass
                
        if detect_preview_raw.exists() and not detect_preview_h264.exists():
            try:
                transcode_to_h264(detect_preview_raw, detect_preview_h264)
            except Exception:
                pass
                
        heatmaps = sorted(OUTPUTS_HEATMAPS_DIR.glob("*.png"))
        has_heatmaps = len(heatmaps) > 0

        outputs = {
            "tracks_json": track_json.exists(),
            "tracks_preview": track_preview_h264.exists(),
            "tracks_preview_file": track_preview_h264.name if track_preview_h264.exists() else None,
            "detect_preview": detect_preview_h264.exists(),
            "detect_preview_file": detect_preview_h264.name if detect_preview_h264.exists() else None,
            "has_heatmaps": has_heatmaps
        }

        results.append({
            "filename": v.name,
            "base_name": base_name,
            "metadata": meta,
            "outputs": outputs
        })
        
    return jsonify({"videos": results})


@app.route("/video/source/<path:filename>")
def stream_source_video(filename):
    """Stream input video with HTTP 206 Range support."""
    file_path = INPUTS_DIR / filename
    if not file_path.exists():
        abort(404, "Source video not found")
    return send_file(str(file_path), mimetype="video/mp4", conditional=True)


@app.route("/video/output/<path:filename>")
def stream_output_video(filename):
    """Stream output video with HTTP 206 Range support."""
    file_path = OUTPUTS_VIDEO_DIR / filename
    if not file_path.exists():
        abort(404, "Output video not found")
    return send_file(str(file_path), mimetype="video/mp4", conditional=True)


@app.route("/api/heatmaps")
def list_heatmaps():
    """List generated heatmap images."""
    heatmaps = sorted(OUTPUTS_HEATMAPS_DIR.glob("*.png"))
    items = []
    for h in heatmaps:
        items.append({
            "name": h.stem.replace("_", " ").title(),
            "filename": h.name,
            "size_kb": round(h.stat().st_size / 1024, 1)
        })
    return jsonify({"heatmaps": items})


@app.route("/heatmaps/<path:filename>")
def get_heatmap_image(filename):
    """Serve a heatmap image."""
    file_path = OUTPUTS_HEATMAPS_DIR / filename
    if not file_path.exists():
        abort(404, "Heatmap not found")
    return send_file(str(file_path), mimetype="image/png")


def run_pipeline_worker(job_id: str, video_name: str, mode: str, max_frames: int, conf: float):
    """Background worker that runs detection/tracking and generates web outputs."""
    job = jobs[job_id]
    job["status"] = "running"
    video_path = INPUTS_DIR / video_name
    base_name = video_path.stem

    try:
        # Determine script and flags
        if mode == "detect":
            script = "video_detect.py"
            cmd = [
                str(PYTHON_BIN), script,
                "--video", str(video_path),
                "--out", "outputs/video",
                "--conf", str(conf)
            ]
            if max_frames > 0:
                cmd.extend(["--max-frames", str(max_frames)])
        else: # "track" or "full"
            script = "video_track.py"
            cmd = [
                str(PYTHON_BIN), script,
                "--video", str(video_path),
                "--out", "outputs/video",
                "--conf", str(conf)
            ]
            if max_frames > 0:
                cmd.extend(["--max-frames", str(max_frames)])

        job["logs"].append(f"Starting {mode} pipeline: {' '.join(cmd)}")
        
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(proc.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                job["logs"].append(clean_line)
                if len(job["logs"]) > 100:
                    job["logs"] = job["logs"][-100:]
                
                # Parse progress e.g. "Frame 29/900 | 30 processed | 7.1 FPS | 5 active tracks"
                if "Frame" in clean_line and "/" in clean_line:
                    try:
                        parts = clean_line.split("|")
                        frame_part = parts[0].strip().replace("Frame", "").strip()
                        cur_frame, total_frame = map(int, frame_part.split("/"))
                        job["current_frame"] = cur_frame
                        job["total_frames"] = total_frame
                        job["progress"] = min(99, int((cur_frame / total_frame) * 100))
                        
                        for p in parts[1:]:
                            if "FPS" in p:
                                job["fps"] = p.strip()
                            elif "active tracks" in p:
                                job["active_tracks"] = p.strip()
                            elif "players" in p:
                                job["players"] = p.strip()
                    except Exception:
                        pass
                        
        proc.wait()
        
        if proc.returncode != 0:
            job["status"] = "error"
            job["error"] = f"Process exited with code {proc.returncode}"
            return

        # Fast H.264 transcode for browser preview
        raw_preview = None
        h264_preview = None
        if mode == "detect":
            raw_preview = OUTPUTS_VIDEO_DIR / f"{base_name}_preview.mp4"
            h264_preview = OUTPUTS_VIDEO_DIR / f"{base_name}_preview_h264.mp4"
        else:
            raw_preview = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks_preview.mp4"
            h264_preview = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks_preview_h264.mp4"

        if raw_preview.exists():
            job["logs"].append("Transcoding preview video to web-native H.264...")
            transcode_to_h264(raw_preview, h264_preview)
            job["logs"].append("Transcoding complete.")
            job["output_video"] = h264_preview.name

        # If mode is full or track, generate heatmaps
        if mode in ("track", "full"):
            tracks_json = OUTPUTS_VIDEO_DIR / f"{base_name}_tracks.json"
            if tracks_json.exists():
                job["logs"].append("Generating movement heatmaps...")
                frame0_path = OUTPUTS_VIDEO_DIR / f"{base_name}_frame0.jpg"
                if not frame0_path.exists():
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(video_path),
                        "-vframes", "1", "-q:v", "2", str(frame0_path)
                    ], capture_output=True)
                
                heatmap_cmd = [
                    str(PYTHON_BIN), "heatmap.py",
                    "--tracks", str(tracks_json),
                    "--out", str(OUTPUTS_HEATMAPS_DIR),
                    "--sigma", "35",
                    "--top-n", "6"
                ]
                if frame0_path.exists():
                    heatmap_cmd.extend(["--bg", str(frame0_path)])
                    
                subprocess.run(heatmap_cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
                job["logs"].append("Heatmaps generated successfully.")

        job["progress"] = 100
        job["status"] = "completed"
        job["logs"].append("Pipeline finished successfully!")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["logs"].append(f"Error: {str(e)}")


@app.route("/api/run", methods=["POST"])
def trigger_pipeline():
    """Trigger pipeline execution on a selected video."""
    data = request.get_json() or {}
    video_name = data.get("video")
    mode = data.get("mode", "track") # track, detect, full
    max_frames = int(data.get("max_frames", 150)) # 150 for 5s preview, 0 for all
    conf = float(data.get("conf", 0.12))

    if not video_name:
        return jsonify({"error": "No video specified"}), 400
        
    video_path = INPUTS_DIR / video_name
    if not video_path.exists():
        return jsonify({"error": f"Video {video_name} not found"}), 404

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "video": video_name,
        "mode": mode,
        "status": "queued",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": "0",
        "active_tracks": "0",
        "players": "0",
        "output_video": None,
        "logs": [],
        "created_at": time.time()
    }

    t = threading.Thread(
        target=run_pipeline_worker,
        args=(job_id, video_name, mode, max_frames, conf),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id, "message": "Pipeline initiated"})


@app.route("/api/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Query progress and logs of an active or finished job."""
    if job_id not in jobs:
        abort(404, "Job not found")
    return jsonify(jobs[job_id])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting VisionX Web Studio on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
