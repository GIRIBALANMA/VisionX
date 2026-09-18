"""
VisionX Video Utilities (FFmpeg / FFprobe)
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Union


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract video metadata (width, height, fps, duration, frames, size_mb)
    using ffprobe.
    """
    video_path = Path(video_path)
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
            "size_mb": size_mb,
        }
    except Exception as e:
        return {
            "width": 0, "height": 0, "fps": 30.0, "duration": 0.0, "frames": 0,
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else 0,
            "error": str(e),
        }


def transcode_to_h264(input_path: Union[str, Path], output_path: Union[str, Path]):
    """
    Fast transcode an OpenCV MP4 video to browser-compliant H.264 (yuv420p)
    using ffmpeg.
    """
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
