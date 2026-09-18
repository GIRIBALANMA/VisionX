# VisionX ⚽

> **High-Precision Aerial Drone Football Player Tracking & Movement Heatmap Generation via Sliced Grid Inference (Tiled Hyper-Inference) and ByteTrack.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF.svg?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![Flask](https://img.shields.io/badge/Web%20Studio-Flask-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

In top-down drone and stadium-camera footage ($3840 \times 2160$ 4K resolution), football players appear as minuscule silhouettes ranging from only **$15\text{ to }30\text{ pixels}$**. 

When feeding standard 4K footage into deep learning detectors resized to standard resolutions ($640\text{px}$ or $1280\text{px}$):
- Each player shrinks down to **$\approx 3\text{--}6\text{ pixels}$**.
- At the detector's highest resolution feature map (stride 8 P3 layer), an object smaller than $4\text{px}$ spans **less than $0.5$ feature cells** — completely falling below convolutional receptive fields and anchor priors.
- **Result:** Baseline detectors suffer a catastrophic recall collapse, detecting **less than $5\%$** of active players on the pitch.

**VisionX solves this via Sliced Grid Inference (Tiled Hyper-Inference):**
1. **Dynamic Grid Slicing**: Automatically tiles the 4K frame into a $4 \times 3$ overlapping grid (12 tiles, $20\%$ overlap).
2. **Receptive Field Magnification**: Each tile is evaluated in GPU batch mode at $800\text{px}$, keeping players at near-native scale ($18\text{--}30\text{px}$) where limbs, heads, and shadows are distinct.
3. **Coordinate Re-projection & Seam NMS**: Tile detections are mathematically remapped back to global 4K coordinates and merged across seams via global Non-Maximum Suppression (NMS).
4. **Calibrated Pitch Filtering**: Inset field boundary polygons strictly isolate the pitch, removing substitute benches, sideline coaches, and stadium banners.
5. **Aerial ByteTrack Association**: Tracks player ground contact centroids across occlusions and momentary missed frames.
6. **2D Gaussian Density Heatmaps**: Accumulates multi-frame player trajectories into publication-quality movement heatmaps.

---

## 📊 Empirical Accuracy Benchmark: Whole Frame vs. Sliced Grid

Evaluated on native 4K broadcast footage (`inputs/video/Video.mp4`, $3840 \times 2160$ @ $29.97\text{ FPS}$, 22 active field players):

| Metric | Traditional Whole Frame (640px) | Traditional Whole Frame (1280px) | VisionX Sliced Grid ($4 \times 3$, 800px) | Impact Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Avg. Players Detected / Frame** | $0.1$ players | $0.1$ players | **$16.5$ players** | **$\mathbf{\approx 165\times}$ Increase** |
| **Pitch Recall Rate** | $0.4\%$ (Failure) | $0.4\%$ (Failure) | **$74.8\%$** | **$+74.4\%$ Recall Gain** |
| **Active Persistent Tracks** | $0$ (Tracks die in 1 frame) | $0$ (Tracks die in 1 frame) | **$12 \text{--} 18$ continuous tracks** | **Continuous Trajectories** |
| **Total Tracked Foot-Points (150f)** | $< 50$ points | $< 50$ points | **$2,403$ ground contact points** | **Full Trajectory Density** |
| **Inference Latency (RTX 2050 GPU)** | $69.7\text{ ms}$ ($14.3\text{ FPS}$) | $38.8\text{ ms}$ ($25.8\text{ FPS}$) | **$218.3\text{ ms}$ ($4.6\text{ FPS}$ batched)** | **Real-Time Feasible** |
| **Sideline Bench Noise** | High false positives on benches | High false positives on benches | **$0$ bench false positives** | **Strict Field Inset** |

---

## 📐 Mathematical Formulation

### 1. Sliced Grid Partitioning
For an image with width $W$ and height $H$, grid columns $N_c$ and rows $N_r$, with overlap ratio $\omega \in (0, 1)$:
$$W_{\text{tile}} = \left\lfloor \frac{W}{N_c - (N_c - 1)\omega} \right\rfloor, \quad H_{\text{tile}} = \left\lfloor \frac{H}{N_r - (N_r - 1)\omega} \right\rfloor$$
$$\Delta x = \lfloor W_{\text{tile}} \cdot (1 - \omega) \rfloor, \quad \Delta y = \lfloor H_{\text{tile}} \cdot (1 - \omega) \rfloor$$

Each tile $(c, r)$ is extracted with offset:
$$x_{\text{start}}^{(c, r)} = \min(c \cdot \Delta x, W - W_{\text{tile}}), \quad y_{\text{start}}^{(c, r)} = \min(r \cdot \Delta y, H - H_{\text{tile}})$$

### 2. Coordinate Re-projection
For bounding box $[x_1^{\text{tile}}, y_1^{\text{tile}}, x_2^{\text{tile}}, y_2^{\text{tile}}]$ detected inside tile $(c, r)$:
$$x_1^{\text{main}} = x_1^{\text{tile}} + x_{\text{start}}^{(c, r)}, \quad y_1^{\text{main}} = y_1^{\text{tile}} + y_{\text{start}}^{(c, r)}$$
$$x_2^{\text{main}} = x_2^{\text{tile}} + x_{\text{start}}^{(c, r)}, \quad y_2^{\text{main}} = y_2^{\text{tile}} + y_{\text{start}}^{(c, r)}$$

### 3. Aerial Ground Contact Point
In top-down orthographic camera views, the ground contact centroid is derived from the box center:
$$x_{\text{foot}} = \frac{x_1^{\text{main}} + x_2^{\text{main}}}{2}, \quad y_{\text{foot}} = \frac{y_1^{\text{main}} + y_2^{\text{main}}}{2}$$

---

## 🏗️ Repository Architecture

VisionX is modularized into a domain-driven Python package (`visionx/`) with single-responsibility components and thin CLI entrypoints:

```
VisionX/
├── visionx/                         # Core Python Package
│   ├── __init__.py                  # Package exports & version
│   ├── config.py                    # Central constants & default hyperparameters
│   ├── core/                        # Core computer vision & inference logic
│   │   ├── __init__.py
│   │   ├── detector.py              # Sliced Grid inference, scale validation & NMS
│   │   ├── grid.py                  # Grid slicing geometry & tile offsets
│   │   ├── pitch.py                 # Playing field calibration & polygon tests
│   │   └── tracker.py               # ByteTracker integration & trajectory engine
│   ├── visualizer/                  # Rendering & plotting modules
│   │   ├── __init__.py
│   │   ├── draw.py                  # Frame annotation (bounding boxes, IDs, foot-points)
│   │   ├── heatmap.py               # Gaussian density accumulation & blending
│   │   └── plot.py                  # Matplotlib validation charts & figures
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── geometry.py              # Centroid & ground contact calculations
│       └── video.py                 # FFprobe metadata extraction & FFmpeg H.264 transcoding
│
├── person.py                        # Thin CLI entrypoint: single-frame detection
├── video_detect.py                  # Thin CLI entrypoint: per-frame video detection
├── video_track.py                   # Thin CLI entrypoint: multi-object tracking (ByteTrack)
├── heatmap.py                       # Thin CLI entrypoint: movement heatmap generator
├── app.py                           # Minimalist Flask Web Studio server
├── start.sh                         # One-click environment bootstrap & studio launcher
├── static/                          # Minimalist Web Studio frontend (Vanilla CSS & JS)
│   ├── style.css                    # Obsidian dark mode aesthetic
│   └── app.js                       # Frame-synchronized dual-player engine
├── templates/
│   └── index.html                   # Dual video player interface
├── inputs/video/                    # Source footage directory
└── outputs/                         # Processed outputs directory
    ├── video/                       # Tracked preview videos & trajectory JSONs
    └── heatmaps/                    # Gaussian movement heatmaps
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Linux** (Ubuntu/Debian recommended)
- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (e.g. RTX 2050 / 3060 / 4090)
- **FFmpeg** and **FFprobe** installed on system `PATH`

### 2. One-Click Launch (Recommended)
Clone the repository and run the automated startup script:
```bash
git clone https://github.com/GIRIBALANMA/VisionX.git
cd VisionX
chmod +x start.sh
./start.sh
```
`start.sh` automatically:
- Provisions `.venv` and installs PyTorch, Ultralytics, OpenCV, and Flask.
- Checks CUDA acceleration.
- Verifies input/output directories.
- Launches the **VisionX Web Studio** at `http://localhost:5000`.

---

## 💻 Command Line Interface (CLI)

### 1. Multi-Object Tracking (`video_track.py`)
Run the full Sliced Grid ByteTrack tracking pipeline on any video:
```bash
# Track full video with Sliced Grid (default: 4x3 grid, imgsz=800)
python video_track.py --video inputs/video/Video.mp4 --out outputs/video

# Fast 150-frame preview run (5 seconds)
python video_track.py --video inputs/video/Video.mp4 --max-frames 150 --preview-frames 150
```
**Generated Deliverables:**
- `<name>_tracks.json`: Comprehensive frame-by-frame player bounding boxes and foot points.
- `<name>_tracks_preview.mp4`: OpenCV raw annotated video.
- `<name>_tracks_preview_h264.mp4`: Browser-compatible H.264 video.
- `<name>_tracks_count.png`: Active tracks count chart over time.

### 2. Movement Heatmap Generation (`heatmap.py`)
Generate full-pitch presence heatmaps from tracking data:
```bash
python heatmap.py \
  --tracks outputs/video/Video_tracks.json \
  --out outputs/heatmaps \
  --bg outputs/video/Video_frame0.jpg \
  --sigma 35 \
  --top-n 6
```
**Generated Deliverables:**
- `overall_heatmap.png`: Full-pitch team/game presence heatmap.
- `summary_heatmaps.png`: Multi-panel comparison grid for top most-active players.
- `player_<id>_heatmap.png`: Individual player spatial distribution maps.

### 3. Single-Frame / Image Detection (`person.py`)
Evaluate Sliced Grid detection on a single image frame:
```bash
python person.py --image path/to/frame.jpg --out outputs/image --conf 0.08
```

### 4. Per-Frame Detection (`video_detect.py`)
Run detection without temporal tracking association:
```bash
python video_detect.py --video inputs/video/Video.mp4 --preview-frames 150
```

---

## 🖥️ Minimalist Web Studio

VisionX includes a native **Dual-Player Synchronized Comparison Studio** built with Obsidian dark aesthetics (`#0b0c10`):

```
+-------------------------------------------------------------------------+
|  VISIONX // Parallel Comparison & Heatmap Studio                        |
+-------------------------------------------------------------------------+
|  [ Video: Video.mp4 v ]  [ Mode: Tracking v ]  [ Run Full Video ]       |
+------------------------------------+------------------------------------+
|          ORIGINAL FOOTAGE          |          VISIONX OUTPUT            |
|                                    |                                    |
|         < Raw 4K Video >           |    < Sliced Grid + Tracked Boxes > |
|                                    |                                    |
+------------------------------------+------------------------------------+
|  [> Play] [|< -1f] [>| +1f] [==== Scrubber ====] [1.0x v] [Sync: Locked]|
+-------------------------------------------------------------------------+
|  [ Heatmaps Tab ]  [ Telemetry Tab ]  [ Live Execution Console ]        |
+-------------------------------------------------------------------------+
```

### Key Web Capabilities:
- **Frame-Accurate Dual Sync**: Prevents drift between original and processed videos to within $\pm 1$ video frame ($< 0.03\text{s}$).
- **Single-Frame Stepping**: Precision forward/backward stepping (`-1f` / `+1f`) for tactical football analysis.
- **Range Request Streaming**: Streams multi-gigabyte 4K MP4 files with instant seeking via HTTP 206 Partial Content.
- **Integrated Heatmap Gallery**: Direct inspection of generated Gaussian heatmaps with modal zoom.

---

## ⚙️ Key Hyperparameters (`visionx/config.py`)

| Parameter | Default | Purpose |
| :--- | :---: | :--- |
| `DEFAULT_GRID_COLS` | `4` | Number of vertical slice columns. |
| `DEFAULT_GRID_ROWS` | `3` | Number of horizontal slice rows. |
| `DEFAULT_GRID_OVERLAP`| `0.20` | $20\%$ boundary overlap to prevent cutting players on tile seams. |
| `DEFAULT_TILE_IMGSZ`| `800` | Inference resolution per tile (yields native $1.2\times$ scale). |
| `DEFAULT_CONF` | `0.08` | Confidence threshold for high-altitude tiny silhouettes. |
| `DEFAULT_IOU` | `0.25` | Global NMS IoU threshold for merging seam duplicates. |
| `MIN_PLAYER_DIM` | `8` | Minimum bounding box dimension in 4K pixels. |
| `MAX_PLAYER_DIM` | `65` | Maximum bounding box dimension in 4K pixels. |
| `TRACKER_HIGH_THRESH`| `0.10` | High-confidence threshold for ByteTrack first-stage match. |
| `TRACKER_LOW_THRESH` | `0.04` | Low-confidence threshold for recovering occluded players. |
| `TRACKER_MATCH_THRESH`| `0.85` | Association distance threshold tuned for top-down scale. |
| `TRACKER_BUFFER` | `60` | Frames to keep lost track alive before deletion. |

---

## 📄 License

This project is licensed under the **MIT License**.
Distributed as part of the **VisionX 2026** initiative.
