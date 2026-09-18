"""
VisionX Configuration & Default Hyperparameters
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

# COCO Target & Overhead Assist Classes
PERSON_CLASS_ID = 0
OVERHEAD_ASSIST_CLASSES = [14, 32, 33]  # bird, sports ball, kite in COCO

# Sliced Grid Parameters
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 3
DEFAULT_GRID_OVERLAP = 0.20
DEFAULT_TILE_IMGSZ = 800

# Detection Thresholds
DEFAULT_CONF = 0.08
DEFAULT_IOU = 0.25
DEFAULT_WEIGHTS = "yolov8s.pt"

# Aerial Player Scale Filters (in 4K pixels)
MIN_PLAYER_DIM = 8
MAX_PLAYER_DIM = 65

# Pitch Boundary Proportions (for standard 4K overhead football footage)
PITCH_X_MIN_RATIO = 0.108  # ~415px on 3840w
PITCH_X_MAX_RATIO = 0.896  # ~3440px on 3840w
PITCH_Y_MIN_RATIO = 0.055  # ~118px on 2160h
PITCH_Y_MAX_RATIO = 0.935  # ~2020px on 2160h

# ByteTracker Tuned Hyperparameters for Aerial View
TRACKER_HIGH_THRESH = 0.10
TRACKER_LOW_THRESH = 0.04
TRACKER_NEW_THRESH = 0.08
TRACKER_BUFFER = 60
TRACKER_MATCH_THRESH = 0.85
TRACKER_FUSE_SCORE = False
