"""
VisionX Visualizer & Frame Annotation Engine
Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np


def draw_detection_frame(
    frame: np.ndarray,
    points: List[Dict[str, Any]],
    frame_num: int,
    pitch_poly: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Renders raw per-frame player detection boxes, foot points, and tags."""
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


def draw_tracking_frame(
    frame: np.ndarray,
    frame_results: List[Dict[str, Any]],
    frame_num: int,
    pitch_poly: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Renders persistent ID tracked bounding boxes, ground centroids, and tags."""
    vis = frame.copy()
    if pitch_poly is not None:
        cv2.polylines(vis, [pitch_poly], True, (0, 220, 255), 2)

    for det in frame_results:
        tid = det["track_id"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        fx, fy = map(int, det["foot_point"])
        conf = det["conf"]

        # Deterministic distinct color per track_id
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
