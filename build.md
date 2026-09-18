# BUILD.md — Player Heatmap Pipeline

Status: Stage 1 (image person detection) — working.
Next: video → tracking → coordinates → heatmap.

---

## Step 1 — Image Detection (DONE)
- `person_detection.py`
- YOLOv8, person-class only, pitch-mask filter, foot-point extraction.
- Output: bbox + foot_point + confidence per image.

---

## Step 2 — Video Detection

Modify Step 1 to run per-frame instead of per-image.

**Changes needed:**
- Read video with `cv2.VideoCapture`, loop frame by frame.
- Run `detect_persons()` on each frame (same model, same conf/iou).
- Keep pitch-mask filter, but compute the pitch contour **once** (first frame or every N frames) — recomputing every frame is wasteful since camera is fixed.
- Store detections per frame: `{frame_num: [{bbox, foot_point, conf}, ...]}`.
- Do NOT assign identity yet — this step is still "detect only," frame-independent.

**Output:** `detections_raw.json` — per-frame detection list, no track IDs yet.

**Watch for:**
- FPS drop with `yolov8s` on full video — benchmark on 100 frames first, drop to `yolov8n` if too slow, or run detection every 2nd frame and interpolate.
- Fixed camera means no need for camera-motion compensation — skip that complexity.

---

## Step 3 — Multi-Object Tracking + Coordinates

Add a tracker on top of Step 2's per-frame detections.

**Tool:** `ByteTrack` (via `ultralytics` built-in `model.track()`, or `supervision` library's ByteTrack).

**Why ByteTrack:** recovers low-confidence detections during partial occlusion instead of dropping the ID — matches your Stage 1 occlusion requirement.

**Implementation:**
- Replace `model.predict()` with `model.track(source=video, persist=True, classes=[0], tracker="bytetrack.yaml")`.
- Each detection now carries a `track_id` that persists across frames.
- For each frame, extract: `{track_id, foot_point, frame_num}`.
- Since camera is fixed, foot_point pixel coords = consistent real-world reference — no homography needed yet (only required if you later want actual meters instead of pixel-space heatmap).

**Output:** `tracks.json` — `{track_id: [(frame_num, x, y), ...]}` for every player.

**Validation checkpoint:**
- Plot track_id count over time — should roughly stay ≈ number of players on pitch (22 + refs). Sudden jumps = ID switches to debug (occlusion/crowd frames).

---

## Step 4 — Heatmap Generation

Input: `tracks.json` foot-point coordinates.

**Per-player heatmap:**
- Take all `(x, y)` foot points for one `track_id` across all frames.
- Build a 2D histogram / KDE over the pitch image dimensions.
- Overlay with `matplotlib` (`seaborn.kdeplot` or `cv2` + Gaussian blur on an accumulator array) on top of a pitch image or blank pitch template.

**Team/overall heatmap:**
- Aggregate all `(x, y)` points from all track_ids together → team/full-game occupation map.
- (No team split yet since Stage 2 jersey classification isn't built — this is just "all players combined" for now.)

**Output:** saved heatmap image(s) — `player_<id>_heatmap.png`, `overall_heatmap.png`.

---

## Order of Execution
1. Get one fixed-camera full-pitch video clip (SoccerTrack wide-view).
2. Run Step 2 on a short clip (~30s) first — sanity check detections are stable frame to frame.
3. Run Step 3 — check track_id stability (no excessive ID switching).
4. Run Step 4 — generate heatmap, visually validate against known play (e.g., keeper heatmap should cluster near goal).
5. Only after this works cleanly → move to Stage 2 (jersey/team classification), per original problem statement.
