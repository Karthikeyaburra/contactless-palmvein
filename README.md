# contactless-palmvein

A contactless biometric authentication system built around near-infrared
palm vein imaging. Runs as a live camera application on a Raspberry Pi 5,
with an offline testing mode and a FastAPI backend for a web/touchscreen
frontend.

Feature extraction follows Ma et al.'s adaptive Gabor filter method (IET
Biometrics, 2017); ROI extraction is landmark-based rather than the
contour-concavity approach most implementations use — see
[Architecture](#architecture) for why.

## What it does

1. Captures a grayscale NIR frame of a palm held above the sensor.
2. Locates the palm using MediaPipe's hand landmark model rather than
   convex-hull concavity detection. Concavity-depth-based finger valley
   detection degrades when fingers aren't spread apart — landmark
   detection doesn't have that failure mode.
3. Extracts a rotation-normalized, scale-invariant 256×256 ROI aligned to
   the index-middle and ring-little finger valleys.
4. Enhances vein contrast with CLAHE, tuned to match the spatial scale of
   the downstream Gabor blocks (see `TUNING.md`).
5. Extracts a binary VeinCode per 256×256 ROI using an adaptive Gabor
   filter bank — orientation, frequency, and bandwidth are estimated
   per 32×32 block from local texture rather than applied uniformly.
6. Matches a probe VeinCode against enrolled templates using a two-layer
   search: a fast signature pre-filter narrows the candidate pool, then a
   parallelized masked Hamming distance (MNHD) scores the survivors.

## Architecture

```
main.py              live camera app: enroll / scan / list / delete via hotkeys
server.py            FastAPI backend exposing the same pipeline over HTTP
ui.py                pygame touchscreen/desktop UI (fallback when not headless)
web/, static/        frontend assets served by server.py

mediapipe_img.py      hand segmentation, landmark detection, ROI extraction
gabor.py              adaptive Gabor VeinCode extraction + matching
search_engine.py       two-layer candidate search (signature filter + MNHD)
db_manager.py          SQLite persistence: users, templates, access log
test_offline.py         pipeline testing against saved images, no camera needed
```

`main.py`, `server.py`, and `ui.py` are three different front ends onto the
same underlying pipeline (`mediapipe_img.py` → `gabor.py` →
`search_engine.py` → `db_manager.py`); none of them duplicate that logic.

### Why landmark-based ROI extraction

The more common approach locates finger valleys from convexity defects in
the hand's contour. That works when the hand is held with fingers
deliberately spread, but the concavity depth between fingers is
proportional to how far apart they actually are — closed or partially
closed fingers produce shallow or undetectable defects, and the ROI
placement becomes unstable or wrong. Using MediaPipe's hand landmark model
(index/middle and ring/little MCP knuckle midpoints as the alignment
baseline) avoids that dependency on finger pose.

### Two-layer matching

Comparing a probe against every enrolled template with the full masked
Hamming distance doesn't scale past a small number of users. Layer 1
computes a cheap 16-float signature per template and prunes the candidate
pool with a Euclidean distance cutoff before the expensive comparison runs.
Layer 2 runs the actual MNHD match — parallelized across workers — only
against what survives layer 1, with a low-N fallback so small databases
don't get over-pruned. Exact constants are in `TUNING.md`, which also
documents the CLAHE and threshold parameters and why they're set where they
are, rather than duplicating that here.

## Setup

```bash
pip install -r requirements.txt

curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

On Raspberry Pi 5 with the NoIR camera, the `picamera2` driver is installed
via apt rather than pip:

```bash
sudo apt install -y python3-picamera2 python3-libcamera
```

`main.py` falls back to `cv2.VideoCapture(0)` automatically if `picamera2`
isn't available, so the live app also runs on a laptop webcam for
development without a Pi.

## Usage

**Live camera app** (Raspberry Pi or any machine with a webcam):

```bash
python main.py
```

`N` enrolls a new user (6 samples across different tilts/heights, with a
per-sample countdown), `S` scans and identifies, `L` lists enrolled users,
`D` deletes one, `Q` quits.

**Offline testing** (no camera — validate the pipeline against saved
images):

```bash
python test_offline.py --enroll <username> img1.png img2.png ...
python test_offline.py --scan probe.png
python test_offline.py --report      # self-match / cross-match summary
```

Useful for iterating on the CV pipeline or matching thresholds without
needing a live capture setup every time.

**Web/API backend**:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Serves the pipeline over a REST API for the frontend in `web/`.

## Documentation

- `TUNING.md` — matching thresholds, search engine constants, CLAHE
  parameters, and why each is set where it is.
- `BACKEND_ARCHITECTURE_REPORT.md` — backend/API design in more depth than
  belongs in this README.

## Status

Core pipeline (segmentation → landmark-based ROI → Gabor feature extraction
→ matching) and all three front ends are functional. Ongoing work is
around matching accuracy validation on a larger, multi-subject dataset
rather than a handful of self-collected samples — `test_offline.py
--report` is the tool for that as more data comes in.

## Reference

Ma, X., Jing, X., Huang, H., Cui, Y., Mu, J. "Palm vein recognition scheme
based on an adaptive Gabor filter." *IET Biometrics*, 2017.
