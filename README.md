# Contactless Palm-Vein Biometric Authentication

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-009688?logo=google&logoColor=white)](https://mediapipe.dev/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A **contactless** near-infrared (NIR) palm vein authentication system with **pose-invariant ROI extraction**, **adaptive Gabor feature encoding**, and **two-layer biometric matching**. Runs full identification in **< 0.25 s on Raspberry Pi 5**.

---

## 🎯 Why This Project

Most palm-vein implementations rely on **convex-hull concavity detection** for finger-valley localization — which fails when fingers aren't perfectly spread. This project replaces that with **MediaPipe hand landmarks** for robust, pose-invariant ROI alignment, and implements the **adaptive Gabor filter bank** from Ma et al. (IET Biometrics, 2017) with per-block parameter estimation.

**Result:** A pipeline that works on a **single NIR frame** from a NoIR camera, requires no cooperative hand pose, and runs on **embedded hardware** (Pi 5) without GPU acceleration.

---

## ⚡ Pipeline Overview

```
NIR Frame (640×480)
        │
        ▼
┌───────────────────────┐
│ MediaPipe Hand        │  ──► 21 3D landmarks (index/middle MCP, ring/little MCP)
│ Landmarker (CPU)      │       Used for rotation + scale normalization
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ ROI Extraction        │  ──► 256×256 aligned crop
│ • Rotation: align index-middle & ring-little MCP midpoints
│ • Scale: fixed inter-knuckle distance → 256px
│ • CLAHE (clip=2.0, tile=8×8) matched to Gabor block size
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Adaptive Gabor        │  ──► Binary VeinCode (256×256 bits)
│ • 32×32 blocks        │
│ • Per-block: orientation θ, frequency f, bandwidth σ from local texture
│ • Gabor filter bank → phase quantization → binary code
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Two-Layer Matching    │  ──► User ID + MNHD score + latency
│ Layer 1: 16-float signature → Euclidean prune (top-K)
│ Layer 2: Masked Normalized Hamming Distance (MNHD) parallelized
└───────────────────────┘
```

---

## 🔬 Technical Highlights

### 1. Pose-Invariant ROI Extraction (Landmark-Based)

| Approach | Failure Mode | This Project |
|----------|--------------|--------------|
| **Convexity defects** | Shallow/undetectable valleys when fingers closed → unstable ROI | MediaPipe landmarks (MCP knuckles) — works at any finger spread |
| **Fixed template** | Fails under rotation/scale variation | Rotation normalized to knuckle baseline; scale normalized to fixed inter-knuckle distance |

### 2. Adaptive Gabor Filter Bank (Ma et al. 2017)

- **Per-block estimation:** Each 32×32 block computes local orientation (structure tensor), frequency (radial FFT peak), bandwidth (autocorrelation decay)
- **No global parameters** — adapts to vein contrast, illumination, and skin texture variation across the palm
- **Output:** Binary `VeinCode` (256×256 bits) + binary mask (valid ridge regions)

### 3. Two-Layer Scalable Matching

| Layer | Method | Complexity | Purpose |
|-------|--------|------------|---------|
| **1. Signature Filter** | 16-float global descriptor (mean/var of block codes) → Euclidean distance | $O(N)$ | Prune gallery to top-K candidates (default K=5) |
| **2. MNHD** | Masked Normalized Hamming Distance, parallelized across workers | $O(K \cdot W \cdot H)$ | Exact biometric comparison on survivors |

**Low-N fallback:** If gallery < 10 templates, skip Layer 1 to avoid over-pruning.

### 4. Embedded Performance (Raspberry Pi 5)

| Stage | Latency (Pi 5, 2.4 GHz Cortex-A76) |
|-------|-------------------------------------|
| MediaPipe Landmarker | ~85 ms |
| ROI + CLAHE | ~12 ms |
| Adaptive Gabor (256×256, 64 blocks) | ~95 ms |
| Two-Layer Match (10 templates) | ~8 ms |
| **Total Identification** | **< 200 ms** (well under 0.25 s budget) |

---

## 🏗️ Repository Structure

```
contactless-palmvein/
├── main.py                 # Live camera app (enroll/scan/list/delete via hotkeys)
├── server.py               # FastAPI backend (REST API for web/touchscreen frontend)
├── ui.py                   # Pygame touchscreen/desktop UI (fallback when not headless)
├── web/                    # Frontend assets served by server.py
│   └── static/
├── mediapipe_img.py        # Hand segmentation, landmark detection, ROI extraction
├── gabor.py                # Adaptive Gabor VeinCode extraction + matching
├── search_engine.py        # Two-layer candidate search (signature filter + MNHD)
├── db_manager.py           # SQLite persistence: users, templates, access log
├── test_offline.py         # Pipeline testing against saved images (no camera)
├── vein.py                 # Core data structures (VeinCode, Template, MatchResult)
├── requirements.txt        # Python dependencies
├── TUNING.md               # Matching thresholds, CLAHE params, search constants + rationale
├── BACKEND_ARCHITECTURE_REPORT.md  # API design, error handling, deployment notes
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Raspberry Pi 5** with NoIR Camera Module (or any webcam for dev)
- **MediaPipe Hand Landmarker model** (auto-downloaded on first run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# MediaPipe model (auto-fetched, but can pre-download):
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 2. Raspberry Pi 5 Camera Setup
```bash
sudo apt install -y python3-picamera2 python3-libcamera
# main.py auto-detects picamera2; falls back to cv2.VideoCapture(0) on laptop
```

### 3. Run Live Camera App
```bash
python main.py
```
**Hotkeys:** `N` = enroll (6 samples, tilted), `S` = scan/identify, `L` = list users, `D` = delete user, `Q` = quit

### 4. Offline Testing (No Camera)
```bash
# Enroll from saved images
python test_offline.py --enroll alice img1.png img2.png ...

# Scan probe image
python test_offline.py --scan probe.png

# Full self-match / cross-match report
python test_offline.py --report
```

### 5. Web/API Backend
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
Serves REST API + static frontend from `web/`.

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| `TUNING.md` | All thresholds (MNHD cutoff, signature prune K, CLAHE clip/tile, Gabor block size) with empirical justification |
| `BACKEND_ARCHITECTURE_REPORT.md` | FastAPI design: request/response schemas, error codes, CORS, deployment (systemd, nginx), security considerations |

---

## 🧪 Testing & Validation

```bash
# Unit tests for core modules (run from repo root)
python -m pytest test_offline.py::test_gabor_determinism -v
python -m pytest test_offline.py::test_mnhd_symmetry -v
python -m pytest test_offline.py::test_roi_repeatability -v

# Accuracy validation on collected dataset
python test_offline.py --report
```
Outputs: False Match Rate (FMR), False Non-Match Rate (FNMR), EER, latency percentiles.

---

## 📐 Reference

> Ma, X., Jing, X., Huang, H., Cui, Y., Mu, J. **"Palm vein recognition scheme based on an adaptive Gabor filter."** *IET Biometrics*, 6(3), 165–173, 2017. DOI: `10.1049/iet-bmt.2016.0113`

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.