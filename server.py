#!/usr/bin/env python3
r"""
server.py
---------
FastAPI + Uvicorn backend serving the Palm Vein Biometrics API & Neobrutalism Web UI.
Run with:
  .\.venv\Scripts\python.exe server.py
"""

import os
import sys
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pipeline imports
from db_manager import (
    init_db, enroll_user, user_exists, list_users,
    delete_user, log_access, get_all_signatures,
    get_templates_by_ids, get_username, compute_signature,
)
from search_engine import SearchEngine
from mediapipe_img import (
    build_landmarker, detect_hand_landmarks,
    extract_valleys_from_landmarks, segment_hand,
    extract_ma2017_scaled_roi, enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD

# App & Directories
app = FastAPI(title="Palm Vein Biometrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAPTURE_DIR = "captures"
ROI_DIR = "roi_clahe"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# Hardware & Engine Init
init_db()
engine = SearchEngine(n_workers=4)

landmarker = None
try:
    landmarker = build_landmarker(DEFAULT_MODEL_PATH)
except Exception as e:
    print(f"Landmarker warning: {e}")

picam2 = None
CAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    preview_cfg = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    still_cfg = picam2.create_still_configuration(main={"size": (1640, 1232), "format": "RGB888"})
    picam2.configure(preview_cfg)
    picam2.start()
    picam2.set_controls({"AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 1.0})
    CAMERA_AVAILABLE = True
except Exception:
    picam2 = None
    CAMERA_AVAILABLE = False

# In-memory temporary enrollment cache: username -> list of veincodes
enrollment_cache = {}


# Helpers
def capture_frame_gray() -> np.ndarray:
    if CAMERA_AVAILABLE and picam2 is not None:
        picam2.switch_mode(still_cfg)
        frame = picam2.capture_array()
        picam2.switch_mode(preview_cfg)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Fallback to local captures for desktop testing
    sample_candidates = [
        os.path.join(CAPTURE_DIR, "myright", "capture_000.png"),
        os.path.join(CAPTURE_DIR, "myleft", "capture_005.png"),
        "1100_2.bmp"
    ]
    for p in sample_candidates:
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    # Synthetic frame
    synth = np.full((480, 640), 120, dtype=np.uint8)
    cv2.circle(synth, (320, 240), 100, 200, -1)
    return synth


def process_image(gray: np.ndarray):
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    hand_mask = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35
    )
    clahe_roi = enhance_roi_vessels(roi_256)
    code = extract_veincode(clahe_roi)
    return clahe_roi, code


# Request Models
class SampleReq(BaseModel):
    username: str
    sample_idx: int = 0


class SaveReq(BaseModel):
    username: str


# API Endpoints
@app.get("/api/status")
def get_status():
    users = list_users()
    return {
        "status": "online",
        "camera_available": CAMERA_AVAILABLE,
        "users_count": len(users),
        "total_templates": sum(u["sample_count"] for u in users),
        "match_threshold": MATCH_THRESHOLD,
    }


@app.get("/api/users")
def get_users():
    return {"users": list_users()}


@app.delete("/api/users/{username}")
def remove_user(username: str):
    if not user_exists(username):
        raise HTTPException(status_code=404, detail="User not found")
    delete_user(username)
    engine.refresh_cache()
    return {"success": True, "deleted": username}


@app.post("/api/scan")
def scan_palm():
    t0 = time.time()
    gray = capture_frame_gray()
    try:
        clahe_roi, probe_code = process_image(gray)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline extraction failed: {e}")

    username, score = engine.identify(probe_code)
    elapsed = int((time.time() - t0) * 1000)
    accepted = (username is not None)

    log_access(user_id=None, score=score, accepted=accepted)

    # Encode CLAHE ROI as base64 thumbnail
    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "accepted": accepted,
        "username": username,
        "score": float(score),
        "threshold": float(MATCH_THRESHOLD),
        "time_ms": elapsed,
        "clahe_base64": b64_roi,
    }


@app.post("/api/enroll/sample")
def enroll_sample(req: SampleReq):
    uname = req.username.strip().lower()
    if not uname:
        raise HTTPException(status_code=400, detail="Username required")

    gray = capture_frame_gray()
    try:
        clahe_roi, code = process_image(gray)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sample extraction failed: {e}")

    # Add to in-memory session cache
    enrollment_cache.setdefault(uname, []).append(code)

    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "sample_count": len(enrollment_cache[uname]),
        "vr_mean": float(code["VR"].mean()),
        "thumb": b64_roi,
    }


@app.post("/api/enroll/save")
def save_enrollment(req: SaveReq):
    uname = req.username.strip().lower()
    if uname not in enrollment_cache or len(enrollment_cache[uname]) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 captured samples to enroll")

    samples = enrollment_cache[uname]
    enroll_user(uname, samples)
    engine.refresh_cache()
    del enrollment_cache[uname]

    return {"success": True, "username": uname, "samples_stored": len(samples)}


@app.get("/api/report")
def get_report():
    users = list_users()
    sig_data = get_all_signatures()
    uid_map = {}
    for uid in sig_data["user_ids"]:
        if uid not in uid_map.values():
            try:
                uid_map[get_username(uid)] = uid
            except KeyError:
                pass

    self_matches = []
    for u in users:
        uid = uid_map.get(u["username"])
        tids = [tid for tid, uid_val in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid_val == uid]
        if len(tids) >= 2:
            tmpl = get_templates_by_ids(tids)
            scores = [match_templates(tmpl[a], tmpl[b]) for a in range(len(tmpl)) for b in range(a + 1, len(tmpl))]
            if scores:
                mn, av, mx = min(scores), sum(scores) / len(scores), max(scores)
                qual = "GOOD" if mx < 0.35 else "WARN"
                self_matches.append((u["username"], mn, av, mx, qual))

    cross_matches = []
    if len(users) >= 2:
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u1, u2 = users[i], users[j]
                t1 = get_templates_by_ids([tid for tid, uid in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid == uid_map.get(u1["username"])][:1])
                t2 = get_templates_by_ids([tid for tid, uid in zip(sig_data["template_ids"], sig_data["user_ids"]) if uid == uid_map.get(u2["username"])][:1])
                if t1 and t2:
                    sc = match_templates(t1[0], t2[0])
                    stat = "OK" if sc > 0.45 else "WARN"
                    cross_matches.append((f"{u1['username']} vs {u2['username']}", sc, stat))

    return {"self_matches": self_matches, "cross_matches": cross_matches}


# Serve Frontend Static Assets
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    print("\n" + "=" * 55)
    print("  🚀 PALM VEIN RECOGNITION — NEOBRUTALISM WEB APP")
    print("  Server running at: http://localhost:8000")
    print("=" * 55 + "\n")
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
