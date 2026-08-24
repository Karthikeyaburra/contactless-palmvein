#!/usr/bin/env python3
r"""
server.py
---------
FastAPI + Uvicorn backend serving the Palm Vein Biometrics API & Neobrutalism Web UI.
Directly connects to Raspberry Pi NoIR Camera (Picamera2) or USB Webcam (OpenCV).
"""

import os
import sys
import time
import base64
import cv2
import numpy as np
import mimetypes
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure proper MIME types on all OS platforms (especially Windows)
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")

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

from contextlib import asynccontextmanager

# Directories
CAPTURE_DIR = "captures"
ROI_DIR = "roi_clahe"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# Hardware & Engine Globals
engine = None
landmarker = None
picam2 = None
cv_cap = None
CAMERA_AVAILABLE = False
CAMERA_TYPE = None
preview_cfg = None
still_cfg = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, landmarker, picam2, cv_cap, CAMERA_AVAILABLE, CAMERA_TYPE, preview_cfg, still_cfg
    init_db()
    engine = SearchEngine(n_workers=4)

    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
    except Exception as e:
        print(f"[!] Landmarker warning: {e}")

    # 1. Try Raspberry Pi NoIR Camera (Picamera2)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        preview_cfg = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        still_cfg = picam2.create_still_configuration(main={"size": (1640, 1232), "format": "RGB888"})
        picam2.configure(preview_cfg)
        picam2.start()
        picam2.set_controls({"AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 1.0})
        CAMERA_AVAILABLE = True
        CAMERA_TYPE = "picamera2"
        print("[+] Picamera2 initialized successfully (Raspberry Pi NoIR Camera).")
    except Exception:
        picam2 = None

    # 2. Fallback to USB Webcam / V4L2 Camera (OpenCV)
    if not CAMERA_AVAILABLE:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cv_cap = cap
                    CAMERA_AVAILABLE = True
                    CAMERA_TYPE = "opencv"
                    print("[+] OpenCV VideoCapture(0) initialized successfully.")
                else:
                    cap.release()
        except Exception as e:
            print(f"[!] OpenCV Camera error: {e}")

    if not CAMERA_AVAILABLE:
        print("[!] No live camera detected. Connect a Pi NoIR camera or USB webcam.")

    yield

    if engine:
        engine.shutdown()
    if CAMERA_TYPE == "picamera2" and picam2 is not None:
        picam2.stop()
    if CAMERA_TYPE == "opencv" and cv_cap is not None:
        cv_cap.release()

# App & Middleware
app = FastAPI(title="Palm Vein Biometrics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory temporary enrollment cache: username -> list of veincodes
enrollment_cache = {}


# Helpers
def capture_frame_gray() -> np.ndarray:
    """Capture a single live grayscale frame from camera."""
    if CAMERA_TYPE == "picamera2" and picam2 is not None:
        picam2.switch_mode(still_cfg)
        frame = picam2.capture_array()
        picam2.switch_mode(preview_cfg)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    if CAMERA_TYPE == "opencv" and cv_cap is not None:
        # Flush any stale buffer frames
        for _ in range(2):
            cv_cap.grab()
        ret, frame = cv_cap.read()
        if not ret or frame is None:
            raise ValueError("Failed to capture frame from webcam.")
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    raise ValueError("No live camera available. Please connect Raspberry Pi camera or webcam.")


def process_image(gray: np.ndarray):
    """
    Extract CLAHE ROI and Gabor VeinCode from a grayscale hand frame.
    Raises ValueError if palm landmarks or valleys cannot be detected.
    """
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    if landmarks is None or len(landmarks) < 21:
        raise ValueError("No hand landmarks detected. Hold palm flat ~10-15cm above camera.")

    pv1, pv2 = extract_valleys_from_landmarks(landmarks)
    if pv1 is None or pv2 is None:
        raise ValueError("Cannot detect finger valley landmarks. Spread fingers slightly.")

    hand_mask = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35
    )
    if roi_256 is None or roi_256.size == 0:
        raise ValueError("Failed to extract palm ROI bounding box.")

    clahe_roi = enhance_roi_vessels(roi_256)
    code = extract_veincode(clahe_roi)
    return clahe_roi, code


def save_capture_to_disk(gray: np.ndarray, roi: np.ndarray, username: str, mode: str, idx: int = 0):
    """Save raw capture and CLAHE ROI to captures/ and roi_clahe/."""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cap_name = f"{username}_{mode}_{idx}_{ts}.png" if mode == "enroll" else f"{username}_{mode}_{ts}.png"
        roi_name = f"{username}_{mode}_{idx}_{ts}_clahe.png" if mode == "enroll" else f"{username}_{mode}_{ts}_clahe.png"
        cv2.imwrite(os.path.join(CAPTURE_DIR, cap_name), gray)
        cv2.imwrite(os.path.join(ROI_DIR, roi_name), roi)
    except Exception as e:
        print(f"[!] Warning: Failed saving capture to disk: {e}")


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
        "camera_type": CAMERA_TYPE or "None",
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
    try:
        gray = capture_frame_gray()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        clahe_roi, probe_code = process_image(gray)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {e}")

    username, score = engine.identify(probe_code)
    elapsed = int((time.time() - t0) * 1000)
    accepted = (username is not None)

    log_access(user_id=None, score=score, accepted=accepted)

    # Save to disk
    save_capture_to_disk(gray, clahe_roi, username or "unknown", "scan")

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

    try:
        gray = capture_frame_gray()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        clahe_roi, code = process_image(gray)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction error: {e}")

    # Add to in-memory session cache
    enrollment_cache.setdefault(uname, []).append(code)
    sample_idx = len(enrollment_cache[uname])

    # Save capture and ROI to disk
    save_capture_to_disk(gray, clahe_roi, uname, "enroll", idx=sample_idx)

    _, buf = cv2.imencode(".png", clahe_roi)
    b64_roi = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "sample_count": sample_idx,
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
            media_type, _ = mimetypes.guess_type(file_path)
            return FileResponse(file_path, media_type=media_type or "application/octet-stream")
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import uvicorn
    import webbrowser
    print("\n" + "=" * 55)
    print("  [+] PALM VEIN RECOGNITION - NEOBRUTALISM WEB APP")
    print("  [+] Server running at: http://localhost:8000")
    print("=" * 55 + "\n")
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
