#!/usr/bin/env python3
"""
main.py
-------
Live Palm Vein Biometrics System on Raspberry Pi 5.

Features:
  - Always-On Live NIR Camera Feed with Alignment HUD
  - Interactive Hotkeys: [N] Enroll, [S] Scan, [L] List Users, [Q] Quit
  - 5-Second Multi-Angle Positioning Countdown rendered on Live Video
  - SQLite Database Vault + 4-Worker Parallel Matcher Engine
"""

import sys
import os
import time
import cv2
import numpy as np

from mediapipe_img import (
    build_landmarker, detect_hand_landmarks,
    extract_valleys_from_landmarks, segment_hand,
    extract_ma2017_scaled_roi, enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD
from db_manager import (
    init_db, enroll_user, delete_user, list_users,
    user_exists, log_access, count_users,
)
from search_engine import SearchEngine

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
CAPTURE_DIR = "captures"
ROI_DIR     = "roi_clahe"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR,     exist_ok=True)

# ---------------------------------------------------------------------------
# Camera Setup
# ---------------------------------------------------------------------------
CAMERA_AVAILABLE = False
picam2 = None
cv_cap = None
CAMERA_TYPE = "None"

try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    preview_cfg = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    still_cfg = picam2.create_still_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam2.configure(preview_cfg)
    picam2.start()
    picam2.set_controls({
        "AeEnable":     False,
        "ExposureTime": 5000,
        "AnalogueGain": 1.0,
    })
    CAMERA_AVAILABLE = True
    CAMERA_TYPE = "picamera2"
    print("[+] Picamera2 initialized successfully.")
except Exception as e:
    print(f"[-] Picamera2 not available: {e}")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cv_cap = cap
                CAMERA_AVAILABLE = True
                CAMERA_TYPE = "opencv"
                print("[+] OpenCV VideoCapture(0) fallback initialized.")
            else:
                cap.release()
    except Exception as e2:
        print(f"[-] OpenCV camera fallback failed: {e2}")

# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------
def process_frame(gray: np.ndarray, landmarker) -> tuple:
    """Raw grayscale frame -> (clahe_roi, veincode)."""
    stretched     = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks     = detect_hand_landmarks(stretched, landmarker)
    pv1, pv2      = extract_valleys_from_landmarks(landmarks)
    hand_mask     = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35,
    )
    clahe_roi = enhance_roi_vessels(roi_256)
    code      = extract_veincode(clahe_roi)
    return clahe_roi, code


def _get_live_frame() -> np.ndarray:
    """Returns a 640x480 BGR frame for live display."""
    if CAMERA_TYPE == "picamera2" and picam2 is not None:
        frame = picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif CAMERA_TYPE == "opencv" and cv_cap is not None:
        ret, frame = cv_cap.read()
        if ret and frame is not None:
            return cv2.resize(frame, (640, 480))
    # Fallback blank frame
    canvas = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, "CAMERA OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return canvas


def _capture_hires_gray() -> np.ndarray:
    """Grabs a high-resolution frame for biometric processing."""
    if CAMERA_TYPE == "picamera2" and picam2 is not None:
        picam2.switch_mode(still_cfg)
        frame = picam2.capture_array()
        picam2.switch_mode(preview_cfg)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif CAMERA_TYPE == "opencv" and cv_cap is not None:
        ret, frame = cv_cap.read()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.zeros((480, 640), dtype=np.uint8)


def _save_capture(gray: np.ndarray, username: str, mode: str, idx: int = 0) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    name = (f"{username}_enroll_{idx}_{ts}.png" if mode == "enroll"
            else f"{username}_scan_{ts}.png")
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, gray)
    return path


def _save_roi(roi: np.ndarray, username: str, mode: str, idx: int = 0) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    name = (f"{username}_enroll_{idx}_{ts}_clahe.png" if mode == "enroll"
            else f"{username}_scan_{ts}_clahe.png")
    path = os.path.join(ROI_DIR, name)
    cv2.imwrite(path, roi)
    return path


# ---------------------------------------------------------------------------
# Visual HUD & Overlay Drawing
# ---------------------------------------------------------------------------
def _draw_hud(bgr: np.ndarray, status_msg: str = "", timer_val: int = None, hint_msg: str = "") -> np.ndarray:
    """Draws target alignment box, countdown timer, and status HUD."""
    h, w, _ = bgr.shape
    cx, cy = w // 2, h // 2
    box_sz = 130

    # Target alignment box
    color = (0, 255, 200) if timer_val is None else (0, 220, 255)
    cv2.rectangle(bgr, (cx - box_sz, cy - box_sz), (cx + box_sz, cy + box_sz), color, 2)
    cv2.putText(bgr, "ALIGN PALM HERE", (cx - 95, cy - box_sz - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Top banner with countdown timer
    cv2.rectangle(bgr, (0, 0), (w, 50), (15, 15, 15), -1)
    if timer_val is not None:
        cv2.putText(bgr, f"CAPTURING IN: {timer_val}s", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(bgr, "STEADY HAND", (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    else:
        cv2.putText(bgr, "PALM VEIN BIOMETRICS", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(bgr, "[N]Enroll  [S]Scan  [Q]Quit", (w - 290, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

    # Bottom guidance footer
    footer_text = hint_msg or status_msg or "Press [N] in window or terminal to enroll, [S] to scan."
    cv2.rectangle(bgr, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.putText(bgr, footer_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return bgr


# ---------------------------------------------------------------------------
# 5-Second Timed Capture Routine
# ---------------------------------------------------------------------------
def _timed_sample_capture(landmarker, username: str, sample_idx: int, hint: str):
    """
    Shows live countdown on the camera window for 5 seconds, then snaps high-res frame.
    """
    win_name = "Live NIR Palm Feed"
    t_end = time.time() + 5.0

    while time.time() < t_end:
        rem = max(1, int(t_end - time.time() + 0.99))
        frame = _get_live_frame()
        hud = _draw_hud(frame, timer_val=rem, hint_msg=f"Sample [{sample_idx+1}/6]: {hint}")
        cv2.imshow(win_name, hud)
        cv2.waitKey(20)

    # Flash white on snap
    flash = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.imshow(win_name, flash)
    cv2.waitKey(60)

    # Capture high-res frame
    t0 = time.time()
    gray = _capture_hires_gray()

    try:
        clahe_roi, code = process_frame(gray, landmarker)
        return gray, clahe_roi, code, time.time() - t0
    except ValueError as e:
        return gray, None, None, str(e)


# ---------------------------------------------------------------------------
# Enrollment Loop (6 Multi-Angle Samples with 5s Timer)
# ---------------------------------------------------------------------------
def opt_enroll(landmarker, engine):
    """Enroll a user with live camera window and 5s countdown per sample."""
    if not CAMERA_AVAILABLE:
        print("  [!] Camera not available.")
        return

    raw = input("\n  Enter username to enroll: ")
    username = raw.strip().lower().replace(" ", "_")
    if not username:
        print("  [!] Invalid username.")
        return

    if user_exists(username):
        print(f"  [!] '{username}' already exists. Delete first to re-enroll.")
        return

    print(f"\n  ========================================================")
    print(f"   Enrolling '{username}' — 6 Multi-Angle Palm Samples")
    print(f"   Look at the 'Live NIR Palm Feed' window on your screen!")
    print(f"  ========================================================\n")

    sample_hints = [
        "Hold palm flat, centered ~10-15cm above sensor",
        "Tilt palm slightly to the LEFT (~5 degrees)",
        "Tilt palm slightly to the RIGHT (~5 degrees)",
        "Raise palm slightly HIGHER (~15-18cm)",
        "Spread fingers slightly wider",
        "Hold palm flat, centered for final confirmation",
    ]

    veincode_list = []
    win_name = "Live NIR Palm Feed"

    for i in range(6):
        hint = sample_hints[i]
        print(f"  Sample [{i+1}/6]: {hint}")
        print(f"  Look at screen... 5-second countdown starting now!")

        for attempt in range(2):
            gray, clahe_roi, code, result = _timed_sample_capture(landmarker, username, i, hint)
            
            if code is not None:
                _save_capture(gray, username, "enroll", idx=i)
                _save_roi(clahe_roi, username, "enroll", idx=i)
                veincode_list.append(code)
                print(f"    ✓ OK Sample [{i+1}/6] captured! (VR={code['VR'].mean():.3f} in {result:.2f}s)\n")
                
                # Show success badge on screen for 1 second
                for _ in range(15):
                    f = _get_live_frame()
                    cv2.putText(f, f"SAMPLE [{i+1}/6] ACCEPTED! OK", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(win_name, f)
                    cv2.waitKey(50)
                break
            else:
                print(f"    [!] Detection failed: {result}")
                if attempt == 0:
                    print("    Retrying in 5 seconds... Adjust hand position.")
                    for _ in range(15):
                        f = _get_live_frame()
                        cv2.putText(f, "REPOSITION HAND - RETRYING...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow(win_name, f)
                        cv2.waitKey(50)

    if len(veincode_list) < 3:
        print(f"\n  [!] Too few valid samples ({len(veincode_list)}/3 minimum). Enrollment cancelled.\n")
        return

    enroll_user(username, veincode_list)
    engine.refresh_cache()
    print(f"\n  [+] ENROLLED: '{username}' successfully with {len(veincode_list)} samples stored in database.\n")


# ---------------------------------------------------------------------------
# Scan & Identify Routine (3s Live Countdown)
# ---------------------------------------------------------------------------
def opt_scan(landmarker, engine):
    """Identifies a user with live camera window and 3s countdown."""
    if not CAMERA_AVAILABLE:
        print("  [!] Camera not available.")
        return

    win_name = "Live NIR Palm Feed"
    print("\n  Starting 3-second Scan Countdown... Align palm inside the box!")

    t_end = time.time() + 3.0
    while time.time() < t_end:
        rem = max(1, int(t_end - time.time() + 0.99))
        frame = _get_live_frame()
        hud = _draw_hud(frame, timer_val=rem, hint_msg="Hold palm steady inside green box to scan")
        cv2.imshow(win_name, hud)
        cv2.waitKey(20)

    # Flash white on snap
    flash = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.imshow(win_name, flash)
    cv2.waitKey(60)

    t0 = time.time()
    gray = _capture_hires_gray()

    try:
        clahe_roi, probe_code = process_frame(gray, landmarker)
    except ValueError as e:
        print(f"\n  [!] Scan failed: {e}\n")
        return

    username, score = engine.identify(probe_code)
    elapsed = time.time() - t0
    accepted = username is not None

    label = username if accepted else "unknown"
    _save_capture(gray, label, "scan")
    _save_roi(clahe_roi, label, "scan")
    log_access(user_id=None, score=score, accepted=accepted)

    print("\n  ──────────────────────────────────────────")
    if accepted:
        print(f"  RESULT : AUTHENTICATED — {username}")
        print(f"  User   : {username}")
    else:
        print(f"  RESULT : NOT RECOGNISED")
    print(f"  Score  : {score:.4f}  (threshold: {MATCH_THRESHOLD:.4f})")
    print(f"  Time   : {elapsed:.2f}s")
    print("  ──────────────────────────────────────────\n")

    # Display result overlay on screen for 2.5 seconds
    badge_color = (0, 255, 0) if accepted else (0, 0, 255)
    badge_text = f"VERIFIED: {username}" if accepted else "NOT RECOGNISED"
    for _ in range(30):
        f = _get_live_frame()
        cv2.rectangle(f, (50, 180), (590, 300), (0, 0, 0), -1)
        cv2.rectangle(f, (50, 180), (590, 300), badge_color, 3)
        cv2.putText(f, badge_text, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.9, badge_color, 2)
        cv2.putText(f, f"MNHD Score: {score:.4f} ({elapsed:.2f}s)", (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win_name, f)
        cv2.waitKey(50)


# ---------------------------------------------------------------------------
# List Users & Delete
# ---------------------------------------------------------------------------
def opt_list():
    users = list_users()
    print("\n  Enrolled Users in Database:")
    if not users:
        print("    (No users registered yet)")
    else:
        for u in users:
            print(f"    - {u['username']} ({u['sample_count']} samples, enrolled: {u['enrolled_at']})")
    print()


def opt_delete(engine):
    raw = input("\n  Enter username to delete: ")
    username = raw.strip().lower()
    if not username:
        return
    if not user_exists(username):
        print(f"  [!] User '{username}' not found.")
        return
    delete_user(username)
    engine.refresh_cache()
    print(f"  [✓] User '{username}' deleted from database.\n")


# ---------------------------------------------------------------------------
# Main Interactive Loop
# ---------------------------------------------------------------------------
def main():
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║          PALM VEIN AUTHENTICATION SYSTEM               ║")
    print("║            Raspberry Pi 5  |  NoIR Camera              ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    print(f"  Camera : {'READY (' + CAMERA_TYPE + ')' if CAMERA_AVAILABLE else 'OFFLINE'}")
    print(f"  Model  : {DEFAULT_MODEL_PATH}")

    print("\n  [1/3] Initialising database...")
    init_db()

    print("  [2/3] Loading 4-worker search engine...")
    engine = SearchEngine(n_workers=4)

    print("  [3/3] Loading MediaPipe landmarker...")
    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
    except Exception as e:
        print(f"  [!] MediaPipe loading error: {e}")
        engine.shutdown()
        sys.exit(1)

    print("\n  ========================================================")
    print("   Live Camera Feed is starting on your monitor screen!")
    print("   Hotkeys in Camera Window or Terminal:")
    print("     [N] = Enroll New User (with 5-second timers)")
    print("     [S] = Scan & Identify (with 3-second timer)")
    print("     [L] = List Enrolled Users")
    print("     [D] = Delete User")
    print("     [Q] = Quit")
    print("  ========================================================\n")

    win_name = "Live NIR Palm Feed"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)

    try:
        while True:
            frame = _get_live_frame()
            hud = _draw_hud(frame)
            cv2.imshow(win_name, hud)

            key = cv2.waitKey(25) & 0xFF

            if key == ord('q') or key == 27:
                print("  Quitting application...")
                break
            elif key == ord('n') or key == ord('1'):
                opt_enroll(landmarker, engine)
            elif key == ord('s') or key == ord('2'):
                opt_scan(landmarker, engine)
            elif key == ord('l') or key == ord('3'):
                opt_list()
            elif key == ord('d') or key == ord('4'):
                opt_delete(engine)

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        engine.shutdown()
        if CAMERA_TYPE == "picamera2" and picam2 is not None:
            picam2.stop()
        elif CAMERA_TYPE == "opencv" and cv_cap is not None:
            cv_cap.release()
        print("\n  System shutdown clean. Goodbye!\n")


if __name__ == "__main__":
    main()
