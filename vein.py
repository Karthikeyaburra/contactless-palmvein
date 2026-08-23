#!/usr/bin/env python3
"""
Palm Vein Capture + Enrollment + Identification

Keys:
    n = enroll a new user
    s = scan / identify a user
    q / ESC = quit

Pipeline for both enrollment and scan:
    camera capture -> MediaPipe ROI extraction -> CLAHE -> Gabor -> VeinCode

Only the original capture and CLAHE ROI are written as images during a
capture. The enrolled biometric template is stored in data/palm_vein.db.
"""

import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2

from mediapipe_img import (
    build_landmarker,
    detect_hand_landmarks,
    extract_valleys_from_landmarks,
    segment_hand,
    extract_ma2017_scaled_roi,
    enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD
from db_manager import init_db, enroll_user, log_access, list_users, user_exists
from search_engine import SearchEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAPTURE_DIR        = "captures"
ROI_DIR            = "roi_clahe"
MODEL_PATH         = DEFAULT_MODEL_PATH
NUM_ENROLL_SAMPLES = 6   # samples collected per enrollment; paper recommends 6

PREVIEW_SIZE = (640, 480)
STILL_SIZE   = (1640, 1232)

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR,     exist_ok=True)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def save_capture(gray: np.ndarray, username: str, mode: str,
                 sample_idx: int = 0) -> str:
    """
    Save a raw grayscale capture to CAPTURE_DIR.
    mode: 'enroll' or 'scan'
    Returns saved path.

    File format:
        captures/<username>_enroll_<idx>_<timestamp>.png   (enrollment)
        captures/<username>_scan_<timestamp>.png            (scan / unknown)
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    if mode == 'enroll':
        name = f"{username}_enroll_{sample_idx}_{ts}.png"
    else:
        name = f"{username}_scan_{ts}.png"
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, gray)
    return path


def save_roi(clahe_roi: np.ndarray, username: str, mode: str,
             sample_idx: int = 0) -> str:
    """
    Save CLAHE ROI to ROI_DIR using the same naming convention.
    Returns saved path.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    if mode == 'enroll':
        name = f"{username}_enroll_{sample_idx}_{ts}_clahe.png"
    else:
        name = f"{username}_scan_{ts}_clahe.png"
    path = os.path.join(ROI_DIR, name)
    cv2.imwrite(path, clahe_roi)
    return path


# ---------------------------------------------------------------------------
# Full biometric pipeline
# ---------------------------------------------------------------------------

def process_frame(gray: np.ndarray, landmarker) -> tuple:
    """
    Full pipeline: raw grayscale frame -> (clahe_roi, veincode)
    Raises ValueError on any pipeline failure (bad frame, no hand, etc.)
    """
    stretched  = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    landmarks  = detect_hand_landmarks(stretched, landmarker)
    pv1, pv2   = extract_valleys_from_landmarks(landmarks)
    hand_mask  = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35,
    )
    clahe_roi = enhance_roi_vessels(roi_256)
    code      = extract_veincode(clahe_roi)
    return clahe_roi, code


# ---------------------------------------------------------------------------
# Enrollment — NUM_ENROLL_SAMPLES samples, SPACE between each
# ---------------------------------------------------------------------------

def enroll(picam2, still_config, preview_config, landmarker, engine):
    """Enroll a new user with up to NUM_ENROLL_SAMPLES palm samples."""
    username = input("Enter new user ID: ").strip().lower()

    if not username or not username.replace("_", "").replace("-", "").isalnum():
        print("Invalid user ID. Use letters, numbers, hyphens, underscores only.")
        return

    if user_exists(username):
        print(f"User '{username}' already enrolled. Delete first to re-enroll.")
        return

    print(f"\nEnrolling '{username}' — {NUM_ENROLL_SAMPLES} palm samples required.")
    print("Place your palm in front of the camera, then press SPACE to capture.")

    veincode_list = []

    for i in range(NUM_ENROLL_SAMPLES):
        print(f"\nSample {i + 1} of {NUM_ENROLL_SAMPLES} — press SPACE to capture, ESC to cancel.")

        if not _wait_for_space_or_esc(picam2, username, i):
            print("Enrollment cancelled.")
            return

        clahe_roi, code = _capture_still(picam2, still_config, preview_config,
                                         landmarker)
        if clahe_roi is None:
            print("Capture failed. Enrollment cancelled.")
            return

        gray = _last_gray
        save_capture(gray, username, 'enroll', sample_idx=i)
        save_roi(clahe_roi, username, 'enroll', sample_idx=i)
        veincode_list.append(code)
        print(f"  Sample {i + 1} captured. VR={code['VR'].mean():.3f}")

    _run_consistency_check(veincode_list)

    enroll_user(username, veincode_list)
    engine.refresh_cache()

    print(f"\nENROLLED: '{username}' with {len(veincode_list)} samples.")
    print(f"Files saved to {CAPTURE_DIR}/ and {ROI_DIR}/")


def _run_consistency_check(veincode_list: list):
    """Compute pairwise MNHD between all collected samples and warn if any pair
    scores above 0.45 — high distance suggests inconsistent hand placement."""
    print("\nChecking template consistency...")
    inconsistent = []
    for a in range(len(veincode_list)):
        for b in range(a + 1, len(veincode_list)):
            score = match_templates(veincode_list[a], veincode_list[b])
            if score > 0.45:
                inconsistent.append((a, b, score))

    if inconsistent:
        print(f"WARNING: {len(inconsistent)} sample pair(s) have high distance:")
        for a, b, s in inconsistent:
            print(f"  Sample {a+1} vs Sample {b+1}: {s:.4f} (threshold 0.45)")
        print("Consider re-enrolling for better accuracy.")
    else:
        print("All samples consistent. Enrollment quality: GOOD")


# Module-level mutable used to share the last captured gray image between
# the SPACE-wait loop and the caller without passing it through the return
# value of the wait helper (which only returns a bool).
_last_gray = None


def _wait_for_space_or_esc(picam2, username: str, sample_idx: int) -> bool:
    """
    Show a live preview overlay and block until SPACE or ESC is pressed.
    Returns True on SPACE, False on ESC.
    """
    while True:
        frame = picam2.capture_array()
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        label = (f"Enroll {username} — sample {sample_idx + 1}/{NUM_ENROLL_SAMPLES}"
                 f" — SPACE to capture")
        cv2.putText(bgr, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Live NIR Palm Feed", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        if key == ord(' '):
            return True


def _capture_still(picam2, still_config, preview_config, landmarker):
    """
    Switch to still mode, capture one frame, run the pipeline.
    Returns (clahe_roi, veincode) on success, (None, None) on failure.
    Side-effect: sets the module-level _last_gray.
    """
    global _last_gray
    picam2.switch_mode(still_config)
    hires = picam2.capture_array()
    gray  = cv2.cvtColor(hires, cv2.COLOR_RGB2GRAY)
    picam2.switch_mode(preview_config)
    _last_gray = gray

    try:
        clahe_roi, code = process_frame(gray, landmarker)
        return clahe_roi, code
    except ValueError as e:
        print(f"Capture failed: {e}. Try again — press 'n' to restart enrollment.")
        return None, None


# ---------------------------------------------------------------------------
# Scan — single capture, identify via SearchEngine
# ---------------------------------------------------------------------------

def scan(picam2, still_config, preview_config, landmarker, engine):
    """Capture one still and identify the user against all enrolled templates."""
    print("Scanning...")

    picam2.switch_mode(still_config)
    hires = picam2.capture_array()
    gray  = cv2.cvtColor(hires, cv2.COLOR_RGB2GRAY)
    picam2.switch_mode(preview_config)

    t0 = time.time()

    try:
        clahe_roi, probe_code = process_frame(gray, landmarker)
    except ValueError as e:
        print(f"Scan failed: {e}")
        return

    username, score = engine.identify(probe_code)
    elapsed = time.time() - t0

    _handle_scan_result(gray, clahe_roi, username, score, elapsed)


def _handle_scan_result(gray, clahe_roi, username, score, elapsed):
    """Log, save files, and print the scan outcome."""
    if username:
        log_access(user_id=None, score=score, accepted=True)
        save_capture(gray, username, 'scan')
        save_roi(clahe_roi, username, 'scan')
        print(f"\nAUTHENTICATED : {username}")
    else:
        log_access(user_id=None, score=score, accepted=False)
        save_capture(gray, 'unknown', 'scan')
        save_roi(clahe_roi, 'unknown', 'scan')
        print(f"\nNOT RECOGNISED")

    print(f"Score         : {score:.4f}  (threshold: {MATCH_THRESHOLD:.4f})")
    print(f"Time          : {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Camera / main loop
# ---------------------------------------------------------------------------

def main():
    """Application entry point. Initialises everything, then runs the event loop."""
    init_db()
    engine     = SearchEngine(n_workers=4)
    landmarker = build_landmarker(MODEL_PATH)

    picam2         = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": PREVIEW_SIZE, "format": "RGB888"}
    )
    still_config = picam2.create_still_configuration(
        main={"size": STILL_SIZE, "format": "RGB888"}
    )

    picam2.configure(preview_config)
    picam2.start()
    picam2.set_controls({
        "AeEnable":    False,
        "ExposureTime": 5000,
        "AnalogueGain": 1.0,
    })

    print("========================================")
    print(" Palm Vein Authentication")
    print("========================================")
    print(" n = enroll new user")
    print(" s = scan / identify")
    print(" q / ESC = quit")
    print("========================================")

    try:
        _run_event_loop(picam2, still_config, preview_config, landmarker, engine)
    finally:
        engine.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        picam2.stop()


def _run_event_loop(picam2, still_config, preview_config, landmarker, engine):
    """Main key-dispatch loop. Runs until q or ESC."""
    while True:
        frame = picam2.capture_array()
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live NIR Palm Feed", bgr)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            enroll(picam2, still_config, preview_config, landmarker, engine)
        elif key == ord('s'):
            scan(picam2, still_config, preview_config, landmarker, engine)


if __name__ == "__main__":
    main()
