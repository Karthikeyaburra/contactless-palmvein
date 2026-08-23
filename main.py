#!/usr/bin/env python3
"""
main.py
-------
Single interactive entry point for the Palm Vein Authentication System.
Run on Raspberry Pi 5:  python3 main.py
Run for offline dev:    python3 main.py  (camera unavailable — options 1/2 disabled)

Menu options:
  1  Enroll new user   (camera required)
  2  Scan & identify   (camera required)
  3  List enrolled users
  4  Delete user
  5  System report
  Q  Quit
"""

import os
import sys
import time
import cv2
import numpy as np

from db_manager import (
    init_db, enroll_user, user_exists, list_users, delete_user,
    log_access, get_all_signatures, get_templates_by_ids,
    get_username, compute_signature,
)
from search_engine import SearchEngine
from mediapipe_img import (
    build_landmarker, detect_hand_landmarks,
    extract_valleys_from_landmarks, segment_hand,
    extract_ma2017_scaled_roi, enhance_roi_vessels,
    DEFAULT_MODEL_PATH,
)
from gabor import extract_veincode, match_templates, MATCH_THRESHOLD

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
CAPTURE_DIR = "captures"
ROI_DIR     = "roi_clahe"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR,     exist_ok=True)

# ---------------------------------------------------------------------------
# Camera initialisation (graceful degradation on non-Pi hardware)
# ---------------------------------------------------------------------------
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
except Exception:
    picam2 = None
    CAMERA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core pipeline (independent copy — no dependency on vein.py)
# ---------------------------------------------------------------------------

def process_frame(gray: np.ndarray, landmarker) -> tuple:
    """
    Raw grayscale frame → (clahe_roi, veincode).
    Raises ValueError on any pipeline failure.
    """
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


# ---------------------------------------------------------------------------
# File-save helpers
# ---------------------------------------------------------------------------

def _save_capture(gray: np.ndarray, username: str, mode: str,
                  idx: int = 0) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    name = (f"{username}_enroll_{idx}_{ts}.png" if mode == "enroll"
            else f"{username}_scan_{ts}.png")
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, gray)
    return path


def _save_roi(roi: np.ndarray, username: str, mode: str,
              idx: int = 0) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    name = (f"{username}_enroll_{idx}_{ts}_clahe.png" if mode == "enroll"
            else f"{username}_scan_{ts}_clahe.png")
    path = os.path.join(ROI_DIR, name)
    cv2.imwrite(path, roi)
    return path


# ---------------------------------------------------------------------------
# Camera capture helper
# ---------------------------------------------------------------------------

def _capture_gray() -> np.ndarray:
    """Switch to still mode, grab one frame, return grayscale ndarray."""
    picam2.switch_mode(still_cfg)
    frame = picam2.capture_array()
    picam2.switch_mode(preview_cfg)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _banner():
    print("\n╔══════════════════════════════════════╗")
    print("║     PALM VEIN AUTHENTICATION         ║")
    print("║     Raspberry Pi 5  |  NIR Camera    ║")
    print("╚══════════════════════════════════════╝\n")
    cam_status = "READY" if CAMERA_AVAILABLE else "OFFLINE (laptop mode)"
    print(f"  Camera  : {cam_status}")
    print(f"  Model   : {DEFAULT_MODEL_PATH}")
    print()


def _menu():
    print("  [1]  Enroll new user")
    print("  [2]  Scan & identify")
    print("  [3]  List enrolled users")
    print("  [4]  Delete user")
    print("  [5]  System report")
    print("  [Q]  Quit")
    print()


def _hr(char="─", width=44):
    print(char * width)


def _no_camera():
    print("  Camera not available — use test_offline.py for offline testing.")


def _validate_username(raw: str) -> str:
    """Strip, lowercase, and validate. Returns clean name or '' on failure."""
    name = raw.strip().lower()
    if not name:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if not all(c in allowed for c in name):
        return ""
    return name


# ---------------------------------------------------------------------------
# Option 1 — Enroll new user
# ---------------------------------------------------------------------------

def _consistency_check(veincode_list: list) -> int:
    """
    Pairwise MNHD check. Prints result and returns number of bad pairs.
    Threshold: 0.50 (scores in genuine matches are typically 0.10-0.45).
    """
    bad = []
    for a in range(len(veincode_list)):
        for b in range(a + 1, len(veincode_list)):
            s = match_templates(veincode_list[a], veincode_list[b])
            if s > 0.50:
                bad.append((a, b, s))
    if bad:
        print(f"  WARNING: {len(bad)} inconsistent pair(s) — consider re-enrolling.")
        for a, b, s in bad:
            print(f"    Sample {a+1} vs Sample {b+1}: {s:.4f}")
    else:
        print("  Quality check: GOOD")
    return len(bad)


def _capture_one_sample(landmarker, username: str, idx: int):
    """
    Capture one palm sample with one retry on failure.
    Returns (gray, clahe_roi, code, elapsed) on success, or None on double fail.
    """
    for attempt in range(2):
        if attempt == 1:
            print("    Retrying — press ENTER when ready.")
            input()
        t0   = time.time()
        gray = _capture_gray()
        try:
            clahe_roi, code = process_frame(gray, landmarker)
            return gray, clahe_roi, code, time.time() - t0
        except ValueError as e:
            print(f"    Failed: {e}")
            if attempt == 0:
                print("    Try again — press ENTER to retry.")
                input()
    print("    Sample failed twice — skipping.")
    return None


def opt_enroll(landmarker, engine):
    """Enroll a new user with up to 6 palm samples."""
    if not CAMERA_AVAILABLE:
        _no_camera()
        return

    raw = input("  Enter username: ")
    username = _validate_username(raw)
    if not username:
        print("  Invalid username. Use a-z, 0-9, underscore, hyphen only.")
        return

    if user_exists(username):
        print(f"  '{username}' is already enrolled. Delete first to re-enroll.")
        return

    print(f"\n  Enrolling '{username}' — 6 palm samples required.")
    print("  Press ENTER before each capture.\n")

    veincode_list = []

    for i in range(6):
        input(f"  Sample [{i+1}/6] — hold palm steady, press ENTER to capture")
        result = _capture_one_sample(landmarker, username, i)
        if result is None:
            continue
        gray, clahe_roi, code, elapsed = result
        _save_capture(gray,     username, "enroll", idx=i)
        _save_roi(clahe_roi,    username, "enroll", idx=i)
        veincode_list.append(code)
        print(f"    OK  VR={code['VR'].mean():.3f}  ({elapsed:.2f}s)")

    if len(veincode_list) < 3:
        print(f"\n  Too few valid samples ({len(veincode_list)}/3 minimum). Cancelled.")
        return

    print()
    _consistency_check(veincode_list)

    enroll_user(username, veincode_list)
    engine.refresh_cache()
    print(f"\n  ENROLLED: '{username}' with {len(veincode_list)} samples.\n")


# ---------------------------------------------------------------------------
# Option 2 — Scan & identify
# ---------------------------------------------------------------------------

def opt_scan(landmarker, engine):
    """Capture one palm and identify the user."""
    if not CAMERA_AVAILABLE:
        _no_camera()
        return

    input("  Place palm in front of camera — press ENTER to scan")

    t0   = time.time()
    gray = _capture_gray()

    try:
        clahe_roi, probe_code = process_frame(gray, landmarker)
    except ValueError as e:
        print(f"  Scan failed: {e}")
        return

    username, score = engine.identify(probe_code)
    elapsed         = time.time() - t0
    accepted        = username is not None

    _print_scan_result(username, score, elapsed)
    log_access(user_id=None, score=score, accepted=accepted)

    label = username if accepted else "unknown"
    _save_capture(gray,     label, "scan")
    _save_roi(clahe_roi,    label, "scan")


def _print_scan_result(username, score: float, elapsed: float):
    result = f"AUTHENTICATED — {username}" if username else "NOT RECOGNISED"
    _hr("─")
    print(f"  RESULT : {result}")
    if username:
        print(f"  User   : {username}")
    print(f"  Score  : {score:.4f}  (threshold: {MATCH_THRESHOLD:.4f})")
    print(f"  Time   : {elapsed:.2f}s")
    _hr("─")
    print()


# ---------------------------------------------------------------------------
# Option 3 — List enrolled users
# ---------------------------------------------------------------------------

def opt_list():
    """Print a formatted table of all enrolled users."""
    users = list_users()
    print()
    if not users:
        print("  No users enrolled.")
        return

    print("  Enrolled Users")
    _hr()
    print(f"  {'#':<4} {'Username':<18} {'Samples':>7}   Enrolled At")
    _hr()
    for n, u in enumerate(users, 1):
        enrolled = u["enrolled_at"][:10]
        print(f"  {n:<4} {u['username']:<18} {u['sample_count']:>7}   {enrolled}")
    _hr()
    total = sum(u["sample_count"] for u in users)
    print(f"  Total: {len(users)} user(s), {total} template(s)")
    print()


# ---------------------------------------------------------------------------
# Option 4 — Delete user
# ---------------------------------------------------------------------------

def opt_delete(engine):
    """Soft-delete an enrolled user after confirmation."""
    opt_list()

    raw = input("  Enter username to delete (or ENTER to cancel): ").strip()
    if not raw:
        print("  Cancelled.")
        return

    username = raw.strip().lower()
    if not user_exists(username):
        print(f"  '{username}' not found or already inactive.")
        return

    confirm = input(f"  Delete '{username}'? This cannot be undone. [y/N]: ")
    if confirm.strip().lower() != "y":
        print("  Cancelled.")
        return

    try:
        delete_user(username)
    except ValueError as e:
        print(f"  Error: {e}")
        return

    engine.refresh_cache()
    print(f"  Deleted: '{username}'.\n")


# ---------------------------------------------------------------------------
# Option 5 — System report
# ---------------------------------------------------------------------------

def _get_uid_map() -> dict:
    """Return {username: user_id} for all active users from the signature cache."""
    sig = get_all_signatures()
    uid_map: dict = {}
    for uid in sig["user_ids"]:
        if uid not in uid_map.values():
            try:
                uid_map[get_username(uid)] = uid
            except KeyError:
                pass
    return uid_map


def _user_tids(uid: int, sig: dict) -> list:
    """Return template IDs belonging to a given user_id."""
    return [tid for tid, u in zip(sig["template_ids"], sig["user_ids"]) if u == uid]


def _self_match_scores(templates: list) -> list:
    """All pairwise MNHD scores for a single user's template list."""
    scores = []
    for a in range(len(templates)):
        for b in range(a + 1, len(templates)):
            scores.append(match_templates(templates[a], templates[b]))
    return scores


def _report_self_match(users: list, uid_map: dict, sig: dict):
    print("  Self-Match (same user, different samples — target: all < 0.35)")
    _hr()
    print(f"  {'Username':<18} {'Min':>6}  {'Avg':>6}  {'Max':>6}  Quality")
    _hr()
    for u in users:
        uid  = uid_map.get(u["username"])
        tids = _user_tids(uid, sig)
        if len(tids) < 2:
            print(f"  {u['username']:<18}  (1 sample — skipped)")
            continue
        tmpl   = get_templates_by_ids(tids)
        scores = _self_match_scores(tmpl)
        mn, av, mx = min(scores), sum(scores) / len(scores), max(scores)
        quality = "GOOD" if mx < 0.35 else "WARN — re-enroll advised"
        print(f"  {u['username']:<18} {mn:>6.4f}  {av:>6.4f}  {mx:>6.4f}  {quality}")
    _hr()
    print()


def _report_cross_match(users: list, uid_map: dict, sig: dict):
    print("  Cross-Match (different users — target: all > 0.45)")
    _hr()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            t1 = get_templates_by_ids(_user_tids(uid_map[u1["username"]], sig)[:1])
            t2 = get_templates_by_ids(_user_tids(uid_map[u2["username"]], sig)[:1])
            if not t1 or not t2:
                continue
            score  = match_templates(t1[0], t2[0])
            status = "OK" if score > 0.45 else "WARN — false-accept risk"
            print(f"  {u1['username']} vs {u2['username']}: {score:.4f}  {status}")
    _hr()
    print()


def _report_last_scan() -> str:
    """Return the timestamp of the most recent access_log entry, or 'never'."""
    import sqlite3
    from db_manager import DB_PATH
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT scan_at FROM access_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "never"
    except Exception:
        return "unknown"


def opt_report():
    """Print self-match and cross-match accuracy report."""
    users = list_users()
    print()
    total_tmpl = sum(u["sample_count"] for u in users)
    print(f"  Users: {len(users)}   Templates: {total_tmpl}")
    print(f"  Last scan: {_report_last_scan()}")
    print()

    if not users:
        print("  No users enrolled — nothing to report.")
        return

    sig     = get_all_signatures()
    uid_map = _get_uid_map()

    _report_self_match(users, uid_map, sig)

    if len(users) >= 2:
        _report_cross_match(users, uid_map, sig)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    _banner()
    print("  Initialising database...")
    init_db()

    print("  Loading search engine...")
    engine = SearchEngine(n_workers=4)

    print("  Loading MediaPipe landmarker...")
    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        engine.shutdown()
        sys.exit(1)

    print("  Ready.\n")

    try:
        _run_loop(landmarker, engine)
    finally:
        engine.shutdown()
        if CAMERA_AVAILABLE and picam2 is not None:
            picam2.stop()
        print("\n  Goodbye.\n")


def _run_loop(landmarker, engine):
    """Interactive menu loop. Runs until the user presses Q."""
    while True:
        _menu()
        choice = input("  Choice: ").strip().lower()
        print()

        if choice == "1":
            opt_enroll(landmarker, engine)
        elif choice == "2":
            opt_scan(landmarker, engine)
        elif choice == "3":
            opt_list()
        elif choice == "4":
            opt_delete(engine)
        elif choice == "5":
            opt_report()
        elif choice in ("q", "quit", "exit"):
            break
        else:
            print("  Unknown option. Enter 1–5 or Q.\n")


if __name__ == "__main__":
    main()
