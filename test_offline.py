#!/usr/bin/env python3
"""
test_offline.py
---------------
Offline palm vein pipeline test — no camera or Raspberry Pi required.
Uses saved NIR palm images to exercise the full pipeline end-to-end.

Usage
-----
# Enroll with 1-6 images (accepts any count ≥ 1):
python3 test_offline.py --enroll alice right_1.png right_2.png right_3.png right_4.png

# Identify a single image:
python3 test_offline.py --scan right_test.png

# Full accuracy report (self-match + cross-match):
python3 test_offline.py --report
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from db_manager import (
    init_db, enroll_user, user_exists, list_users,
    get_all_signatures, get_templates_by_ids, compute_signature,
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
# Pipeline helper
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, landmarker) -> tuple:
    """
    Load one image and run the full biometric pipeline.
    Returns (clahe_roi, veincode, elapsed_seconds).
    Raises ValueError with a descriptive message on any failure.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    t0        = time.time()
    stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    landmarks = detect_hand_landmarks(stretched, landmarker)
    pv1, pv2  = extract_valleys_from_landmarks(landmarks)
    hand_mask = segment_hand(stretched)
    roi_256, _, _ = extract_ma2017_scaled_roi(
        stretched, pv1, pv2, hand_mask,
        target_size=256, scale_factor=1.5, offset_factor=0.35,
    )
    clahe_roi = enhance_roi_vessels(roi_256)
    code      = extract_veincode(clahe_roi)
    elapsed   = time.time() - t0
    return clahe_roi, code, elapsed


# ---------------------------------------------------------------------------
# Consistency check (shared between enroll and report)
# ---------------------------------------------------------------------------

def consistency_check(veincode_list: list) -> list:
    """
    Check pairwise MNHD between all collected templates.
    Returns list of (i, j, score) for pairs with score > 0.45.
    """
    inconsistent = []
    for a in range(len(veincode_list)):
        for b in range(a + 1, len(veincode_list)):
            score = match_templates(veincode_list[a], veincode_list[b])
            if score > 0.45:
                inconsistent.append((a, b, score))
    return inconsistent


# ---------------------------------------------------------------------------
# --enroll
# ---------------------------------------------------------------------------

def cmd_enroll(args, landmarker, engine):
    """Enroll a user from 1-6 image files."""
    username    = args.enroll[0]
    image_paths = args.enroll[1:]

    if not image_paths:
        print("ERROR: Provide at least 1 image path after the username.")
        sys.exit(1)

    if user_exists(username):
        print(f"ERROR: User '{username}' already enrolled. Delete first to re-enroll.")
        sys.exit(1)

    print(f"\nEnrolling '{username}' with {len(image_paths)} image(s)...")
    print("-" * 55)

    veincode_list = []
    for path in image_paths:
        try:
            _, code, elapsed = run_pipeline(path, landmarker)
            print(f"  OK    {os.path.basename(path):<30}  "
                  f"VR={code['VR'].mean():.3f}  {elapsed:.2f}s")
            veincode_list.append(code)
        except ValueError as e:
            print(f"  SKIP  {os.path.basename(path):<30}  {e}")

    if not veincode_list:
        print("\nEnrollment aborted: no images processed successfully.")
        sys.exit(1)

    _print_consistency(veincode_list)

    enroll_user(username, veincode_list)
    engine.refresh_cache()

    print(f"\nENROLLED: '{username}' — {len(veincode_list)} sample(s) stored.")


def _print_consistency(veincode_list: list):
    """Print consistency check results for a set of veincode dicts."""
    print(f"\nConsistency check ({len(veincode_list)} samples)...")
    inconsistent = consistency_check(veincode_list)
    if inconsistent:
        print(f"  WARNING: {len(inconsistent)} pair(s) with high MNHD distance:")
        for a, b, s in inconsistent:
            print(f"    Sample {a+1} vs Sample {b+1}: {s:.4f}  (threshold 0.45)")
        print("  Consider re-enrolling for better accuracy.")
    else:
        print("  All samples consistent — enrollment quality: GOOD")


# ---------------------------------------------------------------------------
# --scan
# ---------------------------------------------------------------------------

def cmd_scan(args, landmarker, engine):
    """Run the identification pipeline against a single image."""
    path = args.scan
    print(f"\nScanning: {path}")
    print("-" * 55)

    try:
        _, probe_code, elapsed_extract = run_pipeline(path, landmarker)
    except ValueError as e:
        print(f"Scan failed: {e}")
        sys.exit(1)

    t_match0              = time.time()
    username, score       = engine.identify(probe_code)
    elapsed_match         = time.time() - t_match0

    _print_scan_result(username, score, elapsed_extract, elapsed_match)


def _print_scan_result(username, score, elapsed_extract, elapsed_match):
    """Print the identification outcome in a human-readable table."""
    if username:
        result_str = f"AUTHENTICATED as {username}"
    else:
        result_str = "NOT RECOGNISED"

    print(f"\nResult    : {result_str}")
    print(f"Score     : {score:.4f}  (threshold: {MATCH_THRESHOLD:.4f})")
    print(f"Extract   : {elapsed_extract:.2f}s")
    print(f"Match     : {elapsed_match:.3f}s")
    print(f"Total     : {elapsed_extract + elapsed_match:.2f}s")


# ---------------------------------------------------------------------------
# --report helpers
# ---------------------------------------------------------------------------

def _get_uid_map() -> dict:
    """
    Return {username -> user_id} for all active users by inspecting the
    signature cache. Avoids adding a direct SQL query outside db_manager.
    """
    from db_manager import get_username
    sig_data = get_all_signatures()
    uid_map  = {}
    for uid in sig_data['user_ids']:
        if uid not in uid_map.values():
            try:
                uname = get_username(uid)
                uid_map[uname] = uid
            except KeyError:
                pass
    return uid_map


def _user_template_ids(user_id: int, sig_data: dict) -> list:
    """Return all template IDs belonging to user_id."""
    return [
        tid for tid, uid in zip(sig_data['template_ids'], sig_data['user_ids'])
        if uid == user_id
    ]


def _self_match_scores(template_list: list) -> list:
    """Return all pairwise MNHD scores for templates of the same user."""
    scores = []
    for a in range(len(template_list)):
        for b in range(a + 1, len(template_list)):
            scores.append(match_templates(template_list[a], template_list[b]))
    return scores


# ---------------------------------------------------------------------------
# --report
# ---------------------------------------------------------------------------

def cmd_report(engine):
    """Print a full accuracy report: enrolled users, self-match, cross-match."""
    users    = list_users()
    sig_data = get_all_signatures()

    _report_users(users)

    if not users:
        return

    uid_map = _get_uid_map()
    _report_self_match(users, uid_map, sig_data)

    if len(users) >= 2:
        _report_cross_match(users, uid_map, sig_data)


def _report_users(users: list):
    """Print the enrolled user table."""
    print("\n=== Enrolled Users ===")
    print(f"{'Username':<20} {'Samples':>7}   {'Enrolled At'}")
    print("-" * 55)
    total = 0
    for u in users:
        print(f"{u['username']:<20} {u['sample_count']:>7}   {u['enrolled_at']}")
        total += u['sample_count']
    print(f"\nTotal: {len(users)} user(s), {total} template(s)")


def _report_self_match(users: list, uid_map: dict, sig_data: dict):
    """Compute and print self-match scores (same user, different samples)."""
    print("\n=== Self-Match Verification ===")
    print("All scores should be < 0.35 for good enrollment quality")
    print(f"{'Username':<20} {'Min':>6}  {'Avg':>6}  {'Max':>6}  {'Quality'}")
    print("-" * 55)

    for u in users:
        uid  = uid_map.get(u['username'])
        tids = _user_template_ids(uid, sig_data)

        if len(tids) < 2:
            print(f"{u['username']:<20}  (only 1 sample \u2014 skip)")
            continue

        templates = get_templates_by_ids(tids)
        scores    = _self_match_scores(templates)

        mn  = min(scores)
        avg = sum(scores) / len(scores)
        mx  = max(scores)
        quality = "GOOD" if mx < 0.35 else "WARN \u2014 re-enroll advised"
        print(f"{u['username']:<20} {mn:>6.4f}  {avg:>6.4f}  {mx:>6.4f}  {quality}")


def _report_cross_match(users: list, uid_map: dict, sig_data: dict):
    """Compare the best template of every pair of different users."""
    print("\n=== Cross-Match Test ===")
    print("All scores should be > 0.45 to avoid false-accept risk")
    print(f"{'Pair':<35} {'Score':>6}  {'Status'}")
    print("-" * 55)

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            uid1   = uid_map.get(u1['username'])
            uid2   = uid_map.get(u2['username'])

            tids1 = _user_template_ids(uid1, sig_data)[:1]
            tids2 = _user_template_ids(uid2, sig_data)[:1]
            if not tids1 or not tids2:
                continue

            t1    = get_templates_by_ids(tids1)[0]
            t2    = get_templates_by_ids(tids2)[0]
            score = match_templates(t1, t2)
            pair  = f"{u1['username']} vs {u2['username']}"
            status = "OK" if score > 0.45 else "WARN \u2014 possible false-accept risk"
            print(f"{pair:<35} {score:>6.4f}  {status}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline palm vein pipeline test \u2014 no camera required"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--enroll', nargs='+', metavar=('USERNAME', 'IMAGE'),
        help='Enroll a user: --enroll <username> <img1> [img2 ... img6]',
    )
    group.add_argument(
        '--scan', metavar='IMAGE',
        help='Identify an image: --scan <image_path>',
    )
    group.add_argument(
        '--report', action='store_true',
        help='Print accuracy report with self-match and cross-match analysis',
    )

    args = parser.parse_args()

    init_db()
    engine = SearchEngine(n_workers=4)

    try:
        if args.enroll:
            landmarker = build_landmarker(DEFAULT_MODEL_PATH)
            cmd_enroll(args, landmarker, engine)

        elif args.scan:
            landmarker = build_landmarker(DEFAULT_MODEL_PATH)
            cmd_scan(args, landmarker, engine)

        elif args.report:
            cmd_report(engine)

    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
