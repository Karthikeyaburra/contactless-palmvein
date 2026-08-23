#!/usr/bin/env python3
"""
Palm Vein ROI Extraction Pipeline -- MediaPipe Hand-Landmark Version
======================================================================

WHY THIS REPLACES THE CONVEXITY-DEFECT APPROACH
-------------------------------------------------
The original contour-defect method reads Pv1/Pv2 (index-middle and
ring-little "valleys") off the concavity DEPTH of the hand's silhouette.
That depth is directly proportional to how far apart your fingers are
spread: closed fingers -> shallow/undetectable valleys -> unstable or
missing landmarks -> ROI jumps around between captures of the same hand.

This version replaces that geometry entirely with MediaPipe Hands, a
model trained to regress 21 hand-joint positions directly from the image.
It does not care whether your fingers are spread or together, and it is
far less sensitive to wrist/forearm framing, shadows, or rotation --
because it's predicting anatomical joints, not measuring silhouette
concavity.

Landmark index reference (MediaPipe Hands, 21 points):
    0  = wrist
    5  = index MCP knuckle     9  = middle MCP knuckle
    13 = ring MCP knuckle      17 = pinky MCP knuckle
(MCP = the knuckle where the finger meets the palm -- these sit right at
the base of each finger, exactly where the inter-finger valleys are.)

  Pv1 (index-middle valley) ~= midpoint(landmark 5, landmark 9)
  Pv2 (ring-little valley)  ~= midpoint(landmark 13, landmark 17)

SETUP (do this once, on a machine with normal internet access):

    pip install mediapipe opencv-python

    # Download the hand landmark model (~ a few MB):
    curl -L -o hand_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Usage:
    python3 mediapipe_img.py path/to/hand_image.jpg [path/to/hand_landmarker.task]
"""

import sys
import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

DEFAULT_MODEL_PATH = "hand_landmarker.task"


# ---------- Stage 0: Landmarker factory (call ONCE at startup) ----------

def build_landmarker(model_path=DEFAULT_MODEL_PATH):
    """
    Creates and returns a persistent HandLandmarker instance.
    Call this ONCE at application startup.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: '{model_path}'\n"
            f"Download: curl -L -o {model_path} "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    return HandLandmarker.create_from_options(options)


# ---------- Stage 1: Hand Landmark Detection ----------

def detect_hand_landmarks(gray_img, landmarker):
    """
    Runs MediaPipe HandLandmarker on the image and returns all 21 (x, y)
    pixel coordinates for the first detected hand.

    landmarker: an already-created HandLandmarker instance.
    Do NOT create or destroy the landmarker inside this function.
    """
    if gray_img.shape[0] < 200 or gray_img.shape[1] < 200:
        raise ValueError(
            f"Image too small for landmark detection: {gray_img.shape}. "
            f"Minimum 200x200 required."
        )

    rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        raise ValueError("No hand detected. Check palm placement and lighting.")

    h, w = gray_img.shape
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]


def extract_valleys_from_landmarks(landmarks_px):
    """
    Derives Pv1 (index-middle) and Pv2 (ring-little) from MCP knuckle
    landmarks. Stable regardless of finger spread, unlike contour defects.
    """
    index_mcp  = np.array(landmarks_px[5],  dtype=float)
    middle_mcp = np.array(landmarks_px[9],  dtype=float)
    ring_mcp   = np.array(landmarks_px[13], dtype=float)
    pinky_mcp  = np.array(landmarks_px[17], dtype=float)

    pv1 = tuple(((index_mcp + middle_mcp) / 2.0).astype(int))
    pv2 = tuple(((ring_mcp  + pinky_mcp)  / 2.0).astype(int))

    return pv1, pv2


# ---------- Stage 1b: Hand Silhouette (still needed for ROI direction check) ----------

def segment_hand(gray_img):
    """Unchanged from the original pipeline -- still used to find the
    palm's distance-transform peak for choosing which side of the
    Pv1-Pv2 baseline the ROI should be cropped toward."""
    norm    = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(norm, (11, 11), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled  = np.zeros_like(clean)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

    return filled


# ---------- Stage 2: Coordinate Alignment & Scaled ROI Extraction ----------

def extract_ma2017_scaled_roi(gray_img, pv1, pv2, binary_mask, target_size=256,
                               scale_factor=1.5, offset_factor=0.35):
    dx = pv2[0] - pv1[0]
    dy = pv2[1] - pv1[1]
    dist_pv   = np.hypot(dx, dy)
    angle_deg = np.degrees(np.arctan2(dy, dx))

    mid_x = (pv1[0] + pv2[0]) / 2.0
    mid_y = (pv1[1] + pv2[1]) / 2.0

    h, w = gray_img.shape
    M            = cv2.getRotationMatrix2D((mid_x, mid_y), angle_deg, 1.0)
    rotated_gray = cv2.warpAffine(gray_img,    M, (w, h), flags=cv2.INTER_LINEAR)
    rotated_bin  = cv2.warpAffine(binary_mask, M, (w, h), flags=cv2.INTER_NEAREST)

    pt_mid_h = np.array([mid_x, mid_y, 1.0])
    rot_mid  = M.dot(pt_mid_h)
    mx_r, my_r = int(rot_mid[0]), int(rot_mid[1])

    dist_map      = cv2.distanceTransform(rotated_bin, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_map)
    direction     = 1 if max_loc[1] > my_r else -1

    L        = int(dist_pv * scale_factor)
    offset_d0 = int(dist_pv * offset_factor)

    x1 = int(mx_r - L / 2)
    x2 = int(mx_r + L / 2)

    if direction > 0:
        y1 = my_r + offset_d0
        y2 = y1 + L
    else:
        y2 = my_r - offset_d0
        y1 = y2 - L

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi_patch = rotated_gray[y1:y2, x1:x2]
    if roi_patch.size == 0 or roi_patch.shape[0] < 10 or roi_patch.shape[1] < 10:
        raise ValueError("Invalid ROI bounding box coordinates.")

    roi_normalized = cv2.resize(roi_patch, (target_size, target_size),
                                interpolation=cv2.INTER_CUBIC)

    return roi_normalized, (x1, y1, x2, y2), rotated_gray


# ---------- Stage 3: Vascular Feature Enhancement ----------

def enhance_roi_vessels(roi_img):
    """
    Applies bilateral filter + CLAHE to enhance vein contrast.
    Returns only the CLAHE-enhanced ROI (clahe_roi).
    """
    stretched   = cv2.normalize(roi_img, None, 0, 255, cv2.NORM_MINMAX)
    smooth      = cv2.bilateralFilter(stretched, d=7, sigmaColor=35, sigmaSpace=35)
    clahe       = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    clahe_roi   = clahe.apply(smooth)
    return clahe_roi


# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mediapipe_img.py <path_to_image> [model_path]")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' does not exist.")
        sys.exit(1)

    raw_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        print(f"Error: Unable to decode image from '{img_path}'.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    try:
        stretched_raw = cv2.normalize(raw_gray, None, 0, 255, cv2.NORM_MINMAX)

        # Build a persistent landmarker for this standalone run.
        landmarker = build_landmarker(model_path)

        # 1. Landmark-based valley detection (replaces convexity defects)
        landmarks_px = detect_hand_landmarks(stretched_raw, landmarker)
        pv1, pv2 = extract_valleys_from_landmarks(landmarks_px)

        # 2. Silhouette used internally for palm-side direction checking.
        hand_mask = segment_hand(stretched_raw)

        # 3. Scaled ROI.
        roi_256, box, rotated_img = extract_ma2017_scaled_roi(
            stretched_raw, pv1, pv2, hand_mask,
            target_size=256, scale_factor=1.5, offset_factor=0.35
        )

        # 4. CLAHE preprocessing. Returns only the CLAHE ROI.
        clahe_roi = enhance_roi_vessels(roi_256)
        cv2.imwrite(f"{base_name}_roi_clahe.png", clahe_roi)

        print(f"Pipeline executed successfully for '{img_path}':")
        print(f" - Landmark Valleys: Pv1={pv1}, Pv2={pv2}")

    except (ValueError, FileNotFoundError) as err:
        print(f"Pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
