#!/usr/bin/env python3
"""
palm_vein_matcher.py
---------------------
Integrated Palm Vein Feature Extraction & Matching Engine
Based on Ma et al. (2017), IET Biometrics.

Features:
  - Structure Tensor Dominant Orientation Estimation (0, 30, 60, 90, 120, 150 deg)
  - Adaptive Sigma & Mu Frequency Mapping
  - DC-Free 2D Gabor Kernel Filtering with Padded Context
  - VeinCode Binarization (VR, VI)
  - Shift-Tolerant Modified Normalized Hamming Distance (MNHD) Matching
"""

import sys
import os
import time
import numpy as np
import cv2
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------
# Configuration Constants
# --------------------------------------------------------------------------
BLOCK_SIZE = 32
ROI_SIZE = 256
GABOR_KSIZE = 15
ORIENTATIONS_DEG = [0, 30, 60, 90, 120, 150]

MAX_DISPLACEMENT = 8
MATCH_THRESHOLD = 0.38  # Standard verification threshold for MNHD


# --------------------------------------------------------------------------
# Stage 1: Adaptive Parameter Estimation
# --------------------------------------------------------------------------
def estimate_orientation(block):
    """
    Computes dominant orientation of a 32x32 block via Structure Tensor
    and quantizes it to the nearest of {0, 30, 60, 90, 120, 150} degrees.
    """
    block_u8 = np.clip(block * 255, 0, 255).astype(np.uint8)

    dx = cv2.Sobel(block_u8, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(block_u8, cv2.CV_64F, 0, 1, ksize=3)

    Jxx = gaussian_filter(dx * dx, sigma=1.0)
    Jxy = gaussian_filter(dx * dy, sigma=1.0)
    Jyy = gaussian_filter(dy * dy, sigma=1.0)

    angle = 0.5 * np.arctan2(2 * Jxy.mean(), (Jxx - Jyy).mean())
    angle_deg = np.rad2deg(angle) % 180

    candidates = np.array(ORIENTATIONS_DEG, dtype=float)
    diffs = np.abs(candidates - angle_deg)
    diffs = np.minimum(diffs, 180.0 - diffs)
    return np.deg2rad(candidates[np.argmin(diffs)])


def estimate_sigma(block):
    """
    Maps local intensity deviation to one of four discrete sigma levels (Eq. 21).
    Expects block normalized in [0, 1].
    """
    D = np.std(block.astype(np.float64))
    if D <= 0.05:
        return 1.0               # Flat / uniform region
    elif D <= 0.10:
        return np.sqrt(2)        # ~1.414 — slowly changing
    elif D <= 0.18:
        return 2.0 * np.sqrt(2)  # ~2.828 — moderate texture
    else:
        return 4.0 * np.sqrt(2)  # ~5.657 — rich vascular texture


_SIGMA_TO_MU_RAW = {
    1.0:            0.00,
    np.sqrt(2):     0.12,
    2.0*np.sqrt(2): 0.80,
    4.0*np.sqrt(2): 2.00,
}

def sigma_to_mu(sigma):
    """Normalized carrier center frequency (cycles/pixel)."""
    closest = min(_SIGMA_TO_MU_RAW, key=lambda k: abs(k - sigma))
    return _SIGMA_TO_MU_RAW[closest] / BLOCK_SIZE


# --------------------------------------------------------------------------
# Stage 2: Gabor Kernel Construction & Filtering
# --------------------------------------------------------------------------
def gabor_kernel(sigma, mu, theta, ksize=GABOR_KSIZE):
    """Generates DC-free real and imaginary 2D Gabor kernels."""
    k = ksize // 2
    y, x = np.mgrid[-k:k+1, -k:k+1].astype(np.float64)

    gauss = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    carrier = 2.0 * np.pi * mu * (x * np.cos(theta) + y * np.sin(theta))

    real = gauss * np.cos(carrier)
    imag = gauss * np.sin(carrier)

    # DC bias removal
    real -= real.mean()
    imag -= imag.mean()

    return real, imag


def extract_padded_block(roi, r, c, pad):
    """Extracts a block with symmetric padding across boundary margins."""
    H, W = roi.shape
    r0, r1 = r - pad, r + BLOCK_SIZE + pad
    c0, c1 = c - pad, c + BLOCK_SIZE + pad

    cr0, cr1 = max(0, r0), min(H, r1)
    cc0, cc1 = max(0, c0), min(W, c1)
    patch = roi[cr0:cr1, cc0:cc1]

    pt = cr0 - r0
    pb = r1 - cr1
    pl = cc0 - c0
    pr = c1 - cc1
    if pt or pb or pl or pr:
        patch = np.pad(patch, ((pt, pb), (pl, pr)), mode='symmetric')
    return patch


# --------------------------------------------------------------------------
# Stage 3: VeinCode Feature Extraction
# --------------------------------------------------------------------------
def extract_veincode(roi_input, ksize=GABOR_KSIZE, debug=False):
    """
    Extracts binary VeinCode templates (VR, VI) from a preprocessed palm ROI.
    Accepts file path, uint8 array, or float array.
    """
    if isinstance(roi_input, str):
        roi = cv2.imread(roi_input, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise FileNotFoundError(f"Cannot load image from {roi_input}")
    else:
        roi = roi_input.copy()

    roi = roi.astype(np.float64)
    if roi.shape != (ROI_SIZE, ROI_SIZE):
        roi = cv2.resize(roi.astype(np.uint8), (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    if roi.max() > 1.0:
        roi = roi / 255.0

    VR = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
    VI = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)

    pad = ksize // 2
    n_blocks = ROI_SIZE // BLOCK_SIZE

    sigma_map = np.zeros((n_blocks, n_blocks))
    theta_map = np.zeros((n_blocks, n_blocks))
    mu_map = np.zeros((n_blocks, n_blocks))

    for bi, r in enumerate(range(0, ROI_SIZE, BLOCK_SIZE)):
        for bj, c in enumerate(range(0, ROI_SIZE, BLOCK_SIZE)):
            block = roi[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]

            theta = estimate_orientation(block)
            sigma = estimate_sigma(block)
            mu = sigma_to_mu(sigma)

            sigma_map[bi, bj] = sigma
            theta_map[bi, bj] = np.rad2deg(theta)
            mu_map[bi, bj] = mu

            real_k, imag_k = gabor_kernel(sigma, mu, theta, ksize=ksize)
            patch = extract_padded_block(roi, r, c, pad)

            CR = fftconvolve(patch, real_k, mode='same')[pad:pad+BLOCK_SIZE, pad:pad+BLOCK_SIZE]
            CI = fftconvolve(patch, imag_k, mode='same')[pad:pad+BLOCK_SIZE, pad:pad+BLOCK_SIZE]

            # Binarize real and imaginary phase components
            VR[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = (CR >= 0).astype(np.uint8)
            if mu == 0.0:
                VI[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = 0
            else:
                VI[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = (CI >= 0).astype(np.uint8)

    result = {'VR': VR, 'VI': VI}
    if debug:
        result.update({'sigma_map': sigma_map, 'theta_map': theta_map, 'mu_map': mu_map})
    return result


# --------------------------------------------------------------------------
# Stage 4: MNHD Template Matching
# --------------------------------------------------------------------------
def _shifted_overlap(P, Q, s, t):
    N = P.shape[0]
    r0 = max(0, t)
    r1 = min(N, N + t)
    c0 = max(0, s)
    c1 = min(N, N + s)
    if r1 <= r0 or c1 <= c0:
        return None, None, 0, 0
    P_crop = P[r0 - t:r1 - t, c0 - s:c1 - s]
    Q_crop = Q[r0:r1, c0:c1]
    return P_crop, Q_crop, (r1 - r0), (c1 - c0)


def normalized_hamming_distance(template, target, s, t):
    """Computes Normalized Hamming Distance between two shifted VeinCodes."""
    PR, PI = template['VR'], template['VI']
    QR, QI = target['VR'], target['VI']

    PR_c, QR_c, h, w = _shifted_overlap(PR, QR, s, t)
    if PR_c is None:
        return 1.0

    PI_c, QI_c, _, _ = _shifted_overlap(PI, QI, s, t)
    xor_R = np.logical_xor(PR_c, QR_c)
    xor_I = np.logical_xor(PI_c, QI_c)

    denom = 2 * h * w
    if denom == 0:
        return 1.0
    return (xor_R.sum() + xor_I.sum()) / denom


def match_templates(template, target, max_disp=MAX_DISPLACEMENT):
    """
    Translates template over [-max_disp, max_disp] in X and Y
    to find minimum Modified Normalized Hamming Distance (MNHD).
    """
    best_dist = 1.0
    for s in range(-max_disp, max_disp + 1):
        for t in range(-max_disp, max_disp + 1):
            d = normalized_hamming_distance(template, target, s, t)
            if d < best_dist:
                best_dist = d
    return best_dist


def verify_pair(template_img, probe_img, threshold=MATCH_THRESHOLD, max_disp=MAX_DISPLACEMENT):
    """Extracts features and compares two palm ROI inputs."""
    code1 = extract_veincode(template_img)
    code2 = extract_veincode(probe_img)
    distance = match_templates(code1, code2, max_disp=max_disp)
    return (distance <= threshold), distance


# --------------------------------------------------------------------------
# CLI & Verification Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_path1 = sys.argv[1]
        img_path2 = sys.argv[2]

        print(f"Comparing:")
        print(f"  Template: {img_path1}")
        print(f"  Probe:    {img_path2}")

        t0 = time.time()
        is_same_person, dist = verify_pair(img_path1, img_path2)
        elapsed = time.time() - t0

        print(f"\n--- Verification Results ---")
        print(f"  MNHD Distance : {dist:.4f}")
        print(f"  Threshold     : {MATCH_THRESHOLD}")
        print(f"  Authentication: {'MATCH (Same Hand)' if is_same_person else 'NON-MATCH (Different Subject)'}")
        print(f"  Execution Time: {elapsed:.3f}s")

    elif len(sys.argv) == 2:
        img_path = sys.argv[1]
        print(f"Extracting VeinCode for {img_path}...")
        t0 = time.time()
        code = extract_veincode(img_path, debug=True)
        print(f"Extraction completed in {time.time() - t0:.3f}s")
        print(f"VR Ones Ratio: {code['VR'].mean():.4f}")
        print(f"VI Ones Ratio: {code['VI'].mean():.4f}")

        # Self-match verification
        self_dist = match_templates(code, code, max_disp=2)
        print(f"Self-Match MNHD: {self_dist:.4f} (Expected: 0.0000)")

    else:
        print("Usage:")
        print("  1. Verification:  python3 palm_vein_matcher.py <template_roi.png> <probe_roi.png>")
        print("  2. Feature Test:  python3 palm_vein_matcher.py <roi.png>")
