#!/usr/bin/env python3
"""
ui.py
-----
Fullscreen Neobrutalism Pygame UI for Palm Vein Recognition System.
Designed for Raspberry Pi 5 with 5-inch portrait display (~480x854px).
Runs windowed on laptop/dev systems and fullscreen on Raspberry Pi.
"""

import os
import sys
import time
import math
import platform
import threading
import pygame
import cv2
import numpy as np

# Core system imports
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

# ---------------------------------------------------------------------------
# Hardware & Display Configuration
# ---------------------------------------------------------------------------
ON_PI = platform.machine().startswith('aarch') or platform.machine() in ('armv7l', 'armv6l')
SCREEN_W = 480
SCREEN_H = 854

TOP_H = 64
NAV_H = 84
CONTENT_TOP = TOP_H
CONTENT_BOTTOM = SCREEN_H - NAV_H
CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP

CAPTURE_DIR = "captures"
ROI_DIR = "roi_clahe"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Neobrutalism Vibrant Color Palette (No dull colors, high saturation)
# ---------------------------------------------------------------------------
BG          = (255, 253, 240)   # #FFFDF0 - Warm Ultra-Clean Cream
BLACK       = (18,  18,  18)    # #121212 - Deep Ink Black
WHITE       = (255, 255, 255)   # #FFFFFF - Crisp Card White
YELLOW      = (255, 222, 89)    # #FFDE59 - Electric Accent Yellow
ELECTRIC_BL = (56,  189, 248)   # #38BDF8 - Vibrant Sky/Cyan Blue
NEON_CYAN   = (0,   240, 255)   # #00F0FF - Electric Cyan pop (Auth Success)
HOT_PINK    = (255, 64,  129)   # #FF4081 - Electric Magenta / Reject Alert
ORANGE      = (255, 122, 0)     # #FF7A00 - Electric Sunburst Orange
PURPLE      = (168, 85,  247)   # #A855F7 - Royal Violet
LIME        = (204, 255, 0)     # #CCFF00 - Ultra Electric Lime
GRAY_LIGHT  = (241, 243, 245)   # #F1F3F5 - Neutral Light Card Gray
GRAY_MUTED  = (140, 145, 155)   # #8C919B - Secondary text gray
CARD_BG     = (255, 255, 255)   # White card faces
NAV_BG      = (255, 253, 240)   # Same as BG
SHADOW      = (18,  18,  18)    # Solid hard black shadow

# ---------------------------------------------------------------------------
# Camera Setup (Graceful Fallback)
# ---------------------------------------------------------------------------
picam2 = None
preview_cfg = None
still_cfg = None
CAMERA_AVAILABLE = False

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
        "AeEnable": False,
        "ExposureTime": 5000,
        "AnalogueGain": 1.0,
    })
    CAMERA_AVAILABLE = True
except Exception:
    picam2 = None
    CAMERA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core Image Processing (Local Copy)
# ---------------------------------------------------------------------------
def process_frame(gray: np.ndarray, landmarker) -> tuple:
    """Raw grayscale frame -> (clahe_roi, veincode). Raises ValueError on failure."""
    if gray.shape[0] < 200 or gray.shape[1] < 200:
        raise ValueError("Image too small for landmark detection.")
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


def capture_gray() -> np.ndarray:
    """Capture raw still image as grayscale array. Returns dummy pattern if no camera."""
    if CAMERA_AVAILABLE and picam2 is not None:
        picam2.switch_mode(still_cfg)
        frame = picam2.capture_array()
        picam2.switch_mode(preview_cfg)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Offline mock/sample image fallback for testing UI without camera
    sample_paths = [
        os.path.join(CAPTURE_DIR, "myright", "capture_000.png"),
        os.path.join(CAPTURE_DIR, "myleft", "capture_005.png"),
        "1100_2.bmp"
    ]
    for p in sample_paths:
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    # Synthetic frame if no sample exists
    synthetic = np.full((480, 640), 128, dtype=np.uint8)
    cv2.circle(synthetic, (320, 240), 120, 220, -1)
    return synthetic


def _save_capture(gray: np.ndarray, username: str, mode: str, idx: int = 0) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{username}_enroll_{idx}_{ts}.png" if mode == "enroll" else f"{username}_scan_{ts}.png"
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, gray)
    return path


def _save_roi(roi: np.ndarray, username: str, mode: str, idx: int = 0) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{username}_enroll_{idx}_{ts}_clahe.png" if mode == "enroll" else f"{username}_scan_{ts}_clahe.png"
    path = os.path.join(ROI_DIR, name)
    cv2.imwrite(path, roi)
    return path


# ---------------------------------------------------------------------------
# Background Pipeline Worker (Non-blocking UI)
# ---------------------------------------------------------------------------
class PipelineWorker(threading.Thread):
    def __init__(self, gray: np.ndarray, landmarker, task_type="scan", extra=None):
        super().__init__(daemon=True)
        self.gray = gray
        self.landmarker = landmarker
        self.task_type = task_type
        self.extra = extra
        self.result = None
        self.error = None
        self.done = False

    def run(self):
        try:
            self.result = process_frame(self.gray, self.landmarker)
        except Exception as e:
            self.error = str(e)
        finally:
            self.done = True


# ---------------------------------------------------------------------------
# Font Manager
# ---------------------------------------------------------------------------
class FontBook:
    def __init__(self):
        pygame.font.init()
        # High impact Neobrutalist bold fonts
        candidates = ["Archivo Black", "Arial Black", "Trebuchet MS", "Impact", "Helvetica", "DejaVu Sans"]
        self.font_name = None
        for name in candidates:
            if name.lower() in [f.lower() for f in pygame.font.get_fonts()]:
                self.font_name = name
                break
        
        self.huge  = self._load_font(52, bold=True)
        self.title = self._load_font(28, bold=True)
        self.h2    = self._load_font(22, bold=True)
        self.body  = self._load_font(16, bold=True)
        self.small = self._load_font(13, bold=True)
        self.mono  = pygame.font.SysFont("Consolas", 14, bold=True) or self.small

    def _load_font(self, size, bold=True):
        if self.font_name:
            return pygame.font.SysFont(self.font_name, size, bold=bold)
        return pygame.font.Font(None, size)


# ---------------------------------------------------------------------------
# Neobrutalism Drawing Primitives
# ---------------------------------------------------------------------------
def draw_neo_card(surface, rect, fill_color, border_color=BLACK,
                  shadow_offset=5, border_radius=12, border_width=3):
    """Draws hard-shadow neobrutalist card with bold border."""
    if shadow_offset > 0:
        s_rect = pygame.Rect(rect.x + shadow_offset, rect.y + shadow_offset, rect.width, rect.height)
        pygame.draw.rect(surface, SHADOW, s_rect, border_radius=border_radius)
    pygame.draw.rect(surface, fill_color, rect, border_radius=border_radius)
    if border_width > 0:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=border_radius)


def draw_neo_button(surface, rect, fill_color, text, font, text_color=BLACK,
                    shadow_offset=4, border_radius=10, is_pressed=False, is_disabled=False):
    """Renders chunky neobrutalism clickable button with press depression effect."""
    draw_rect = rect.copy()
    current_shadow = shadow_offset
    if is_disabled:
        fill_color = GRAY_LIGHT
        text_color = GRAY_MUTED
        current_shadow = 2
    elif is_pressed:
        draw_rect.x += shadow_offset - 1
        draw_rect.y += shadow_offset - 1
        current_shadow = 1

    draw_neo_card(surface, draw_rect, fill_color, BLACK, current_shadow, border_radius, 3)
    
    txt_surf = font.render(text, True, text_color)
    txt_rect = txt_surf.get_rect(center=draw_rect.center)
    surface.blit(txt_surf, txt_rect)
    return draw_rect


def draw_decorations(surface):
    """Draws iconic geometric neobrutalist graphic accents."""
    # Dot matrix grid
    for row in range(5):
        for col in range(6):
            pygame.draw.circle(surface, (215, 210, 190), (380 + col * 14, 90 + row * 14), 2)
    
    # Graphic Crosses (+)
    def draw_cross(x, y, size=12, color=ORANGE, thickness=3):
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), thickness)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), thickness)
    
    draw_cross(50, 140, 10, ORANGE, 3)
    draw_cross(430, 420, 12, PURPLE, 3)
    draw_cross(40, 680, 10, HOT_PINK, 3)
    
    # Geometric shapes
    pygame.draw.circle(surface, YELLOW, (440, 160), 22)
    pygame.draw.circle(surface, BLACK, (440, 160), 22, 3)
    
    pygame.draw.circle(surface, ELECTRIC_BL, (35, 340), 16)
    pygame.draw.circle(surface, BLACK, (35, 340), 16, 3)
    
    # Angled accent pill
    pill_rect = pygame.Rect(18, 520, 26, 44)
    pygame.draw.rect(surface, LIME, pill_rect, border_radius=13)
    pygame.draw.rect(surface, BLACK, pill_rect, 3, border_radius=13)


# ---------------------------------------------------------------------------
# UI State & Screen Views
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.current_screen = 0  # 0=SCAN, 1=ENROLL, 2=USERS, 3=ADMIN
        self.overlay = None      # None, 'result', 'confirm_delete', 'report_modal'
        self.result_data = {}
        self.result_timer = 0
        self.toast_msg = ""
        self.toast_color = YELLOW
        self.toast_timer = 0
        
        # Enrollment session state
        self.enroll_username = ""
        self.enroll_input_active = True
        self.enroll_samples = []       # list of veincode dicts
        self.enroll_roi_thumbs = []    # list of pygame surfaces
        self.last_quality_msg = ""
        self.last_quality_vr = 0.0
        self.is_processing = False
        
        # Users & Delete state
        self.user_list = []
        self.delete_target = None
        self.user_scroll_offset = 0
        
        # Admin / Report state
        self.report_data = {}
        self.last_scan_info = "None"
        
        # Animation states
        self.scan_pulse_angle = 0.0
        self.cursor_blink = 0


# ---------------------------------------------------------------------------
# Screen Renderers
# ---------------------------------------------------------------------------
def render_top_bar(surface, title, font_book, now):
    """Renders crisp neobrutalism top header."""
    top_rect = pygame.Rect(0, 0, SCREEN_W, TOP_H)
    pygame.draw.rect(surface, BLACK, top_rect)
    pygame.draw.line(surface, BLACK, (0, TOP_H), (SCREEN_W, TOP_H), 3)

    # Title text with icon
    t_surf = font_book.h2.render(title, True, WHITE)
    surface.blit(t_surf, (20, (TOP_H - t_surf.get_height()) // 2))

    # Camera status badge
    badge_rect = pygame.Rect(SCREEN_W - 120, 16, 104, 32)
    cam_color = LIME if CAMERA_AVAILABLE else ORANGE
    draw_neo_card(surface, badge_rect, cam_color, BLACK, 0, 8, 2)
    
    dot_x = badge_rect.x + 14
    dot_y = badge_rect.centery
    pygame.draw.circle(surface, BLACK, (dot_x, dot_y), 5)
    
    status_txt = "NIR CAM" if CAMERA_AVAILABLE else "NO CAM"
    s_surf = font_book.small.render(status_txt, True, BLACK)
    surface.blit(s_surf, (badge_rect.x + 28, badge_rect.centery - s_surf.get_height() // 2))


def render_bottom_nav(surface, state, font_book):
    """Renders neobrutalism pill navigation bar matching the reference style."""
    nav_rect = pygame.Rect(0, SCREEN_H - NAV_H, SCREEN_W, NAV_H)
    pygame.draw.rect(surface, BG, nav_rect)
    pygame.draw.line(surface, BLACK, (0, SCREEN_H - NAV_H), (SCREEN_W, SCREEN_H - NAV_H), 3)

    items = [
        {"label": "SCAN",   "icon": "⚡", "screen": 0},
        {"label": "ENROLL", "icon": "➕", "screen": 1},
        {"label": "USERS",  "icon": "👥", "screen": 2},
        {"label": "ADMIN",  "icon": "⚙️", "screen": 3},
    ]

    item_w = SCREEN_W // len(items)
    for i, item in enumerate(items):
        cx = i * item_w + item_w // 2
        cy = SCREEN_H - NAV_H + NAV_H // 2
        is_active = (state.current_screen == item["screen"])

        if is_active:
            # Active highlighter pill (Neobrutalism badge)
            pill_w = item_w - 18
            pill_h = 44
            pill_rect = pygame.Rect(cx - pill_w // 2, cy - pill_h // 2, pill_w, pill_h)
            draw_neo_card(surface, pill_rect, YELLOW, BLACK, 3, 22, 2)

            txt = font_book.body.render(item["label"], True, BLACK)
            surface.blit(txt, txt.get_rect(center=pill_rect.center))
        else:
            txt = font_book.body.render(item["label"], True, GRAY_MUTED)
            surface.blit(txt, txt.get_rect(center=(cx, cy)))


# ---------------------------------------------------------------------------
# SCREEN 0: SCAN
# ---------------------------------------------------------------------------
def render_screen_scan(surface, state, font_book, now):
    # Main Hero Title
    title = font_book.title.render("PALM VEIN AUTH", True, BLACK)
    surface.blit(title, (24, CONTENT_TOP + 18))
    
    sub = font_book.small.render("Hold palm 10-15cm above sensor", True, BLACK)
    surface.blit(sub, (24, CONTENT_TOP + 52))

    # Scanner Viewport Neo Card
    card_rect = pygame.Rect(32, CONTENT_TOP + 86, SCREEN_W - 64, 300)
    draw_neo_card(surface, card_rect, CARD_BG, BLACK, 6, 16, 4)

    # Scanner graphic / Pulsing Biometric Target
    center = card_rect.center
    state.scan_pulse_angle += 0.05
    pulse_r = int(75 + 8 * math.sin(state.scan_pulse_angle))

    # Concentric target circles
    pygame.draw.circle(surface, BG, center, 95)
    pygame.draw.circle(surface, BLACK, center, 95, 2)
    pygame.draw.circle(surface, ELECTRIC_BL if not state.is_processing else ORANGE, center, pulse_r, 4)
    pygame.draw.circle(surface, YELLOW, center, 32)
    pygame.draw.circle(surface, BLACK, center, 32, 3)

    # Palm symbol / crosshairs inside scanner
    pygame.draw.line(surface, BLACK, (center[0] - 22, center[1]), (center[0] + 22, center[1]), 3)
    pygame.draw.line(surface, BLACK, (center[0], center[1] - 22), (center[0], center[1] + 22), 3)

    # Corner bracket indicators on card
    b_len = 18
    corners = [
        (card_rect.left + 14, card_rect.top + 14),
        (card_rect.right - 14, card_rect.top + 14),
        (card_rect.left + 14, card_rect.bottom - 14),
        (card_rect.right - 14, card_rect.bottom - 14),
    ]
    for cx, cy in corners:
        dx = 1 if cx < card_rect.centerx else -1
        dy = 1 if cy < card_rect.centery else -1
        pygame.draw.line(surface, BLACK, (cx, cy), (cx + dx * b_len, cy), 3)
        pygame.draw.line(surface, BLACK, (cx, cy), (cx, cy + dy * b_len), 3)

    # Status text under scanner card
    status_label = "SENSOR READY" if not state.is_processing else "EXTRACTING BIOMETRICS..."
    s_col = BLACK if not state.is_processing else ORANGE
    st_surf = font_book.body.render(status_label, True, s_col)
    surface.blit(st_surf, st_surf.get_rect(center=(SCREEN_W // 2, card_rect.bottom - 24)))

    # Primary Action "SCAN NOW" Button
    btn_rect = pygame.Rect(32, card_rect.bottom + 26, SCREEN_W - 64, 66)
    btn_text = "SCANNING..." if state.is_processing else "SCAN PALM NOW  →"
    btn_color = YELLOW if not state.is_processing else LIME
    draw_neo_button(surface, btn_rect, btn_color, btn_text, font_book.h2, BLACK, 5, 14, state.is_processing)

    # Last Scan Log Card
    log_rect = pygame.Rect(32, btn_rect.bottom + 22, SCREEN_W - 64, 110)
    draw_neo_card(surface, log_rect, WHITE, BLACK, 4, 12, 3)

    hdr = font_book.small.render("LAST ACTIVITY LOG", True, GRAY_MUTED)
    surface.blit(hdr, (log_rect.x + 16, log_rect.y + 14))

    last_user = state.result_data.get("username") or "None (Standby)"
    last_score = state.result_data.get("score")
    score_str = f"Score: {last_score:.4f}" if last_score is not None else "Threshold: < 0.3800"
    
    u_txt = font_book.body.render(f"User: {last_user}", True, BLACK)
    s_txt = font_book.small.render(score_str, True, BLACK)
    surface.blit(u_txt, (log_rect.x + 16, log_rect.y + 40))
    surface.blit(s_txt, (log_rect.x + 16, log_rect.y + 70))


# ---------------------------------------------------------------------------
# SCREEN 1: ENROLL
# ---------------------------------------------------------------------------
def render_screen_enroll(surface, state, font_book, now):
    title = font_book.title.render("ENROLL NEW USER", True, BLACK)
    surface.blit(title, (24, CONTENT_TOP + 16))

    # 1. Username Input Box
    lbl1 = font_book.small.render("ENTER USERNAME / ID", True, BLACK)
    surface.blit(lbl1, (32, CONTENT_TOP + 56))

    input_rect = pygame.Rect(32, CONTENT_TOP + 78, SCREEN_W - 64, 52)
    in_color = WHITE if state.enroll_input_active else GRAY_LIGHT
    draw_neo_card(surface, input_rect, in_color, BLACK, 4, 10, 3)

    # Display entered text + cursor
    disp_text = state.enroll_username if state.enroll_username else "type username here..."
    text_color = BLACK if state.enroll_username else GRAY_MUTED
    txt_surf = font_book.body.render(disp_text, True, text_color)
    surface.blit(txt_surf, (input_rect.x + 16, input_rect.centery - txt_surf.get_height() // 2))

    if state.enroll_input_active and (now // 500) % 2 == 0:
        cx = input_rect.x + 18 + (txt_surf.get_width() if state.enroll_username else 0)
        pygame.draw.line(surface, BLACK, (cx, input_rect.y + 12), (cx, input_rect.bottom - 12), 2)

    # 2. 6-Sample Progress Indicator Card
    prog_card = pygame.Rect(32, input_rect.bottom + 18, SCREEN_W - 64, 100)
    draw_neo_card(surface, prog_card, WHITE, BLACK, 4, 12, 3)

    p_title = font_book.small.render(f"SAMPLE PROGRESS ({len(state.enroll_samples)} / 6)", True, BLACK)
    surface.blit(p_title, (prog_card.x + 16, prog_card.y + 14))

    # 6 Large Neobrutalist Check Circles
    dot_spacing = (prog_card.width - 32) // 6
    for i in range(6):
        dot_x = prog_card.x + 24 + i * dot_spacing
        dot_y = prog_card.y + 60
        is_captured = i < len(state.enroll_samples)
        
        d_color = LIME if is_captured else GRAY_LIGHT
        pygame.draw.circle(surface, SHADOW, (dot_x + 2, dot_y + 2), 16)
        pygame.draw.circle(surface, d_color, (dot_x, dot_y), 16)
        pygame.draw.circle(surface, BLACK, (dot_x, dot_y), 16, 2)
        
        num_str = "✓" if is_captured else str(i + 1)
        n_surf = font_book.small.render(num_str, True, BLACK)
        surface.blit(n_surf, n_surf.get_rect(center=(dot_x, dot_y)))

    # 3. Capture Action Button
    cap_btn_rect = pygame.Rect(32, prog_card.bottom + 20, SCREEN_W - 64, 60)
    samples_count = len(state.enroll_samples)
    cap_btn_text = f"CAPTURE SAMPLE [{samples_count + 1}/6]  📸" if samples_count < 6 else "ALL 6 SAMPLES CAPTURED ✓"
    cap_disabled = samples_count >= 6 or state.is_processing
    draw_neo_button(surface, cap_btn_rect, HOT_PINK, cap_btn_text, font_book.h2, WHITE, 4, 12, state.is_processing, cap_disabled)

    # 4. Save Enrollment Button
    can_save = len(state.enroll_samples) >= 3 and bool(state.enroll_username.strip()) and not state.is_processing
    save_btn_rect = pygame.Rect(32, cap_btn_rect.bottom + 18, SCREEN_W - 64, 60)
    save_btn_text = "SAVE ENROLLMENT TO DATABASE  💾"
    draw_neo_button(surface, save_btn_rect, YELLOW, save_btn_text, font_book.h2, BLACK, 4, 12, False, not can_save)

    # 5. Live Feedback / Quality Card
    fb_card = pygame.Rect(32, save_btn_rect.bottom + 18, SCREEN_W - 64, 110)
    draw_neo_card(surface, fb_card, ELECTRIC_BL, BLACK, 4, 12, 3)

    fb_title = font_book.small.render("SAMPLE QUALITY FEEDBACK", True, BLACK)
    surface.blit(fb_title, (fb_card.x + 16, fb_card.y + 14))

    q_text = state.last_quality_msg if state.last_quality_msg else "Present palm & tap Capture Sample"
    q_surf = font_book.body.render(q_text, True, BLACK)
    surface.blit(q_surf, (fb_card.x + 16, fb_card.y + 44))

    rule_txt = "Min 3 samples required (6 recommended)"
    r_surf = font_book.small.render(rule_txt, True, BLACK)
    surface.blit(r_surf, (fb_card.x + 16, fb_card.y + 76))


# ---------------------------------------------------------------------------
# SCREEN 2: USERS
# ---------------------------------------------------------------------------
def render_screen_users(surface, state, font_book, now):
    title = font_book.title.render("ENROLLED USERS", True, BLACK)
    surface.blit(title, (24, CONTENT_TOP + 16))

    cnt_str = f"{len(state.user_list)} Users Active"
    sub = font_book.small.render(cnt_str, True, BLACK)
    surface.blit(sub, (24, CONTENT_TOP + 52))

    if not state.user_list:
        empty_rect = pygame.Rect(32, CONTENT_TOP + 90, SCREEN_W - 64, 180)
        draw_neo_card(surface, empty_rect, YELLOW, BLACK, 5, 14, 3)
        e1 = font_book.h2.render("No Enrolled Users", True, BLACK)
        e2 = font_book.body.render("Tap ENROLL to register palm templates.", True, BLACK)
        surface.blit(e1, e1.get_rect(center=(SCREEN_W // 2, empty_rect.centery - 16)))
        surface.blit(e2, e2.get_rect(center=(SCREEN_W // 2, empty_rect.centery + 18)))
        return

    # Render User List Cards
    card_y = CONTENT_TOP + 85
    visible_height = CONTENT_H - 100
    card_h = 88

    for i, u in enumerate(state.user_list):
        if card_y + card_h > CONTENT_BOTTOM - 10:
            break
        
        c_rect = pygame.Rect(32, card_y, SCREEN_W - 64, card_h)
        draw_neo_card(surface, c_rect, WHITE, BLACK, 4, 12, 3)

        # Avatar circle
        av_rect = pygame.Rect(c_rect.x + 14, c_rect.y + 16, 48, 48)
        pygame.draw.rect(surface, ELECTRIC_BL, av_rect, border_radius=24)
        pygame.draw.rect(surface, BLACK, av_rect, 2, border_radius=24)
        av_txt = font_book.h2.render(u["username"][:1].upper(), True, BLACK)
        surface.blit(av_txt, av_txt.get_rect(center=av_rect.center))

        # Username & Details
        u_name = font_book.h2.render(u["username"], True, BLACK)
        u_date = font_book.small.render(f"Samples: {u['sample_count']}  •  {u['enrolled_at'][:10]}", True, GRAY_MUTED)
        surface.blit(u_name, (c_rect.x + 72, c_rect.y + 18))
        surface.blit(u_date, (c_rect.x + 72, c_rect.y + 48))

        # Delete Button
        del_rect = pygame.Rect(c_rect.right - 80, c_rect.centery - 18, 68, 36)
        draw_neo_button(surface, del_rect, HOT_PINK, "DEL", font_book.small, WHITE, 2, 8)

        card_y += card_h + 14


# ---------------------------------------------------------------------------
# SCREEN 3: ADMIN & DIAGNOSTICS
# ---------------------------------------------------------------------------
def render_screen_admin(surface, state, font_book, now):
    title = font_book.title.render("SYSTEM DIAGNOSTICS", True, BLACK)
    surface.blit(title, (24, CONTENT_TOP + 16))

    # 1. Accuracy Report Action Card
    rep_rect = pygame.Rect(32, CONTENT_TOP + 64, SCREEN_W - 64, 110)
    draw_neo_card(surface, rep_rect, YELLOW, BLACK, 5, 14, 3)
    r1 = font_book.h2.render("BIOMETRIC REPORT  📊", True, BLACK)
    r2 = font_book.body.render("Self-Match & Cross-Match Matrix", True, BLACK)
    r3 = font_book.small.render("Tap to view intra/inter separation →", True, BLACK)
    surface.blit(r1, (rep_rect.x + 18, rep_rect.y + 16))
    surface.blit(r2, (rep_rect.x + 18, rep_rect.y + 48))
    surface.blit(r3, (rep_rect.x + 18, rep_rect.y + 78))

    # 2. Database Stats Card
    db_rect = pygame.Rect(32, rep_rect.bottom + 16, SCREEN_W - 64, 120)
    draw_neo_card(surface, db_rect, ELECTRIC_BL, BLACK, 5, 14, 3)
    d1 = font_book.h2.render("DATABASE ENGINE", True, BLACK)
    total_templates = sum(u["sample_count"] for u in state.user_list)
    d2 = font_book.body.render(f"Enrolled Users: {len(state.user_list)}", True, BLACK)
    d3 = font_book.body.render(f"Stored Templates: {total_templates}", True, BLACK)
    d4 = font_book.small.render("Storage: SQLite with zlib compression", True, BLACK)
    surface.blit(d1, (db_rect.x + 18, db_rect.y + 14))
    surface.blit(d2, (db_rect.x + 18, db_rect.y + 42))
    surface.blit(d3, (db_rect.x + 18, db_rect.y + 68))
    surface.blit(d4, (db_rect.x + 18, db_rect.y + 94))

    # 3. Hardware & Pipeline Status Card
    hw_rect = pygame.Rect(32, db_rect.bottom + 16, SCREEN_W - 64, 125)
    draw_neo_card(surface, hw_rect, LIME, BLACK, 5, 14, 3)
    h1 = font_book.h2.render("HARDWARE STACK", True, BLACK)
    h2 = font_book.body.render("Camera: " + ("RPi NoIR v2 (Active)" if CAMERA_AVAILABLE else "Mock / Offline"), True, BLACK)
    h3 = font_book.body.render("Matcher: 4-Worker Parallel MNHD", True, BLACK)
    h4 = font_book.small.render("Threshold: MNHD <= 0.3800", True, BLACK)
    surface.blit(h1, (hw_rect.x + 18, hw_rect.y + 14))
    surface.blit(h2, (hw_rect.x + 18, hw_rect.y + 42))
    surface.blit(h3, (hw_rect.x + 18, hw_rect.y + 68))
    surface.blit(h4, (hw_rect.x + 18, hw_rect.y + 96))

    # 4. Payment / Extension Card (Neobrutalism Accent)
    p_rect = pygame.Rect(32, hw_rect.bottom + 16, SCREEN_W - 64, 90)
    draw_neo_card(surface, p_rect, PURPLE, BLACK, 5, 14, 3)
    p1 = font_book.h2.render("VEIN-PAY GATEWAY", True, WHITE)
    p2 = font_book.small.render("Offline Biometric Payment Module (Ready)", True, WHITE)
    surface.blit(p1, (p_rect.x + 18, p_rect.y + 18))
    surface.blit(p2, (p_rect.x + 18, p_rect.y + 52))


# ---------------------------------------------------------------------------
# OVERLAYS (Full-screen Popups & Modal Alerts)
# ---------------------------------------------------------------------------
def render_result_overlay(surface, state, font_book):
    """Full-screen Neobrutalist high-energy result banner."""
    accepted = state.result_data.get("accepted", False)
    username = state.result_data.get("username", "Unknown")
    score = state.result_data.get("score", 1.0)
    elapsed = state.result_data.get("elapsed", 0.35)

    bg_color = NEON_CYAN if accepted else HOT_PINK
    surface.fill(bg_color)
    
    # Outer frame border
    pygame.draw.rect(surface, BLACK, (0, 0, SCREEN_W, SCREEN_H), 8)

    # Result Card Modal
    m_rect = pygame.Rect(36, SCREEN_H // 2 - 220, SCREEN_W - 72, 440)
    draw_neo_card(surface, m_rect, WHITE, BLACK, 8, 20, 5)

    # Big Icon
    icon_char = "✓" if accepted else "✕"
    icon_col = LIME if accepted else HOT_PINK
    i_surf = font_book.huge.render(icon_char, True, BLACK)
    
    icon_box = pygame.Rect(m_rect.centerx - 45, m_rect.y + 24, 90, 90)
    draw_neo_card(surface, icon_box, icon_col, BLACK, 4, 45, 4)
    surface.blit(i_surf, i_surf.get_rect(center=icon_box.center))

    # Result Header
    res_title = "AUTHENTICATED" if accepted else "NOT RECOGNISED"
    r_surf = font_book.title.render(res_title, True, BLACK)
    surface.blit(r_surf, r_surf.get_rect(center=(m_rect.centerx, m_rect.y + 145)))

    # Username or Subtitle
    sub_str = f"Welcome, {username}" if accepted else "Palm does not match records"
    s_surf = font_book.h2.render(sub_str, True, BLACK)
    surface.blit(s_surf, s_surf.get_rect(center=(m_rect.centerx, m_rect.y + 185)))

    # Score Gauge Bar
    g_rect = pygame.Rect(m_rect.x + 30, m_rect.y + 230, m_rect.width - 60, 36)
    draw_neo_card(surface, g_rect, GRAY_LIGHT, BLACK, 0, 8, 2)
    
    # Fill proportional to confidence
    fill_ratio = max(0.0, min(1.0, 1.0 - (score / 0.50)))
    fill_w = int((g_rect.width - 4) * fill_ratio)
    if fill_w > 0:
        f_rect = pygame.Rect(g_rect.x + 2, g_rect.y + 2, fill_w, g_rect.height - 4)
        pygame.draw.rect(surface, LIME if accepted else HOT_PINK, f_rect, border_radius=6)

    score_txt = font_book.body.render(f"MNHD Score: {score:.4f}  (Req: < 0.3800)", True, BLACK)
    surface.blit(score_txt, score_txt.get_rect(center=(m_rect.centerx, m_rect.y + 295)))

    time_txt = font_book.small.render(f"Pipeline Time: {elapsed:.2f}s  •  Edge Pi 5", True, GRAY_MUTED)
    surface.blit(time_txt, time_txt.get_rect(center=(m_rect.centerx, m_rect.y + 330)))

    # Auto-dismiss note
    d_txt = font_book.small.render("Auto-closing in 2s...", True, BLACK)
    surface.blit(d_txt, d_txt.get_rect(center=(m_rect.centerx, m_rect.bottom - 36)))


def render_confirm_delete_modal(surface, state, font_book):
    """Modal confirmation dialog for user deletion."""
    # Dim background
    dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))

    modal_rect = pygame.Rect(40, SCREEN_H // 2 - 140, SCREEN_W - 80, 280)
    draw_neo_card(surface, modal_rect, WHITE, BLACK, 8, 16, 4)

    t1 = font_book.title.render("DELETE USER?", True, BLACK)
    surface.blit(t1, t1.get_rect(center=(SCREEN_W // 2, modal_rect.y + 36)))

    u_txt = font_book.h2.render(f"'{state.delete_target}'", True, HOT_PINK)
    surface.blit(u_txt, u_txt.get_rect(center=(SCREEN_W // 2, modal_rect.y + 78)))

    w_txt = font_book.small.render("This will deactivate all enrolled templates.", True, BLACK)
    surface.blit(w_txt, w_txt.get_rect(center=(SCREEN_W // 2, modal_rect.y + 115)))

    # Action Buttons
    cancel_rect = pygame.Rect(modal_rect.x + 20, modal_rect.bottom - 74, 130, 50)
    confirm_rect = pygame.Rect(modal_rect.right - 150, modal_rect.bottom - 74, 130, 50)

    draw_neo_button(surface, cancel_rect, GRAY_LIGHT, "CANCEL", font_book.body, BLACK, 3, 10)
    draw_neo_button(surface, confirm_rect, HOT_PINK, "DELETE", font_book.body, WHITE, 3, 10)


def render_report_modal(surface, state, font_book):
    """Full accuracy diagnostics report popup."""
    dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 180))
    surface.blit(dim, (0, 0))

    m_rect = pygame.Rect(24, 60, SCREEN_W - 48, SCREEN_H - 120)
    draw_neo_card(surface, m_rect, WHITE, BLACK, 6, 16, 4)

    h = font_book.title.render("ACCURACY REPORT", True, BLACK)
    surface.blit(h, (m_rect.x + 20, m_rect.y + 20))

    # Self-Match Summary
    y = m_rect.y + 68
    t_sm = font_book.h2.render("Self-Match (Intra-User)", True, BLACK)
    surface.blit(t_sm, (m_rect.x + 20, y))
    y += 28

    if "self_matches" in state.report_data and state.report_data["self_matches"]:
        for uname, mn, av, mx, qual in state.report_data["self_matches"]:
            row_txt = font_book.small.render(f"{uname:<12} min:{mn:.3f} avg:{av:.3f} max:{mx:.3f}", True, BLACK)
            surface.blit(row_txt, (m_rect.x + 20, y))
            y += 22
    else:
        surface.blit(font_book.small.render("Need >= 2 samples per user for self-match", True, GRAY_MUTED), (m_rect.x + 20, y))
        y += 24

    # Cross-Match Summary
    y += 16
    t_cm = font_book.h2.render("Cross-Match (Separation)", True, BLACK)
    surface.blit(t_cm, (m_rect.x + 20, y))
    y += 28

    if "cross_matches" in state.report_data and state.report_data["cross_matches"]:
        for pair_name, sc, stat in state.report_data["cross_matches"]:
            col = LIME if stat == "OK" else ORANGE
            row_txt = font_book.small.render(f"{pair_name}: {sc:.4f} [{stat}]", True, BLACK)
            surface.blit(row_txt, (m_rect.x + 20, y))
            y += 22
    else:
        surface.blit(font_book.small.render("Need >= 2 users for cross-match analysis", True, GRAY_MUTED), (m_rect.x + 20, y))
        y += 24

    # Close Button
    close_rect = pygame.Rect(m_rect.centerx - 70, m_rect.bottom - 60, 140, 44)
    draw_neo_button(surface, close_rect, YELLOW, "CLOSE", font_book.body, BLACK, 3, 10)


def render_toast(surface, state, font_book, now):
    """Renders floating top toast notification."""
    if now < state.toast_timer and state.toast_msg:
        t_rect = pygame.Rect(32, TOP_H + 12, SCREEN_W - 64, 46)
        draw_neo_card(surface, t_rect, state.toast_color, BLACK, 4, 10, 3)
        msg_surf = font_book.body.render(state.toast_msg, True, BLACK)
        surface.blit(msg_surf, msg_surf.get_rect(center=t_rect.center))


def show_toast(state, msg, color=YELLOW, duration_ms=2200):
    state.toast_msg = msg
    state.toast_color = color
    state.toast_timer = pygame.time.get_ticks() + duration_ms


# ---------------------------------------------------------------------------
# Core UI Event & Logic Handlers
# ---------------------------------------------------------------------------
def handle_screen_click(pos, state, landmarker, engine):
    """Processes touch & mouse clicks on the current screen."""
    now = pygame.time.get_ticks()

    # If an overlay modal is active
    if state.overlay == "result":
        state.overlay = None
        state.current_screen = 0
        return

    if state.overlay == "confirm_delete":
        modal_rect = pygame.Rect(40, SCREEN_H // 2 - 140, SCREEN_W - 80, 280)
        cancel_rect = pygame.Rect(modal_rect.x + 20, modal_rect.bottom - 74, 130, 50)
        confirm_rect = pygame.Rect(modal_rect.right - 150, modal_rect.bottom - 74, 130, 50)
        if cancel_rect.collidepoint(pos):
            state.overlay = None
            state.delete_target = None
        elif confirm_rect.collidepoint(pos):
            if state.delete_target:
                try:
                    delete_user(state.delete_target)
                    engine.refresh_cache()
                    state.user_list = list_users()
                    show_toast(state, f"User '{state.delete_target}' deleted", HOT_PINK)
                except Exception as e:
                    show_toast(state, f"Error: {e}", HOT_PINK)
            state.overlay = None
            state.delete_target = None
        return

    if state.overlay == "report_modal":
        m_rect = pygame.Rect(24, 60, SCREEN_W - 48, SCREEN_H - 120)
        close_rect = pygame.Rect(m_rect.centerx - 70, m_rect.bottom - 60, 140, 44)
        if close_rect.collidepoint(pos) or not m_rect.collidepoint(pos):
            state.overlay = None
        return

    # Bottom Navigation Click Handling
    if pos[1] >= SCREEN_H - NAV_H:
        item_w = SCREEN_W // 4
        clicked_idx = pos[0] // item_w
        if 0 <= clicked_idx <= 3:
            state.current_screen = clicked_idx
            if clicked_idx == 2:
                state.user_list = list_users()
            elif clicked_idx == 3:
                state.user_list = list_users()
        return

    # SCREEN 0: SCAN
    if state.current_screen == 0:
        btn_rect = pygame.Rect(32, CONTENT_TOP + 86 + 300 + 26, SCREEN_W - 64, 66)
        if btn_rect.collidepoint(pos) and not state.is_processing:
            state.is_processing = True
            gray = capture_gray()
            worker = PipelineWorker(gray, landmarker, task_type="scan")
            worker.start()
            return worker

    # SCREEN 1: ENROLL
    elif state.current_screen == 1:
        input_rect = pygame.Rect(32, CONTENT_TOP + 78, SCREEN_W - 64, 52)
        if input_rect.collidepoint(pos):
            state.enroll_input_active = True
            return

        prog_card = pygame.Rect(32, input_rect.bottom + 18, SCREEN_W - 64, 100)
        cap_btn_rect = pygame.Rect(32, prog_card.bottom + 20, SCREEN_W - 64, 60)
        save_btn_rect = pygame.Rect(32, cap_btn_rect.bottom + 18, SCREEN_W - 64, 60)

        # Capture Sample Clicked
        if cap_btn_rect.collidepoint(pos) and len(state.enroll_samples) < 6 and not state.is_processing:
            username = state.enroll_username.strip().lower()
            if not username:
                show_toast(state, "Please enter a username first!", ORANGE)
                return
            if user_exists(username):
                show_toast(state, f"User '{username}' already exists!", HOT_PINK)
                return
            
            state.is_processing = True
            gray = capture_gray()
            worker = PipelineWorker(gray, landmarker, task_type="enroll_sample")
            worker.start()
            return worker

        # Save Enrollment Clicked
        if save_btn_rect.collidepoint(pos) and len(state.enroll_samples) >= 3 and not state.is_processing:
            username = state.enroll_username.strip().lower()
            try:
                enroll_user(username, state.enroll_samples)
                engine.refresh_cache()
                show_toast(state, f"ENROLLED '{username}' with {len(state.enroll_samples)} samples!", LIME, 3000)
                # Reset enrollment state
                state.enroll_username = ""
                state.enroll_samples = []
                state.enroll_roi_thumbs = []
                state.last_quality_msg = ""
                state.user_list = list_users()
                state.current_screen = 2  # Go to users screen
            except Exception as e:
                show_toast(state, f"Enrollment error: {e}", HOT_PINK)

    # SCREEN 2: USERS
    elif state.current_screen == 2:
        card_y = CONTENT_TOP + 85
        card_h = 88
        for u in state.user_list:
            if card_y + card_h > CONTENT_BOTTOM - 10:
                break
            c_rect = pygame.Rect(32, card_y, SCREEN_W - 64, card_h)
            del_rect = pygame.Rect(c_rect.right - 80, c_rect.centery - 18, 68, 36)
            if del_rect.collidepoint(pos):
                state.delete_target = u["username"]
                state.overlay = "confirm_delete"
                return
            card_y += card_h + 14

    # SCREEN 3: ADMIN
    elif state.current_screen == 3:
        rep_rect = pygame.Rect(32, CONTENT_TOP + 64, SCREEN_W - 64, 110)
        if rep_rect.collidepoint(pos):
            # Compute report data on the fly
            state.report_data = generate_report_data()
            state.overlay = "report_modal"

    return None


def generate_report_data():
    """Generates self-match and cross-match accuracy metrics."""
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


def handle_worker_completion(worker, state, engine):
    """Invoked on main thread when a background pipeline task completes."""
    state.is_processing = False

    if worker.error:
        show_toast(state, f"Processing Error: {worker.error}", HOT_PINK, 2800)
        return

    clahe_roi, veincode = worker.result

    # Handle Scan Result
    if worker.task_type == "scan":
        t0 = time.time()
        username, score = engine.identify(veincode)
        elapsed = time.time() - t0 + 0.15  # Include pipeline time estimate
        accepted = (username is not None)

        log_access(user_id=None, score=score, accepted=accepted)
        lbl = username if accepted else "unknown"
        _save_capture(worker.gray, lbl, "scan")
        _save_roi(clahe_roi, lbl, "scan")

        state.result_data = {
            "accepted": accepted,
            "username": username or "Unknown",
            "score": score,
            "elapsed": elapsed
        }
        state.overlay = "result"
        state.result_timer = pygame.time.get_ticks()

    # Handle Enrollment Sample
    elif worker.task_type == "enroll_sample":
        idx = len(state.enroll_samples)
        username = state.enroll_username.strip().lower()
        _save_capture(worker.gray, username, "enroll", idx)
        _save_roi(clahe_roi, username, "enroll", idx)
        
        state.enroll_samples.append(veincode)
        state.last_quality_vr = float(veincode["VR"].mean())
        state.last_quality_msg = f"Sample #{idx + 1} Accepted (VR: {state.last_quality_vr:.3f})"
        show_toast(state, f"Sample {idx + 1}/6 Captured ✓", LIME, 1600)


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------
def run_ui():
    """Main application lifecycle and Pygame loop."""
    pygame.init()
    
    if ON_PI:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

    pygame.display.set_caption("Palm Vein Biometrics — Neobrutalism UI")
    clock = pygame.time.Clock()

    font_book = FontBook()
    state = AppState()

    # Pre-render background with static decorations
    bg_surface = pygame.Surface((SCREEN_W, SCREEN_H))
    bg_surface.fill(BG)
    draw_decorations(bg_surface)

    # Initialize Engine & Database
    init_db()
    engine = SearchEngine(n_workers=4)
    state.user_list = list_users()

    try:
        landmarker = build_landmarker(DEFAULT_MODEL_PATH)
    except Exception as e:
        print(f"Landmarker initialization warning: {e}")
        landmarker = None

    active_worker = None

    titles = ["PALM VEIN SCAN", "ENROLL PALM", "USER DIRECTORY", "SYSTEM DIAGNOSTICS"]

    running = True
    try:
        while running:
            dt = clock.tick(60)
            now = pygame.time.get_ticks()

            # ── Event Dispatch ──────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if state.overlay:
                            state.overlay = None
                        else:
                            running = False
                    elif state.current_screen == 1 and state.enroll_input_active:
                        if event.key == pygame.K_BACKSPACE:
                            state.enroll_username = state.enroll_username[:-1]
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            state.enroll_input_active = False
                        elif event.unicode and len(state.enroll_username) < 18:
                            ch = event.unicode.lower()
                            if ch in "abcdefghijklmnopqrstuvwxyz0123456789_-":
                                state.enroll_username += ch

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    new_worker = handle_screen_click(event.pos, state, landmarker, engine)
                    if new_worker:
                        active_worker = new_worker

            # ── Worker Thread Check ──────────────────────────────────────
            if active_worker and active_worker.done:
                handle_worker_completion(active_worker, state, engine)
                active_worker = None

            # ── Auto-dismiss Result Overlay after 2.5s ───────────────────
            if state.overlay == "result" and (now - state.result_timer > 2500):
                state.overlay = None
                state.current_screen = 0

            # ── Rendering Pipeline ───────────────────────────────────────
            screen.blit(bg_surface, (0, 0))
            render_top_bar(screen, titles[state.current_screen], font_book, now)

            if state.current_screen == 0:
                render_screen_scan(screen, state, font_book, now)
            elif state.current_screen == 1:
                render_screen_enroll(screen, state, font_book, now)
            elif state.current_screen == 2:
                render_screen_users(screen, state, font_book, now)
            elif state.current_screen == 3:
                render_screen_admin(screen, state, font_book, now)

            # Bottom Navigation Bar
            render_bottom_nav(screen, state, font_book)

            # Modals & Overlays
            if state.overlay == "result":
                render_result_overlay(screen, state, font_book)
            elif state.overlay == "confirm_delete":
                render_confirm_delete_modal(screen, state, font_book)
            elif state.overlay == "report_modal":
                render_report_modal(screen, state, font_book)

            # Floating Toasts
            render_toast(screen, state, font_book, now)

            pygame.display.flip()

    finally:
        engine.shutdown()
        if CAMERA_AVAILABLE and picam2 is not None:
            picam2.stop()
        pygame.quit()


if __name__ == "__main__":
    run_ui()
