import os
import cv2
import numpy as np

# =====================================================
# Calibration paths (robust: try multiple locations)
# =====================================================
BASE_CANDIDATES = []

if os.environ.get("VISION_BASE"):
    BASE_CANDIDATES.append(os.environ["VISION_BASE"])

BASE_CANDIDATES += [
    "/home/ubuntu/jetarm-passion-project/test/Vision testing",
    "/home/ubuntu/jetarm-passion-project/test/Vision/Vision testing",
]

BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(os.path.join(b, "homography_sheet.npy")) and os.path.exists(os.path.join(b, "affine_sheet_to_robot.npy")):
        BASE = b
        break

if BASE is None:
    raise FileNotFoundError(
        "Could not find homography_sheet.npy and affine_sheet_to_robot.npy in any known BASE path. "
        "Set VISION_BASE env var to the folder containing them."
    )

H_sheet = np.load(os.path.join(BASE, "homography_sheet.npy"))
A_robot = np.load(os.path.join(BASE, "affine_sheet_to_robot.npy"))

# =====================================================
# ROI SETTINGS (DETECTION ONLY)
# =====================================================
ROI_ENABLED = True

# Fractions of the frame to KEEP (center region). Tune these.
# Example: remove left/right and bottom clutter.
ROI_X0_FRAC = 0.18
ROI_X1_FRAC = 0.82
ROI_Y0_FRAC = 0.08
ROI_Y1_FRAC = 0.78

# Skip contours that touch ROI border (often partial junk)
SKIP_TOUCHING_BORDER = True
BORDER_PX = 2

# =====================================================
# Coordinate transforms
# =====================================================
def pixel_to_sheet(u, v):
    pt = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_sheet)[0][0]
    return float(out[0]), float(out[1])

def sheet_to_robot(xs, ys):
    vec = np.array([xs, ys, 1.0], dtype=np.float32)
    xr, yr = A_robot @ vec
    return float(xr), float(yr)

def pixel_to_robot(u, v):
    xs, ys = pixel_to_sheet(u, v)
    xr_mm, yr_mm = sheet_to_robot(xs, ys)

    # mm -> cm
    return xr_mm / 10.0, yr_mm / 10.0

# =====================================================
# ROI helpers
# =====================================================
def get_roi_rect(frame):
    h, w = frame.shape[:2]

    x0 = int(w * ROI_X0_FRAC)
    x1 = int(w * ROI_X1_FRAC)
    y0 = int(h * ROI_Y0_FRAC)
    y1 = int(h * ROI_Y1_FRAC)

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    return x0, y0, x1, y1

def draw_roi(frame):
    x0, y0, x1, y1 = get_roi_rect(frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

# =====================================================
# Color detection (RED / GREEN / BLUE / NEUTRAL)
# =====================================================
def detect_color(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "NEUTRAL"

    # small brightness/contrast boost for darker scenes
    roi = cv2.convertScaleAbs(roi, alpha=1.20, beta=15)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    v_med = float(np.median(V))

    # very dark -> neutral
    if v_med < 45:
        return "NEUTRAL"

    # white/gray: bright but low saturation
    V_BRIGHT_MIN = 70
    S_NEUTRAL_MAX = 35

    bright = (V > V_BRIGHT_MIN)
    bright_count = int(bright.sum())

    if bright_count >= 30:
        neutral_like = bright & (S < S_NEUTRAL_MAX)
        neutral_ratio = int(neutral_like.sum()) / max(1, bright_count)
        if neutral_ratio > 0.70:
            return "NEUTRAL"

    # color pixels mask (adaptive V threshold helps lighting changes)
    V_TH = max(25, v_med * 0.55)
    S_TH = 20

    good = (S > S_TH) & (V > V_TH)
    good_ratio = int(good.sum()) / max(1, int(H.size))

    if good_ratio < 0.06:
        return "NEUTRAL"

    H_good = H[good].astype(np.uint8)
    hist = cv2.calcHist([H_good], [0], None, [180], [0, 180])
    h_peak = int(np.argmax(hist))

    parents = {"RED": 0, "GREEN": 60, "BLUE": 120}

    def hue_dist(a, b):
        d = abs(a - b)
        return min(d, 180 - d)

    return min(parents, key=lambda k: hue_dist(h_peak, parents[k]))

# =====================================================
# Brick detection (single frame)
# =====================================================
def detect_bricks(frame):
    """
    Detect bricks ONLY in ROI, but convert pixel centers back to full-frame
    before calling pixel_to_robot(). This preserves your existing homography.
    """
    if ROI_ENABLED:
        x0, y0, x1, y1 = get_roi_rect(frame)
        work = frame[y0:y1, x0:x1]
        off_x, off_y = x0, y0
    else:
        work = frame
        off_x, off_y = 0, 0

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bricks = []
    wh, ww = work.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        if SKIP_TOUCHING_BORDER:
            if x <= BORDER_PX or y <= BORDER_PX or (x + bw) >= (ww - BORDER_PX) or (y + bh) >= (wh - BORDER_PX):
                continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect

        if rh > rw:
            angle += 90
        angle = angle % 180

        # Convert ROI-local pixel center -> full-frame pixel center
        cx_full = float(cx + off_x)
        cy_full = float(cy + off_y)

        # Convert ROI-local bounding box -> full-frame bbox
        x_full = int(x + off_x)
        y_full = int(y + off_y)

        # Shrink ROI for color (reduce background influence)
        pad = int(min(bw, bh) * 0.08)
        pad = max(2, min(pad, 6))

        x2 = x_full + pad
        y2 = y_full + pad
        bw2 = max(1, bw - 2 * pad)
        bh2 = max(1, bh - 2 * pad)

        color = detect_color(frame, x2, y2, bw2, bh2)

        Xr, Yr = pixel_to_robot(cx_full, cy_full)

        bricks.append({
            "x": float(Xr),
            "y": float(Yr),
            "angle": float(angle),
            "color": str(color),
        })

    return bricks
