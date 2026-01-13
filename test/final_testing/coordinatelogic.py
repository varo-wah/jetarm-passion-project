# coordinates.py
import cv2
import numpy as np
import os

# =====================================================
# Calibration paths
# =====================================================
BASE = "/home/ubuntu/jetarm-passion-project/test/Vision/Vision testing"

H_sheet = np.load(os.path.join(BASE, "homography_sheet.npy"))
A_robot = np.load(os.path.join(BASE, "affine_sheet_to_robot.npy"))

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
    xr_mm, yr_mm = sheet_to_robot(xs, ys)   # currently mm

    # Convert mm -> cm (move decimal one place left)
    xr_cm = xr_mm / 10.0
    yr_cm = yr_mm / 10.0

    return xr_cm, yr_cm

# =====================================================
# Color detection
# =====================================================

# [42] Replace detect_color() with this 4-bucket version
def detect_color(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "NEUTRAL"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # 1) Build a "color pixel" mask (lower thresholds first; tune later)
    good = (S > 25) & (V > 40)
    good_count = int(good.sum())
    total = int(H.size)
    good_ratio = good_count / max(1, total)

    # If almost no pixels look like real color, call it NEUTRAL
    if good_ratio < 0.08:   # tune 0.05–0.15
        return "NEUTRAL"

    H_good = H[good].astype(np.uint8)

    # 2) Dominant hue
    hist = cv2.calcHist([H_good], [0], None, [180], [0, 180])
    h_peak = int(np.argmax(hist))

    # 3) Nearest parent hue (RED/GREEN/BLUE)
    parents = {"RED": 0, "GREEN": 60, "BLUE": 120}

    def hue_dist(a, b):
        d = abs(a - b)
        return min(d, 180 - d)

    return min(parents, key=lambda k: hue_dist(h_peak, parents[k]))


# =====================================================
# SNAPSHOT brick detection (FOR AUTOMATION)
# =====================================================

def detect_bricks(frame):
    """
    Process ONE image and return brick data.
    NO smoothing. NO looping.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bricks = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if h > w:
            angle += 90
        angle = angle % 180

        x, y, bw, bh = cv2.boundingRect(c)

        # Shrink ROI to reduce background/edges
        pad = int(min(bw, bh) * 0.08)
        pad = max(2, min(pad, 6))   # cap at 6px
        x2 = x + pad
        y2 = y + pad
        bw2 = max(1, bw - 2 * pad)
        bh2 = max(1, bh - 2 * pad)

        color = detect_color(frame, x2, y2, bw2, bh2)

        Xr, Yr = pixel_to_robot(cx, cy)

        bricks.append({
            "x": Xr,
            "y": Yr,
            "angle": angle,
            "color": color
        })

    return bricks
