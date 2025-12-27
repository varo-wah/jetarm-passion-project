# coordinates.py
import cv2
import numpy as np
import os

# =====================================================
# Calibration paths
# =====================================================
BASE = "/home/ubuntu/jetarm-passion-project/test/Vision testing"

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
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Filter out background/shadow pixels
    good = (S > 50) & (V > 50)
    good_count = int(good.sum())
    if good_count < 30:
        # Too little signal → treat as neutral/unknown
        return "neutral"

    H_good = H[good].astype(np.uint8)

    # Dominant hue via histogram peak (0..179)
    hist = cv2.calcHist([H_good], [0], None, [180], [0, 180])
    h_peak = int(np.argmax(hist))

    # 4-bucket mapping (covers ALL hues)
    # Note: red wraps around 0 and 179, so include both ends in WARM
    if h_peak < 35 or h_peak >= 170:
        return "WARM"      # red/orange/yellow family (broad)
    if 35 <= h_peak < 85:
        return "GREEN"
    if 85 <= h_peak < 130:
        return "BLUE"      # cyan/blue
    return "PURPLE"        # purple/pink

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
        pad = max(2, int(min(bw, bh) * 0.12))  # tune 0.08–0.18
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
