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
    return sheet_to_robot(xs, ys)

# =====================================================
# Color detection
# =====================================================

def detect_color(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:, :, 0])
    avg_s = np.mean(hsv[:, :, 1])

    if 140 <= avg_h <= 170 and avg_s > 80:
        return "pink"
    if 20 <= avg_h <= 35 and avg_s > 80:
        return "yellow"
    if 85 <= avg_h <= 105 and avg_s > 80:
        return "cyan"

    return "unknown"

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
        color = detect_color(frame, x, y, bw, bh)

        Xr, Yr = pixel_to_robot(cx, cy)

        bricks.append({
            "x": Xr,
            "y": Yr,
            "angle": angle,
            "color": color
        })

    return bricks
