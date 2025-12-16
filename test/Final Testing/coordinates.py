import cv2
import numpy as np
import os
BASE = "/home/ubuntu/jetarm-passion-project/test/Vision testing"

# NOTE: smoothing buffers are global and ONLY for visual debugging
# Do NOT reuse this logic for automated picking
# Smoothing buffers for coordinates
X_history = []
Y_history = []
SMOOTH_N = 5   # number of frames to smooth

# =====================================================
# Load calibration matrices
# =====================================================

# Pixel → sheet_mm
H_sheet = np.load(os.path.join(BASE, "homography_sheet.npy"))

# Sheet_mm → robot_mm (2×3 affine matrix)
A_robot = np.load(os.path.join(BASE, "affine_sheet_to_robot.npy"))


# =====================================================
# Coordinate helpers
# =====================================================

def pixel_to_sheet(u, v):
    """Convert (pixel u,v) → (sheet_x, sheet_y) mm."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_sheet)[0][0]
    return float(out[0]), float(out[1])


def sheet_to_robot(xs, ys):
    """Convert sheet mm → robot mm using affine transform."""
    vec = np.array([xs, ys, 1.0], dtype=np.float32)
    xr, yr = A_robot @ vec
    return float(xr), float(yr)


def pixel_to_robot(u, v):
    """Full chain: pixel → sheet_mm → robot_mm."""
    xs, ys = pixel_to_sheet(u, v)
    return sheet_to_robot(xs, ys)


# =====================================================
# COLOR DETECTION
# =====================================================

def detect_color(frame, x, y, w, h):
    """Detect LEGO pink, yellow, cyan."""
    roi = frame[y:y+h, x:x+w]

    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # compute average HSV values
    avg_h = np.mean(hsv[:,:,0])
    avg_s = np.mean(hsv[:,:,1])
    avg_v = np.mean(hsv[:,:,2])

    # ---- LEGO COLORS ----

    # Pink (magenta range)
    if 140 <= avg_h <= 170 and avg_s > 80:
        return "pink"

    # Yellow
    if 20 <= avg_h <= 35 and avg_s > 80:
        return "yellow"

    # Cyan / light blue
    if 85 <= avg_h <= 105 and avg_s > 80:
        return "cyan"

    return "unknown"



# =====================================================
# Object + angle detection pipeline
# =====================================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:
            continue

        # -------------------------
        #   ANGLE LOGIC
        # -------------------------
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if h > w:
            angle += 90
        angle = angle % 180

        # Compute bounding box for color
        x, y, bw, bh = cv2.boundingRect(c)
        color_name = detect_color(frame, x, y, bw, bh)

        # Draw rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # Center dot
        cx_i, cy_i = int(cx), int(cy)
        cv2.circle(frame, (cx_i, cy_i), 4, (0, 0, 255), -1)

        # -----------------------------------------
        # Pixel → robot (raw)
        # -----------------------------------------
        Xr_raw, Yr_raw = pixel_to_robot(cx, cy)

        # -----------------------------------------
        # Add to smoothing history
        # -----------------------------------------
        Xr = Xr_raw
        Yr = Yr_raw

        # Limit history length
        if len(X_history) > SMOOTH_N:
            X_history.pop(0)
        if len(Y_history) > SMOOTH_N:
            Y_history.pop(0)

        # # Compute smoothed values
        # Xr = sum(X_history) / len(X_history)
        # Yr = sum(Y_history) / len(Y_history)

        # ----------------------------------------
        # Display smoothed result
        # -----------------------------------------
        cv2.putText(
            frame,
            f"Coordinates ({Xr:.1f}, {Yr:.1f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"Angle {angle:.1f}",
            (x, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Color: {color_name}",
            (x, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow("Coordinates → Robot", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()