# viewer.py
import cv2
import numpy as np
from coordinatelogic import pixel_to_robot, detect_color

# =====================================================
# Smoothing (DISPLAY ONLY)
# =====================================================
X_history = []
Y_history = []
angle_history = []
ROI_X0_FRAC = 0.18
ROI_X1_FRAC = 0.82
ROI_Y0_FRAC = 0.05
ROI_Y1_FRAC = 0.62
SMOOTH_N = 5

def smooth(val, history):
    history.append(val)
    if len(history) > SMOOTH_N:
        history.pop(0)
    return sum(history) / len(history)

# =====================================================
# Live camera viewer
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

    # -----------------------------
    # ROI LIMIT: ignore edges/bottom
    # -----------------------------
    Ht, Wt = thresh.shape[:2]
    x0 = int(Wt * ROI_X0_FRAC); x1 = int(Wt * ROI_X1_FRAC)
    y0 = int(Ht * ROI_Y0_FRAC); y1 = int(Ht * ROI_Y1_FRAC)

    roi_mask = np.zeros_like(thresh)
    roi_mask[y0:y1, x0:x1] = 255
    thresh = cv2.bitwise_and(thresh, roi_mask)

    # Draw ROI on the viewer (so you can tune it)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(frame, "DETECTION ROI", (x0, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        if h > w:
            angle += 90
        angle = angle % 180

        angle_raw = angle

        angle = smooth(angle_raw, angle_history)

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(min(bw, bh) * 0.08)
        pad = max(2, min(pad, 6))
        x2 = x + pad
        y2 = y + pad
        bw2 = max(1, bw - 2 * pad)
        bh2 = max(1, bh - 2 * pad)

        color = detect_color(frame, x2, y2, bw2, bh2)

        # RAW coordinate
        Xr_raw, Yr_raw = pixel_to_robot(cx, cy)

        # SMOOTHED (display only)
        Xr = smooth(Xr_raw, X_history)
        Yr = smooth(Yr_raw, Y_history)

        cv2.putText(frame, f"({Xr:.1f}, {Yr:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        cv2.putText(frame, f"Angle {angle:.1f}",
                    (x, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        cv2.putText(frame, f"{color}",
                    (x, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

    cv2.imshow("JetArm Viewer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
