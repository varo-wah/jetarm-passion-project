# ui_server/viewer_overlay.py
import cv2
import numpy as np
from final_testing.coordinatelogic import pixel_to_robot, detect_color


# =====================================================
# Smoothing (DISPLAY ONLY)
# Keep these module-level so smoothing persists across frames
# =====================================================
X_history = []
Y_history = []
angle_history = []
SMOOTH_N = 5


def smooth(val, history):
    history.append(val)
    if len(history) > SMOOTH_N:
        history.pop(0)
    return sum(history) / len(history)


def annotate_frame(frame):
    """
    Takes a BGR frame (numpy array), draws overlays on it, and returns it.
    No camera access, no imshow, no infinite loop.
    """
    if frame is None:
        return frame

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
        color = detect_color(frame, x, y, bw, bh)

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

    return frame
