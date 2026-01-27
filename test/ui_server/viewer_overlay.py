import cv2
import numpy as np
from final_testing.coordinatelogic import pixel_to_robot, detect_color

# =====================================================
# ROI (WEBSITE OVERLAY DETECTION AREA)
# Edit these to make ROI bigger/smaller
# =====================================================
ROI_X0_FRAC = 0.12
ROI_X1_FRAC = 0.88
ROI_Y0_FRAC = 0.03
ROI_Y1_FRAC = 0.75

ROI_DRAW_BOX = True          # draw the ROI rectangle on the stream
ROI_MASK_DISPLAY = False     # if True, black out everything outside ROI (visual only)
SKIP_TOUCHING_BORDER = True  # helps ignore partial junk touching ROI edges
BORDER_PX = 2

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


def _roi_bounds(frame):
    h, w = frame.shape[:2]
    x0 = int(w * ROI_X0_FRAC)
    x1 = int(w * ROI_X1_FRAC)
    y0 = int(h * ROI_Y0_FRAC)
    y1 = int(h * ROI_Y1_FRAC)

    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w - 1))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h - 1))
    return x0, y0, x1, y1


def annotate_frame(frame):
    """
    Takes a BGR frame (numpy array), draws overlays on it, and returns it.
    No camera access, no imshow, no infinite loop.
    """
    if frame is None:
        return frame

    x0, y0, x1, y1 = _roi_bounds(frame)

    # Optional: visually hide everything outside ROI (does NOT affect mapping)
    if ROI_MASK_DISPLAY:
        masked = np.zeros_like(frame)
        masked[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        frame = masked

    if ROI_DRAW_BOX:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(
            frame, "DETECTION ROI", (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

    # --- Build threshold from full frame (then ROI-mask it) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )

    # ROI mask on the binary image BEFORE findContours (hard filter)
    roi_mask = np.zeros_like(thresh)
    roi_mask[y0:y1, x0:x1] = 255
    thresh = cv2.bitwise_and(thresh, roi_mask)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:
            continue

        # Reject partial junk touching ROI border
        if SKIP_TOUCHING_BORDER:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= x0 + BORDER_PX or by <= y0 + BORDER_PX or (bx + bw) >= x1 - BORDER_PX or (by + bh) >= y1 - BORDER_PX:
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

        # Bounding box + color (shrink ROI to reduce background influence)
        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(min(bw, bh) * 0.08)
        pad = max(2, min(pad, 6))

        color = detect_color(
            frame,
            x + pad,
            y + pad,
            max(1, bw - 2 * pad),
            max(1, bh - 2 * pad),
        )

        # RAW coordinate from full-frame pixels (homography stays valid)
        Xr_raw, Yr_raw = pixel_to_robot(cx, cy)

        # SMOOTHED (display only)
        Xr = smooth(Xr_raw, X_history)
        Yr = smooth(Yr_raw, Y_history)

        cv2.putText(
            frame, f"({Xr:.1f}, {Yr:.1f})",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 0), 1
        )

        cv2.putText(
            frame, f"Angle {angle:.1f}",
            (x, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 1
        )

        cv2.putText(
            frame, f"{color}",
            (x, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1
        )

    return frame
