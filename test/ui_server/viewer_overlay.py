import cv2
import numpy as np
from final_testing.coordinatelogic import pixel_to_robot, detect_color

# =========================
# ROI (TUNE THESE)
# Bigger ROI: decrease X0/Y0, increase X1/Y1
# =========================
ROI_X0_FRAC = 0.12
ROI_X1_FRAC = 0.88
ROI_Y0_FRAC = 0.03
ROI_Y1_FRAC = 0.75

ROI_DRAW_BOX = True
ROI_MASK_DISPLAY = False   # visual only (does not affect detection)

MIN_AREA = 400
MAX_AREA = 20000

# Clean-up to stabilize contours under changing lighting
MORPH_ON = True
MORPH_KERNEL = (3, 3)


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
    if frame is None:
        return frame

    # Use a clean source for detection, draw on a separate output
    src = frame
    out = frame.copy()

    x0, y0, x1, y1 = _roi_bounds(src)

    # Optional: visually hide outside ROI on the stream
    if ROI_MASK_DISPLAY:
        masked = np.zeros_like(out)
        masked[y0:y1, x0:x1] = out[y0:y1, x0:x1]
        out = masked

    # --- DETECT ONLY INSIDE ROI (crop for detection stability) ---
    work = src[y0:y1, x0:x1]
    if work.size == 0:
        return out

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )

    if MORPH_ON:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect

        if rh > rw:
            angle += 90
        angle = angle % 180

        # ROI-local -> full-frame pixels
        cx_full = float(cx + x0)
        cy_full = float(cy + y0)

        # Draw rotated box (offset ROI)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box[:, 0] += x0
        box[:, 1] += y0
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)

        # Bounding box for text + color ROI (offset ROI)
        bx, by, bw, bh = cv2.boundingRect(c)
        bx_full = bx + x0
        by_full = by + y0

        pad = int(min(bw, bh) * 0.08)
        pad = max(2, min(pad, 6))

        color = detect_color(
            src,
            bx_full + pad,
            by_full + pad,
            max(1, bw - 2 * pad),
            max(1, bh - 2 * pad),
        )

        Xr, Yr = pixel_to_robot(cx_full, cy_full)

        cv2.putText(out, f"({Xr:.1f}, {Yr:.1f})",
                    (bx_full, max(20, by_full - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(out, f"Angle {angle:.1f}",
                    (bx_full, by_full + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(out, f"{color}",
                    (bx_full, by_full + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw ROI box LAST (so it never affects thresholding)
    if ROI_DRAW_BOX:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(out, "DETECTION ROI", (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out
