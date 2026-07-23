import cv2
import numpy as np

from jetarm.vision.coordinatelogic import detect_color, pixel_to_robot

MIN_AREA = 400
MAX_AREA = 20000

# Clean-up to stabilize contours under changing lighting
MORPH_ON = True
MORPH_KERNEL = (3, 3)


def annotate_frame(frame):
    if frame is None:
        return frame

    # Use a clean source for detection, draw on a separate output
    src = frame
    out = frame.copy()

    # OpenCV feed intentionally scans the full frame for exhibition comparison.
    work = src
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

        cx_full = float(cx)
        cy_full = float(cy)

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)

        # Bounding box for text + color sampling.
        bx, by, bw, bh = cv2.boundingRect(c)
        bx_full = bx
        by_full = by

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

    return out
