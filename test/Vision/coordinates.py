import cv2
import numpy as np

# Load homography
H = np.load("autohomography.npy")

def pixel_to_world(u, v):
    pt = np.array([[[u, v]]], dtype=np.float32)
    real = cv2.perspectiveTransform(pt, H)
    X, Y = real[0][0]
    return float(X), float(Y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast-based segmentation
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

        # --- key part: minAreaRect for tilt ---
        rect = cv2.minAreaRect(c)
        # rect = ((cx, cy), (w, h), angle)
        (cx, cy), (w, h), angle = rect

        # Normalize angle so it always follows the LONGEST rectangle side
        # (gripper left–right direction) and lies within [0, 180).
        # OpenCV reports the angle for the short side in [-90, 0).
        if h > w:
            angle += 90.0

        angle = angle % 180.0

        # convert box points to int and draw rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        # center point
        cx_i, cy_i = int(cx), int(cy)
        cv2.circle(frame, (cx_i, cy_i), 4, (0, 0, 255), -1)

        # pixel → world
        Xw, Yw = pixel_to_world(cx, cy)

        # optional camera offset, if you still use it:
        # Xw += offset_x_mm
        # Yw += offset_y_mm

        cv2.putText(
            frame,
            f"Center ({Xw:.1f},{Yw:.1f})",
            (cx_i + 5, cy_i - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"angle {angle:.1f} deg",
            (cx_i + 5, cy_i + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        # --------------------------------------

    cv2.imshow("Tilt detection (minAreaRect)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
