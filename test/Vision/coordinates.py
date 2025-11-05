import cv2
import numpy as np

# Load homography
H = np.load("homography.npy")

def pixel_to_world(u, v):
    pt = np.array([[[u, v]]], dtype=np.float32)
    real = cv2.perspectiveTransform(pt, H)
    X, Y = real[0][0]
    return float(X), float(Y)

# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive threshold: highlight objects that differ in brightness
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 5)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 20000:  # skip tiny noise and full-frame shadows
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = int(x + w/2), int(y + h/2)

        X, Y = pixel_to_world(cx, cy)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
        cv2.putText(frame, f"Center ({X:.1f},{Y:.1f})", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.imshow("Contrast-based Detection → Real Coordinates", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
