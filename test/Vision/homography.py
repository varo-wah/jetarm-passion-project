# calibrate_homography_auto.py
import cv2
import numpy as np

# Real-world coordinates in mm for your 30 cm × 22 cm workspace
world_pts = np.array([
    [-150,   130],     # bottom-left
    [150, 130],     # bottom-right
    [150, 350],   # top-right
    [-150],   350]    # top-left
], dtype=np.float32)

clicked = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append([x, y])
        print(f"Point {len(clicked)}: ({x}, {y})")

cap = cv2.VideoCapture(0)   # change index if JetArm camera is not 0
ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera error — no frame captured.")
    exit()

cv2.namedWindow("Click 4 corners (BL, BR, TR, TL)")
cv2.setMouseCallback("Click 4 corners (BL, BR, TR, TL)", on_click)

while True:
    disp = frame.copy()
    for p in clicked:
        cv2.circle(disp, tuple(p), 5, (0,255,0), -1)
    cv2.imshow("Click 4 corners (BL, BR, TR, TL)", disp)
    k = cv2.waitKey(1) & 0xFF

    if len(clicked) == 4:
        img_pts = np.array(clicked, dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, world_pts)
        np.save("homography.npy", H)
        print("✅ Homography saved as homography.npy")
        break

    if k == 27:  # ESC to cancel
        break

cv2.destroyAllWindows()
