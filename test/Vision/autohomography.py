import cv2
import numpy as np

# Real-world coordinates in mm
world_pts = np.array([
    [-147.5,  80.0],   # bottom-left
    [ 147.5,  80.0],   # bottom-right
    [ 147.5, 302.0],   # top-right  (80 + 222)
    [-147.5, 302.0]    # top-left
], dtype=np.float32)

# Open camera and grab one frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera error — no frame captured.")
    exit()

h, w = frame.shape[:2]

# Image coordinates = edges of the frame
img_pts = np.array([
    [0, h-1],      # bottom-left
    [w-1, h-1],    # bottom-right
    [w-1, 0],      # top-right
    [0, 0]         # top-left
], dtype=np.float32)

# Compute and save homography
H, _ = cv2.findHomography(img_pts, world_pts)
np.save("autohomography.npy", H)

print("✅ Homography saved as autohomography.npy")
print("Frame size:", w, "x", h)
