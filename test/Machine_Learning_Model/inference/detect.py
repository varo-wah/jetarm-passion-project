# detect.py

from ultralytics import YOLO
import cv2

# ===== LOAD MODEL =====
model = YOLO("../models/yolov8n.pt")  # adjust if needed

# ===== START CAMERA =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

print("Starting YOLO live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== RUN YOLO =====
    results = model(frame)

    # ===== DRAW RESULTS =====
    annotated_frame = results[0].plot()

    # ===== SHOW =====
    cv2.imshow("YOLO Detection", annotated_frame)

    # ===== PRINT DETECTIONS (for debugging) =====
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected class {cls} with confidence {conf:.2f}")

    # ===== EXIT KEY =====
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()