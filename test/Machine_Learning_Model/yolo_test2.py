import cv2
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n.pt"
CONF = 0.5
CAMERA_INDEX = 0

print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame")
        break

    results = model(frame, conf=CONF, verbose=False)
    result = results[0]

    annotated = result.plot()

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # draw center point
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            print({
                "class": name,
                "confidence": round(conf, 3),
                "center_px": (cx, cy)
            })

    cv2.imshow("Live YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()