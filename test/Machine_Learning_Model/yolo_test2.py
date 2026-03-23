import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("models/yolov8n.pt")
print("Model loaded.")

IMAGE_PATH = "sample_images/test.jpg"

img = cv2.imread(IMAGE_PATH)
if img is None:
    print("ERROR: image not found:", IMAGE_PATH)
    raise SystemExit(1)

results = model(img)
result = results[0]
annotated = result.plot()

cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

if result.boxes is not None:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls]
        print({
            "class": name,
            "confidence": round(conf, 3)
        })