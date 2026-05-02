import cv2
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parent.parent / "dataset" / "images" / "raw"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Error: Cannot access camera")
    raise SystemExit

count = len(list(SAVE_DIR.glob("*.jpg")))

print("Dataset Capture")
print("SPACE = save image")
print("q = quit")
print(f"Saving to: {SAVE_DIR}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = SAVE_DIR / f"lego_{count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
