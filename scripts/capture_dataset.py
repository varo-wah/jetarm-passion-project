"""
Capture raw LEGO dataset images into data/raw/lego.
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jetarm.config.paths import RAW_LEGO_IMAGES_DIR  # noqa: E402


CAMERA_INDEX = 0


def main():
    RAW_LEGO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        raise SystemExit(1)

    count = len(list(RAW_LEGO_IMAGES_DIR.glob("*.jpg")))

    print("Dataset Capture")
    print("SPACE = save image")
    print("q = quit")
    print(f"Saving to: {RAW_LEGO_IMAGES_DIR}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filename = RAW_LEGO_IMAGES_DIR / f"lego_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved: {filename}")
            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
