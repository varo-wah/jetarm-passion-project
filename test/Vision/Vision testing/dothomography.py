import cv2
import numpy as np
from pathlib import Path

ROWS = 7
COLS = 10
DOT_SPACING = 30.0  # mm
PATTERN_SIZE = (COLS, ROWS)

OUTPUT_DIR = Path(__file__).resolve().parent
H_SHEET_PATH = OUTPUT_DIR / "homography_sheet.npy"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, centers = cv2.findCirclesGrid(
            gray,
            PATTERN_SIZE,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )

        display = frame.copy()

        if found:
            cv2.drawChessboardCorners(display, PATTERN_SIZE, centers, found)
            print("Dot grid detected.")

            img_pts = centers.reshape(-1, 2)

            sheet_coords = []
            for r in range(ROWS):
                for c in range(COLS):
                    sheet_coords.append([c * DOT_SPACING, r * DOT_SPACING])
            sheet_coords = np.array(sheet_coords, dtype=np.float32)

            H_sheet, _ = cv2.findHomography(img_pts, sheet_coords)
            np.save(H_SHEET_PATH, H_sheet)
            print("Saved pixel→sheet homography to:", H_SHEET_PATH)

            cv2.imshow("Dot grid", display)
            cv2.waitKey(0)
            break
        else:
            cv2.putText(
                display, "Dot grid NOT found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow("Dot grid", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
