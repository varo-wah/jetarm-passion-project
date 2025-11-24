"""Detect a symmetric circle grid and compute an image-to-world homography.

Running ``python3 dothomography.py`` will:
- Open the default camera.
- Detect a 7x10 symmetric circle grid with 30 mm dot spacing.
- Map image-space corners to world coordinates centered on the grid with an
  80 mm offset on the Y axis.
- Save the resulting homography to ``Vision/dothomography.npy``.
- Display the camera feed with detected circle centers drawn.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Grid definition
ROWS = 7
COLS = 10
DOT_SPACING_MM = 30.0
PATTERN_SIZE: Tuple[int, int] = (COLS, ROWS)  # (columns, rows) for cv2.findCirclesGrid

# Output path
OUTPUT_PATH = Path(__file__).resolve().parent / "dothomography.npy"


def reshape_grid_points(centers: np.ndarray) -> np.ndarray:
    """Reshape detected centers into (rows, cols, 2) using OpenCV's ordering.

    ``cv2.findCirclesGrid`` returns the points in row-major order (top-to-bottom,
    left-to-right) irrespective of slant or how much of the camera frame the
    paper occupies. By reshaping directly, we avoid mis-ordering when the grid
    appears slanted in the image.
    """

    flat = centers.reshape(-1, 2)
    return flat.reshape(ROWS, COLS, 2)


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera 0.")
        return 1

    visible_width_mm = (COLS - 1) * DOT_SPACING_MM
    visible_height_mm = (ROWS - 1) * DOT_SPACING_MM
    half_w = visible_width_mm / 2.0
    y0 = 80.0
    y1 = y0 + visible_height_mm

    world_pts = np.array(
        [
            [-half_w, y0],  # bottom-left
            [half_w, y0],  # bottom-right
            [half_w, y1],  # top-right
            [-half_w, y1],  # top-left
        ],
        dtype=np.float32,
    )

    print(f"Visible grid width (mm): {visible_width_mm:.2f}")
    print(f"Visible grid height (mm): {visible_height_mm:.2f}")
    print("World coordinates (mm):")
    print(world_pts)

    homography_computed = False

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame from camera.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, centers = cv2.findCirclesGrid(
                gray,
                PATTERN_SIZE,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            )

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, PATTERN_SIZE, centers, found)
                grid_points = reshape_grid_points(centers)

                bottom_left = grid_points[-1, 0]
                bottom_right = grid_points[-1, -1]
                top_right = grid_points[0, -1]
                top_left = grid_points[0, 0]

                img_pts = np.array(
                    [bottom_left, bottom_right, top_right, top_left],
                    dtype=np.float32,
                )

                H, _ = cv2.findHomography(img_pts, world_pts)
                if H is None:
                    print("Warning: Homography computation failed.")
                else:
                    print("Homography matrix:")
                    print(H)
                    reprojected = cv2.perspectiveTransform(
                        img_pts.reshape(-1, 1, 2), H
                    ).reshape(-1, 2)
                    diff = reprojected - world_pts
                    rms_error = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                    print(f"RMS reprojection error (mm): {rms_error:.4f}")
                    np.save(OUTPUT_PATH, H)
                    print(f"Homography saved to: {OUTPUT_PATH}")
                    homography_computed = True

            else:
                cv2.putText(
                    display,
                    "Circle grid not found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Circle Grid Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if homography_computed or key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not homography_computed:
        print("Circle grid was not detected. Homography not saved.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
