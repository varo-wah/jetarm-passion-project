"""Compute homography with centered X=0 and bottom Y=10 mm reference.

Running ``python3 dotcheckhomography.py`` will:
- Open the default camera.
- Detect a 7x10 symmetric circle grid with 30 mm dot spacing.
- Map image-space corners to world coordinates where X=0 is at the camera
  center and the bottom of the frame is Y=10 mm.
- Save the resulting homography to ``Vision/dotcheckhomography.npy``.
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
OUTPUT_PATH = Path(__file__).resolve().parent / "dotcheckhomography.npy"


def reshape_grid_points(centers: np.ndarray) -> np.ndarray:
    """Reshape detected centers into (rows, cols, 2) using OpenCV's ordering."""

    return centers.reshape(ROWS, COLS, 2)


def build_world_grid(y_bottom_mm: float) -> np.ndarray:
    """Construct world coordinates for every grid dot.

    X is centered so that the mid-column lies at 0. Y starts at ``y_bottom_mm``
    for the bottom row and increases upward by the dot spacing.
    """

    half_w = ((COLS - 1) * DOT_SPACING_MM) / 2.0
    visible_height_mm = (ROWS - 1) * DOT_SPACING_MM
    y_top = y_bottom_mm + visible_height_mm

    x_coords = (np.arange(COLS) - (COLS - 1) / 2.0) * DOT_SPACING_MM
    y_coords = y_top - np.arange(ROWS) * DOT_SPACING_MM

    xs, ys = np.meshgrid(x_coords, y_coords)
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera 0.")
        return 1

    visible_width_mm = (COLS - 1) * DOT_SPACING_MM
    visible_height_mm = (ROWS - 1) * DOT_SPACING_MM
    half_w = visible_width_mm / 2.0

    y_bottom = 10.0
    y_top = y_bottom + visible_height_mm

    world_pts = np.array(
        [
            [-half_w, y_bottom],  # bottom-left
            [half_w, y_bottom],  # bottom-right
            [half_w, y_top],  # top-right
            [-half_w, y_top],  # top-left
        ],
        dtype=np.float32,
    )

    print(f"Visible grid width (mm): {visible_width_mm:.2f}")
    print(f"Visible grid height (mm): {visible_height_mm:.2f}")
    print("World coordinates (mm) for corners:")
    print(world_pts)

    world_grid = build_world_grid(y_bottom)
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

                    all_img_pts = centers.reshape(-1, 1, 2)
                    reprojected = cv2.perspectiveTransform(all_img_pts, H).reshape(-1, 2)
                    world_flat = world_grid.reshape(-1, 2)
                    diff = reprojected - world_flat
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
