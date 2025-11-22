"""Compute a homography from a detected symmetric circle grid.

This script opens the default camera, searches for a 7x10 circle grid with
30 mm spacing and 10 mm diameter dots, estimates a homography that maps image
coordinates to real-world coordinates, reports RMS error, and saves the
homography matrix to ``Vision/dothomography.npy``.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# Grid definition
ROWS = 7
COLS = 10
DOT_SPACING_MM = 30.0
DOT_DIAMETER_MM = 10.0  # Not used directly but documented for clarity
PATTERN_SIZE = (COLS, ROWS)  # (columns, rows) for cv2.findCirclesGrid

# Output path
OUTPUT_PATH = Path(__file__).resolve().parent / "dothomography.npy"


def generate_world_points(rows: int, cols: int, spacing: float) -> np.ndarray:
    """Create world coordinates for the symmetric grid.

    The origin is the top-left dot. Points are ordered row-major to match
    ``cv2.findCirclesGrid`` output.
    """
    # columns change fastest, then rows (x corresponds to columns, y to rows)
    column_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    x_coords = column_indices.astype(np.float32) * spacing
    y_coords = row_indices.astype(np.float32) * spacing
    world_points = np.stack((x_coords, y_coords), axis=-1)
    return world_points.reshape(-1, 2)


def compute_rms_error(homography: np.ndarray, img_pts: np.ndarray, world_pts: np.ndarray) -> float:
    """Project image points using the homography and compute RMS error."""
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), homography)
    residuals = world_pts.reshape(-1, 1, 2) - projected
    squared_error = np.square(residuals).sum(axis=2)
    rms = np.sqrt(np.mean(squared_error))
    return float(rms)


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera 0.")
        return 1

    world_points = generate_world_points(ROWS, COLS, DOT_SPACING_MM)
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
                # Draw detected centers for visualization
                cv2.drawChessboardCorners(display, PATTERN_SIZE, centers, found)

                # Compute homography once a valid detection is found
                H, mask = cv2.findHomography(centers.reshape(-1, 2), world_points, method=0)
                if H is not None:
                    rms_error = compute_rms_error(H, centers.reshape(-1, 2), world_points)
                    print(f"Homography computed. RMS projection error: {rms_error:.3f} mm")
                    np.save(OUTPUT_PATH, H)
                    print(f"Homography saved to: {OUTPUT_PATH}")
                    homography_computed = True
                else:
                    print("Warning: Homography computation failed.")
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
            if key == ord("q"):
                break
            if homography_computed:
                # Stop after computing the first valid homography
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not homography_computed:
        print("Circle grid was not detected. Homography not saved.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
