"""Compute homography with centered X=0 and bottom Y=100 mm reference.

Running ``python3 dotcheckhomography.py`` will:
- Open the default camera.
- Detect a 7x10 symmetric circle grid with 30 mm dot spacing.
- Map image-space corners to world coordinates where X=0 is at the camera
  center and the bottom of the frame is Y=100 mm.
- Save the resulting homography to ``Vision/dotcheckhomography.npy``.
- Display the camera feed with detected circle centers drawn.

Note:
- If you previously saw a world Y value around -10 when the bottom reference
  was 10 mm, that happens when the grid-derived pixel-to-mm scale places the
  detected corners slightly "below" the assumed bottom. With the bottom fixed
  to a given millimeter height, any portion of the grid that the scale
  extrapolates beneath that line will show as a negative offset even though
  the camera frame itself is intact.
"""

from pathlib import Path
from typing import Callable, Tuple

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


def build_pixel_to_world(  # noqa: PLR0913
    frame_shape: Tuple[int, int],
    grid_corners: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    visible_width_mm: float,
    visible_height_mm: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a converter from pixel coordinates to world coordinates.

    World axes are anchored to the camera frame so that:
    - X=0 lies at the horizontal center of the camera.
    - Y=100 mm lies at the bottom of the camera frame and increases upward.

    Scaling comes from the detected grid so the mapping remains consistent
    even when the printed pattern occupies only part of the frame.
    """

    bottom_left, bottom_right, top_right, top_left = grid_corners

    grid_pixel_width = float(bottom_right[0] - bottom_left[0])
    grid_pixel_height = float(top_left[1] - bottom_left[1])
    if grid_pixel_width == 0 or grid_pixel_height == 0:
        raise ValueError("Invalid grid dimensions detected in pixels.")

    mm_per_px_x = visible_width_mm / grid_pixel_width
    mm_per_px_y = visible_height_mm / abs(grid_pixel_height)

    frame_height, frame_width = frame_shape[:2]
    cx = frame_width / 2.0
    bottom_px = frame_height - 1

    BOTTOM_Y_MM = 100.0

    def pixel_to_world(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        px = pts[..., 0]
        py = pts[..., 1]

        wx = (px - cx) * mm_per_px_x
        wy = BOTTOM_Y_MM + (bottom_px - py) * mm_per_px_y
        return np.stack([wx, wy], axis=-1).astype(np.float32)

    return pixel_to_world


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera 0.")
        return 1

    visible_width_mm = (COLS - 1) * DOT_SPACING_MM
    visible_height_mm = (ROWS - 1) * DOT_SPACING_MM

    print(f"Visible grid width (mm): {visible_width_mm:.2f}")
    print(f"Visible grid height (mm): {visible_height_mm:.2f}")

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

                try:
                    pixel_to_world = build_pixel_to_world(
                        frame.shape,
                        (bottom_left, bottom_right, top_right, top_left),
                        visible_width_mm,
                        visible_height_mm,
                    )
                except ValueError as exc:
                    print(f"Warning: {exc}")
                    pixel_to_world = None

                if pixel_to_world is not None:
                    world_pts = pixel_to_world(img_pts)
                    world_grid = pixel_to_world(grid_points)

                    print("World coordinates (mm) for corners:")
                    print(world_pts)

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
