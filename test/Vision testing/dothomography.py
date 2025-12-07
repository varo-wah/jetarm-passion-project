import cv2
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROWS = 7
COLS = 10
DOT_SPACING = 30.0   # mm
PATTERN_SIZE = (COLS, ROWS)

OUTPUT_DIR = Path(__file__).resolve().parent
H_SHEET_PATH = OUTPUT_DIR / "homography_sheet.npy"
AFFINE_PATH   = OUTPUT_DIR / "affine_sheet_to_robot.npy"

# Your 3 measured anchor points (sheet → robot)
sheet_pts = np.array([
    [270, 180],   # Dot A sheet
    [0,   180],   # Dot B sheet
    [60,   90],   # Dot C sheet
], dtype=np.float32)

robot_pts = np.array([
    [125, 120],    # Dot A robot
    [-180,120],    # Dot B robot
    [-110,225],    # Dot C robot
], dtype=np.float32)

# ============================================================
# Helper: pixel → sheet using homography
# ============================================================

def pixel_to_sheet(u, v, H_sheet):
    pt = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_sheet)[0][0]
    return float(out[0]), float(out[1])

# ============================================================
# Helper: sheet → robot using affine transform
# ============================================================

def sheet_to_robot(Xs, Ys, A):
    v = np.array([[Xs, Ys, 1.0]], dtype=np.float32).T
    r = A @ v
    return float(r[0]), float(r[1])

# ============================================================
# Calibration Script
# ============================================================

def main():
    print("Opening camera...")
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

            # ------------------------------------------------------------
            # 1. Compute pixel → sheet homography
            # ------------------------------------------------------------
            img_pts = centers.reshape(-1,2)

            sheet_coords = []
            for r in range(ROWS):
                for c in range(COLS):
                    sheet_coords.append([c * DOT_SPACING, r * DOT_SPACING])
            sheet_coords = np.array(sheet_coords, dtype=np.float32)

            H_sheet, _ = cv2.findHomography(img_pts, sheet_coords)
            np.save(H_SHEET_PATH, H_sheet)
            print("Saved pixel→sheet homography to:", H_SHEET_PATH)

            # ------------------------------------------------------------
            # 2. Compute sheet → robot affine transform
            # ------------------------------------------------------------
            # We need a 2x3 matrix A where:
            # [Xr, Yr]^T = A * [Xs, Ys, 1]^T

            # construct linear system
            M = []
            b = []
            for i in range(3):
                Xs, Ys = sheet_pts[i]
                Xr, Yr = robot_pts[i]

                M.append([Xs, Ys, 1, 0,  0, 0])
                M.append([0,  0,  0, Xs, Ys, 1])
                b.append(Xr)
                b.append(Yr)

            M = np.array(M, dtype=np.float32)
            b = np.array(b, dtype=np.float32).reshape(-1,1)

            params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            A = params.reshape(2,3)
            np.save(AFFINE_PATH, A)
            print("Saved sheet→robot affine transform to:", AFFINE_PATH)
            print("Affine matrix:\n", A)

            # ------------------------------------------------------------
            # 3. Test RMS error (pixel→sheet→robot)
            # ------------------------------------------------------------
            total_err = 0
            count = 0

            for i in range(len(img_pts)):
                u, v = img_pts[i]

                xs, ys = pixel_to_sheet(u, v, H_sheet)
                xr, yr = sheet_to_robot(xs, ys, A)

                # predict robot coords of this dot
                # what SHOULD the sheet coords be?
                r = i // COLS
                c = i % COLS
                Xs_true = c * DOT_SPACING
                Ys_true = r * DOT_SPACING

                # convert true sheet coords → robot coords
                Xr_true, Yr_true = sheet_to_robot(Xs_true, Ys_true, A)

                err = ( (xr - Xr_true)**2 + (yr - Yr_true)**2 )**0.5
                total_err += err
                count += 1

            rms = total_err / count
            print(f"RMS robot-space error: {rms:.3f} mm")

            # ------------------------------------------------------------
            # Display
            # ------------------------------------------------------------
            cv2.imshow("dothomography", display)
            cv2.waitKey(0)
            break

        else:
            cv2.putText(display, "Dot grid NOT found", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("dothomography", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
