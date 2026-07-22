import numpy as np
from pathlib import Path

# 3 anchor points: sheet_mm → robot_mm
# sheet coordinates from dot indices (spacing = 30 mm, origin = top-left dot)
sheet_pts = np.array([
    [210.0, 150.0],   # pink   (col 7, row 5)
    [ 60.0, 120.0],   # blue   (col 2, row 4)
    [120.0,  90.0],   # yellow (col 4, row 3)
], dtype=np.float32)

# robot coordinates you measured, converted from cm to mm
robot_pts = np.array([
    [  65.0, 155.0],   # pink
    [-100.0, 195.0],   # blue
    [ -35.0, 220.0],   # yellow
], dtype=np.float32)

# Build linear system for 2×3 affine matrix A
# [Xr, Yr]^T = A · [Xs, Ys, 1]^T
M = []
b = []
for (Xs, Ys), (Xr, Yr) in zip(sheet_pts, robot_pts):
    M.append([Xs, Ys, 1, 0,  0, 0])
    M.append([0,  0, 0, Xs, Ys, 1])
    b.append(Xr)
    b.append(Yr)

M = np.array(M, dtype=np.float32)
b = np.array(b, dtype=np.float32).reshape(-1, 1)

params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
A = params.reshape(2, 3)

out_path = Path(__file__).resolve().parent / "affine_sheet_to_robot.npy"
np.save(out_path, A)

# quick sanity check: project the three sheet points and compare
pred = []
for Xs, Ys in sheet_pts:
    v = np.array([Xs, Ys, 1.0], dtype=np.float32)
    xr, yr = (A @ v).tolist()
    pred.append([xr, yr])

pred = np.array(pred)
err = np.linalg.norm(pred - robot_pts, axis=1)
rms = np.sqrt(np.mean(err**2))

print("Affine matrix A (sheet → robot):")
print(A)
print("Saved to:", out_path)
print("Per-point error (mm):", err)
print("RMS error (mm):", rms)
