import numpy as np
from pathlib import Path

# 3 anchor points: sheet_mm → robot_mm
sheet_pts = np.array([
    [150.0,  60.0],   # tall blue (row 2, col 5)
    [ 90.0, 120.0],   # small blue (row 4, col 3)
    [210.0, 150.0],   # red (row 5, col 7)
], dtype=np.float32)

robot_pts = np.array([
    [  -2.5, 250.0],  # tall blue
    [ -70.0, 190.0],  # small blue
    [  62.5, 155.0],  # red
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

print("Affine matrix A (sheet → robot):")
print(A)
print("Saved to:", out_path)
