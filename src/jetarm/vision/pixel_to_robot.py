from typing import Dict, List, Tuple  # L002

import cv2  # L004
import numpy as np  # L005

from jetarm.config.paths import CALIBRATION_DIR
from jetarm.config.vision_config import (
    BRIGHTNESS_ALPHA,
    BRIGHTNESS_BETA,
    GOOD_RATIO_MIN,
    MAX_AREA,
    MIN_AREA,
    NEUTRAL_BRIGHT_RATIO,
    ROI_SHRINK_FRAC,
    ROI_SHRINK_MAX_PX,
    ROI_SHRINK_MIN_PX,
    ROI_X0_FRAC,
    ROI_X1_FRAC,
    ROI_Y0_FRAC,
    ROI_Y1_FRAC,
    S_NEUTRAL_MAX,
    S_THRESHOLD,
    V_BRIGHT_MIN,
    V_DARK_NEUTRAL,
)

# =====================================================  # L007
# CONFIG  # L008
# =====================================================  # L009

BASE = CALIBRATION_DIR  # L012

HOMOGRAPHY_FILE = "homography_sheet.npy"  # L014
AFFINE_FILE = "affine_sheet_to_robot.npy"  # L015

S_TH = S_THRESHOLD  # L044

# =====================================================  # L047
# CALIBRATION LOAD  # L048
# =====================================================  # L049

def _load_calibration(base) -> Tuple[np.ndarray, np.ndarray]:  # L051
    """Load homography + affine matrices. Raises a clear error if missing."""  # L052
    h_path = base / HOMOGRAPHY_FILE  # L053
    a_path = base / AFFINE_FILE  # L054

    if not h_path.exists():  # L056
        raise FileNotFoundError(f"Missing calibration file: {h_path}")  # L057
    if not a_path.exists():  # L058
        raise FileNotFoundError(f"Missing calibration file: {a_path}")  # L059

    H_sheet = np.load(h_path)  # L061
    A_robot = np.load(a_path)  # L062

    if H_sheet.shape != (3, 3):  # L064
        raise ValueError(f"homography must be 3x3, got {H_sheet.shape}")  # L065
    if A_robot.shape != (2, 3):  # L066
        raise ValueError(f"affine must be 2x3, got {A_robot.shape}")  # L067

    return H_sheet, A_robot  # L069

H_sheet, A_robot = _load_calibration(BASE)  # L071

# =====================================================  # L073
# COORDINATE TRANSFORMS  # L074
# =====================================================  # L075

def pixel_to_sheet(u: float, v: float) -> Tuple[float, float]:  # L077
    pt = np.array([[[u, v]]], dtype=np.float32)  # L078
    out = cv2.perspectiveTransform(pt, H_sheet)[0][0]  # L079
    return float(out[0]), float(out[1])  # L080

def sheet_to_robot(xs: float, ys: float) -> Tuple[float, float]:  # L082
    vec = np.array([xs, ys, 1.0], dtype=np.float32)  # L083
    xr, yr = A_robot @ vec  # L084
    return float(xr), float(yr)  # L085

def pixel_to_robot(u: float, v: float) -> Tuple[float, float]:  # L087
    """Return robot X,Y in cm."""  # L088
    xs, ys = pixel_to_sheet(u, v)  # L089
    xr_mm, yr_mm = sheet_to_robot(xs, ys)  # calibration outputs mm  # L090
    return xr_mm / 10.0, yr_mm / 10.0      # mm -> cm  # L091

# =====================================================  # L093
# ROI HELPERS  # L094
# =====================================================  # L095

def roi_bounds_from_shape(shape: Tuple[int, int]) -> Tuple[int, int, int, int]:  # L097
    """Compute ROI bounds (x0,y0,x1,y1) from a (height,width) shape."""  # L098
    h, w = shape[:2]  # L099
    x0 = int(w * ROI_X0_FRAC)  # L100
    x1 = int(w * ROI_X1_FRAC)  # L101
    y0 = int(h * ROI_Y0_FRAC)  # L102
    y1 = int(h * ROI_Y1_FRAC)  # L103
    return x0, y0, x1, y1  # L104

def apply_roi_mask(binary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:  # L106
    """Zero out everything outside the configured ROI."""  # L107
    x0, y0, x1, y1 = roi_bounds_from_shape(binary.shape[:2])  # L108

    roi_mask = np.zeros_like(binary)  # L110
    roi_mask[y0:y1, x0:x1] = 255  # L111

    masked = cv2.bitwise_and(binary, roi_mask)  # L113
    return masked, (x0, y0, x1, y1)  # L114

# =====================================================  # L116
# COLOR DETECTION (returns: NEUTRAL / RED / GREEN / BLUE)  # L117
# =====================================================  # L118

def detect_color(frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:  # L120
    roi = frame[y:y+h, x:x+w]  # L121
    if roi.size == 0:  # L122
        return "NEUTRAL"  # L123

    # Improve dark-brick visibility (software boost)  # L125
    roi = cv2.convertScaleAbs(roi, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)  # L126

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # L128
    H = hsv[:, :, 0]  # L129
    S = hsv[:, :, 1]  # L130
    V = hsv[:, :, 2]  # L131

    # -----------------------------  # L133
    # A) Explicit NEUTRAL detection  # L134
    # -----------------------------  # L135
    v_med = float(np.median(V))  # L136

    # Very dark -> neutral (shadow/black)  # L138
    if v_med < V_DARK_NEUTRAL:  # L139
        return "NEUTRAL"  # L140

    # White/gray -> many pixels are bright but low saturation  # L142
    bright = (V > V_BRIGHT_MIN)  # L143
    bright_count = int(bright.sum())  # L144

    if bright_count >= 30:  # L146
        neutral_like = bright & (S < S_NEUTRAL_MAX)  # L147
        neutral_ratio = int(neutral_like.sum()) / max(1, bright_count)  # L148
        if neutral_ratio > NEUTRAL_BRIGHT_RATIO:  # L149
            return "NEUTRAL"  # L150

    # --------------------------------  # L152
    # B) Colored classification (nearest of RED/GREEN/BLUE)  # L153
    # --------------------------------  # L154
    V_TH = max(25.0, v_med * 0.55)  # L155
    good = (S > S_TH) & (V > V_TH)  # L156
    good_ratio = int(good.sum()) / max(1, int(H.size))  # L157

    if good_ratio < GOOD_RATIO_MIN:  # L159
        return "NEUTRAL"  # L160

    H_good = H[good].astype(np.uint8)  # L162
    hist = cv2.calcHist([H_good], [0], None, [180], [0, 180])  # L163
    h_peak = int(np.argmax(hist))  # L164

    parents = {"RED": 0, "GREEN": 60, "BLUE": 120}  # L166

    def hue_dist(a: int, b: int) -> int:  # L168
        d = abs(a - b)  # L169
        return min(d, 180 - d)  # L170

    return min(parents, key=lambda k: hue_dist(h_peak, parents[k]))  # L172

# =====================================================  # L174
# SNAPSHOT BRICK DETECTION (FOR AUTOMATION)  # L175
# =====================================================  # L176

def detect_bricks(frame: np.ndarray) -> List[Dict[str, float]]:  # L178
    """Process ONE image and return detected bricks with robot coords."""  # L179
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # L180
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # L181

    thresh = cv2.adaptiveThreshold(  # L183
        blur, 255,  # L184
        cv2.ADAPTIVE_THRESH_MEAN_C,  # L185
        cv2.THRESH_BINARY_INV,  # L186
        25, 5  # L187
    )  # L188

    # Apply ROI restriction BEFORE contour detection  # L190
    thresh, (rx0, ry0, rx1, ry1) = apply_roi_mask(thresh)  # L191

    contours, _ = cv2.findContours(  # L193
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # L194
    )  # L195

    bricks: List[Dict[str, float]] = []  # L197

    for c in contours:  # L199
        area = cv2.contourArea(c)  # L200
        if area < MIN_AREA or area > MAX_AREA:  # L201
            continue  # L202

        rect = cv2.minAreaRect(c)  # L204
        (cx, cy), (rw, rh), angle = rect  # L205

        # Extra guard: reject anything whose center is outside ROI  # L207
        if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1):  # L208
            continue  # L209

        # Normalize angle  # L211
        if rh > rw:  # L212
            angle += 90  # L213
        angle = angle % 180  # L214

        x, y, bw, bh = cv2.boundingRect(c)  # L216

        # Shrink ROI for color sampling to avoid edges/background  # L218
        pad = int(min(bw, bh) * ROI_SHRINK_FRAC)  # L219
        pad = max(ROI_SHRINK_MIN_PX, min(pad, ROI_SHRINK_MAX_PX))  # L220

        x2 = x + pad  # L222
        y2 = y + pad  # L223
        bw2 = max(1, bw - 2 * pad)  # L224
        bh2 = max(1, bh - 2 * pad)  # L225

        color = detect_color(frame, x2, y2, bw2, bh2)  # L227

        Xr, Yr = pixel_to_robot(cx, cy)  # L229

        bricks.append({  # L231
            "x": float(Xr),  # L232
            "y": float(Yr),  # L233
            "angle": float(angle),  # L234
            "color": color,  # L235
        })  # L236

    return bricks  # L238
