ROI_X0_FRAC = 0.2
ROI_X1_FRAC = 0.875
ROI_Y0_FRAC = 0.25
ROI_Y1_FRAC = 0.90

ROI_DRAW_BOX = True
ROI_MASK_DISPLAY = False


def roi_bounds_from_shape(shape):
    h, w = shape[:2]
    x0 = int(w * ROI_X0_FRAC)
    x1 = int(w * ROI_X1_FRAC)
    y0 = int(h * ROI_Y0_FRAC)
    y1 = int(h * ROI_Y1_FRAC)

    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w - 1))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h - 1))
    return x0, y0, x1, y1


def point_in_roi(shape, x, y):
    x0, y0, x1, y1 = roi_bounds_from_shape(shape)
    return x0 <= x <= x1 and y0 <= y <= y1
