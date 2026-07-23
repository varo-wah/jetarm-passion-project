from typing import Any, Dict, Tuple

from jetarm.config.detection_roi import roi_bounds_from_shape


EDGE_MARGIN_FRAC = 0.08
EDGE_MARGIN_MIN_PX = 24


def choose_safe_wrist_angle(
    brick: Dict[str, Any],
    frame_shape: Tuple[int, int],
) -> Tuple[float, str]:
    detected_angle = float(brick.get("angle", 90.0))
    required_keys = ("center_x", "center_y", "x1", "y1", "x2", "y2")

    if frame_shape is None:
        return detected_angle, "clear"

    if not all(key in brick and brick[key] is not None for key in required_keys):
        return detected_angle, "clear"

    rx0, ry0, rx1, ry1 = roi_bounds_from_shape(frame_shape)
    roi_w = max(1, rx1 - rx0)
    roi_h = max(1, ry1 - ry0)
    margin = max(EDGE_MARGIN_MIN_PX, int(min(roi_w, roi_h) * EDGE_MARGIN_FRAC))

    cx = float(brick["center_x"])
    cy = float(brick["center_y"])
    x1 = float(brick["x1"])
    y1 = float(brick["y1"])
    x2 = float(brick["x2"])
    y2 = float(brick["y2"])

    edge_distances = {
        "left": min(abs(x1 - rx0), abs(cx - rx0)),
        "right": min(abs(rx1 - x2), abs(rx1 - cx)),
        "top": min(abs(y1 - ry0), abs(cy - ry0)),
        "bottom": min(abs(ry1 - y2), abs(ry1 - cy)),
    }

    edge, distance = min(edge_distances.items(), key=lambda item: item[1])
    if distance > margin:
        return detected_angle, "clear"

    if edge in ("left", "right"):
        return 90.0, edge

    return 0.0, edge
