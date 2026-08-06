"""Natural display output shared by the annotated camera views."""

from __future__ import annotations

import numpy as np


def natural_display_frame(frame: np.ndarray) -> np.ndarray:
    """Return an exact copy; no gamma, saturation, or color filter is applied."""

    if frame is None or frame.size == 0:
        return frame
    return frame.copy()
