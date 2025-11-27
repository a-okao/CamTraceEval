from typing import Optional, Tuple

import numpy as np


def pixel_to_world(u: Optional[float], v: Optional[float], calib_params: dict) -> Tuple[Optional[float], Optional[float]]:
    if u is None or v is None:
        return None, None
    model = calib_params.get("model", "scale_offset")
    if model == "scale_offset":
        so = calib_params.get("scale_offset", {})
        sx = float(so.get("scale_x_mm_per_px", 1.0))
        sy = float(so.get("scale_y_mm_per_px", 1.0))
        ox = float(so.get("offset_x_mm", 0.0))
        oy = float(so.get("offset_y_mm", 0.0))
        x = u * sx + ox
        y = v * sy + oy
        return x, y
    elif model == "homography":
        H = np.array(calib_params.get("homography", {}).get("matrix", np.eye(3)), dtype=float)
        pt = np.array([u, v, 1.0], dtype=float)
        mapped = H @ pt
        if mapped[2] == 0:
            return None, None
        x = mapped[0] / mapped[2]
        y = mapped[1] / mapped[2]
        return x, y
    else:
        raise ValueError(f"Unknown calibration model: {model}")
