from typing import Optional, Tuple

import numpy as np


def build_two_point_calibration(p0_px, p1_px, length_mm: float = 100.0) -> dict:
    """Build a runtime calibration from two clicked points defining a straight line of length_mm."""
    p0 = np.array(p0_px, dtype=float)
    p1 = np.array(p1_px, dtype=float)
    v = p1 - p0
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Calibration points must be distinct")
    axis = v / norm
    perp = np.array([-axis[1], axis[0]])
    scale = length_mm / norm  # mm per pixel along the line direction
    return {
        "model": "line_two_point",
        "origin": p0,
        "axis": axis,
        "perp": perp,
        "scale": scale,
        "length_mm": length_mm,
    }


def build_circle_calibration(center_px, edge_px, radius_mm: float) -> dict:
    c = np.array(center_px, dtype=float)
    e = np.array(edge_px, dtype=float)
    r_px = np.linalg.norm(e - c)
    if r_px == 0:
        raise ValueError("Circle calibration requires distinct center and edge point")
    scale = radius_mm / r_px  # mm per pixel
    return {
        "model": "circle_center_radius",
        "center_px": c,
        "radius_px": r_px,
        "radius_mm": radius_mm,
        "scale": scale,
    }


def pixel_to_world_runtime(u: float, v: float, runtime_calib: dict) -> Tuple[float, float]:
    p = np.array([u, v], dtype=float)
    origin = np.array(runtime_calib["origin"], dtype=float)
    axis = np.array(runtime_calib["axis"], dtype=float)
    perp = np.array(runtime_calib["perp"], dtype=float)
    scale = float(runtime_calib["scale"])
    delta = p - origin
    x_mm = float(np.dot(delta, axis) * scale)  # along the line (0 -> length_mm)
    y_mm = float(np.dot(delta, perp) * scale)  # perpendicular to the line
    return x_mm, y_mm


def pixel_to_world(u: Optional[float], v: Optional[float], calib_params: dict, runtime_calib: Optional[dict] = None) -> Tuple[Optional[float], Optional[float]]:
    if u is None or v is None:
        return None, None

    if runtime_calib is not None:
        try:
            model = runtime_calib.get("model")
            if model == "line_two_point":
                return pixel_to_world_runtime(float(u), float(v), runtime_calib)
            elif model == "circle_center_radius":
                c = np.array(runtime_calib["center_px"], dtype=float)
                scale = float(runtime_calib["scale"])
                delta = np.array([u, v], dtype=float) - c
                return float(delta[0] * scale), float(delta[1] * scale)
        except Exception:
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
