from typing import Optional, Tuple

import cv2
import numpy as np


def _build_mask(hsv: np.ndarray, marker_cfg: dict) -> np.ndarray:
    ranges = marker_cfg.get("hsv_ranges")
    if isinstance(ranges, list) and ranges:
        mask = None
        for r in ranges:
            lower = np.array(r.get("lower", [0, 0, 0]), dtype=np.uint8)
            upper = np.array(r.get("upper", [179, 255, 255]), dtype=np.uint8)
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            return cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8))
        return mask

    hsv_lower = np.array(marker_cfg.get("hsv_lower", [0, 0, 0]), dtype=np.uint8)
    hsv_upper = np.array(marker_cfg.get("hsv_upper", [179, 255, 255]), dtype=np.uint8)
    return cv2.inRange(hsv, hsv_lower, hsv_upper)


def detect_marker(frame, marker_cfg: dict) -> Tuple[Optional[int], Optional[int]]:
    """Detect marker center in pixel coordinates; returns (u, v) or (None, None)."""
    blur_k = int(marker_cfg.get("blur_kernel", 0))
    morph_k = int(marker_cfg.get("morph_kernel", 0))
    min_area = float(marker_cfg.get("min_area", 0))

    image = frame
    if blur_k and blur_k > 1 and blur_k % 2 == 1:
        image = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _build_mask(hsv, marker_cfg)

    if morph_k and morph_k > 1 and morph_k % 2 == 1:
        kernel = np.ones((morph_k, morph_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None, None

    m = cv2.moments(best)
    if m["m00"] == 0:
        return None, None
    u = int(m["m10"] / m["m00"])
    v = int(m["m01"] / m["m00"])
    return u, v
