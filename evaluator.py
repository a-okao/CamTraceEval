from typing import List, Optional, Tuple

import numpy as np


def compute_errors(xs: List[Optional[float]], ys: List[Optional[float]], trajectory) -> List[float]:
    errors = []
    for x, y in zip(xs, ys):
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            errors.append(np.nan)
        else:
            errors.append(float(trajectory.distance(x, y)))
    return errors


def error_stats(errors: List[float]) -> Tuple[float, float]:
    arr = np.array(errors, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean(valid ** 2)))
    max_err = float(np.max(valid))
    return rmse, max_err


def movement_duration(t_start: Optional[float], t_end: Optional[float]) -> float:
    if t_start is None or t_end is None:
        return float("nan")
    return float(t_end - t_start)
