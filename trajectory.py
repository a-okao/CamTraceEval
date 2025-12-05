from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LineTrajectory:
    p0: np.ndarray
    p1: np.ndarray

    def __post_init__(self):
        self.vector = self.p1 - self.p0
        self.norm = np.linalg.norm(self.vector)
        if self.norm == 0:
            raise ValueError("LineTrajectory requires two distinct points")

    def distance(self, x: float, y: float) -> float:
        p = np.array([x, y], dtype=float)
        
        # Project p onto the line segment defined by p0 and p1
        # v = p1 - p0 (self.vector)
        # w = p - p0
        # t = (w . v) / (v . v)
        
        w = p - self.p0
        c1 = np.dot(w, self.vector)
        c2 = np.dot(self.vector, self.vector) # This is self.norm**2
        
        if c2 <= 0:
            return np.linalg.norm(p - self.p0)
            
        t = c1 / c2
        t = max(0.0, min(1.0, t))
        
        closest = self.p0 + t * self.vector
        return np.linalg.norm(p - closest)

    def ideal_points(self) -> np.ndarray:
        return np.vstack([self.p0, self.p1])


@dataclass
class CircleTrajectory:
    center: np.ndarray
    radius: float
    samples: int = 360

    def distance(self, x: float, y: float) -> float:
        dx = x - self.center[0]
        dy = y - self.center[1]
        d = np.hypot(dx, dy)
        return abs(d - self.radius)

    def ideal_points(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, self.samples, endpoint=True)
        xs = self.center[0] + self.radius * np.cos(angles)
        ys = self.center[1] + self.radius * np.sin(angles)
        return np.vstack([xs, ys]).T


def build_trajectory(mode: str, cfg: dict):
    traj_cfg = cfg.get("trajectory", {})
    samples = int(traj_cfg.get("ideal_plot_samples", 360))
    if mode.upper() == "LINE":
        line_cfg = traj_cfg.get("line", {})
        p0 = np.array(line_cfg.get("p0", [0.0, 0.0]), dtype=float)
        p1 = np.array(line_cfg.get("p1", [100.0, 0.0]), dtype=float)
        traj = LineTrajectory(p0=p0, p1=p1)
    elif mode.upper() == "CIRCLE":
        circle_cfg = traj_cfg.get("circle", {})
        center = np.array(circle_cfg.get("center", [0.0, 0.0]), dtype=float)
        radius = float(circle_cfg.get("radius", 40.0))
        traj = CircleTrajectory(center=center, radius=radius, samples=samples)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return traj


def fit_circle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a circle to points (x, y) using least squares.
    Returns (center, radius).
    """
    # Equation: x^2 + y^2 + Dx + Ey + F = 0
    # Rewrite as: Dx + Ey + F = -(x^2 + y^2)
    # Solve A w = b for w=[D, E, F]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)

    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = w

    x_c = -D / 2
    y_c = -E / 2
    radius = np.sqrt(x_c**2 + y_c**2 - F)

    return np.array([x_c, y_c]), radius


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a line segment to points (x, y).
    Uses PCA to find the direction and projects points to find the endpoints.
    Returns (p0, p1).
    """
    points = np.column_stack((x, y))
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # PCA
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    # Eigenvector with largest eigenvalue is the direction
    direction = evecs[:, np.argmax(evals)]
    
    # Project points onto the line defined by mean and direction
    # t = dot(point - mean, direction)
    projections = np.dot(centered, direction)
    
    t_min = np.min(projections)
    t_max = np.max(projections)
    
    p0 = mean + t_min * direction
    p1 = mean + t_max * direction
    
    return p0, p1
