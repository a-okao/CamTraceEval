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
        w = p - self.p0
        cross = abs(self.vector[0] * w[1] - self.vector[1] * w[0])
        return cross / self.norm

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
