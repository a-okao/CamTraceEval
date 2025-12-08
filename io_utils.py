import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(records: List[dict], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = ["time", "u", "v", "x", "y", "error"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: ("" if rec.get(k) is None else rec.get(k)) for k in fieldnames}
            writer.writerow(row)


def load_csv(path: Path) -> List[dict]:
    records = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {}
            for k, v in row.items():
                if v == "":
                    rec[k] = None
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v  # Keep as string if not float (shouldn't happen for expected fields)
            records.append(rec)
    return records


def plot_trajectory(ideal: np.ndarray, actual: Iterable[Tuple[float, float]], output_path: Path, mode: str) -> None:
    ensure_dir(output_path.parent)
    ideal = np.asarray(ideal, dtype=float)
    real_pts = np.array([(x, y) for x, y in actual if x is not None and y is not None and not (np.isnan(x) or np.isnan(y))], dtype=float)
    plt.figure(figsize=(6, 6))
    # Draw measured points first
    if real_pts.size > 0:
        plt.plot(real_pts[:, 0], real_pts[:, 1], label="Measured", color="tab:orange", marker="o", markersize=3, linewidth=1)
    # Then draw ideal trajectory on top
    plt.plot(ideal[:, 0], ideal[:, 1], label=f"Ideal {mode}", color="tab:blue")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_error(times: List[float], errors: List[float], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    t = np.array(times, dtype=float)
    e = np.array(errors, dtype=float)
    mask = ~np.isnan(t) & ~np.isnan(e)
    plt.figure(figsize=(7, 4))
    plt.plot(t[mask], e[mask], color="tab:red", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Error [mm]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
