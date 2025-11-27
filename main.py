import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from calib import pixel_to_world
from camera import Camera
from evaluator import compute_errors, error_stats, movement_duration
from io_utils import ensure_dir, plot_error, plot_trajectory, save_csv
from marker import detect_marker
from trajectory import build_trajectory


def key_code(value) -> int:
    """Convert config value to a single key code."""
    if isinstance(value, int):
        return value
    value_str = str(value)
    return ord(value_str[0]) if value_str else 0


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Camera + marker trajectory evaluation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["LINE", "CIRCLE"], default="LINE", help="Evaluation mode")
    parser.add_argument("--label", type=str, help="Label for output files (e.g., before/after)")
    parser.add_argument("--output-dir", type=Path, help="Base directory for outputs")
    parser.add_argument("--device", type=int, help="Camera device ID override")
    return parser.parse_args()


def format_key(char_value: int) -> str:
    return chr(char_value) if 0 <= char_value <= 255 else "?"


def main():
    args = parse_args()
    config = load_config(args.config)

    output_cfg = config.get("output", {})
    label = args.label or output_cfg.get("label", "run")
    base_dir = args.output_dir or Path(output_cfg.get("base_dir", "outputs"))
    mode = args.mode.upper()

    camera_cfg = config.get("camera", {})
    marker_cfg = config.get("marker", {})
    calib_cfg = config.get("calibration", {})
    display_cfg = config.get("display", {})

    start_key = key_code(camera_cfg.get("start_key", "s"))
    stop_key = key_code(camera_cfg.get("stop_key", "e"))
    quit_key = key_code(camera_cfg.get("quit_key", "q"))

    trajectory = build_trajectory(mode, config)

    cam = Camera(
        device_id=args.device if args.device is not None else int(camera_cfg.get("device_id", 0)),
        width=camera_cfg.get("frame_width"),
        height=camera_cfg.get("frame_height"),
        window_name=camera_cfg.get("window_name", "LiveView"),
    )

    records = []
    measuring = False
    t_start = None
    t_last = None

    try:
        while True:
            frame, ts = cam.get_frame()
            if frame is None or ts is None:
                print("Failed to read frame; stopping capture.")
                if measuring:
                    break
                else:
                    return

            u, v = detect_marker(frame, marker_cfg)
            x, y = pixel_to_world(u, v, calib_cfg)

            draw_frame = frame.copy()
            if u is not None and v is not None:
                radius = int(display_cfg.get("draw_marker_radius", 8))
                cv2.circle(draw_frame, (int(u), int(v)), radius, (0, 0, 255), thickness=-1)  # red filled marker overlay

            status_text = "REC" if measuring else "READY"
            status_color = (0, 255, 0) if measuring else (0, 255, 255)
            font_scale = float(display_cfg.get("font_scale", 0.5))
            text_color = tuple(display_cfg.get("text_color", [0, 255, 0]))
            cv2.putText(draw_frame, f"{status_text} mode={mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 1, cv2.LINE_AA)
            cv2.putText(
                draw_frame,
                f"[{format_key(start_key)}]=start  [{format_key(stop_key)}]=stop  [{format_key(quit_key)}]=quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

            uv_text = f"u,v: {u:.0f}, {v:.0f} px" if u is not None and v is not None else "u,v: N/A"
            if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                xy_text = f"x,y: {x:.1f}, {y:.1f} mm"
            else:
                xy_text = "x,y: N/A"
            cv2.putText(draw_frame, uv_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
            cv2.putText(draw_frame, xy_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

            key = cam.show(draw_frame)

            if key == start_key and not measuring:
                measuring = True
                t_start = ts
                records.clear()
                print("Measurement started.")
            elif key == stop_key and measuring:
                t_last = ts
                measuring = False
                print("Measurement stopped.")
                break
            elif key == quit_key:
                if measuring:
                    t_last = ts
                print("Quit requested by user.")
                break
            elif key == 27:  # ESC
                if measuring:
                    t_last = ts
                print("ESC pressed, quitting.")
                break

            if measuring:
                rel_time = ts - t_start if t_start is not None else ts
                records.append(
                    {
                        "time": rel_time,
                        "u": float(u) if u is not None else None,
                        "v": float(v) if v is not None else None,
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                    }
                )
                t_last = ts
    finally:
        cam.release()

    if not records:
        print("No data recorded. Use start/stop keys during capture.")
        return

    if t_start is None:
        t_start = 0.0
    if t_last is None and records:
        t_last = t_start + records[-1]["time"]

    xs = [rec["x"] for rec in records]
    ys = [rec["y"] for rec in records]
    errors = compute_errors(xs, ys, trajectory)
    for rec, err in zip(records, errors):
        if err is None or (isinstance(err, float) and math.isnan(err)):
            rec["error"] = None
        else:
            rec["error"] = float(err)

    rmse, max_err = error_stats(errors)
    duration = movement_duration(t_start, t_last)

    ensure_dir(base_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"mode_{mode.lower()}_{label}_{run_id}"
    csv_path = base_dir / f"{prefix}.csv"
    traj_plot_path = base_dir / f"{prefix}_trajectory.png"
    err_plot_path = base_dir / f"{prefix}_error.png"

    save_csv(records, csv_path)
    ideal_points = trajectory.ideal_points()
    actual_points = [(rec["x"], rec["y"]) for rec in records]
    plot_trajectory(ideal_points, actual_points, traj_plot_path, mode)
    times = [rec["time"] for rec in records]
    plot_error(times, errors, err_plot_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved trajectory plot: {traj_plot_path}")
    print(f"Saved error plot: {err_plot_path}")
    print(f"Samples: {len(records)}")
    print(f"Duration: {duration:.3f} s (t_start={t_start:.3f}, t_end={t_last:.3f})")
    print(f"RMSE: {rmse:.3f} mm, Max error: {max_err:.3f} mm")


if __name__ == "__main__":
    main()
