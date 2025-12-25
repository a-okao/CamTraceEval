import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from calib import build_circle_calibration, build_two_point_calibration, pixel_to_world
from camera import Camera
from evaluator import compute_errors, error_stats, movement_duration
from io_utils import ensure_dir, plot_error, plot_trajectory, save_csv, load_csv
from marker import detect_marker
from trajectory import CircleTrajectory, LineTrajectory, build_trajectory, fit_line_trajectory, fit_circle_trajectory


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
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup cycles (default: 3)")
    parser.add_argument("--cycles", type=int, default=5, help="Number of recording cycles to auto-stop (default: 5)")
    parser.add_argument("--load", type=Path, help="Path to a CSV file to load and re-evaluate")
    parser.add_argument("--fit-line", action="store_true", help="When loading LINE mode data, fit an ideal line (100mm) to the recorded points for evaluation.")
    parser.add_argument("--fit-circle", action="store_true", help="When loading CIRCLE mode data, fit an ideal circle to the recorded points for evaluation.")
    return parser.parse_args()


def process_results(records: list, trajectory, mode: str, label: str, base_dir: Path, round_trip_durations: list = None, lap_durations: list = None):
    if not records:
        print("No data recorded.")
        return

    # Assuming t_start is the time of the first record if not provided, 
    # but strictly speaking t_start is implicit in relative times if they start from 0.
    # However, records['time'] are relative to start.
    t_start = 0.0
    t_last = records[-1]["time"]

    xs = [rec["x"] for rec in records]
    ys = [rec["y"] for rec in records]
    errors = compute_errors(xs, ys, trajectory)
    
    # Update errors in records
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
    print(f"Duration: {duration:.3f} s")
    print(f"RMSE: {rmse:.3f} mm, Max error: {max_err:.3f} mm")

    result_lines = [
        f"Mode: {mode}",
        f"Samples: {len(records)}",
        f"Duration: {duration:.3f} s",
        f"RMSE: {rmse:.3f} mm",
        f"Max error: {max_err:.3f} mm",
    ]

    if mode == "LINE" and round_trip_durations:
        avg_rt = np.mean(round_trip_durations)
        min_rt = min(round_trip_durations)
        max_rt = max(round_trip_durations)
        print(f"Round Trips: {len(round_trip_durations)} completed.")
        print(f"  Avg: {avg_rt:.3f}s, Min: {min_rt:.3f}s, Max: {max_rt:.3f}s")
        result_lines.append(f"Round Trips: {len(round_trip_durations)}")
        result_lines.append(f"  Avg: {avg_rt:.3f}s, Min: {min_rt:.3f}s, Max: {max_rt:.3f}s")
    elif mode == "CIRCLE" and lap_durations:
        avg_lap = np.mean(lap_durations)
        min_lap = min(lap_durations)
        max_lap = max(lap_durations)
        print(f"Laps: {len(lap_durations)} completed.")
        print(f"  Avg: {avg_lap:.3f}s, Min: {min_lap:.3f}s, Max: {max_lap:.3f}s")
        result_lines.append(f"Laps: {len(lap_durations)}")
        result_lines.append(f"  Avg: {avg_lap:.3f}s, Min: {min_lap:.3f}s, Max: {max_lap:.3f}s")

    result_lines.extend([
        f"CSV: {csv_path.name}",
        f"Trajectory plot: {traj_plot_path.name}",
        f"Error plot: {err_plot_path.name}",
        "Press any key to close",
    ])

    # Result window (separate from camera view)
    res_h = 30 + 25 * len(result_lines)
    res_w = 640
    result_img = np.ones((res_h, res_w, 3), dtype=np.uint8) * 255
    y = 30
    for line in result_lines:
        cv2.putText(result_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 25
    cv2.imshow("Result", result_img)

    # Show saved plots in separate windows (press any key to close all)
    traj_img = cv2.imread(str(traj_plot_path))
    if traj_img is not None:
        cv2.imshow("Trajectory Plot", traj_img)
    else:
        print(f"Failed to open trajectory plot image: {traj_plot_path}")
    err_img = cv2.imread(str(err_plot_path))
    if err_img is not None:
        cv2.imshow("Error Plot", err_img)
    else:
        print(f"Failed to open error plot image: {err_plot_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    traj_cfg = config.get("trajectory", {})

    start_key = key_code(camera_cfg.get("start_key", "s"))
    stop_key = key_code(camera_cfg.get("stop_key", "e"))
    quit_key = key_code(camera_cfg.get("quit_key", "q"))
    calibrate_key = key_code(camera_cfg.get("calibrate_key", "c"))
    color_calibrate_key = key_code(camera_cfg.get("color_calibrate_key", "k"))

    calib_line_length = float(calib_cfg.get("two_point_length_mm", 100.0))
    endpoint_threshold = float(traj_cfg.get("endpoint_threshold_mm", 5.0))

    trajectory = build_trajectory(mode, config)

    # If loading from file, skip camera setup and loop
    if args.load:
        print(f"Loading data from {args.load}...")
        records = load_csv(args.load)
        if not records:
            print("Failed to load records or file is empty.")
            return
        
        if mode == "LINE" and args.fit_line: # Only fit if --fit-line is provided
            # Recalculate ideal trajectory based on loaded data for LINE mode
            print("Fitting ideal line (translation only) to loaded data...")
            xs = [rec["x"] for rec in records]
            ys = [rec["y"] for rec in records]
            try:
                # Use the direction and length from the configured ideal trajectory
                ideal_direction = trajectory.vector
                ideal_length = trajectory.norm
                
                trajectory = fit_line_trajectory(
                    np.array(xs, dtype=float), 
                    np.array(ys, dtype=float), 
                    length_mm=ideal_length,
                    fixed_direction=ideal_direction
                )
                print(f"New ideal line: p0={trajectory.p0}, p1={trajectory.p1}")
            except Exception as e:
                print(f"Failed to fit line: {e}. Using default config trajectory.")

        elif mode == "CIRCLE" and args.fit_circle: # Only fit if --fit-circle is provided
            # Recalculate ideal trajectory based on loaded data for CIRCLE mode
            print("Fitting ideal circle to loaded data...")
            xs = [rec["x"] for rec in records]
            ys = [rec["y"] for rec in records]
            try:
                trajectory = fit_circle_trajectory(np.array(xs, dtype=float), np.array(ys, dtype=float))
                print(f"New ideal circle: center={trajectory.center}, radius={trajectory.radius:.2f}mm")
            except Exception as e:
                print(f"Failed to fit circle: {e}. Using default config trajectory.")

        # We don't have round_trip/lap durations when loading from CSV
        # unless we re-calculate them or store them separately.
        # For now, we'll just pass empty lists or None.
        process_results(records, trajectory, mode, label, base_dir, None, None)
        return

    # Application states
    STATE_VIEW = "VIEW"
    STATE_GEOMETRY_CALIBRATION = "GEOMETRY_CALIBRATION"
    STATE_COLOR_CALIBRATION = "COLOR_CALIBRATION"
    app_state = STATE_VIEW

    runtime_calib = None
    calib_points_px = []
    calib_line_px = None
    calib_circle_draw = None
    runtime_traj = trajectory

    records = []
    measuring = False
    t_start = None
    t_last = None

    # Round trip measurement state (for LINE mode)
    RT_STATE_AWAITING_DEPARTURE = "AWAITING_DEPARTURE"
    RT_STATE_GOING_TO_ENDPOINT = "GOING_TO_ENDPOINT"
    RT_STATE_RETURNING_TO_STARTPOINT = "RETURNING_TO_STARTPOINT"
    round_trip_state = RT_STATE_AWAITING_DEPARTURE
    round_trip_start_time = None
    round_trip_durations = []

    # Lap timing state (for CIRCLE mode)
    LAP_STATE_AWAITING_START = "AWAITING_START"
    LAP_STATE_AWAITING_HALFWAY = "AWAITING_HALFWAY"
    LAP_STATE_AWAITING_FINISH = "AWAITING_FINISH"
    lap_state = LAP_STATE_AWAITING_START
    lap_start_time = None
    lap_durations = []

    WARMUP_CYCLES = args.warmup
    RECORD_CYCLES = args.cycles
    warmup_done = False

    cam = Camera(
        device_id=args.device if args.device is not None else int(camera_cfg.get("device_id", 0)),
        width=camera_cfg.get("frame_width"),
        height=camera_cfg.get("frame_height"),
        window_name=camera_cfg.get("window_name", "LiveView"),
    )

    # Store the current frame for the callback
    current_frame = None

    def on_geometry_calib_mouse(event, x, y, flags, param):
        nonlocal app_state, calib_points_px, runtime_calib, runtime_traj, calib_line_px, calib_circle_draw
        if app_state != STATE_GEOMETRY_CALIBRATION:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(calib_points_px) >= 2:
                return
            
            # Force horizontal line for the second point in LINE mode
            if mode == "LINE" and len(calib_points_px) == 1:
                y = calib_points_px[0][1]
                print(f"Forcing horizontal alignment: y set to {y}")
            # Force horizontal alignment for the edge point in CIRCLE mode
            elif mode == "CIRCLE" and len(calib_points_px) == 1:
                y = calib_points_px[0][1]
                print(f"Forcing horizontal alignment for circle edge: y set to {y}")

            calib_points_px.append((x, y))
            print(f"Calibration point {len(calib_points_px)}: ({x}, {y})")
            if len(calib_points_px) == 2:
                try:
                    if mode == "LINE":
                        runtime_calib = build_two_point_calibration(calib_points_px[0], calib_points_px[1], calib_line_length)
                        calib_line_px = (calib_points_px[0], calib_points_px[1])
                        runtime_traj = LineTrajectory(np.array([0.0, 0.0]), np.array([calib_line_length, 0.0]))
                        calib_circle_draw = None
                        print("Line calibration updated from two points.")
                    elif mode == "CIRCLE":
                        # User clicks edge then center
                        edge_pt = calib_points_px[0]
                        center_pt = calib_points_px[1]
                        
                        radius_mm = float(config.get("trajectory", {}).get("circle", {}).get("radius", 40.0))
                        runtime_calib = build_circle_calibration(center_pt, edge_pt, radius_mm=radius_mm)
                        runtime_traj = CircleTrajectory(center=np.array([0.0, 0.0]), radius=radius_mm, samples=int(config.get("trajectory", {}).get("ideal_plot_samples", 360)))
                        # store for drawing: center(px), radius(px)
                        r_px = np.linalg.norm(np.array(edge_pt, dtype=float) - np.array(center_pt, dtype=float))
                        calib_circle_draw = (center_pt, r_px)
                        calib_line_px = None
                        print(f"Circle calibration updated: center={center_pt}, radius_px={r_px:.2f} -> radius_mm={radius_mm}")
                    else:
                        print(f"Calibration not supported for mode: {mode}")
                except Exception as e:
                    print(f"Calibration failed: {e}")
                    runtime_calib = None
                    calib_line_px = None
                    calib_circle_draw = None
                app_state = STATE_VIEW
                cam.set_mouse_callback(None)

    def on_color_calib_mouse(event, x, y, flags, param):
        nonlocal app_state, marker_cfg
        if app_state != STATE_COLOR_CALIBRATION or current_frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_frame[y, x]
            h_int, s_int, v_int = int(h), int(s), int(v)
            print(f"Color picked at ({x}, {y}): BGR={current_frame[y,x].tolist()}, HSV=[{h_int}, {s_int}, {v_int}]")

            h_spread = 10
            s_spread = 50
            v_spread = 50

            hsv_lower = np.array([h_int - h_spread, s_int - s_spread, v_int - v_spread], dtype=int)
            hsv_upper = np.array([h_int + h_spread, s_int + s_spread, v_int + v_spread], dtype=int)
            
            # Clamp values to be in valid range
            hsv_lower = np.clip(hsv_lower, [0, 40, 40], [179, 255, 255])
            hsv_upper = np.clip(hsv_upper, [0, 40, 40], [179, 255, 255])

            marker_cfg["hsv_lower"] = hsv_lower.tolist()
            marker_cfg["hsv_upper"] = hsv_upper.tolist()
            # Also update hsv_ranges to keep it consistent
            marker_cfg["hsv_ranges"] = [{"lower": hsv_lower.tolist(), "upper": hsv_upper.tolist()}]

            print(f"New HSV range set:")
            print(f"  lower: {hsv_lower.tolist()}")
            print(f"  upper: {hsv_upper.tolist()}")

            app_state = STATE_VIEW
            cam.set_mouse_callback(None)

    try:
        while True:
            frame, ts = cam.get_frame()
            if frame is None or ts is None:
                print("Failed to read frame; stopping capture.")
                if measuring:
                    break
                else:
                    return
            
            current_frame = frame

            u, v = detect_marker(frame, marker_cfg)
            x, y = pixel_to_world(u, v, calib_cfg, runtime_calib=runtime_calib)
            current_error = None
            if runtime_traj and x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                current_error = runtime_traj.distance(x, y)

            if measuring and x is not None and y is not None:
                if mode == "LINE" and isinstance(runtime_traj, LineTrajectory):
                    p0 = runtime_traj.p0
                    p1 = runtime_traj.p1
                    pos = np.array([x, y])
                    dist_to_p0 = np.linalg.norm(pos - p0)
                    dist_to_p1 = np.linalg.norm(pos - p1)

                    if round_trip_state == RT_STATE_AWAITING_DEPARTURE:
                        if dist_to_p0 > endpoint_threshold:
                            round_trip_state = RT_STATE_GOING_TO_ENDPOINT
                            round_trip_start_time = ts
                            print(f"Round trip started at t={ts:.3f}")
                    elif round_trip_state == RT_STATE_GOING_TO_ENDPOINT:
                        if dist_to_p1 < endpoint_threshold:
                            round_trip_state = RT_STATE_RETURNING_TO_STARTPOINT
                    elif round_trip_state == RT_STATE_RETURNING_TO_STARTPOINT:
                        if dist_to_p0 < endpoint_threshold:
                            duration = ts - round_trip_start_time
                            round_trip_durations.append(duration)
                            print(f"Round trip {len(round_trip_durations)} completed in {duration:.3f}s")
                            
                            if not warmup_done and len(round_trip_durations) >= WARMUP_CYCLES:
                                warmup_done = True
                                t_start = ts
                                round_trip_durations.clear()
                                records.clear()
                                print(f"Warmup of {WARMUP_CYCLES} cycles complete. Recording started.")
                            elif warmup_done and len(round_trip_durations) >= RECORD_CYCLES:
                                print(f"Auto-stop: Reached target of {RECORD_CYCLES} cycles.")
                                measuring = False
                                t_last = ts
                                break
                            
                            round_trip_state = RT_STATE_AWAITING_DEPARTURE
                elif mode == "CIRCLE" and isinstance(runtime_traj, CircleTrajectory):
                    center = runtime_traj.center
                    angle = math.atan2(y - center[1], x - center[0])
                    angle_deg = math.degrees(angle)

                    # Angle thresholds for lap detection (e.g., start at 0, halfway at 180)
                    start_angle_thresh = 15  # +/- 15 degrees around 0
                    halfway_angle_thresh = 15 # +/- 15 degrees around 180

                    if lap_state == LAP_STATE_AWAITING_START:
                        if abs(angle_deg) < start_angle_thresh:
                            lap_state = LAP_STATE_AWAITING_HALFWAY
                            lap_start_time = ts
                            print(f"Lap started at t={ts:.3f}")
                    elif lap_state == LAP_STATE_AWAITING_HALFWAY:
                        if abs(angle_deg) > (180 - halfway_angle_thresh):
                            lap_state = LAP_STATE_AWAITING_FINISH
                    elif lap_state == LAP_STATE_AWAITING_FINISH:
                        if abs(angle_deg) < start_angle_thresh:
                            duration = ts - lap_start_time
                            lap_durations.append(duration)
                            print(f"Lap {len(lap_durations)} completed in {duration:.3f}s")
                            
                            if not warmup_done and len(lap_durations) >= WARMUP_CYCLES:
                                warmup_done = True
                                t_start = ts
                                lap_durations.clear()
                                records.clear()
                                print(f"Warmup of {WARMUP_CYCLES} laps complete. Recording started.")
                            elif warmup_done and len(lap_durations) >= RECORD_CYCLES:
                                print(f"Auto-stop: Reached target of {RECORD_CYCLES} laps.")
                                measuring = False
                                t_last = ts
                                break

                            lap_state = LAP_STATE_AWAITING_START

            draw_frame = frame.copy()
            if u is not None and v is not None:
                radius = int(display_cfg.get("draw_marker_radius", 8))
                cv2.drawMarker(
                    draw_frame,
                    (int(u), int(v)),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=radius,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            is_calibrating_geometry = app_state == STATE_GEOMETRY_CALIBRATION
            is_calibrating_color = app_state == STATE_COLOR_CALIBRATION
            
            if is_calibrating_geometry:
                status_text = "G-CAL"
                status_color = (0, 165, 255)
            elif is_calibrating_color:
                status_text = "C-CAL"
                status_color = (0, 165, 255)
            elif measuring:
                if not warmup_done:
                    current_count = len(round_trip_durations) if mode == "LINE" else len(lap_durations)
                    status_text = f"WARMUP {current_count}/{WARMUP_CYCLES}"
                    status_color = (0, 255, 255) # Yellow for warmup
                else:
                    current_count = len(round_trip_durations) if mode == "LINE" else len(lap_durations)
                    status_text = f"REC {current_count}/{RECORD_CYCLES}"
                    status_color = (0, 255, 0) # Green for recording
            else:
                status_text = "READY"
                status_color = (0, 255, 255)

            font_scale = float(display_cfg.get("font_scale", 0.5))
            text_color = tuple(display_cfg.get("text_color", [0, 255, 0]))
            cv2.putText(draw_frame, f"{status_text} mode={mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 1, cv2.LINE_AA)
            cv2.putText(
                draw_frame,
                f"[{format_key(start_key)}]=start [{format_key(stop_key)}]=stop [{format_key(calibrate_key)}]=g-cal [{format_key(color_calibrate_key)}]=c-cal [{format_key(quit_key)}]=quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

            text_y_base = 60
            if is_calibrating_geometry:
                if mode == "LINE":
                    calib_msg = f"Calibrate LINE: click 2 points ({calib_line_length:.0f}mm)"
                else:
                    calib_msg = "Calibrate CIRCLE: click center then edge"
                cv2.putText(draw_frame, calib_msg, (10, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 1, cv2.LINE_AA)
                text_y_base += 20
            elif is_calibrating_color:
                cv2.putText(draw_frame, "COLOR CALIBRATION: Click on the color to track", (10, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, 1, cv2.LINE_AA)
                text_y_base += 20

            uv_text = f"u,v: {u:.0f}, {v:.0f} px" if u is not None and v is not None else "u,v: N/A"
            if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                xy_text = f"x,y: {x:.1f}, {y:.1f} mm"
            else:
                xy_text = "x,y: N/A"
            cv2.putText(draw_frame, uv_text, (10, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
            cv2.putText(draw_frame, xy_text, (10, text_y_base + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

            # Error display under coordinates
            err_text_y = text_y_base + 40
            if current_error is not None and runtime_calib is not None:
                cv2.putText(
                    draw_frame,
                    f"err: {current_error:.1f} mm",
                    (10, err_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Draw calibration clicks and line overlay
            if calib_points_px and is_calibrating_geometry:
                for pt in calib_points_px:
                    cv2.circle(draw_frame, pt, 4, (255, 0, 0), thickness=-1)
            if calib_line_px and mode == "LINE":
                cv2.line(draw_frame, calib_line_px[0], calib_line_px[1], (255, 255, 0), thickness=2)
            if calib_circle_draw and mode == "CIRCLE":
                center_pt, r_px = calib_circle_draw
                cv2.circle(draw_frame, center_pt, int(r_px), (255, 255, 0), thickness=2)
                cv2.circle(draw_frame, center_pt, 4, (0, 255, 255), thickness=-1)

            key = cam.show(draw_frame)

            if key == calibrate_key:
                if measuring:
                    print("Stop measurement before entering calibration.")
                else:
                    app_state = STATE_GEOMETRY_CALIBRATION
                    cam.set_mouse_callback(on_geometry_calib_mouse)
                    calib_points_px = []
                    calib_line_px = None
                    calib_circle_draw = None
                    if mode == "LINE":
                        print(f"Calibration mode (LINE): click two points for {calib_line_length}mm reference.")
                    else:
                        print("Calibration mode (CIRCLE): click edge then center point.")
            elif key == color_calibrate_key:
                if measuring:
                    print("Stop measurement before entering color calibration.")
                else:
                    app_state = STATE_COLOR_CALIBRATION
                    cam.set_mouse_callback(on_color_calib_mouse)
                    print("Color calibration mode: Click on the color to track.")
            elif key == start_key and app_state == STATE_VIEW:
                measuring = True
                t_start = ts
                records.clear()
                warmup_done = False
                if mode == "LINE":
                    round_trip_state = RT_STATE_AWAITING_DEPARTURE
                    round_trip_durations.clear()
                elif mode == "CIRCLE":
                    lap_state = LAP_STATE_AWAITING_START
                    lap_durations.clear()
                print("Measurement started (Warmup phase).")
            elif key == start_key and app_state != STATE_VIEW:
                print("Finish calibration before starting measurement.")
            elif key == stop_key and measuring:
                t_last = ts
                measuring = False
                if not warmup_done:
                    print("Measurement stopped during warmup. No data recorded.")
                    records.clear() # Ensure no partial warmup data is saved
                else:
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

            if measuring and warmup_done:
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

    traj_for_eval = runtime_traj if runtime_traj is not None else trajectory
    
    process_results(records, traj_for_eval, mode, label, base_dir, round_trip_durations, lap_durations)


if __name__ == "__main__":
    main()
