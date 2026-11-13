"""Render recorded FK, synthetic returns, and the online posterior as an MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FPS = 12
SCAN_SECONDS = 18.0
HOLD_SECONDS = 4.0
WIDTH, HEIGHT = 1280, 720


def _as_float(value, fallback=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def _interval(value):
    if value is None:
        return np.nan, np.nan
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) < 2 or not np.isfinite(arr[:2]).all():
        return np.nan, np.nan
    return float(arr[0]), float(arr[1])


def _field_lateral(frame):
    """Get the filtered total lateral displacement recorded at one exposure."""
    value = frame.get("offset_mean_mm")
    if value is not None:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if len(arr) and np.isfinite(arr[0]):
            return float(arr[0])
    for key in ("total_lateral_mean_mm", "estimated_lateral_mm"):
        value = _as_float(frame.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def _case_and_arm(summary, case_index, arm):
    cases = summary.get("cases", [])
    if case_index < 0 or case_index >= len(cases):
        raise ValueError(f"case index {case_index} is outside 0..{len(cases) - 1}")
    case = cases[case_index]
    arms = case.get("arms", {})
    if arm not in arms:
        raise ValueError(f"arm {arm!r} is unavailable. Recorded arms are {sorted(arms)}")
    frames = arms[arm].get("frames", [])
    if not frames:
        raise ValueError("the selected arm has no exposure frames")
    return case, arms[arm], frames


def _trace_key(traces, case_index, arm, suffix):
    candidates = (
        f"{case_index}_{arm}_{suffix}",
        f"{case_index}_{arm}_poses" if suffix == "fk_m" else "",
        f"{case_index}_{arm}_fk" if suffix == "fk_m" else "",
    )
    for key in candidates:
        if key and key in traces:
            return key
    raise KeyError(f"could not find the executed {suffix} trace for {case_index}_{arm}")


def _local_vessels(vessels_m, first_pose):
    vessels_m = np.asarray(vessels_m, dtype=float)
    if vessels_m.shape != (65, 2, 3):
        raise ValueError(f"nominal_vessels_m must have shape (65, 2, 3), got {vessels_m.shape}")
    rotation = np.asarray(first_pose, dtype=float)[:3, :3]
    origin = np.asarray(first_pose, dtype=float)[:3, 3]
    local = (vessels_m - origin) @ rotation
    return local[..., :2] * 1000.0


def _shifted_vessels(vessels_m, first_pose, offset_axes, physical_shift_mm):
    """Return the current physical anatomy in the first-pose local view.

    ``physical_shift_mm`` is the logged chart-frame shift after the calibration
    bias is removed.  The transport axes are part of the trace, so this does
    not invent a screen-space translation when the probe frame is tilted.
    """
    if offset_axes is None:
        return _local_vessels(vessels_m, first_pose)
    axes = np.asarray(offset_axes, dtype=float)
    shift = np.asarray(physical_shift_mm, dtype=float).reshape(-1)
    if axes.shape != (3, 2) or len(shift) < 2 or not np.isfinite(shift[:2]).all():
        return _local_vessels(vessels_m, first_pose)
    shift_world = shift[:2] @ axes.T * .001
    shifted = np.asarray(vessels_m, dtype=float) + shift_world
    return _local_vessels(shifted, first_pose)


def _local_path(poses, first_pose):
    rotation = np.asarray(first_pose, dtype=float)[:3, :3]
    origin = np.asarray(first_pose, dtype=float)[:3, 3]
    local = (np.asarray(poses)[:, :3, 3] - origin) @ rotation
    return local[:, :2] * 1000.0


def _load_skin_outline(path, first_pose):
    """Load an optional transported skin polygon without exposing a raw mesh.

    Accepted NPZ keys are ``skin_outline_m`` (N×3 robot-frame metres),
    or ``skin_outline_local_mm``/``outline_local_mm`` (N×2 first-pose local millimetres). The outline is only
    a visual context layer; it is never used by the controller.
    """
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as loaded:
        key = next((name for name in ("skin_outline_m", "skin_outline_local_mm", "outline_local_mm")
                    if name in loaded.files), None)
        if key is None:
            raise KeyError("skin outline NPZ must contain skin_outline_m or outline_local_mm")
        outline = np.asarray(loaded[key], dtype=float)
        if "reference_fk" in loaded and not np.allclose(loaded["reference_fk"], first_pose, rtol=0., atol=1.e-9):
            raise ValueError("skin outline uses a different reference probe frame")
    if outline.ndim != 2 or outline.shape[0] < 3 or outline.shape[1] not in (2, 3):
        raise ValueError(f"skin outline must be (N, 2) or (N, 3), got {outline.shape}")
    if outline.shape[1] == 3 and key not in ("skin_outline_local_mm", "outline_local_mm"):
        origin = np.asarray(first_pose, dtype=float)[:3, 3]
        rotation = np.asarray(first_pose, dtype=float)[:3, :3]
        outline = (outline - origin) @ rotation
        return outline[:, :2] * 1000.0
    return outline[:, :2] if key in ("skin_outline_local_mm", "outline_local_mm") else outline[:, :2] * 1000.0


def _series(frames, traces, case_index, case=None, frame_dt_s=0.1):
    step = _as_float(frame_dt_s, 0.1)
    if not np.isfinite(step) or step <= 0.0:
        step = 0.1
    time = np.array([_as_float(row.get("t_s"), i * step) for i, row in enumerate(frames)], dtype=float)
    estimate = np.array([_field_lateral(row) for row in frames], dtype=float)
    low, high = zip(*[_interval(row.get("offset_interval90_mm")) for row in frames], strict=True)
    low, high = np.asarray(low), np.asarray(high)
    dynamic = np.array([_as_float(row.get("dynamic_lateral_mean_mm")) for row in frames])
    z_key = f"{case_index}_z_mm"
    z = np.asarray(traces[z_key], dtype=float).reshape(-1) if z_key in traces else np.full(len(frames), np.nan)
    z_time = time[0] + step * np.arange(len(z))
    # The tape is a chart-relative dynamic residual.  For the physical anatomy
    # audit use the same convention as atlas_scan.py: w + [z, 0] - bias.
    effective_truth = np.full((len(z), 2), np.nan)
    physical_truth = np.full((len(z), 2), np.nan)
    if case is not None and len(z):
        static = np.asarray(case.get("hidden_static_offset_mm", [0.0, 0.0]), dtype=float).reshape(-1)
        bias = np.asarray(case.get("hidden_calibration_bias_mm", [0.0, 0.0]), dtype=float).reshape(-1)
        if len(static) >= 2 and len(bias) >= 2 and np.isfinite(static[:2]).all() and np.isfinite(bias[:2]).all():
            effective_truth[:] = static[:2]
            effective_truth[:, 0] += z
            physical_truth = effective_truth - bias[:2]
    return time, estimate, low, high, dynamic, z_time, z, effective_truth, physical_truth


def _status(frame):
    if frame.get("wrong", False):
        return "wrong identity"
    if frame.get("correct", False):
        return "artery accepted"
    if frame.get("accepted", False):
        return "return accepted"
    if frame.get("packet_mm"):
        return "new return"
    return "no detection"


def render(summary_path: Path, trace_path: Path, output: Path, case_index=0, arm="mpc_ou", skin_outline_path=None):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case, _, frames = _case_and_arm(summary, case_index, arm)
    with np.load(trace_path, allow_pickle=False) as loaded:
        traces = {key: loaded[key].copy() for key in loaded.files}
    pose_key = _trace_key(traces, case_index, arm, "fk_m")
    poses = np.asarray(traces[pose_key], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"executed pose trace must have shape (N, 4, 4), got {poses.shape}")
    vessels = traces.get("nominal_vessels_m")
    if vessels is None:
        raise KeyError("trace has no nominal_vessels_m array")
    vessel_xy = _local_vessels(vessels, poses[0])
    path_xy = _local_path(poses, poses[0])
    offset_axes = traces.get("offset_axes")
    skin_xy = _load_skin_outline(skin_outline_path, poses[0])
    frame_dt_s = float(summary["frame_dt_s"])
    time, estimate, low, high, dynamic, z_time, z, effective_truth, physical_truth = _series(
        frames, traces, case_index, case, frame_dt_s
    )

    finite_low = low[np.isfinite(low)]
    finite_high = high[np.isfinite(high)]
    finite_est = estimate[np.isfinite(estimate)]
    lower_extent = float(finite_low.min()) if len(finite_low) else float(finite_est.min()) if len(finite_est) else -2.0
    upper_extent = float(finite_high.max()) if len(finite_high) else float(finite_est.max()) if len(finite_est) else 2.0
    # Keep the main vessel view zoomed; a wide leg outline, when supplied, is
    # rendered only in a compact inset overview below.
    map_x = [vessel_xy[..., 0].ravel(), path_xy[:, 0]]
    map_y = [vessel_xy[..., 1].ravel(), path_xy[:, 1]]
    x_min = min(float(part.min()) for part in map_x) - 8.0
    x_max = max(float(part.max()) for part in map_x) + 8.0
    y_min = min(min(float(part.min()) for part in map_y), lower_extent) - 6.0
    y_max = max(max(float(part.max()) for part in map_y), upper_extent) + 6.0
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        raise ValueError("trace does not contain finite plot limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), (WIDTH, HEIGHT), fps=FPS, codec="libx264", bitrate="2600k",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
    )
    writer.send(None)
    fig = plt.figure(figsize=(WIDTH / 100.0, HEIGHT / 100.0), dpi=100, facecolor="#f6f8fa")
    grid = fig.add_gridspec(2, 5, height_ratios=(1.55, 1.0), hspace=0.29,
                            wspace=.33, left=.08, right=.96, top=.89, bottom=.10)
    top = fig.add_subplot(grid[0, :4] if skin_xy is not None else grid[0, :])
    bottom = fig.add_subplot(grid[1, :])
    skin_inset = fig.add_subplot(grid[0, 4]) if skin_xy is not None else None
    fig.text(.08, .945, "Leg scan · MPC + ProDMP", fontsize=17, weight="bold")
    fig.text(.96, .947, "SIMULATION REPLAY", fontsize=9, color="#526476", ha="right")
    band_values = np.concatenate([value[np.isfinite(value)]
                                  for value in (low, high, estimate, dynamic, effective_truth[:, 0])])
    band_min, band_max = (float(band_values.min()), float(band_values.max())) if len(band_values) else (-1., 1.)
    margin = .12 * max(2., band_max - band_min)
    total_frames = max(1, round((SCAN_SECONDS + HOLD_SECONDS) * FPS))
    exposures = len(frames)
    for output_index in range(total_frames):
        progress = min(1.0, output_index / max(1, SCAN_SECONDS * FPS))
        exposure = int(progress * (exposures - 1))
        pose_index = int(np.clip(round(progress * (len(poses) - 1)), 0, len(poses) - 1))
        probe_index = int(np.clip(round(exposure * (len(poses) - 1) / max(1, exposures - 1)),
                                  0, len(poses) - 1))
        top.clear()
        bottom.clear()
        top.set_facecolor("#ffffff")
        if skin_inset is not None and skin_xy is not None:
            skin_inset.clear()
            skin_inset.fill(skin_xy[:, 0], skin_xy[:, 1], color="#dbe6ef", alpha=.70)
            skin_inset.plot(skin_xy[:, 0], skin_xy[:, 1], color="#8aa4b8", lw=.8)
            skin_inset.plot(path_xy[:pose_index + 1, 0], path_xy[:pose_index + 1, 1], color="#008a88", lw=1.4)
            skin_inset.scatter(path_xy[pose_index, 0], path_xy[pose_index, 1], s=14, color="#111827", zorder=3)
            skin_inset.set_aspect("equal")
            skin_inset.set_title("Leg atlas", fontsize=10)
            skin_inset.axis("off")
        if len(physical_truth):
            anatomy_shift = physical_truth[min(exposure, len(physical_truth) - 1)]
            physical_vessels = _shifted_vessels(vessels, poses[0], offset_axes, anatomy_shift)
        else:
            physical_vessels = vessel_xy
        # The faint nominal traces show the fixed transported chart; the bold
        # lines are the current physical artery/vein after w+z-bias.
        top.plot(vessel_xy[:, 0, 0], vessel_xy[:, 0, 1], color="#cf6e28", lw=1.0, ls=":", alpha=.35)
        top.plot(vessel_xy[:, 1, 0], vessel_xy[:, 1, 1], color="#6a55a5", lw=1.0, ls=":", alpha=.35)
        top.plot(physical_vessels[:, 0, 0], physical_vessels[:, 0, 1], color="#cf6e28", lw=3.0, label="artery")
        top.plot(physical_vessels[:, 1, 0], physical_vessels[:, 1, 1], color="#6a55a5", lw=2.5, label="vein")
        top.plot(path_xy[: pose_index + 1, 0], path_xy[: pose_index + 1, 1], color="#008a88", lw=2.4)
        top.scatter(path_xy[pose_index, 0], path_xy[pose_index, 1], s=62, marker="s",
                    color="#111827", zorder=5, label="probe")
        row = frames[exposure]
        packets = np.asarray(row.get("packet_mm", []), dtype=float).reshape(-1, 2)
        if len(packets):
            # Image packets are emitted at exposure k.  Use the corresponding
            # FK sample (k*5 in the standard 201/41 trace), while the dark
            # probe marker continues smoothly along the executed path.
            current_pose = poses[probe_index]
            points = current_pose[:3, 3] + 1e-3 * (packets @ current_pose[:3, 1:3].T)
            local = 1000. * ((points - poses[0, :3, 3]) @ poses[0, :3, :3])
            top.scatter(local[:, 0], local[:, 1], s=34, color="#111827", edgecolor="white",
                        lw=.7, zorder=6, label="return")
        top.set_xlim(x_min, x_max)
        top.set_ylim(y_min, y_max)
        top.set_xlabel("local longitudinal axis (mm)", fontsize=10)
        top.set_ylabel("local lateral axis (mm)", fontsize=10)
        top.grid(alpha=.16)
        top.legend(loc="upper right", frameon=False, ncol=4, fontsize=9)
        top.set_title(f"{_status(row)}  ·  {exposure + 1}/{exposures}", loc="left", fontsize=11, color="#155e75")

        bottom.set_facecolor("#ffffff")
        end = exposure + 1
        shown_time = time[:end]
        if np.isfinite(low[:end]).any() and np.isfinite(high[:end]).any():
            bottom.fill_between(shown_time, low[:end], high[:end], color="#5b8db8", alpha=.19,
                                linewidth=0, label="90% band")
        if np.isfinite(estimate[:end]).any():
            bottom.plot(shown_time, estimate[:end], color="#075985", lw=2.5, label="filtered offset")
        if np.isfinite(dynamic[:end]).any():
            bottom.plot(shown_time, dynamic[:end], color="#008a88", lw=1.8, label="filtered z")
        truth_lateral = effective_truth[:, 0]
        truth_end = min(end, len(truth_lateral))
        if truth_end and np.isfinite(truth_lateral[:truth_end]).any():
            bottom.plot(z_time[:truth_end], truth_lateral[:truth_end], color="#9ca3af", lw=1.1,
                        ls="--", label="effective truth (w+z)")
        current_time = time[min(exposure, len(time) - 1)]
        bottom.axvline(current_time, color="#111827", lw=1.0, alpha=.55)
        bottom.set_xlim(time[0], max(time[0] + frame_dt_s, time[-1]))
        bottom.set_ylim(band_min - margin, band_max + margin)
        bottom.set_xlabel("exposure time (s)", fontsize=10)
        bottom.set_ylabel("lateral displacement (mm)", fontsize=10)
        bottom.grid(alpha=.16)
        if bottom.get_legend_handles_labels()[1]:
            bottom.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
        bottom.set_title("Online posterior", loc="left", fontsize=12, weight="bold")
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
        writer.send(np.ascontiguousarray(image))
    writer.close()
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--arm", default="mpc_ou")
    parser.add_argument("--skin-outline", type=Path, default=None,
                        help="optional NPZ with a derived 2-D/3-D transported skin polygon")
    args = parser.parse_args(argv)
    path = render(args.summary, args.traces, args.out, args.case, args.arm, args.skin_outline)
    print(path)


if __name__ == "__main__":
    main()
