"""Read-only audit of requested, published and measured angular motion.

Candidate minus nominal includes all QP effects; it is not a counterfactual
visual-only response. Measured rotation includes nominal control and path motion.
No hardware interfaces are imported or used.
"""
import argparse
import ast
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def analyze(root):
    session = json.loads((root / "session.json").read_text())
    clock = session["clock"]
    epoch = clock["anchor_time_ns"] - clock["anchor_monotonic_ns"]
    results = []
    for trial in session["trials"]:
        for attempt in trial["attempts"]:
            directory = root / attempt["directory"]
            if not (directory / "contact_qp.jsonl").is_file():
                continue
            controls, published = [], {}
            with (directory / "contact_qp.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["event"] == "control_sample":
                        episode = row["repair_episode"]
                        row["episode"] = ast.literal_eval(episode) if isinstance(episode, str) else episode
                        controls.append(row)
                    elif row["event"] == "publication" and row["success"]:
                        published[row["control_id"]] = row
            with h5py.File(directory / "raw.h5") as raw:
                pose_t = raw["tcp/timestamp_mono_ns"][:] * 1e-9
                pose = raw["tcp/pose_m_rad"][:]
                force_end = float(raw["force/timestamp_mono_ns"][-1]) * 1e-9
            assert np.all(np.diff(pose_t) > 0)
            rotations = Rotation.from_euler("xyz", pose[:, 3:])
            slerp = Slerp(pose_t, rotations)
            start = (attempt["stages"]["tracking_observed"] - epoch) * 1e-9
            end = (attempt["stages"]["path_done"] - epoch) * 1e-9 if "path_done" in attempt["stages"] else force_end
            times = np.array([r["record_monotonic_s"] for r in controls])
            assert np.all(np.diff(times) > 0)
            dt = np.maximum(0, np.minimum(np.r_[times[1:], end], end) - np.maximum(times, start))
            scan = (times >= start) & (times <= end)
            allowed = np.array([r["episode"]["repair_allowed"] for r in controls])
            requested = np.array([r["allocation_diagnostics"]["differential_request_m_s"] > 0 for r in controls])
            nominal = np.array([r["nominal_twist_tool"][4] for r in controls])
            candidate = np.array([r["candidate_twist_tool"][4] for r in controls])
            final = np.array([published[r["control_id"]]["final_command_model_tool"][4] if r["control_id"] in published else np.nan for r in controls])
            active = scan & requested & np.isfinite(final)
            sign = np.array([r["allocation_diagnostics"]["differential_sign"] for r in controls])
            # Transparent returns omit the constant, geometry-derived row.
            observed_rows = [r["allocation_diagnostics"]["differential_row"] for r in controls
                             if "differential_row" in r["allocation_diagnostics"]]
            assert np.allclose(observed_rows, observed_rows[0], rtol=0, atol=1e-12)
            row_y = np.full(len(controls), observed_rows[0][4])
            desired_direction = sign * row_y
            quality = np.array([r["feature"]["quality"] if r["feature"] else [np.nan]*3 for r in controls])
            exhausted = np.array([r["episode"]["exhausted"] for r in controls])
            force_gate = np.array([r["allocation_diagnostics"]["repair_force_gate"] for r in controls])
            deadband = attempt["contact_qp"]["config"]["qp"]["differential_repair"]["balance_deadband"]
            latent_direction = (np.abs(quality[:, 2]-quality[:, 0]) > deadband) & (force_gate > 0)
            result = dict(
                attempt=attempt["directory"],
                scan_s=end-start,
                requested_cycles=int((scan & requested).sum()),
                requested_successfully_published_cycles=int(active.sum()),
                permission_s=float(dt[allowed].sum()),
                permission_without_differential_request_s=float(dt[allowed & ~requested].sum()),
                requested_s=float(dt[requested].sum()),
                exhausted_with_latent_direction_s=float(dt[exhausted & latent_direction].sum()),
                # Numerical-zero exclusion for this audit, not a control parameter.
                qp_changed_nominal_wy_over_1e_6_rad_s_cycles=int(np.sum(active & (np.abs(candidate-nominal) > 1e-6))),
                final_abs_wy_deg_s_p50_p95_max=np.degrees(np.quantile(np.abs(final[active]), [.5, .95, 1])).tolist(),
                final_minus_nominal_abs_wy_deg_s_p50_p95_max=np.degrees(np.quantile(np.abs(final[active]-nominal[active]), [.5, .95, 1])).tolist(),
                qp_adjustment_in_requested_direction_fraction=float(np.mean(desired_direction[active]*(candidate[active]-nominal[active]) > 1e-9)),
                final_wy_in_requested_direction_fraction=float(np.mean(desired_direction[active]*final[active] > 1e-9)),
                final_candidate_max_wy_difference_deg_s=float(np.degrees(np.max(np.abs(final[active]-candidate[active])))),
                repair_episodes=[],
            )
            # Episode intervals end on a healthy reset or the last scan sample.
            episode_start = None
            for i, row in enumerate(controls):
                ep = row["episode"]
                if ep["armed"] and episode_start is None:
                    episode_start = i
                closed = episode_start is not None and (not ep["armed"] or i == len(controls)-1)
                if not closed:
                    continue
                a, b = episode_start, i
                episode_start = None
                if not np.any(requested[a:b+1] & scan[a:b+1]) and not np.any(allowed[a:b+1] & scan[a:b+1]):
                    continue
                t0, t1 = max(times[a], start), min(times[b], end)
                if t1 <= t0 or t0 < pose_t[0] or t1 > pose_t[-1]:
                    continue
                pair = slerp([t0, t1])
                measured_y = (pair[0].inv() * pair[1]).as_rotvec()[1]
                sl = slice(a,b+1)
                reset = not ep["armed"]
                last = controls[b-1]["episode"] if reset else ep
                result["repair_episodes"].append(dict(
                    start_scan_s=t0-start, end_scan_s=t1-start,
                    healthy_reset=reset,
                    requested_cycles=int((requested[sl] & scan[sl]).sum()),
                    measured_relative_tool_y_deg=float(np.degrees(measured_y)),
                    commanded_qp_minus_nominal_integral_deg=float(np.degrees(np.sum((candidate[sl]-nominal[sl])*dt[sl]))),
                    commanded_total_integral_deg=float(np.degrees(np.sum(candidate[sl]*dt[sl]))),
                    charged_permission_s=last["permission_elapsed_s"],
                    charged_angle_travel_deg=float(np.degrees(last["measured_angle_travel_rad"])),
                    start_quality_lcr=quality[a].tolist(), end_quality_lcr=quality[b].tolist(),
                    exhausted_s=float(sum(dt[j] for j in range(a,b+1) if controls[j]["episode"]["exhausted"])),
                    exhausted_with_latent_direction_s=float(dt[sl][exhausted[sl] & latent_direction[sl]].sum()),
                ))
            results.append(result)
    return dict(
        method="Publication joined by control_id; scan times use session shared clock. Pose rotation uses xyz RPY and SO(3) interpolation, not Euler subtraction.",
        limits="Final-minus-nominal includes all QP effects. Measured episode rotation is total relative rotation, not visual-only attribution. Image changes during translation are not causal proof of angular repair. No hardware execution or controller modification.",
        attempts=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
