"""Frozen-tick box-feasibility check; never invokes native code."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import linprog, lsq_linear

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NATIVE = ROOT / "planning" / "native_subset"
sys.path.insert(0, str(ROOT / "capability"))
from capability import CapabilityEvaluator  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def one(cap, file_name, *, pick="max"):
    z = np.load(NATIVE / file_name, allow_pickle=True)
    i = int(np.argmax(z["qp_equality_residual_max"])) if pick == "max" else int(pick)
    J = np.asarray(cap.kin.jacobian(z["q"][i]))
    v = np.asarray(z["v_cmd_received"][i])
    vf = np.asarray(z["v_cmd_feasible"][i])
    lo, hi = np.asarray(z["box_lo"][i]), np.asarray(z["box_hi"][i])
    vmax = np.asarray([.4] + [2.512] * 7)
    singular = np.linalg.svd(J, compute_uv=False)
    result = dict(file=file_name, tick=i, rank=int(np.linalg.matrix_rank(J)),
                  singular_values=singular.tolist(),
                  recorded_qp_residual=float(z["qp_equality_residual_max"][i]),
                  v_received_norm=float(np.linalg.norm(v)),
                  v_feasible_norm=float(np.linalg.norm(vf)),
                  recorded_qp_residual_relative_to_received=float(
                      z["qp_equality_residual_max"][i] / max(np.linalg.norm(v), 1e-12)),
                  saved_qdot_to_feasible_inf=float(np.max(abs(J @ z["qdot"][i] - vf))),
                  saved_feasible_to_tcp_inf=float(np.max(abs(vf - z["v_tcp_estimated"][i]))),
                  box_lo=lo.tolist(), box_hi=hi.tolist(), qdot=z["qdot"][i].tolist(),
                  box_width_median_arm=float(np.median(hi[1:] - lo[1:])))
    for name, bounds in (("trace_box", (lo, hi)), ("physical_velocity", (-vmax, vmax))):
        lp = linprog(np.zeros(8), A_eq=J, b_eq=v,
                     bounds=list(zip(*bounds)), method="highs")
        fit = lsq_linear(J, v, bounds=bounds, lsq_solver="exact",
                         tol=1e-14, lsmr_tol=1e-14, max_iter=1000)
        fit_f = lsq_linear(J, vf, bounds=bounds, lsq_solver="exact",
                           tol=1e-14, lsmr_tol=1e-14, max_iter=1000)
        result[name] = dict(
            linprog_success=bool(lp.success),
            linprog_message=str(lp.message),
            exact_residual=(None if lp.x is None else float(np.max(abs(J @ lp.x - v)))),
            bounded_lsq_residual=float(np.linalg.norm(J @ fit.x - v)),
            bounded_lsq_feasible_twist_residual=float(np.linalg.norm(J @ fit_f.x - vf)),
            bounded_lsq_active=int(np.sum((fit.x <= bounds[0] + 1e-8) | (fit.x >= bounds[1] - 1e-8))),
            bounded_lsq_optimality=float(fit.optimality),
        )
    return result


def clock_check():
    x = json.loads((NATIVE / "RESULTS.json").read_text())
    gaps, increments, nonpositive = [], [], 0
    for episode in x["episodes"]:
        previous_end = None
        for segment in episode["segments"]:
            t = np.load(NATIVE / segment["trace_file"], allow_pickle=True)["native_model_t_mono"]
            d = np.diff(t)
            increments.extend(d.tolist())
            nonpositive += int(np.sum(d <= 0))
            if previous_end is not None:
                gaps.append(float(t[0] - previous_end))
            previous_end = float(t[-1])
    return dict(min_increment=float(min(increments)), max_increment=float(max(increments)),
                nonpositive_increments=nonpositive, cross_segment_gaps=gaps)


def main():
    cap = CapabilityEvaluator()
    selected = [
        "h0_a0_identity_0_trace.npz", "h1_a0_identity_2_trace.npz",
        "h0_a0_t2_-0.08m_5_trace.npz", "h1_a0_t2_-0.08m_5_trace.npz",
    ]
    output = dict(
        scope="four frozen maximum-residual ticks; no native rerun and no source edits",
        source_sha256={
            "native_results": digest(NATIVE / "RESULTS.json"),
            "runner": digest(HERE.parent / "run_native_subset.py"),
            "protocol": digest(HERE.parent / "NATIVE_SUBSET_PROTOCOL.txt"),
        },
        clock=clock_check(),
        ticks=[one(cap, f) for f in selected],
        interpretation=[
            "The trace qdot box makes the exact received Cartesian equality infeasible on all four selected ticks; this is established by linprog, not inferred from box-active counts.",
            "The physical joint-speed box alone admits the exact received twist on three of four ticks. The remaining nominal tick has a bounded least-squares residual of about 0.0102 in the saved J/frame convention, so it is not treated as a pure physical-speed proof.",
            "The saved Jacobian has rank six on every selected tick. The smallest singular values are nonzero, so a singularity is not the explanation.",
            "Model time is strictly increasing at 5 ms per tick and cross-segment gaps are also 5 ms. This rules out a nonmonotone/repeated model-time failure in these traces, while not proving every historical clock path correct.",
            "The exact active source cannot be identified from these fields alone: the box combines rate/history constraints and the source code is needed to separate acceleration, jerk, and other terms. The robust conclusion is local box infeasibility plus missing execution gating.",
        ],
    )
    (HERE / "BOX_FEASIBILITY.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = [
        "Frozen QP box-feasibility diagnosis",
        "===================================",
        "Four maximum-residual ticks were checked with scipy linprog and bounded least squares.",
        "",
        "Result",
        "------",
        "For all four ticks, exact J(q) dq = v_cmd_received is infeasible under the recorded trace box_lo/box_hi. The bounded least-squares residuals are approximately 0.229, 0.400, 0.0469, and 0.0433 (the first two nominal segments, then two stress holds).",
        "The physical velocity box ±[0.4, 2.512 x 7] admits exact equality on both stress holds and the first nominal tick. The second nominal tick retains a bounded residual of about 0.0102 in this saved Jacobian/frame convention, so it is reported as a residual case rather than a pure speed-limit proof.",
        "All four Jacobians have rank six with nonzero smallest singular values; this is not a rank-deficiency diagnosis. Model time increments are 0.005 s throughout and across segment boundaries, with no nonpositive increment.",
        "",
        "Interpretation",
        "--------------",
        "The recorded local dynamic qdot box, not the static physical velocity limit by itself, is sufficient to make the requested task equality infeasible at the selected failure ticks. The native QP can still report solved because it returns a constrained/slack solution; the execution layer does not convert the large task residual or zero task progress into a failure/pause.",
        "The trace does not identify whether the box width comes from acceleration, jerk, or another history term. The next smallest check is to log those box terms for one of these exact ticks and compare the unconstrained and box-constrained solutions; no broad retuning is needed for the AB conclusion.",
    ]
    (HERE / "BOX_FEASIBILITY_REPORT.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"ticks": len(selected), "clock": output["clock"], "output": str(HERE / "BOX_FEASIBILITY.json")}, indent=2))


if __name__ == "__main__":
    main()
