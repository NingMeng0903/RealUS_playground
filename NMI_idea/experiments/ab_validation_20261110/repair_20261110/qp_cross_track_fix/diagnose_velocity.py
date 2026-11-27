"""Independent instantaneous feasibility of the six-dimensional QP request."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dird"))
import qp_replay as replay


def main():
    bridge = replay.QPReplayBridge("native", "urdf", native_clock="deterministic")
    kin, inner, cfg = bridge.make_controller()
    vlim = inner.limits.v_max.copy()
    inner.close()
    out = {"scope": "Necessary feasibility only: measured-rail affine map, original logged velocity box; no extra CBF rows.", "runs": {}}
    row_scale = np.r_[np.full(3, 1/.02), np.full(3, 1/.4)]
    for name in ("original_2cm", "6d_2cm", "6d_retime_0p5cm"):
        z = np.load(HERE.parent / "qp_clock_repair" / f"{name}_trace.npz")
        records = []
        ids = np.unique(np.r_[np.arange(0, len(z["q"]), 4), np.argmax(z["position_error_m"])])
        max_prediction_error = 0.
        for i in ids:
            q = z["q_meas_before"][i]
            J = np.asarray(kin.jacobian(q)).copy()
            rail_actual = z["qdot_prev_used"][i, 0]
            offset = J[:, 0] * rail_actual
            Jtask = J.copy()
            Jtask[:, 0] = 0.
            req = z["v_cmd_received"][i]
            b = (req-offset)*row_scale
            A = Jtask*row_scale[:,None]
            bounds = list(zip(z["box_lo"][i], z["box_hi"][i]))
            box = linprog(np.zeros(8), A_eq=A, b_eq=b, bounds=bounds, method="highs")
            linear = linprog(np.zeros(8), A_eq=A[:3], b_eq=b[:3], bounds=bounds, method="highs")
            angular = linprog(np.zeros(8), A_eq=A[3:], b_eq=b[3:], bounds=bounds, method="highs")
            speed = linprog(np.zeros(8), A_eq=A, b_eq=b,
                bounds=list(zip(-vlim, vlim)), method="highs")
            predicted = Jtask @ z["qdot"][i] + offset
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(predicted-z["v_tcp_estimated"][i]))))
            records.append(dict(tick=int(i), box_feasible=bool(box.success),
                linear_only_feasible=bool(linear.success), angular_only_feasible=bool(angular.success),
                speed_only_feasible=bool(speed.success),
                position_error_m=float(z["position_error_m"][i]),
                linear_residual_m_s=float(np.linalg.norm(req[:3]-predicted[:3])),
                angular_residual_rad_s=float(np.linalg.norm(req[3:]-predicted[3:])),
                req=req.tolist(), achieved=predicted.tolist(),
                box_lo=z["box_lo"][i].tolist(), box_hi=z["box_hi"][i].tolist()))
        out["runs"][name] = dict(samples=len(records), box_feasible=sum(x["box_feasible"] for x in records),
            linear_only_feasible=sum(x["linear_only_feasible"] for x in records),
            angular_only_feasible=sum(x["angular_only_feasible"] for x in records),
            speed_only_feasible=sum(x["speed_only_feasible"] for x in records),
            J_prediction_max_error=max_prediction_error, records=records)
        print(name, {k:v for k,v in out["runs"][name].items() if k!="records"}, flush=True)
    (HERE/"velocity_feasibility.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
