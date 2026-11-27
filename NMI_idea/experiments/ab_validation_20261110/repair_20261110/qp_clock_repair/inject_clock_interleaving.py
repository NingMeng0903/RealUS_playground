"""One offline fault injection: expose wall time before legacy hook cleanup."""
from pathlib import Path
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qp_reference_followup"))
import run_reference_protocol as reference_run
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import NativeWbcClient
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P

INJECT_TICK = 80  # Fixed before running; nonzero rail velocity in frozen trace.


class FaultBridge(reference_run.replay.QPReplayBridge):
    def make_controller(self):
        kin, inner, cfg = super().make_controller()
        native = inner._native
        clean_notify = native._notify_request
        state = {"tick": 0, "injected": False, "snapshots": []}
        self.injection = state

        def notify():
            if int(native._in["cmd"][0]) == P.CMD_STEP and state["tick"] == INJECT_TICK:
                seq = int(native._in["seq"][0])
                published = float(native._in["t_mono"][0])
                # Native polls seq even without this notification. Wait for
                # its acknowledgement before the legacy hook rewrites SHM.
                deadline = time.monotonic() + .015
                while int(native._out["seq"][0]) != seq and time.monotonic() < deadline:
                    time.sleep(.0001)
                consumed = int(native._out["seq"][0]) == seq
                native._in["t_mono"][0] = 1000. + state["tick"] * cfg.dt
                state["snapshots"].append(dict(tick=state["tick"], seq=seq,
                    published_timestamp=published,
                    acknowledged_before_cleanup=consumed,
                    post_hook_timestamp=float(native._in["t_mono"][0])))
                state["injected"] = consumed
            return clean_notify()

        def update(*args, **kwargs):
            kwargs["model_time_s"] = (time.monotonic() if state["tick"] == INJECT_TICK
                                      else 1000. + state["tick"] * cfg.dt)
            before = native._seq
            try:
                return NativeWbcClient.update(native, *args, **kwargs)
            finally:
                if native._seq != before:
                    state["tick"] += 1

        native._notify_request = notify
        native.update = update
        return kin, inner, cfg


if __name__ == "__main__":
    frozen = np.load(reference_run.INPUT, allow_pickle=False)
    reference_run.replay.WorldPolylineReference = reference_run.Analytic6DReference
    bridge = FaultBridge("native", "urdf", native_clock="deterministic")
    raw = bridge.query_pose_path(frozen["branch4_path0_T"], frozen["branch4_path0_q_seed"],
        speed_m_s=.02, settle_s=2., return_trace=True, return_state=False,
        provenance="deliberate old-clock publication interleaving at fixed tick80; NOT nominal performance")
    trace = raw.pop("trace")
    np.savez_compressed(HERE / "fault_trace.npz", **{
        key: np.asarray([row[key] for row in trace]) for key in trace[0]})
    result = dict(injection=bridge.injection, result=raw)
    (HERE / "fault_injection.json").write_text(json.dumps(reference_run.clean_json(result),
        indent=2, allow_nan=False))
    print("injected", bridge.injection["injected"], "ticks", raw["n_steps"],
          "failures", len(raw["failures"]), flush=True)
