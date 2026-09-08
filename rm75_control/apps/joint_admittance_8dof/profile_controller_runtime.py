#!/usr/bin/env python3
"""Offline native + CSV + plotting load benchmark with ideal robot feedback.

No hardware is opened. A CSV supplies only the initial measured pose; this is
a repeatable runtime benchmark, not a replay of the physical force experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/joint_admittance_8dof.yaml"


@contextmanager
def _bind_control_thread(cpu: int | None):
    """Temporarily bind only this calling thread to ``cpu``.

    ``sched_setaffinity(0, ...)`` operates on the calling Linux thread.  Save
    the current mask before the tick loop and restore it even when the
    benchmark exits through a solver error.  A missing/inaccessible CPU is a
    diagnostic result rather than a reason to mutate a wider process mask.
    """

    info = {
        "requested": None if cpu is None else int(cpu),
        "applied": False,
        "mask_before": None,
        "mask_during": None,
        "mask_after": None,
        "error": None,
    }
    original = None
    try:
        original = set(os.sched_getaffinity(0))
        info["mask_before"] = sorted(original)
    except (AttributeError, OSError, ValueError) as exc:
        info["error"] = f"getaffinity:{type(exc).__name__}"

    if cpu is None or original is None:
        try:
            yield info
        finally:
            if original is not None:
                try:
                    info["mask_after"] = sorted(os.sched_getaffinity(0))
                except (AttributeError, OSError, ValueError):
                    pass
        return

    try:
        os.sched_setaffinity(0, {int(cpu)})
        info["applied"] = True
        info["mask_during"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError, TypeError) as exc:
        info["error"] = f"setaffinity:{type(exc).__name__}"

    try:
        yield info
    finally:
        if info["applied"]:
            try:
                os.sched_setaffinity(0, original)
            except (AttributeError, OSError, ValueError):
                info["error"] = info["error"] or "restoreaffinity:failed"
        try:
            info["mask_after"] = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError, ValueError):
            pass


def _plot_load(stop):
    from rm75_control.control.admittance_common.observer_runtime import prepare_observer_process
    prepare_observer_process()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    x = np.linspace(0, 10, 1000)
    lines = [ax.plot(x, np.sin(x))[0] for ax in axes.flat]
    while not stop.is_set():
        t0 = time.monotonic()
        for i, line in enumerate(lines):
            line.set_ydata(np.sin(x + t0 + i))
        fig.canvas.draw()
        stop.wait(max(0.0, .05 - (time.monotonic() - t0)))
    plt.close(fig)


def run_case(
    seed,
    *,
    ticks,
    log_path=None,
    plot=False,
    collision_urdf=None,
    config: Path | None = None,
    control_cpu: int | None = None,
    native_cpu: int | None = None,
):
    import numpy as np
    import yaml
    from rm75_control.control.admittance_common.controller import AdmittanceConfig, AdmittanceController
    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController, _TickLogger
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

    config_path = (Path(config) if config is not None else DEFAULT_CONFIG).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    timing = raw.setdefault("timing", {})
    if control_cpu is not None:
        timing["control_cpu"] = int(control_cpu)
    if native_cpu is not None:
        timing["native_cpu"] = int(native_cpu)
    cfg = build_joint_ik_config(raw)
    cfg.backend = "native"
    cfg.native_shm_prefix = "runtime_profile_" + uuid.uuid4().hex
    if collision_urdf:
        cfg.collision.collision_urdf = Path(collision_urdf).resolve()
    controller = JointIkController(RobotKinematics(), cfg)
    logger = _TickLogger(str(log_path)) if log_path else None
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    worker = ctx.Process(target=_plot_load, args=(stop,)) if plot else None
    if worker:
        worker.start()
    fields = ("qp_kinematics_ms", "qp_collision_ms", "qp_assembly_ms", "qp1_solve_ms",
              "qp2_solve_ms", "qpik_total_ms", "native_dispatch_ms", "native_roundtrip_ms",
              "native_transport_ms")
    timings = {name: [] for name in (*fields, "log_enqueue_ms", "tick_work_ms")}
    statuses = Counter()
    cbf = []
    first_failure = None
    affinity_info = None
    q = np.array(seed)
    previous = np.zeros(8)
    # Include the real force-controller telemetry object so CSV admission and
    # formatting exercise the full schema, including the outer snapshot.
    outer = SimpleNamespace(controller=AdmittanceController(cfg.dt, AdmittanceConfig()),
                            last_pose_d=controller.kin.fk_pose(q))
    try:
        controller.reset(q)
        controller.enable()
        # Warm imports/worker startup before collecting steady-state timing.
        if logger or worker:
            time.sleep(1.0)
        with _bind_control_thread(cfg.control_cpu) as affinity_info:
            t0 = deadline = time.monotonic()
            for index in range(ticks):
                started = time.monotonic()
                step = controller.update(np.zeros(6), q_meas=q,
                                         rail_exec_vel_m_s=float(previous[0]), auto_commit=False)
                statuses[step.fallback_reason or step.qp1_status] += 1
                cbf.append(step.n_cbf_active)
                for key in fields:
                    timings[key].append(float(getattr(step, key)))
                log_t0 = time.monotonic()
                if logger:
                    logger.write(started - t0, "runtime_profile", index * cfg.dt,
                                 step, q, controller.kin.fk_pose(q), np.zeros(6), outer=outer)
                timings["log_enqueue_ms"].append((time.monotonic() - log_t0) * 1000)
                timings["tick_work_ms"].append((time.monotonic() - started) * 1000)
                if step.solver_fault_latched:
                    native = getattr(controller, "_native", None)
                    try:
                        timeout_diagnostics = native.timeout_diagnostics() if native is not None else None
                    except Exception as exc:
                        timeout_diagnostics = f"unavailable:{type(exc).__name__}: {exc}"
                    first_failure = {"tick": index, "q_meas": q.tolist(),
                        "qp1_status": step.qp1_status, "qp2_status": step.qp2_status,
                        "qp1_iterations": int(controller._native._out["qp1_iter"][0]),
                        "qp2_iterations": int(controller._native._out["qp2_iter"][0]),
                        "reason": step.fallback_reason,
                        "hard_residual": float(step.qpik_hard_residual_max),
                        "equality_residual": float(step.qpik_equality_residual_max),
                        "timeout_diagnostics": timeout_diagnostics}
                    for key in ("hard_residual", "equality_residual"):
                        if not np.isfinite(first_failure[key]):
                            first_failure[key] = None
                    break
                controller.commit_publication(step.qdot)
                q, previous = step.q_send.copy(), step.qdot.copy()
                deadline += cfg.dt
                time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        controller.close()
        if logger:
            logger.close()
        stop.set()
        if worker:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
    result = {
        "config": str(config_path),
        "control_cpu": cfg.control_cpu,
        "native_cpu": cfg.native_cpu,
        "control_thread_affinity": affinity_info,
        "ticks": len(cbf),
        "statuses": dict(statuses),
        "max_active_cbf": max(cbf, default=0),
        "log_dropped": logger.dropped if logger else 0,
        "first_failure": first_failure,
        "timing": {},
    }
    for name, values in timings.items():
        finite = np.asarray(values)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            result["timing"][name] = dict(zip(("p50_ms", "p99_ms", "max_ms"),
                                                    np.percentile(finite, [50, 99, 100]).tolist()))
            result["timing"][name]["over_5ms"] = int(np.sum(finite > 5.0))
    return result


def main():
    # This is an offline benchmark entrypoint.  Bound numerical pools before
    # importing NumPy/SciPy/native support, while keeping module imports inert.
    # The benchmark itself retains normal priority; only its CSV/plot child
    # uses prepare_observer_process().
    from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads

    limit_numeric_threads(threads=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--collision-urdf", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"joint-admittance YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--control-cpu", type=int, default=None,
                        help="pin only the benchmark calling thread during ticks")
    parser.add_argument("--native-cpu", type=int, default=None,
                        help="override the native wbc_rt child CPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    import yaml

    with config_path.open() as handle:
        config_raw = yaml.safe_load(handle) or {}
    timing = config_raw.get("timing") or {}
    effective_control_cpu = (
        args.control_cpu if args.control_cpu is not None else timing.get("control_cpu")
    )
    effective_native_cpu = (
        args.native_cpu if args.native_cpu is not None else timing.get("native_cpu")
    )
    print(
        f"[profile] config={config_path} control_cpu={effective_control_cpu} "
        f"native_cpu={effective_native_cpu} ideal_feedback=True",
        flush=True,
    )
    with args.seed_csv.open() as handle:
        row = next(csv.DictReader(handle))
    seed = [float(row[f"q_meas_{i}"]) for i in range(8)]
    result = {
        "seed_csv": str(args.seed_csv),
        "config": str(config_path),
        "control_cpu": effective_control_cpu,
        "native_cpu": effective_native_cpu,
        "ideal_feedback": True,
        "force_telemetry_snapshot": True,
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="controller_runtime_profile_") as directory:
        for name, log, plot in (("control", False, False), ("csv", True, False), ("csv_and_plot", True, True)):
            result["cases"][name] = run_case(seed, ticks=args.ticks,
                log_path=Path(directory) / f"{name}.csv" if log else None, plot=plot,
                collision_urdf=args.collision_urdf, config=config_path,
                control_cpu=args.control_cpu, native_cpu=args.native_cpu)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
