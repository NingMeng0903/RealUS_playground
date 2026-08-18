#!/usr/bin/env python3
"""cProfile the QPIK assembly path via strict replay or a synthetic tick train.

CBF ``build_cbf_rows`` now uses the sphere broadphase (activation+hysteresis
band) instead of a full 28-pair narrow-phase.  Offline, collision-off
assembly is ~0.45 ms; collision-on is ~2.4 ms (was ~3.8 ms on HW).  The
leftover is ``CollisionModel.update`` / HPP-FCL, not the ProxQP solve
(~0.1 ms).  Do not raise ``grad_period_ticks`` to spend that budget.

    source env.sh
    python apps/joint_admittance_8dof/profile_qpik_assembly.py \\
        apps/logs/ellipse_track/run_YYYYMMDD_HHMMSS.csv \\
        --max-rows 400

    python apps/joint_admittance_8dof/profile_qpik_assembly.py --synthetic 400
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np


def _profile_synthetic(n_ticks: int, *, collision: bool = False) -> None:
    from rm75_control.control.joint_admittance_8dof.collision_model import (
        CollisionConfig,
    )
    from rm75_control.control.joint_admittance_8dof.model import (
        RobotKinematics,
        full_q_from_arm,
    )
    from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
        QpConfig,
        QpIkController,
    )
    from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(
        kin, v_scale=0.8, a_max=np.concatenate(([0.6], np.full(7, 3.0)))
    )
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            backend="proxqp",
            collision=CollisionConfig(enabled=bool(collision)),
            use_cpp_kernel=True,
        ),
    )
    ctrl.reset()
    q = full_q_from_arm(np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40)
    twist = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    for i in range(int(n_ticks)):
        phase = 2.0 * np.pi * (i / max(n_ticks, 1))
        twist[1] = 0.03 * np.cos(phase)
        r = ctrl.step(q, twist, 0.005, q_meas=q)
        q = q + np.asarray(r.qdot, dtype=float) * 0.005


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="Profile N offline QP ticks instead of replaying a CSV",
    )
    parser.add_argument(
        "--collision",
        action="store_true",
        help="Enable CBF collision assembly in --synthetic (matches HW cost).",
    )
    args = parser.parse_args()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        if args.synthetic > 0 or args.csv is None:
            n = args.synthetic if args.synthetic > 0 else 200
            _profile_synthetic(n, collision=bool(args.collision))
        else:
            from replay_strict_qpik import main as replay_main

            argv = [
                str(args.csv),
                "--mode",
                "free-running",
                "--max-rows",
                str(args.max_rows),
            ]
            if args.config is not None:
                argv.extend(["--config", str(args.config)])
            replay_main(argv)
    finally:
        profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print("\n=== cProfile (top 40 by cumulative time) ===")
    stats.print_stats(40)
    stats.sort_stats("tottime")
    print("\n=== cProfile (top 20 by tottime) ===")
    stats.print_stats(20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
