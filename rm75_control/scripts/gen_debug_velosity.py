#!/usr/bin/env python3
"""Regenerate MD/debug_velosity.md — verbatim velocity-admittance source mirror."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "MD" / "debug_velosity.md"

SECTIONS: list[tuple[str, Path]] = [
    # --- velocity CANFD I/O ---
    ("rm75_control/motion/canfd.py", REPO / "rm75_control/motion/canfd.py"),
    # --- core stack (hybrid_motion = velocity admittance implementation) ---
    ("rm75_control/control/hybrid_motion/__init__.py", REPO / "rm75_control/control/hybrid_motion/__init__.py"),
    ("rm75_control/control/hybrid_motion/paths.py", REPO / "rm75_control/control/hybrid_motion/paths.py"),
    ("rm75_control/control/hybrid_motion/async_state.py", REPO / "rm75_control/control/hybrid_motion/async_state.py"),
    ("rm75_control/control/hybrid_motion/reference.py", REPO / "rm75_control/control/hybrid_motion/reference.py"),
    ("rm75_control/control/hybrid_motion/reference_shaper.py", REPO / "rm75_control/control/hybrid_motion/reference_shaper.py"),
    ("rm75_control/control/hybrid_motion/adaptive_ke.py", REPO / "rm75_control/control/hybrid_motion/adaptive_ke.py"),
    ("rm75_control/control/hybrid_motion/controller.py", REPO / "rm75_control/control/hybrid_motion/controller.py"),
    ("rm75_control/control/hybrid_motion/observer.py", REPO / "rm75_control/control/hybrid_motion/observer.py"),
    ("rm75_control/control/hybrid_motion/scan_log.py", REPO / "rm75_control/control/hybrid_motion/scan_log.py"),
    ("rm75_control/control/hybrid_motion/rm_algo.py", REPO / "rm75_control/control/hybrid_motion/rm_algo.py"),
    ("rm75_control/control/hybrid_motion/loop.py", REPO / "rm75_control/control/hybrid_motion/loop.py"),
    # --- deprecated re-export shims (import path compatibility) ---
    ("rm75_control/control/velocity_admittance/__init__.py", REPO / "rm75_control/control/velocity_admittance/__init__.py"),
    ("rm75_control/control/velocity_admittance/async_state.py", REPO / "rm75_control/control/velocity_admittance/async_state.py"),
    ("rm75_control/control/velocity_admittance/controller.py", REPO / "rm75_control/control/velocity_admittance/controller.py"),
    ("rm75_control/control/velocity_admittance/loop.py", REPO / "rm75_control/control/velocity_admittance/loop.py"),
    ("rm75_control/control/velocity_admittance/observer.py", REPO / "rm75_control/control/velocity_admittance/observer.py"),
    ("rm75_control/control/velocity_admittance/paths.py", REPO / "rm75_control/control/velocity_admittance/paths.py"),
    ("rm75_control/control/velocity_admittance/rm_algo.py", REPO / "rm75_control/control/velocity_admittance/rm_algo.py"),
    ("rm75_control/control/velocity_admittance/scan_log.py", REPO / "rm75_control/control/velocity_admittance/scan_log.py"),
    # --- force compensation (φ regression + slot moves, used by loop/observer) ---
    ("rm75_control/force/compensation/paths.py", REPO / "rm75_control/force/compensation/paths.py"),
    ("rm75_control/force/compensation/id_config.py", REPO / "rm75_control/force/compensation/id_config.py"),
    ("rm75_control/force/compensation/regressor.py", REPO / "rm75_control/force/compensation/regressor.py"),
    ("rm75_control/force/compensation/collection.py", REPO / "rm75_control/force/compensation/collection.py"),
    ("rm75_control/force/compensation/tool_pose.py", REPO / "rm75_control/force/compensation/tool_pose.py"),
    ("rm75_control/force/compensation/excitation.py", REPO / "rm75_control/force/compensation/excitation.py"),
    # --- demo entry points + trajectory plugins ---
    ("tmp/Velocity_Admittance/run_admittance.py", REPO / "tmp/Velocity_Admittance/run_admittance.py"),
    ("tmp/Velocity_Admittance/demo_stack.py", REPO / "tmp/Velocity_Admittance/demo_stack.py"),
    ("tmp/Velocity_Admittance/demo/trajectory_builtin.py", REPO / "tmp/Velocity_Admittance/demo/trajectory_builtin.py"),
    ("tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py", REPO / "tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py"),
    ("tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py", REPO / "tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py"),
    ("tmp/Velocity_Admittance/plot_scan_log.py", REPO / "tmp/Velocity_Admittance/plot_scan_log.py"),
    # --- YAML configs ---
    ("tmp/Velocity_Admittance/config/admittance.yaml", REPO / "tmp/Velocity_Admittance/config/admittance.yaml"),
    ("tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml", REPO / "tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml"),
    ("tmp/Velocity_Admittance/demo/config/d_to_a_sin_tool_y.yaml", REPO / "tmp/Velocity_Admittance/demo/config/d_to_a_sin_tool_y.yaml"),
    ("tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml", REPO / "tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml"),
    ("tmp/Velocity_Admittance/demo/config/adaptive_critical_damping.yaml", REPO / "tmp/Velocity_Admittance/demo/config/adaptive_critical_damping.yaml"),
    ("configs/force_compensation/poses.yaml", REPO / "configs/force_compensation/poses.yaml"),
    ("configs/force_compensation/force_id.yaml", REPO / "configs/force_compensation/force_id.yaml"),
]


def lang(path: Path) -> str:
    if path.suffix in (".yaml", ".yml"):
        return "yaml"
    if path.suffix == ".sh":
        return "bash"
    return "python"


def embed(rel: str, path: Path) -> str:
    body = path.read_text(encoding="utf-8")
    return f"## FILE: `{rel}`\n\n```{lang(path)}\n{body}```\n\n"


def main() -> None:
    missing = [rel for rel, p in SECTIONS if not p.is_file()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))

    header = (
        "# Velocity Admittance Controller — Full Source Dump\n\n"
        f"Generated from workspace `{REPO}`. {len(SECTIONS)} files, verbatim.\n\n"
        "Regenerate: `python scripts/gen_debug_velosity.py`\n\n"
        "## Implementation Overview\n\n"
        "Velocity Admittance stack: **Cartesian velocity commands** (`rm_movev_canfd`), "
        "outer-loop PBAC + force-admittance sleeve fusion, no inner-loop WBC/QP.\n\n"
        "| Module | File | Role |\n"
        "|---|---|---|\n"
        "| Main loop | `hybrid_motion/loop.py` | 10ms CANFD cycle: read state -> phi-compensated force -> outer-loop v_cmd -> movev |\n"
        "| Outer controller | `hybrid_motion/controller.py` | PBAC pose tracking + tool-Z 2nd-order admittance + Dimeas variable damping + sleeve fusion |\n"
        "| Adaptive damping | `hybrid_motion/adaptive_ke.py` | delta-F/delta-x EWMA -> K_hat_e -> b_d = 2*zeta*sqrt(m*K_hat_e) |\n"
        "| Force observer | `hybrid_motion/observer.py` | phi regression compensation + causal LPF -> f_ext |\n"
        "| Trajectory ref | `hybrid_motion/reference.py` + `reference_shaper.py` | MotionReference source + low-pass/shaping |\n"
        "| Vendor trajectory | `hybrid_motion/rm_algo.py` | rm_algo_pose_move wrapper (optional in demos) |\n"
        "| Scan log | `hybrid_motion/scan_log.py` | pose_d vs pose_act NPZ logging and jerk stats |\n"
        "| CANFD I/O | `motion/canfd.py` | movev init/velocity send/quiescence handoff |\n"
        "| Demo entry | `tmp/Velocity_Admittance/run_admittance.py` | Generic YAML-driven entry point |\n"
        "| Phase demo | `demo/d_to_a_sin_tool_y.py` | MoveJ->D + sin_tool_y scan |\n\n"
        "**Difference from joint_admittance (WBC position loop):**\n"
        "- velocity: outer loop outputs **6D Cartesian velocity** -> `rm_movev_canfd`\n"
        "- joint_admittance: outer loop outputs twist -> WBC QP -> **joint angles** -> `rm_movej_canfd`\n\n"
        "The `velocity_admittance/` package is only a deprecated re-export; the real "
        "implementation lives in `hybrid_motion/`.\n\n"
        "---\n\n"
    )
    parts = [header]
    for rel, path in SECTIONS:
        parts.append(embed(rel, path))
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(SECTIONS)} files)")


if __name__ == "__main__":
    main()
