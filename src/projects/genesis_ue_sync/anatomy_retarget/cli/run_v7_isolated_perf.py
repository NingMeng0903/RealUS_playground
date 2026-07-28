"""Measure one V7 stage in an isolated cold-started process.

Timing V7 from inside a warm session is not evidence: the operator, the subject
and every numpy kernel are already resident, and a pose cache or an importable
Blender would change the answer again.  This entry point therefore does exactly
one stage per process, refuses to run when Blender or a blend file is reachable,
and prints a single JSON object so the caller can aggregate several cold starts
without ever holding two of them in one interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _reject_blender_visibility() -> dict[str, Any]:
    if importlib.util.find_spec("bpy") is not None:
        raise ValueError("bpy is importable; isolated V7 timing must hide Blender")
    if "bpy" in sys.modules:
        raise ValueError("bpy is already imported; isolated V7 timing must hide Blender")
    return {"bpy_importable": False}


def _load_pose(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.zeros((55, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
    with np.load(path) as data:
        pose = np.asarray(data["pose_axis_angle"], dtype=np.float32).reshape(55, 3)
        transl = (
            np.asarray(data["transl"], dtype=np.float32).reshape(3)
            if "transl" in data.files
            else np.zeros(3, dtype=np.float32)
        )
    return pose, transl


def _time_apply_pose(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
        apply_subject_pose,
        load_subject_asset,
    )

    import_seconds = time.perf_counter() - started
    load_started = time.perf_counter()
    subject = load_subject_asset(args.subject)
    load_seconds = time.perf_counter() - load_started
    pose, transl = _load_pose(args.pose)
    solve_started = time.perf_counter()
    vertices = apply_subject_pose(
        subject, pose_axis_angle=pose, transl=transl, validate=False
    )
    solve_seconds = time.perf_counter() - solve_started
    return {
        "stage": "apply-pose",
        "import_seconds": float(import_seconds),
        "load_seconds": float(load_seconds),
        "solve_seconds": float(solve_seconds),
        "cold_start_seconds": float(load_seconds + solve_seconds),
        "wall_clock_seconds": float(time.perf_counter() - started),
        "vertex_count": int(len(vertices)),
        "vertex_checksum": float(np.sum(np.asarray(vertices, dtype=np.float64))),
        # A float sum hides compensating deltas, so cross-process determinism is
        # judged on the byte digest instead.
        "vertex_digest": hashlib.sha256(
            np.ascontiguousarray(vertices, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def _time_materialize_beta(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
        load_patella_oracle_v7,
    )
    from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
        load_source_operator,
        materialize_subject,
    )

    import_seconds = time.perf_counter() - started
    load_started = time.perf_counter()
    operator = load_source_operator(args.operator)
    load_seconds = time.perf_counter() - load_started
    with np.load(args.betas_file) as data:
        key = "shapes" if "shapes" in data.files else "betas"
        betas = np.asarray(data[key], dtype=np.float64).reshape(-1)[:10]
    law = load_patella_oracle_v7(args.patella_oracle)
    solve_started = time.perf_counter()
    subject = materialize_subject(
        operator, betas=betas, gender=str(args.gender), patella_law=law
    )
    solve_seconds = time.perf_counter() - solve_started
    return {
        "stage": "materialize-beta",
        "import_seconds": float(import_seconds),
        "load_seconds": float(load_seconds),
        "solve_seconds": float(solve_seconds),
        "cold_start_seconds": float(load_seconds + solve_seconds),
        "wall_clock_seconds": float(time.perf_counter() - started),
        "subject_digest": str(subject.content_digest()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("apply-pose", "materialize-beta"))
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--betas-file", type=Path)
    parser.add_argument("--patella-oracle", type=Path)
    parser.add_argument("--gender", default="male")
    args = parser.parse_args(argv)
    environment = _reject_blender_visibility()
    if args.stage == "apply-pose":
        if args.subject is None:
            raise ValueError("apply-pose needs --subject")
        result = _time_apply_pose(args)
    else:
        if args.operator is None or args.betas_file is None:
            raise ValueError("materialize-beta needs --operator and --betas-file")
        result = _time_materialize_beta(args)
    result["environment"] = environment
    result["measurement"] = "isolated cold start, single process, no pose cache"
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
