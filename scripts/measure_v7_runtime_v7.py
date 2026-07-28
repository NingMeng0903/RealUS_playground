"""Aggregate isolated cold-start timings for the V7 runtime stages.

Spec section 6 judges performance on isolated cold starts and determinism
*across* processes, so this script never measures anything itself: it launches
``cli.run_v7_isolated_perf`` once per measurement in a fresh interpreter and only
aggregates the JSON those processes print. An earlier revision timed the stages
inline and derived "determinism" from repeated calls inside a single process,
which cannot detect state that survives a process boundary, and asserted
``blender_blocked`` as a literal rather than reading the entry point's own probe.

``apply-pose`` is run three times so the three vertex digests can be compared
bit for bit; the reported cold start includes the SubjectAsset load, which is
what the 1.0 s limit covers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

APPLY_POSE_COLD_LIMIT_S = 1.0
DETERMINISM_LIMIT_M = 1e-6
_REPEATS = 3


def _run_isolated(repo: Path, argv: list[str]) -> dict:
    """Run one stage in its own interpreter and return its JSON object."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "projects.genesis_ue_sync.anatomy_retarget.cli.run_v7_isolated_perf",
            *argv,
        ],
        cwd=str(repo),
        env={
            "PYTHONPATH": "src",
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "OPENBLAS_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-3000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--patella-oracle", required=True)
    parser.add_argument("--betas-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]

    runs = [
        _run_isolated(
            repo,
            ["apply-pose", "--subject", args.subject, "--pose", args.pose],
        )
        for _ in range(_REPEATS)
    ]

    digests = sorted({str(run["vertex_digest"]) for run in runs})
    checksums = sorted({repr(run["vertex_checksum"]) for run in runs})
    deterministic = len(digests) == 1 and len(checksums) == 1
    cold = [float(run["cold_start_seconds"]) for run in runs]

    materialize = _run_isolated(
        repo,
        [
            "materialize-beta",
            "--operator", args.operator,
            "--betas-file", args.betas_file,
            "--patella-oracle", args.patella_oracle,
        ],
    )

    blender_hidden = all(
        run.get("environment", {}).get("bpy_importable") is False for run in runs
    )
    report = {
        "measurement": (
            "isolated cold start: one stage per freshly spawned interpreter, "
            "wall clock via time.perf_counter, no warm reuse and no parallelism; "
            "determinism judged across the separate processes below"
        ),
        "apply_pose": {
            "cold_start_seconds_runs": cold,
            "cold_start_seconds_max": max(cold),
            "limit_seconds": APPLY_POSE_COLD_LIMIT_S,
            "pass": max(cold) <= APPLY_POSE_COLD_LIMIT_S,
            "includes_asset_load": True,
            "load_seconds_max": max(float(r["load_seconds"]) for r in runs),
            "solve_seconds_max": max(float(r["solve_seconds"]) for r in runs),
            "vertex_count": int(runs[0]["vertex_count"]),
        },
        "materialize_beta": {
            "cold_start_seconds": float(materialize["cold_start_seconds"]),
            "note": "isolated single-process value; reported, not limited",
        },
        "determinism": {
            "scope": f"{_REPEATS} independent cold-start processes",
            "vertex_digest": digests[0] if deterministic else digests,
            "vertex_checksum": runs[0]["vertex_checksum"],
            "max_vertex_delta_m": 0.0 if deterministic else None,
            "limit_m": DETERMINISM_LIMIT_M,
            "pass": deterministic,
        },
        "blender_blocked": blender_hidden,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_reports": {"apply_pose_runs": runs, "materialize_beta": materialize},
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
