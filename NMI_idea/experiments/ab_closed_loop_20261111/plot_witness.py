#!/usr/bin/env python3
"""Plot one audited same-first witness from closed-loop result files.

The script only draws trajectories whose route IDs resolve through the sibling
``execution/MOTION.json`` and whose referenced ``.npz`` files exist.  If no
qualified witness is present, the plot contains the aggregate bars and an
explicit no-witness annotation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent


def _result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("RESULT.json")) if root.is_dir() else ([root] if root.is_file() else [])


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _motion_manifest(result_path: Path) -> Path | None:
    candidates = (
        result_path.parent / "execution" / "MOTION.json",
        result_path.parent / "MOTION.json",
        result_path.parent.parent / "execution" / "MOTION.json",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _route_arrays(result_path: Path, route_id: str) -> tuple[list[np.ndarray], list[str], str | None]:
    manifest = _motion_manifest(result_path)
    if manifest is None:
        return [], [], None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return [], [], str(manifest)
    matches = [route for route in payload.get("routes", [])
               if isinstance(route, dict) and str(route.get("route", "")) == route_id]
    arrays: list[np.ndarray] = []
    paths: list[str] = []
    for route in matches:
        steps = route.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("record"), str):
                continue
            candidate = (manifest.parent / f"{step['record']}.npz").resolve()
            if not candidate.is_file():
                continue
            try:
                with np.load(candidate) as archive:
                    if "local_xyz_m" not in archive:
                        continue
                    points = np.asarray(archive["local_xyz_m"], dtype=float)
            except Exception:
                continue
            if points.ndim == 2 and points.shape[1] >= 2 and len(points):
                arrays.append(points[:, :2])
                paths.append(str(candidate))
    return arrays, paths, str(manifest)


def _event_route_ids(row: dict[str, Any]) -> list[str]:
    events = row.get("events", [])
    if not isinstance(events, list):
        return []
    return [str(event["route_id"]) for event in events
            if isinstance(event, dict) and isinstance(event.get("route_id"), str) and event["route_id"]]


def _terminal_event(row: dict[str, Any]) -> dict[str, Any] | None:
    events = row.get("events", [])
    if not isinstance(events, list):
        return None
    terminal = [event for event in events if isinstance(event, dict) and event.get("kind") == "terminal"]
    if terminal:
        return terminal[-1]
    return events[-1] if events and isinstance(events[-1], dict) else None


def _find_witness(results: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any] | None:
    # Group the two arms together.  The arm is deliberately *not* part of the
    # key: a same-first witness is a paired comparison, not two independent
    # records.  The fixture name stays in the key so a case ID reused by a
    # mirrored fixture cannot be paired across fixtures.
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, payload in results:
        for row in payload.get("records", []) if isinstance(payload.get("records", []), list) else []:
            if not isinstance(row, dict) or row.get("variant") != "witness":
                continue
            arm = str(row.get("arm", ""))
            if arm not in {"same_fork_minus_I0", "same_fork_minus_I1"}:
                continue
            case_id = row.get("case_id")
            fixture = str(payload.get("fixture", {}).get("name", path.parent.name))
            grouped[(str(case_id), fixture)].append((path, row))
    candidates = []
    for (case_id, fixture), _rows in grouped.items():
        zero = next((item for item in _rows if item[1].get("arm") == "same_fork_minus_I0"), None)
        one = next((item for item in _rows if item[1].get("arm") == "same_fork_minus_I1"), None)
        if zero is None or one is None:
            continue
        path0, row0 = zero
        path1, row1 = one
        if bool(row0.get("success")) == bool(row1.get("success")):
            continue
        if row0.get("same_first_exit") != row1.get("same_first_exit"):
            continue
        routes0, routes1 = _event_route_ids(row0), _event_route_ids(row1)
        if not routes0 or not routes1 or routes0[0] != routes1[0]:
            continue
        if routes0[1:] == routes1[1:]:
            continue
        prefix_arrays, prefix_paths, manifest = _route_arrays(path0, routes0[0])
        branch0_arrays, branch0_paths, _ = _route_arrays(path0, routes0[-1])
        branch1_arrays, branch1_paths, _ = _route_arrays(path1, routes1[-1])
        if not prefix_arrays or not branch0_arrays or not branch1_arrays:
            continue
        candidates.append({
            "case_id": str(case_id), "fixture": fixture, "result0": str(path0), "result1": str(path1),
            "row0": row0, "row1": row1, "prefix_route": routes0[0], "terminal_route_i0": routes0[-1],
            "terminal_route_i1": routes1[-1], "prefix_arrays": prefix_arrays, "branch0_arrays": branch0_arrays,
            "branch1_arrays": branch1_arrays, "trace_paths": sorted(set(prefix_paths + branch0_paths + branch1_paths)),
            "motion_manifest": manifest,
        })
    def natural_case_id(value: str) -> tuple[int, Any]:
        # RESULT rows preserve case IDs as strings for JSON portability, but
        # deterministic witness selection should use numeric order when the
        # runner supplied integer IDs (case 5 before case 100).
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))
    return (sorted(candidates, key=lambda item: (item["fixture"], natural_case_id(item["case_id"])))[0]
            if candidates else None)


def _join(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0, 2))
    result = arrays[0]
    for array in arrays[1:]:
        result = np.vstack((result, array[1:] if len(result) and len(array) else array))
    return result


def _bar_records(results: list[tuple[Path, dict[str, Any]]]) -> tuple[list[str], list[float], list[int], list[int]]:
    arms = ("B_I1_F1", "A_greedy", "pair_fixed", "I1_F0")
    labels, rates, counts, totals = [], [], [], []
    for variant in ("primary", "witness"):
        for arm in arms:
            values = [bool(row.get("success")) for _, payload in results
                      for row in payload.get("records", []) if isinstance(row, dict)
                      and row.get("variant") == variant and row.get("arm") == arm]
            if values:
                labels.append(f"{variant}\n{arm}")
                rates.append(float(np.mean(values)))
                counts.append(int(np.sum(values)))
                totals.append(len(values))
    return labels, rates, counts, totals


def _outcome_text(row: dict[str, Any]) -> str:
    event = _terminal_event(row) or {}
    time_s = event.get("elapsed_s", row.get("first_success_s", row.get("restricted_time_s", "?")))
    return f"{'success' if row.get('success') else 'fail'} @ {time_s}s"


def plot(root: Path, output: Path) -> dict[str, Any]:
    files = _result_files(root)
    loaded = [(path, payload) for path in files if (payload := _load_result(path)) is not None]
    witness = _find_witness(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    ax = axes[0]
    audit: dict[str, Any] = {"schema": "ab_closed_loop_witness_plot_v1", "result_files": [str(path) for path, _ in loaded],
                             "witness": None, "output": str(output)}
    if witness is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No qualified witness\n(same_fork_minus_I1 succeeds and I0 fails)\nwith resolved actual traces",
                ha="center", va="center", fontsize=12)
    else:
        row0, row1 = witness["row0"], witness["row1"]
        prefix = _join(witness["prefix_arrays"])
        branch0 = _join(witness["branch0_arrays"])
        branch1 = _join(witness["branch1_arrays"])
        fixture = next(payload.get("fixture", {}) for path, payload in loaded if str(path) == witness["result0"])
        wall = fixture.get("wall_tcp_envelope_m")
        if isinstance(wall, (list, tuple)) and len(wall) == 4:
            x0, x1, y0, y1 = map(float, wall)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.75", edgecolor="0.25", alpha=0.65, label="public baffle"))
        ax.plot(prefix[:, 0], prefix[:, 1], color="black", linewidth=3.0, linestyle="--", label="common executed prefix")
        ax.plot(branch1[:, 0], branch1[:, 1], color="#1565c0", linewidth=2.1, label=f"I1 terminal {_outcome_text(row1)}")
        ax.plot(branch0[:, 0], branch0[:, 1], color="#c62828", linewidth=2.1, label=f"I0 terminal {_outcome_text(row0)}")
        ax.scatter(prefix[0, 0], prefix[0, 1], color="black", s=28, zorder=5)
        ax.set_title(f"Constructed public-geometry witness, case {witness['case_id']}")
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("local y (m)")
        ax.grid(alpha=0.25)
        ax.axis("equal")
        ax.legend(fontsize=8, loc="best")
        audit["witness"] = {
            "case_id": witness["case_id"], "fixture": witness["fixture"],
            "result_i0": witness["result0"], "result_i1": witness["result1"],
            "same_first_exit": row0.get("same_first_exit"), "prefix_route": witness["prefix_route"],
            "terminal_route_i0": witness["terminal_route_i0"], "terminal_route_i1": witness["terminal_route_i1"],
            "motion_manifest": witness["motion_manifest"], "trace_paths": witness["trace_paths"],
            "outcomes": {"I0": _outcome_text(row0), "I1": _outcome_text(row1)},
            "deadline_s": row0.get("deadline_s"),
        }
    labels, rates, counts, totals = _bar_records(loaded)
    bar_ax = axes[1]
    if labels:
        x = np.arange(len(labels))
        bars = bar_ax.bar(x, rates, color=["#1565c0" if "B_I1_F1" in label else "#757575" for label in labels])
        bar_ax.set_xticks(x, labels, rotation=35, ha="right")
        bar_ax.set_ylim(0.0, 1.0)
        bar_ax.set_ylabel("timely first accepted acquisition rate")
        bar_ax.set_title("Audited result counts")
        for rect, rate, count, total in zip(bars, rates, counts, totals):
            bar_ax.text(rect.get_x() + rect.get_width() / 2.0, min(0.98, rate + 0.03),
                        f"{count}/{total}\n{rate:.2f}", ha="center", va="bottom", fontsize=8)
        bar_ax.grid(axis="y", alpha=0.25)
    else:
        bar_ax.axis("off")
        bar_ax.text(0.5, 0.5, "No primary/witness bar records", ha="center", va="center")
    figure.suptitle("Closed-loop B witness audit")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    output.with_name("WITNESS_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=HERE / "closed_loop_reviewed_run")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.run_root / "WITNESS.png")
    audit = plot(args.run_root, output)
    print(json.dumps({"output": str(output), "result_files": len(audit["result_files"]), "witness": audit["witness"] is not None}))


if __name__ == "__main__":
    main()
