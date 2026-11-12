"""Truth-only headroom diagnostic for the frozen moving-Y study.

This script never changes the pinned checkout and never feeds evaluator labels
to a planner.  It reuses the frozen acoustic world and archived preflight graph
to replay every legal first/tail pair from each observed common-probe state.
The resulting ceilings answer a narrow question: could the existing action
library have produced more success if an oracle selected among actions after
seeing their actual outcomes?

The oracle is diagnostic only.  It is not a policy result, because it uses the
terminal evaluator label after enumerating all candidate outcomes.  The
ordinary online planner is read from the archived run and checked against the
same replay.
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = HERE / "RealUS_playground" / "NMI_idea" / "experiments" / "ab_tro_y_continuation_20261112"
HOLDOUT = STUDY / "holdout"


def load_xz(name: str):
    with lzma.open(HOLDOUT / f"{name}.jsonl.xz", "rt") as stream:
        return [json.loads(line) for line in stream]


def import_frozen_executor():
    """Import run_study without optional PyTorch learner dependencies."""

    # closed_loop imports offline_policies at module import time.  This
    # diagnostic only uses MapWorld, MotionModel and the executor helpers, so
    # a small placeholder avoids changing or installing anything in the
    # reference checkout.
    if "learners" not in sys.modules:
        sys.modules["learners"] = types.ModuleType("learners")
    sys.path.insert(0, str(STUDY))
    import run_study as source_executor  # noqa: E402

    # The source commit deliberately records two large generated assets on
    # the mounted archive drive.  Their absolute paths are part of the frozen
    # manifest, while the working clone contains the portable source only.
    # Point the imported modules at those manifest-checked assets without
    # copying or modifying the reference checkout.
    archive = Path("/media/camp/EXT_DRIVE/RealUS_playground")
    import dynamic_scan  # noqa: E402
    import knee_world  # noqa: E402

    dynamic_scan.PREVIOUS = archive / "NMI_idea" / "experiments" / "ab_leg_bifurcation_20261112"
    geometry = archive / "NMI_idea" / "experiments" / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
    class NoTorchEdges:
        """Edge gate used only to construct the acoustic world.

        The legal action graph is loaded from the frozen manifest.  No path
        is certified here, and no native execution result is used.
        """

        robot = None

        def __init__(self, scan):
            self.q = scan.q

        def allow(self, start, goal):
            return True

    knee_world.Edges = NoTorchEdges
    import certification  # noqa: E402
    certification.Edges = NoTorchEdges

    from closed_loop import prepare  # noqa: E402
    from map_model import MapWorld  # noqa: E402

    world = MapWorld(geometry=geometry, mode="stress")
    route = tuple((int(frame["site"]), int(frame["view"])) for frame in load_xz("COMMON")[0]["prefix_frames"])
    world.initial_route = route
    world.nominal_fork_arrival_s, world.nominal_route_end_s = world._fixed_route_times()

    @dataclass(frozen=True)
    class FrozenPath:
        start: tuple[int, int]
        goal: tuple[int, int]
        duration: float
        stroke_mm: float
        legal: bool
        reason: str
        cost_mm: float

        @property
        def id(self):
            return f"{self.start[0]}:{self.start[1]}>{self.goal[0]}:{self.goal[1]}"

    manifest = json.loads((STUDY / "FROZEN_MANIFEST.json").read_text())
    preflight = manifest["mechanic_preflight"]
    first_paths = tuple(
        FrozenPath(tuple(path["start"]), tuple(path["goal"]), float(path["duration"]),
                   float(path["stroke_mm"]), bool(path["legal"]), str(path["reason"]),
                   float(path["cost_mm"]))
        for path in preflight["first_paths"]
    )
    tail_paths = {
        int(view): tuple(
            FrozenPath(tuple(path["start"]), tuple(path["goal"]), float(path["duration"]),
                       float(path["stroke_mm"]), bool(path["legal"]), str(path["reason"]),
                       float(path["cost_mm"]))
            for path in paths
        )
        for view, paths in preflight["tail_paths"].items()
    }

    def legal_graph(reserve):
        first, tails = [], {}
        for path in first_paths:
            if path.legal and path.stroke_mm <= reserve + 1.e-9:
                eligible = tuple(
                    tail for tail in tail_paths[path.goal[1]]
                    if tail.legal and tail.stroke_mm + path.stroke_mm <= reserve + 1.e-9
                )
                if eligible:
                    first.append(path)
                    tails[path.goal[1]] = eligible
        return tuple(first), tails

    # Keep the source executor's public evaluator functions, but avoid its
    # MotionModel and PyTorch-dependent capability setup.
    source_executor.RESERVES_MM = {"narrow": 10.0, "wide": 26.0}
    source_executor.WORLD = world
    return types.SimpleNamespace(
        WORLD=world,
        RESERVES_MM=source_executor.RESERVES_MM,
        POST_CAP_MM=60.0,
        POST_CAP_DEG=60.0,
        FIRST_DURATION_S=0.65,
        TAIL_DURATION_S=0.85,
        TIME_TOL=1.e-9,
        prepare=prepare,
        update=lambda state, frame: state.updated(world, frame, retain=world.identity_site(frame.site)),
        target_claim=source_executor.target_claim,
        evaluate_claim=source_executor.evaluate_claim,
        choose_real_tail=source_executor.choose_real_tail,
        prefix_hash=source_executor.prefix_hash,
        legal_graph=legal_graph,
        probe_duration=2.50,
        first_paths=first_paths,
        tail_paths=tail_paths,
    )


def public_frame(frame):
    return dict(
        site=frame.site,
        view=frame.view,
        t=frame.t,
        q=frame.q,
        velocity=frame.velocity,
        centroids_mm=frame.centroids.tolist(),
    )


def replay_case(executor, common):
    """Replay all legal candidate pairs for one common observed prefix."""

    seed = int(common["seed"])
    case = executor.WORLD.case(seed)
    prefix = executor.prepare(executor.WORLD, case)
    probe_t = float(common["probe_motion"]["t_end"])
    probe_frame = executor.WORLD.observe(case, probe_t, 3, 3)
    expected_probe = common["common_probe_frame"]
    observed_probe = public_frame(probe_frame)
    if observed_probe["site"] != expected_probe["site"] or observed_probe["view"] != expected_probe["view"]:
        raise AssertionError("replayed common probe goal differs from archive")
    if observed_probe["t"] != expected_probe["t"] or observed_probe["centroids_mm"] != expected_probe["centroids_mm"]:
        raise AssertionError("replayed common probe is not byte-identical to archive")

    start_state = executor.update(prefix, probe_frame)
    rows = []
    for reserve_name in executor.RESERVES_MM:
        first_paths, tails_by_view = executor.legal_graph(executor.RESERVES_MM[reserve_name])
        for first_path in first_paths:
            first_t = probe_t + float(first_path.duration)
            first_frame = executor.WORLD.observe(case, first_t, *first_path.goal)
            state_after_first = None
            if first_frame is not None:
                state_after_first = executor.update(start_state, first_frame)
            tails = tails_by_view[first_path.goal[1]]
            for tail_path in tails:
                row = {
                    "reserve": reserve_name,
                    "first_view": int(first_path.goal[1]),
                    "tail_view": int(tail_path.goal[1]),
                    "first_path": first_path.id,
                    "tail_path": tail_path.id,
                    "first_capture_t": None if first_frame is None else float(first_frame.t),
                    "terminal_capture_t": None,
                    "first_visible": bool(first_frame is not None and 1 in first_frame.labels),
                    "terminal_visible": False,
                    "legal_completed": False,
                    "correct_fresh": False,
                    "strict": False,
                    "false_confirm": False,
                    "label": None,
                    "confidence": 0.0,
                    "failure": "first_capture_failed" if first_frame is None else None,
                }
                if state_after_first is None:
                    rows.append(row)
                    continue
                terminal_t = first_t + float(tail_path.duration)
                terminal_frame = executor.WORLD.observe(case, terminal_t, *tail_path.goal)
                row["terminal_capture_t"] = float(terminal_frame.t)
                row["terminal_visible"] = bool(terminal_frame is not None and 1 in terminal_frame.labels)
                if terminal_frame is None:
                    row["failure"] = "terminal_capture_failed"
                    rows.append(row)
                    continue
                state = executor.update(state_after_first, terminal_frame)
                claim = executor.target_claim(state, 30)
                # The archived preflight marks every candidate retained by the
                # graph as legal.  This acoustic replay therefore uses the
                # graph's path legality and stroke/time gates.  It does not
                # reconstruct FK travel/turn costs, which are reported as an
                # explicit limitation below.
                travel = float(first_path.cost_mm + tail_path.cost_mm)
                turn = None
                duration = float(first_path.duration + tail_path.duration)
                stroke = float(first_path.stroke_mm + tail_path.stroke_mm)
                completed = bool(
                    terminal_frame is not None
                    and stroke <= executor.RESERVES_MM[reserve_name] + executor.TIME_TOL
                    and duration <= executor.FIRST_DURATION_S + executor.TAIL_DURATION_S + executor.TIME_TOL
                    and first_path.legal and tail_path.legal
                    and travel <= executor.POST_CAP_MM + executor.TIME_TOL
                )
                outcome = executor.evaluate_claim(claim, terminal_frame, completed)
                row.update(
                    legal_completed=bool(completed),
                    correct_fresh=bool(outcome["correct_fresh"]),
                    strict=bool(outcome["strict"]),
                    false_confirm=bool(outcome["false_confirm"]),
                    label=outcome["label_for_evaluation"],
                    confidence=float(claim["confidence"]),
                    failure=None if completed else "path_or_shared_cap_did_not_complete",
                    travel_mm=travel,
                    turn_deg=turn,
                    duration_s=duration,
                    stroke_mm=stroke,
                    claim_confirmed=bool(claim["confirmed"]),
                )
                rows.append(row)
    return rows


def bool_sum(rows, key):
    return int(sum(bool(row[key]) for row in rows))


def summarize_case(case_rows, archived_rows):
    result = {}
    for reserve_name in ("narrow", "wide"):
        rows = [row for row in case_rows if row["reserve"] == reserve_name]
        archived = archived_rows[reserve_name]
        by_pair = {(row["first_view"], row["tail_view"]): row for row in rows}
        baseline_key = (int(archived["first_view"]), int(archived["actual_tail_view"]))
        if baseline_key not in by_pair:
            raise AssertionError(f"archived planner pair missing from legal graph: {baseline_key}")
        baseline = by_pair[baseline_key]
        archived_stages = archived.get("stages", [])
        if len(archived_stages) == 2:
            first_frame = archived_stages[0].get("frame")
            terminal_frame = archived_stages[1].get("frame")
            if first_frame is not None and baseline["first_capture_t"] is not None:
                if abs(float(first_frame["t"]) - baseline["first_capture_t"]) > 1.e-10:
                    raise AssertionError("first capture time differs from archive")
            if terminal_frame is not None and baseline["terminal_capture_t"] is not None:
                if abs(float(terminal_frame["t"]) - baseline["terminal_capture_t"]) > 1.e-10:
                    raise AssertionError("terminal capture time differs from archive")
        for key in ("strict", "correct_fresh", "false_confirm", "terminal_visible"):
            archived_key = "target_present" if key == "terminal_visible" else key
            expected = bool(archived["strict"] if key == "strict" else archived[archived_key])
            if bool(baseline[key]) != expected:
                raise AssertionError(
                    f"replay mismatch case={archived['index']} reserve={reserve_name} key={key}"
                )
        result[reserve_name] = {
            "candidate_count": len(rows),
            "baseline_pair": [int(baseline_key[0]), int(baseline_key[1])],
            "baseline_strict": bool(baseline["strict"]),
            "baseline_correct_fresh": bool(baseline["correct_fresh"]),
            "baseline_false_confirm": bool(baseline["false_confirm"]),
            "baseline_terminal_visible": bool(baseline["terminal_visible"]),
            "oracle_target_visible": any(row["terminal_visible"] and row["legal_completed"] for row in rows),
            "oracle_correct_fresh": any(row["correct_fresh"] for row in rows),
            "oracle_strict": any(row["strict"] for row in rows),
            "oracle_false_safe_strict": any(row["strict"] and not row["false_confirm"] for row in rows),
            "strict_pairs": [[int(row["first_view"]), int(row["tail_view"])] for row in rows if row["strict"]],
            "target_visible_pairs": [[int(row["first_view"]), int(row["tail_view"])] for row in rows if row["terminal_visible"] and row["legal_completed"]],
            "wrong_pairs": [[int(row["first_view"]), int(row["tail_view"])] for row in rows if row["false_confirm"]],
            "max_confidence": float(max(row["confidence"] for row in rows)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128, help="number of archived cases from the front of the holdout")
    parser.add_argument("--out", type=Path, default=HERE / "legacy_headroom")
    args = parser.parse_args()
    if not (0 < args.n <= 128):
        raise ValueError("--n must be in 1..128")
    args.out.mkdir(parents=True, exist_ok=True)

    common = load_xz("COMMON")[: args.n]
    runs = load_xz("RUNS")
    archived = {(int(row["index"]), row["reserve"], row["arm"]): row for row in runs}
    executor = import_frozen_executor()

    all_cases = []
    for position, shared in enumerate(common):
        index = int(shared["index"])
        historical = {
            reserve: archived[index, reserve, "online_B"]
            for reserve in executor.RESERVES_MM
        }
        candidate_rows = replay_case(executor, shared)
        summary = summarize_case(candidate_rows, historical)
        all_cases.append({"index": index, "seed": int(shared["seed"]), "summary": summary, "candidates": candidate_rows})
        print(f"{position + 1}/{len(common)} case {index}", flush=True)

    output = {
        "schema": "moving_y_same_action_truth_only_headroom_v1",
        "source_commit": "cdb1705a02f60382ffbe9e42f2ed420e27a28ddb",
        "n_cases": len(all_cases),
        "diagnostic_only": True,
        "ordinary_planner_uses_public_observations": True,
        "oracle_uses_evaluator_labels": True,
        "same_acoustic_world_and_capture_clock": True,
        "same_legal_action_library": True,
        "assets": {
            "geometry_and_rebake": "/media/camp/EXT_DRIVE/RealUS_playground (paths and hashes in FROZEN_MANIFEST.json)",
            "motion_execution": "not reconstructed in this diagnostic",
            "capture_times": "frozen preflight durations added to the archived common-probe clock",
        },
        "limitations": [
            "The oracle enumerates evaluator labels after the fact and is not an admissible policy.",
            "Robot FK, collision, sampled travel and turn are not rerun because the optional PyTorch capability bridge is unavailable.",
            "The graph action legality, stroke/time gates and preflight path costs are loaded from the frozen manifest. The report therefore gives an optimistic action-support ceiling.",
            "No new benchmark claim is made. The historical online-B/MPC decisions remain the archived public-observation decisions.",
        ],
        "cases": all_cases,
    }
    for reserve in executor.RESERVES_MM:
        summaries = [case["summary"][reserve] for case in all_cases]
        n = len(summaries)
        output.setdefault("pooled", {})[reserve] = {
            "n_cases": n,
            "baseline_strict": sum(item["baseline_strict"] for item in summaries),
            "baseline_correct_fresh": sum(item["baseline_correct_fresh"] for item in summaries),
            "baseline_false_confirm": sum(item["baseline_false_confirm"] for item in summaries),
            "baseline_terminal_visible": sum(item["baseline_terminal_visible"] for item in summaries),
            "oracle_target_visible_cases": sum(item["oracle_target_visible"] for item in summaries),
            "oracle_correct_fresh_cases": sum(item["oracle_correct_fresh"] for item in summaries),
            "oracle_strict_cases": sum(item["oracle_strict"] for item in summaries),
            "oracle_false_safe_strict_cases": sum(item["oracle_false_safe_strict"] for item in summaries),
            "headroom_strict_cases": sum(item["oracle_strict"] and not item["baseline_strict"] for item in summaries),
            "headroom_correct_fresh_cases": sum(item["oracle_correct_fresh"] and not item["baseline_correct_fresh"] for item in summaries),
            "baseline_failed_but_any_target_visible": sum(item["oracle_target_visible"] and not item["baseline_strict"] for item in summaries),
            "baseline_failed_but_any_strict": sum(item["oracle_strict"] and not item["baseline_strict"] for item in summaries),
            "cases_with_any_wrong_pair": sum(bool(item["wrong_pairs"]) for item in summaries),
        }
    out_file = args.out / f"HEADROOM_{len(all_cases)}.json"
    out_file.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    compact = {"schema": output["schema"], "n_cases": output["n_cases"], "pooled": output["pooled"], "output": str(out_file)}
    (args.out / f"SUMMARY_{len(all_cases)}.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
