"""Train action advantages from matched legal continuations of a frozen student."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

V13 = Path(__file__).resolve().parent.parent / "leg_scan_student_history_20261124_v13"
sys.path.insert(1, str(V13))

import trajectory_v13 as tr
from closed_scan import ACTOR_DIM
from evidence import fork_on_tape
from region_policy import MAX_MENU, VALUE_WEIGHTS, fit_actor, pack_teacher, paired_targets, terminal_value
from run_v11 import outcome


SEARCH = (227_041_151, 227_053_217)
PROOF = 227_078_709
TARGETS = (3, 4, 5)
EPS = 1.e-6


def _prefix(case, controller, follow, target):
    live = tr.live_case(case, augment=case.split == "v15_teach")
    boundary = int(live.ends[target])
    next_boundary = int(live.ends[target + 1]) if target < TARGETS[-1] else None
    fallback = selected_menu = None

    def reached(node, read, menu, seen):
        nonlocal fallback, selected_menu
        if node.frontier < boundary:
            return False
        if next_boundary is not None and node.frontier >= next_boundary:
            return True
        if any(action.region == target for action in menu):
            selected_menu = tuple(menu)
            return True
        if fallback is None or (menu and not fallback[1]):
            fallback = tr.Result(node.fork(), read, [], frozenset(seen), prefix=True), tuple(menu)
        return False

    result = controller.roll(live, follow, stop=reached)
    if selected_menu is not None:
        return result, selected_menu
    if fallback is not None:
        return fallback
    raise RuntimeError(f"{case.identifier} never reached a decision in R{target + 1}")


def _pose(live):
    node = live.scan.node
    return [int(node.station), int(node.sector), int(node.tick)]


def _branch(prefix, shared_tape, task, controller, follow, offset):
    read, seen = prefix.read, prefix.seen
    live = shared_tape.fork()
    origin = _pose(live)
    if live.done or origin != _pose(prefix.live):
        raise RuntimeError("The branch must begin at the actual unsubmitted prefix")
    try:
        if task is None:
            detail = controller.resume(live, read)
            if detail is None:
                raise RuntimeError("The normal continuation did not execute")
            skill = detail["skill"]
            if not live.done and _pose(live)[2] <= origin[2]:
                raise RuntimeError("The normal continuation did not advance the sample clock")
            if live.done != (skill == "submit"):
                raise RuntimeError("The normal continuation has an inconsistent submission state")
            first_read = read if live.done else None
            execution = {"skill": skill, "start": origin, "end": _pose(live),
                         "goal_region": detail["goal_region"], "goal_station": detail["goal_station"]}
        else:
            live, detail = controller.execute(live, read, task)
            end = _pose(live)
            if live.done or origin[:2] != end[:2] or not origin[2] < end[2]:
                raise RuntimeError("The review must acquire new frames and rejoin the continuation pose")
            if origin[2] != detail["start_tick"] or end[2] != detail["end_tick"]:
                raise RuntimeError("The legal review details disagree with the sample clock")
            seen = seen | {task.key}
            first_read = None
            execution = {"skill": task.name, "start": origin, "end": end, "path": detail}

        terminal = controller.roll(live, follow, seen=seen, first_read=first_read)
        grade = controller.grade(terminal)
        if not terminal.live.done or not grade["completed_legal"]:
            raise RuntimeError("The frozen continuation did not legally submit")
        execution["suffix_reviews"] = [[entry["region"], entry["program"],
                                        entry["start_tick"], entry["end_tick"]]
                                       for entry in terminal.actions]
        return outcome(grade), execution
    except (ValueError, RuntimeError) as error:
        action = "continue_or_submit" if task is None else f"R{task.region + 1}/{task.name}"
        raise RuntimeError(f"{live.case.identifier}, tape +{offset}, {action}, pose {origin}: {error}") from error


def _comparison(reference, candidate):
    old = np.asarray(reference["selected_ids"])[3:]
    new = np.asarray(candidate["selected_ids"])[3:]
    return {
        "strict_gain": bool(candidate["strict"] and not reference["strict"]),
        "strict_loss": bool(reference["strict"] and not candidate["strict"]),
        "wrong_removed": bool(reference["wrong_branch"] and not candidate["wrong_branch"]),
        "wrong_added": bool(candidate["wrong_branch"] and not reference["wrong_branch"]),
        "region_corrected": ((old != 0) & (new == 0)).tolist(),
        "region_lost": ((old == 0) & (new != 0)).tolist(),
        "region_wrong_removed": ((old > 0) & (new <= 0)).tolist(),
        "region_wrong_added": ((old <= 0) & (new > 0)).tolist(),
    }


def teacher_job(case, controller, frozen_actor, rule, version):
    """Return one action menu and the mean paired search-tape advantages."""
    target = TARGETS[tr._ordinal(case) % len(TARGETS)]
    follow = controller.chooser(frozen_actor, rule)
    prefix, programs = _prefix(case, controller, follow, target)
    legal = (None, *programs)
    x = controller.vectors(prefix.live, prefix.read, legal)
    if len(legal) > MAX_MENU or x.shape != (len(legal), ACTOR_DIM) or not np.isfinite(x).all():
        raise RuntimeError(f"{case.identifier} has an invalid high-level menu shape {x.shape}")

    matrices, search = [], []
    for offset in SEARCH:
        base = fork_on_tape(prefix.live, offset)
        branches = [_branch(prefix, base, task, controller, follow, offset) for task in legal]
        reference = branches[0][0]
        matrices.append([terminal_value(grade) for grade, _ in branches])
        search.append({"offset": offset, "branches": [
            {"outcome": grade, "execution": execution, "comparison": _comparison(reference, grade)}
            for grade, execution in branches
        ]})
    delta = paired_targets(matrices)
    utility = delta @ VALUE_WEIGHTS
    chosen = int(np.argmax(utility))

    base = fork_on_tape(prefix.live, PROOF)
    reference, initial = _branch(prefix, base, None, controller, follow, PROOF)
    proof, selected = (_branch(prefix, base, legal[chosen], controller, follow, PROOF)
                       if chosen else (reference, initial))
    comparison = _comparison(reference, proof)
    live = prefix.live
    meta = {"case": asdict(case), "layout_id": live.scan.tape.layout.identifier, "version": version,
            "target_region": target + 1, "target_eligible": any(a.region == target for a in programs),
            "frontier": int(live.frontier), "prefix": _pose(live), "before_last": bool(live.frontier < live.ends[5]),
            "seen": sorted(prefix.seen), "actions": [[0, "continue_or_submit"], *[
                [action.region + 1, action.name] for action in programs]], "chosen": chosen,
            "utility_delta": utility.tolist(), "search": search, "proof_offset": PROOF, "proof": proof,
            "proof_continue": reference, "proof_execution": selected, "proof_continue_execution": initial,
            "proof_comparison": comparison}
    return x, delta, meta


def summarize_teacher(rows):
    """Count action eligibility, signed search labels and independent changes by region."""
    totals = Counter()
    targets = {str(r + 1): Counter() for r in TARGETS}
    programs = {str(r + 1): Counter() for r in TARGETS}
    proof_regions = {str(r + 1): Counter() for r in TARGETS}
    for row in rows:
        selected = row["actions"][row["chosen"]]
        paired = row["proof_comparison"]
        totals.update(prefixes=1, before_last=int(row["before_last"]), after_review=int(bool(row["seen"])),
                      real_candidates=len(row["actions"]) - 1, selected_program=int(bool(row["chosen"])),
                      proof_rescue=int(paired["strict_gain"]), proof_loss=int(paired["strict_loss"]),
                      proof_strict=int(row["proof"]["strict"]), continue_strict=int(row["proof_continue"]["strict"]),
                      proof_wrong=int(row["proof"]["wrong_branch"]), continue_wrong=int(row["proof_continue"]["wrong_branch"]),
                      proof_wrong_removed=int(paired["wrong_removed"]), proof_wrong_added=int(paired["wrong_added"]))
        targets[str(row["target_region"])].update(prefixes=1, before_last=int(row["before_last"]),
                                                with_legal_action=int(row["target_eligible"]),
                                                after_review=int(bool(row["seen"])))
        for action, utility in zip(row["actions"][1:], row["utility_delta"][1:], strict=True):
            count = programs[str(action[0])]
            count.update(legal=1, positive=int(utility > EPS), negative=int(utility < -EPS),
                         neutral=int(abs(utility) <= EPS), selected=int(action == selected))
        for i, region in enumerate(proof_regions.values()):
            region.update(corrected=int(paired["region_corrected"][i]), lost=int(paired["region_lost"][i]),
                          wrong_removed=int(paired["region_wrong_removed"][i]),
                          wrong_added=int(paired["region_wrong_added"][i]))
    return dict(totals) | {"targets": {r: dict(c) for r, c in targets.items()},
                           "programs": {r: dict(c) for r, c in programs.items()},
                           "proof_regions": {r: dict(c) for r, c in proof_regions.items()}}


def fit_student(fit_rows, cal_rows, out, seed, frozen_actor, *, epochs=25):
    """Select CUDA weights on teacher calibration before closed-loop rule development."""
    fit = fit_rows if isinstance(fit_rows, dict) else pack_teacher(fit_rows)
    calibration = cal_rows if isinstance(cal_rows, dict) else pack_teacher(cal_rows)
    if not len(fit["features"]) or not len(calibration["features"]):
        raise ValueError("Student training and teacher calibration require separate nonempty collections")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return fit_actor(fit, calibration, out, "student", seed, start=frozen_actor, epochs=epochs)
