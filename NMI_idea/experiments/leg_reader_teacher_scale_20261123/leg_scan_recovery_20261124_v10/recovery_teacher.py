"""Paired full-program search and teacher labels on the student's own visits."""

from dataclasses import asdict

import numpy as np

from recovery_task import SEARCH_TAPES, TEACH_PROOF, Live, options, stats, table, utility


def focus_prefix(case, tape=None):
    live = Live(case, tape)
    # This curriculum also samples routine, previously successful prefixes.
    number = int(case.identifier.rsplit("_", 1)[-1])
    regions = (5, 5, 4) if case.task == "femoral" else (3, 4, 5, 4, 5, 2)
    focus = regions[number % len(regions)]
    live.forward(live.ends[focus] if number % 5 else live.scan.spec[2])
    return live, focus


def _end(live):
    child = live.fork()
    # The whole remaining route is physically executed on the assigned tape.
    for _ in range(8):
        if child.done:
            break
        child.take(child.base_goal())
    if not child.done:
        raise RuntimeError("The reserved MPC suffix did not reach legal submission")
    result = child.grade()
    if not result["completed_legal"]:
        raise RuntimeError("Teacher scored an illegal terminal pose")
    return child, result


def _replay(live, goals, tape):
    branched = live.fork(tape)
    for goal in goals:
        if branched.done or branched.take(goal) is None:
            return None
    return _end(branched)


def improvement(next_grade, baseline, focus):
    before, after = baseline["selected_ids"], next_grade["selected_ids"]
    lost = sum(a == 0 and b != 0 for a, b in zip(before, after, strict=True))
    new_named = sum(a <= 0 and b > 0 for a, b in zip(before, after, strict=True))
    return utility(next_grade, focus) - utility(baseline, focus) - .11 * lost - .11 * new_named


def _best_continuations(live, first_choices, base, focus, width, teach_continuations):
    """Find two consecutive options on a real tape from this public prefix."""
    candidates = {}
    first = []
    for idx, (goal, traj, _) in enumerate(first_choices):
        child = live.fork()
        if child.take(goal, traj) is None:
            raise RuntimeError("An admitted teacher choice could not execute")
        _, end = _end(child)
        score = improvement(end, base, focus)
        candidates[idx] = [((goal,), score)]
        if not child.done:
            first.append((score, idx, goal, child, end))
    # Expansion uses observations after the first choice, so it can solve
    # multiple deficient zones and can decide to stop repairing.
    after = []
    for _, idx, first_goal, child, otherwise in sorted(first, key=lambda x: x[0], reverse=True)[:width]:
        next_choices = options(child, focus=focus)
        x, mask = table(next_choices)
        values = np.zeros(len(mask), dtype=np.float32)
        for j, (second, traj, _) in enumerate(next_choices):
            if not second.is_repair:
                continue
            grandchild = child.fork()
            grandchild.take(second, traj)
            _, end = _end(grandchild)
            candidates[idx].append(((first_goal, second), improvement(end, base, focus)))
            values[j] = improvement(end, otherwise, second.region)
        if teach_continuations and first_goal.is_repair:
            after.append((x, mask, values))
    return candidates, after


def strong(live, focus, *, width=2, keep_snapshots=False, split="train", teach_continuations=False):
    """Fit on paired selection tapes and test the selected program on a third."""
    computed = options(live, focus)
    features, valid = table(computed)
    first_branches = live.fork(SEARCH_TAPES[0])
    baseline0 = _end(first_branches)[1]
    baseline1 = _end(live.fork(SEARCH_TAPES[1]))[1]
    searches, after = _best_continuations(first_branches, computed, baseline0, focus, width, teach_continuations)
    targets = np.zeros(len(valid), dtype=np.float32)
    best_program = ()
    best = 0.
    for index, options_at_root in searches.items():
        # Always recheck the first action and the two best complete programs.
        nominees = [options_at_root[0]]
        nominees.extend(sorted(options_at_root[1:], key=lambda x: x[1], reverse=True)[:2])
        ranked = []
        for program, score in nominees:
            checked = _replay(live, program, SEARCH_TAPES[1])
            if checked is not None:
                average = .5 * (score + improvement(checked[1], baseline1, focus))
                ranked.append((average, program))
        target, program = max(ranked, key=lambda x: x[0]) if ranked else (0., ())
        targets[index] = target
        if target > best + 1.e-7:
            best_program, best = program, target
    _, ref = _end(live.fork(TEACH_PROOF))
    selected = _replay(live, best_program, TEACH_PROOF)
    valid_program = selected is not None
    final, grade = selected or _end(live.fork(TEACH_PROOF))
    # The learner sees quantitative action outcomes on the search tapes; the
    # third tape is frozen proof for the whole chosen program.
    episode = {"x": features, "mask": valid, "values": targets}
    if teach_continuations:
        blank = (np.zeros_like(features), np.zeros_like(valid), np.zeros_like(targets))
        states = after + [blank] * (width - len(after))
        episode |= {key: np.stack(a) for key, a in zip(("follow_x", "follow_mask", "follow_values"),
                                                      zip(*states, strict=True), strict=True)}
    diagnostics = {
        "case": live.case.identifier, "route": live.case.task, "layout_id": ref["layout_id"],
        "focus": focus + 1, "station": int(live.scan.node.station), "tick": int(live.scan.node.tick),
        "search_goals": [asdict(c[0]) for c in computed], "search_advantages": targets[:len(computed)].tolist(),
        "sequence": [asdict(goal) for goal in best_program], "valid_replay": valid_program,
        "selection_gain": float(best), "proof_gain": improvement(grade, ref, focus),
        "post_recovery_labels": len(after),
        "proof_baseline": ref, "proof_teacher": stats(final, grade),
    }
    snapshots = []
    if keep_snapshots:
        baseline_scan, _ = _end(live.fork(TEACH_PROOF))
        snapshots.extend((live.snapshot(split, "difficulty_prefix"),
                          baseline_scan.snapshot(split, "mpc_terminal"), final.snapshot(split, "strong_recovery_terminal")))
    return episode, diagnostics, snapshots


def adapt(live, candidates, selected, weights, seed, rollout):
    """Compare two real first options followed by the same frozen learner."""
    if selected == 0:
        alternatives = list(range(1, len(candidates)))
        if not alternatives:
            return None
        selected = int(np.random.default_rng(seed).choice(alternatives))
    diffs = []
    bands = []
    for index, tape in enumerate(SEARCH_TAPES):
        endings = []
        for j in (0, selected):
            branch = live.fork(tape + 90_000 + (seed % 35_003))
            branch.take(*candidates[j][:2])
            final = rollout(branch, weights, seed + index, False, record=False)
            endings.append(final["metrics"])
        ref, alt = endings
        diffs.append(improvement(alt, ref, candidates[selected][0].region))
        bands.append({"reference_strict": ref["strict"], "alternative_strict": alt["strict"],
                      "reference_wrong_branch": ref["wrong_branch"], "alternative_wrong_branch": alt["wrong_branch"]})
    return {"alternative": selected, "measured_gain": float(np.mean(diffs)), "by_tape": diffs, "outcomes": bands,
            "station": int(live.scan.node.station), "tick": int(live.scan.node.tick),
            "layout_id": live.scan.tape.layout.identifier, "goal": asdict(candidates[selected][0])}
