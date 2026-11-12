"""Recompute inverse-view outcomes and paired estimates from checked traces."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import lzma
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta


TASKS = {"daughter": ((4, 3), 1), "trunk": ((6, 7), 0), "NONBRANCH": ((6, 7), 0)}
FIELDS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s", "rail_mm",
          "captures", "combined_s")
MARGINS = np.array([2., 2., .10])


def seed(namespace, index):
    return int(np.random.SeedSequence([namespace, index]).generate_state(1)[0])


def noise(tape_seed, n=7):
    return np.random.default_rng(np.random.SeedSequence([tape_seed, 68401])).random((n, 7))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def packet(detect, cdf, uniforms, distance):
    visible = np.flatnonzero(uniforms[:3] < detect)
    if not visible.size:
        return 0, -1
    bins = np.array([min(np.searchsorted(cdf[k], uniforms[k + 3]), len(distance) - 1)
                     for k in visible])
    distances = distance[bins]
    ties = np.flatnonzero(np.abs(distances - distances.min()) < 1.e-12)
    winner = ties[int(uniforms[6] * ties.size)]
    return 1 + bins[winner] + len(distance) * (visible.size > 1), visible[winner]


def fits_route(start, path, edges, nodes, targets):
    totals = np.zeros(4)
    progress = 0
    for goal in path:
        if (start, goal) not in edges or progress >= len(targets):
            return False
        totals += edges[start, goal]
        progress += nodes[goal]["site"] == targets[progress]
        start = goal
    return progress == len(targets) and len(path) <= 6 and np.all(totals <= [65. + 1e-8, 90. + 1e-8, 250. + 1e-8, 9. + 1e-8])


def trace_values(record, arm, nodes, edges, start, targets, packets):
    at, progress = start, 0
    totals = np.zeros(4)
    wrong = set()
    trace = record["trace"]
    for index, step in enumerate(trace):
        source, goal, obs, confidence, claim, kind, mm, turn, rail, tick, slot, forced = step
        source, goal, kind = map(int, (source, goal, kind))
        assert index == slot and source == at and progress < len(targets)
        cost = edges[source, goal]
        totals += cost
        assert np.isclose(totals[3], tick * 1.5, rtol=0, atol=1.e-10)
        np.testing.assert_allclose([mm, turn, rail], cost[:3], rtol=0, atol=1.e-10)
        assert packets[int(tick), goal] == (obs, kind)
        assert 0 <= confidence <= 1. + 1.e-10
        expected_claim = obs > 0 and confidence >= .87 and nodes[goal]["site"] == targets[progress]
        assert bool(claim) == expected_claim
        if expected_claim:
            progress += 1
            if kind != 0:
                wrong.add("other_lumen" if kind == 1 else "false_structure")
        assert bool(forced) == (arm is not None and index == 0)
        if forced:
            assert goal == arm
        at = goal
    assert len(trace) <= 6 and np.all(totals <= [65. + 1e-8, 90. + 1e-8, 250. + 1e-8, 9. + 1e-8])
    tcp, turn, rail, duration = totals
    return [int(progress == len(targets) and not wrong), int(bool(wrong)), tcp, turn, duration, rail, len(trace)], wrong


def compare(left, right, archived):
    a, b = np.asarray(left), np.asarray(right)
    n, repeats, _ = a.shape
    assert a.shape == b.shape and archived["n_cases"] == n and archived["n_repeats"] == repeats
    tape_diff = a - b
    case_diff = tape_diff.mean(axis=1)
    assert np.allclose(case_diff.mean(axis=0), [archived["delta"][name] for name in FIELDS], rtol=0, atol=1.e-10)
    rng = np.random.default_rng(57216)
    sample = np.concatenate([case_diff[rng.integers(n, size=(batch, n))].mean(axis=1)
                             for batch in [400] * 50])
    ci = np.quantile(sample, [.025, .975], axis=0).T
    cost_ci = np.quantile(sample[:, 2:5], [.05 / 6., 1. - .05 / 6.], axis=0).T
    np.testing.assert_allclose(ci, [archived["delta_ci95_case_clustered"][name] for name in FIELDS], rtol=0, atol=1.e-10)
    np.testing.assert_allclose(cost_ci, [archived["cost_bonferroni_ci95"][name] for name in FIELDS[2:5]], rtol=0, atol=1.e-10)
    assert archived["all_costs_equivalent"] == bool(np.all(cost_ci[:, 0] >= -MARGINS) and np.all(cost_ci[:, 1] <= MARGINS))
    assert archived["all_costs_noninferior_with_margin"] == bool(np.all(cost_ci[:, 1] <= MARGINS))
    for col, key in ((0, "success_wins_losses_ties"), (1, "wrong_harm_benefit_ties_cases")):
        np.testing.assert_array_equal(archived[key], [np.sum(case_diff[:, col] > 0.),
                                                     np.sum(case_diff[:, col] < 0.),
                                                     np.sum(case_diff[:, col] == 0.)])
    harms = int(np.sum(tape_diff[..., 1] > 0.))
    if repeats == 1:
        expected = float(beta.ppf(.975, harms + 1, n - harms)) if harms < n else 1.
        assert np.isclose(expected, archived["wrong_harm_975_upper"], rtol=0, atol=1.e-12)
    else:
        assert archived["wrong_harm_975_upper"] is None
    return dict(correct_counts=[int(x[..., 0].sum()) for x in (a, b)],
                wrong_counts=[int(x[..., 1].sum()) for x in (a, b)],
                completion_wins=int(np.sum(tape_diff[..., 0] > 0.)),
                completion_losses=int(np.sum(tape_diff[..., 0] < 0.)),
                harmful_wrong_pairs=harms, favorable_wrong_pairs=int(np.sum(tape_diff[..., 1] < 0.)),
                correct_delta=float(case_diff[:, 0].mean()), correct_ci95=ci[0].tolist(),
                wrong_delta=float(case_diff[:, 1].mean()), wrong_ci95=ci[1].tolist(),
                mean_cost_delta=case_diff[:, 2:5].mean(axis=0).tolist(), cost_simultaneous_ci95=cost_ci.tolist(),
                strict_equivalence=archived["all_costs_equivalent"], no_worse=archived["all_costs_noninferior_with_margin"],
                mean_rail_mm=[float(x[..., 5].mean()) for x in (a, b)],
                mean_captures=[float(x[..., 6].mean()) for x in (a, b)],
                mean_cpu_s=[float((x[..., 7] - x[..., 4]).mean()) for x in (a, b)],
                cpu_plus_motion_ci95=ci[-1].tolist())


def choose(values, candidates, ranking, constrained):
    incumbent = values["MPC"][ranking]
    choices = ["MPC"]
    for name in candidates:
        change = (values[name][ranking] - incumbent).mean(axis=0)
        if name != "MPC" and (not constrained or np.all(change[2:5] <= MARGINS + 1.e-12)):
            choices.append(name)
    return max(choices, key=lambda name: values[name][ranking, 0].sum())


def load_bank(path):
    data = path.read_bytes()
    return lzma.decompress(data) if path.suffix == ".xz" else data


def audit(args):
    study = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(study), str(args.repo / "NMI_idea/experiments/ab_full_b_rollout_20261113")]
    import acoustics
    from world import ANGLES, Geometry

    settings = json.loads((args.cohort / "settings.json").read_text())
    bank_bytes, base_bytes = load_bank(args.motion), load_bank(args.base)
    assert digest(bank_bytes) == settings["bank_sha256"] and digest(base_bytes) == settings["base_sha256"]
    bank, base = json.loads(bank_bytes), json.loads(base_bytes)
    nodes = {node["id"]: node for node in bank["nodes"]}
    old_nodes = {node["id"] for node in base["nodes"]}
    old_edges = {(edge["start"], edge["goal"]): edge for edge in base["edges"]}
    new_edges = {(edge["start"], edge["goal"]): edge for edge in bank["edges"]}
    for pair, edge in old_edges.items():
        assert json.dumps(new_edges[pair], sort_keys=True) == json.dumps(edge, sort_keys=True)
    for node in base["nodes"]:
        expanded = nodes[node["id"]]
        assert json.dumps({k: expanded[k] for k in node}, sort_keys=True) == json.dumps(node, sort_keys=True)
    assert all(n["view"] in ANGLES for n in base["nodes"])
    assert tuple(settings["fields"]) == FIELDS
    world = acoustics.setup_world(args.repo)
    geometries = {task: Geometry(acoustics, world, bank["nodes"], label)
                  for task, (_, label) in TASKS.items()}
    costs = {key: np.array([edge[k] for k in ("travel_mm", "turn_deg", "rail_mm", "duration_s")])
             for key, edge in new_edges.items()}
    coarse = {key: value for key, value in costs.items() if set(key) <= old_nodes}
    original = {key: costs[key] for key in old_edges}
    repeats = settings["repeats"]
    cases = args.cohort / "cases.jsonl"
    stream = cases.open() if cases.exists() else lzma.open(cases.with_suffix(".jsonl.xz"), "rt")
    reported = json.loads((args.cohort / "summary.json").read_text())
    results, seen = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    verification = Counter()
    mechanism = defaultdict(Counter)
    comparisons = {}
    tape_hashes = set()
    frozen = settings.get("source_sha256", {})
    inherited = args.repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"
    previous = args.repo / "NMI_idea/experiments/ab_continuation_b_20261113"
    source_paths = {"legacy": inherited, "continuation": previous}
    changed = []
    archived_source = {}
    for name, expected in frozen.items():
        scope, _, file = name.rpartition("/")
        path = source_paths.get(scope, study) / file
        snapshot = study / "motion/motion_at_screen.py"
        if name == "motion.py" and snapshot.exists() and digest(snapshot.read_bytes()) == expected:
            path = snapshot
            archived_source[name] = str(snapshot.relative_to(study))
        if path.exists() and digest(path.read_bytes()) == expected:
            verification["source_hashes_verified"] += 1
        else:
            changed.append(name)
    if settings["split"] != "smoke":
        assert not changed, changed
        freeze = json.loads((study / "PROTOCOL_FREEZE.json").read_text())
        assert digest((study / "PROTOCOL.md").read_bytes()) == freeze["protocol_sha256"]
    with stream:
        for line in stream:
            row = json.loads(line)
            task, index = row["task"], row["index"]
            assert index not in seen[task]
            seen[task].add(index)
            targets, _ = TASKS[task]
            geometry, start = geometries[task], bank["task_starts"][task]
            case_seed = seed(settings["namespace"], index)
            assert row["case_seed"] == case_seed
            scenario = acoustics.make_scenario(world, case_seed)
            _, _, detect, cdf = acoustics.probabilities(geometry, scenario, np.arange(7) * 1.5, "joint")
            addr = dict(zip((n["id"] for n in bank["nodes"]), geometry.node_to_address))
            anchor_noise = noise(case_seed)[0]
            obs, _ = packet(detect[0, addr[start]], cdf[0, addr[start]], anchor_noise, acoustics.DISTANCE)
            assert row["anchor"] == obs
            verification["anchor_packets"] += 1
            proposed = row["proposals"]["selected"]
            menus = {"MPC": proposed, **row.get("menu_controls", {})}
            menu_costs = {}
            for policy, menu in menus.items():
                assert all(len(menu[site]) == len(selected) for site, selected in proposed.items())
                allowed = old_nodes | {n["id"] for n in nodes.values() if n["view"] in menu.get(str(n["site"]), [])}
                menu_costs[policy] = {key: value for key, value in costs.items() if set(key) <= allowed}
            augmented = menu_costs["MPC"]
            for site, scores in row["proposals"]["scores"].items():
                objectives = ("immediate_claim_utility", "decision_information")
                picks = [max(scores, key=lambda x: (x[obj], -abs(x["angle"]), -x["angle"]))["view"]
                         for obj in objectives] if scores else []
                assert list(dict.fromkeys(picks)) == proposed[site]
                for score in scores:
                    assert -1.e-10 <= score["decision_information"] <= score["perfect_information_claim_gap"] + 1.e-10
                    for witness in score["ideal_completion_witnesses"]:
                        if witness is not None:
                            assert fits_route(start, witness, costs, nodes, targets)
                            assert any(nodes[n]["site"] == int(site) and nodes[n]["view"] == score["view"] for n in witness)
                            if score["view"] in proposed[site]:
                                assert fits_route(start, witness, augmented, nodes, targets)
                            verification["proposal_witnesses"] += 1
            for arm, desc in row["candidates"].items():
                if arm == "MPC":
                    assert not desc["prefix"] and desc["witness"] is None
                    continue
                goal, = desc["prefix"]
                assert int(arm) == goal and desc["witness"][0] == goal
                assert goal != row["mpc_command"] and nodes[goal]["site"] == targets[0]
                assert fits_route(start, desc["witness"], augmented, nodes, targets)
                verification["joint_resource_witnesses"] += 1
                mechanism[task]["witnessed_first_commands"] += 1
                if goal not in old_nodes:
                    mechanism[task]["witnessed_novel_first_commands"] += 1
            values = {arm: [] for arm in row["measurements"]}
            wrong_types = {arm: [] for arm in values}
            for r in range(repeats):
                suffix_seed = seed(case_seed, 930000 + r)
                tape = noise(suffix_seed)
                tape[0] = anchor_noise
                assert row["repeat_seeds"][r] == suffix_seed and row["tape_sha256"][r] == digest(tape.tobytes())
                tape_hashes.add(row["tape_sha256"][r])
                verification["task_suffix_tapes"] += 1
                packets = {}
                for arm in values:
                    record = row["paths"][arm][r]
                    for step in record["trace"]:
                        tick, goal = int(step[9]), int(step[1])
                        assert nodes[goal]["site"] in {5, *targets}
                        if (tick, goal) not in packets:
                            i = addr[goal]
                            packets[tick, goal] = packet(detect[tick, i], cdf[tick, i], tape[tick], acoustics.DISTANCE)
                    active = menu_costs.get(arm, augmented)
                    forced = None
                    if arm == "coarse":
                        active = coarse
                    elif arm == "original":
                        active = original
                    elif arm == "fixed" and row["fixed_command"] >= 0:
                        forced = row["fixed_command"]
                    elif arm in row["candidates"] and arm != "MPC":
                        forced, = row["candidates"][arm]["prefix"]
                    if arm == "MPC":
                        goal = int(record["trace"][0][1]) if record["trace"] else -1
                        assert goal == row["mpc_command"]
                    checked, kinds = trace_values(record, forced, nodes, active, start, targets, packets)
                    observed = row["measurements"][arm][r]
                    np.testing.assert_allclose(checked, observed[:7], rtol=0, atol=1.e-9)
                    assert observed[7] >= checked[4] - 1.e-10
                    values[arm].append(checked + [observed[7]])
                    wrong_types[arm].append(kinds)
                    verification["paths"] += 1
                    verification["trace_packets"] += len(record["trace"])
                    if arm in ("MPC", "claim_only", "fixed_angles"):
                        mechanism[task][f"{arm}_paths_using_novel_angles"] += any(
                            int(step[1]) not in old_nodes for step in record["trace"])
                    for kind in kinds:
                        verification[f"wrong:{task}:{arm}:{kind}"] += 1
            values = {name: np.array(result) for name, result in values.items()}
            for arm in ("MPC", "coarse", "original", "fixed", "claim_only", "fixed_angles"):
                if arm in values:
                    results[task][arm].append(values[arm])
            if repeats > 1:
                halfway = repeats // 2
                assert halfway * 2 == repeats
                archived = next(record for record in reported[task]["selections"] if record["index"] == index)
                for selection, constrained in (("choice", True), ("unconstrained", False)):
                    selected = values["MPC"].copy()
                    paired_choices = []
                    for i, (rank, test) in enumerate(((slice(None, halfway), slice(halfway, None)),
                                                     (slice(halfway, None), slice(None, halfway)))):
                        choice = choose(values, row["candidates"], rank, constrained)
                        assert archived["halves"][i][selection] == choice
                        paired_choices.append(choice)
                        selected[test] = values[choice][test]
                        if constrained and choice != "MPC":
                            mechanism[task]["cost_screened_non_mpc_half_selections"] += 1
                            if int(choice) not in old_nodes:
                                mechanism[task]["cost_screened_novel_half_selections"] += 1
                        if constrained:
                            for wrong in wrong_types[choice][test]:
                                for kind in wrong:
                                    verification[f"wrong:{task}:split_cost_screened:{kind}"] += 1
                    if constrained and paired_choices[0] == paired_choices[1]:
                        mechanism[task]["cost_screened_halves_agree_cases"] += 1
                    results[task][selection].append(selected)
                attainable = np.stack([values[arm][:, 0] for arm in row["candidates"]]).max(axis=0)
                results[task]["hindsight"].append(attainable - values["MPC"][:, 0])
    available = {"split_cost_screened_vs_augmented": ("choice", "MPC"),
                 "split_hard_caps_vs_augmented": ("unconstrained", "MPC"),
                 "split_cost_screened_vs_fixed": ("choice", "fixed"),
                 "augmented_vs_coarse": ("MPC", "coarse"),
                 "coarse_vs_original": ("coarse", "original"),
                 "augmented_vs_original": ("MPC", "original"),
                 "augmented_vs_claim_only": ("MPC", "claim_only"),
                 "augmented_vs_fixed_angles": ("MPC", "fixed_angles"),
                 "fixed_vs_augmented": ("fixed", "MPC")}
    for task, cell in results.items():
        assert seen[task] == set(range(settings["n"]))
        archived = reported[task]["comparisons"]
        comparisons[task] = {name: compare(cell[left], cell[right], archived[name])
                             for name, (left, right) in available.items() if name in archived}
        if "hindsight" in cell:
            comparisons[task]["same_tape_hindsight_gain"] = float(np.mean(cell["hindsight"]))
    assert len(results) == 2
    assert len(tape_hashes) == settings["n"] * repeats
    return dict(verification=dict(verification), independent_case_suffix_tapes=len(tape_hashes),
                different_source_hashes=changed, archived_frozen_sources=archived_source,
                mechanism={task: dict(c) for task, c in mechanism.items()},
                comparisons=comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "motion", "base", "cohort"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))
