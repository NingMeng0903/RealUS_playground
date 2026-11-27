"""Independently reconstruct held-out topology trajectories and endpoints."""
import argparse
from collections import defaultdict
import hashlib
import json
import lzma
from pathlib import Path
from statistics import mean
import sys

import numpy as np

from audit_bank import recheck
from audit_observation import fine_to_coarse


def uniform_tape(namespace, index, shape):
    seed = np.random.SeedSequence([namespace, index]).generate_state(1)[0]
    return np.random.default_rng(int(seed)).random(shape)


def observe(truth, index, time, node, site, tape, mapping, packet_numbers):
    u = tape[time, site - 3]
    bins, kinds = [], []
    for kind in range(3):
        if u[kind] < truth.detect[time, node, index, kind]:
            fine = min(48, int(np.searchsorted(truth.cdf[time, node, index, kind], u[kind + 3])))
            bins.append(int(mapping[fine]))
            kinds.append(kind)
        else:
            bins.append(0)
            kinds.append(-1)
    order = np.argsort(bins, kind="stable")
    bins = [bins[i] for i in order]
    kinds = [kinds[i] for i in order]
    if bins[0] > 0 and len(set(bins)) == 1:
        first = min(2, int(3 * u[6]))
        swap = int(6 * u[6]) % 2
        kinds = [kinds[first], kinds[(first + 1 + swap) % 3], kinds[(first + 2 - swap) % 3]]
    elif u[6] >= 0.5:
        for i in (0, 1):
            if bins[i] > 0 and bins[i] == bins[i + 1]:
                kinds[i], kinds[i + 1] = kinds[i + 1], kinds[i]
                break
    return packet_numbers[tuple(bins)], kinds


def records(path):
    if path.is_dir():
        for child in sorted(path.glob("*.jsonl")):
            yield from records(child)
        return
    opener = lzma.open if path.suffix == ".xz" else Path.open
    with opener(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "arms" in row:
                for name, result in row["arms"].items():
                    yield name, {"index": row["index"], **result}
            else:
                yield row.get("arm", row.get("policy")), row


def endpoint(row, truth, trace, sites, prefix_length, fork_tol):
    index = row["index"]
    label = int(truth.labels[index])
    predicted = int(row["prediction"])
    assert row["truth"] == label and row["family"] == truth.families[index]
    assert bool(row["positive"]) == bool(label > 0)
    identity = False
    fork_error = None
    choice = row["chosen"]
    if predicted > 0:
        t0, t1 = choice["frames"]
        assert t0 > prefix_length and t1 == t0 + 1
        close_site = 4 + (predicted - 1) // 2
        assert [sites[node] for node in choice["nodes"]] == [close_site, close_site - 1]
        for event, node, packet, slot in zip(choice["frames"], choice["nodes"], choice["packets"],
                                            choice["slots"], strict=True):
            audited = trace[event]
            assert audited[:2] == (node, packet)
            assert 0 <= slot < len(audited[2])
        identity = all(trace[event][2][slot] == 0
                       for event, slot in zip(choice["frames"], choice["slots"], strict=True))
        if label > 0:
            fork_error = abs(choice["fork_hat_mm"] - truth.fork_mm[index])
            assert np.isclose(row["fork_error_mm"], fork_error, atol=1e-10, rtol=0.0)
    else:
        assert choice is None
    assert (row["fork_error_mm"] is None) == (fork_error is None)
    success = predicted == label and (label == 0 or (identity and fork_error <= fork_tol))
    wrong = predicted >= 0 and not success
    assert bool(row["success"]) == success
    assert bool(row["wrong"]) == wrong
    assert bool(row["affirmative_error"]) == (predicted > 0 and wrong)
    assert bool(row["correct_images"]) == identity
    assert bool(row["correct_declaration"]) == (predicted == label)


def posterior_endpoint(model, belief, pairs, fork_tolerance, legacy_mean=False):
    labels, forks = model.labels, model.forks
    probabilities = np.zeros(5)
    positions = np.zeros(5)
    slots = np.full((5, 2), -1)
    probabilities[0] = belief[labels == 0].sum()
    for group, pair in enumerate(pairs):
        if pair is None:
            continue
        (t0, a0, p0), (t1, a1, p1) = pair
        conditional_left = model.conditional[t0, model.addresses[a0], p0]
        conditional_right = model.conditional[t1, model.addresses[a1], p1]
        for label in (2 * group + 1, 2 * group + 2):
            best = -1.0
            for slot0 in np.flatnonzero(model.bins[p0]):
                for slot1 in np.flatnonzero(model.bins[p1]):
                    weights = belief * (labels == label) * conditional_left[slot0] * conditional_right[slot1]
                    total = weights.sum()
                    if legacy_mean:
                        position = weights @ forks / total if total > 0 else 0.0
                        probability = weights[np.abs(forks - position) <= fork_tolerance].sum()
                    else:
                        x = forks[labels == label]
                        w = weights[labels == label]
                        if total > 0:
                            # An optimal interval can put its left boundary at
                            # an observed fork. Enumerate those starts instead
                            # of reproducing the optimized sliding window.
                            delta = x[None, :] - x[:, None]
                            masses = ((delta >= 0.) & (delta <= 2. * fork_tolerance)) @ w
                            start = int(np.argmax(masses))
                            probability = masses[start]
                            position = x[start] + fork_tolerance
                        else:
                            probability, position = 0., 0.
                    if probability > best:
                        best = probability
                        probabilities[label] = probability
                        positions[label] = position
                        slots[label] = slot0, slot1
    return probabilities, positions, slots


def check_posterior_endpoint(row, model, belief, pairs, fork_tol, legacy_mean):
    probabilities, positions, slots = posterior_endpoint(model, belief, pairs, fork_tol, legacy_mean)
    mass = np.bincount(model.labels, weights=belief, minlength=5)
    np.testing.assert_allclose(row["posterior"], mass, atol=1e-10, rtol=0.0)
    predicted = row["prediction"]
    if predicted >= 0:
        assert np.isclose(row["probability"], probabilities[predicted], atol=1e-10, rtol=0.0)
    if predicted <= 0:
        return
    chosen = row["chosen"]
    selected = pairs[(predicted - 1) // 2]
    assert chosen["frames"] == [pair[0] for pair in selected]
    if legacy_mean:
        assert np.isclose(chosen["fork_hat_mm"], positions[predicted], atol=1e-8, rtol=0.0)
        np.testing.assert_array_equal(chosen["slots"], slots[predicted])
        return
    (t0, a0, p0), (t1, a1, p1) = selected
    i, j = chosen["slots"]
    assert model.bins[p0, i] > 0 and model.bins[p1, j] > 0
    weights = (belief * (model.labels == predicted) * model.conditional[t0, model.addresses[a0], p0, i]
               * model.conditional[t1, model.addresses[a1], p1, j])
    achieved = weights[np.abs(model.forks - chosen["fork_hat_mm"]) <= fork_tol + 1e-10].sum()
    assert np.isclose(achieved, probabilities[predicted], atol=1e-10, rtol=0.0)


def suffix_diagnostic(row, trace, pair_by_group, sites, truth, index, fork_tol, full_trace=None):
    """Evaluator-only attainable evidence and mutually exclusive observed failures."""
    true_label = int(truth.labels[index])
    if true_label <= 0:
        return None
    group = (true_label - 1) // 2
    near, far = 4 + group, 3 + group
    candidate, visible, supported_pair = False, False, False
    for t0, (a0, p0, labels0, supported0) in trace.items():
        next_frame = trace.get(t0 + 1)
        if next_frame is None:
            continue
        a1, p1, labels1, supported1 = next_frame
        if sites[a0] == near and sites[a1] == far:
            candidate = True
            supported_pair |= supported0 and supported1 and p0 > 0 and p1 > 0
            visible |= 0 in labels0 and 0 in labels1
    original_visible = visible
    if full_trace is not None:
        original_visible = False
        for t0, (a0, _, labels0, _) in full_trace.items():
            next_frame = full_trace.get(t0 + 1)
            if next_frame is not None:
                a1, _, labels1, _ = next_frame
                original_visible |= sites[a0] == near and sites[a1] == far and 0 in labels0 and 0 in labels1
    last = pair_by_group[group]
    stored_visible = last is not None and all(0 in trace[frame[0]][2] for frame in last)
    chosen = row["chosen"]
    chosen_pair_visible = chosen is not None and all(0 in trace[frame][2] for frame in chosen["frames"])
    geometry_ok = (row["prediction"] == true_label and row["fork_error_mm"] is not None
                   and row["fork_error_mm"] <= fork_tol)
    if row["success"]:
        first_failure = "success"
    elif not candidate:
        first_failure = "no_ordered_near_far_moves"
    elif not visible:
        first_failure = "no_pair_with_two_target_returns"
    elif not supported_pair:
        first_failure = "no_supported_public_pair"
    elif not stored_visible:
        first_failure = "target_pair_lost_by_last_pair_retention"
    elif row["prediction"] < 0:
        first_failure = "abstained_despite_stored_target_pair"
    elif row["prediction"] == 0:
        first_failure = "declared_absent_despite_stored_target_pair"
    elif row["prediction"] != true_label:
        first_failure = "wrong_fork_interval_or_side"
    elif not row["correct_images"]:
        first_failure = "wrong_slots"
    else:
        first_failure = "fork_localization_outside_tolerance"
    return {
        "index": index,
        "first_failure": first_failure,
        "ordered_physical_pair": candidate,
        "any_supported_public_pair": supported_pair,
        "any_pair_with_both_target_returns": visible,
        "full_packet_pair_with_both_target_returns_on_same_route": bool(original_visible),
        "retained_pair_with_both_target_returns": stored_visible,
        "chosen_pair_with_both_target_returns": chosen_pair_visible,
        "slot_only_oracle": bool(geometry_ok and chosen_pair_visible),
        "fork_coordinate_only_oracle": bool(row["prediction"] == true_label and row["correct_images"]),
        "label_and_fork_oracle_same_scan_sites": bool(chosen is not None and chosen_pair_visible
                                                     and (row["prediction"] - 1) // 2 == group and row["correct_images"]),
        "correct_label": bool(row["prediction"] == true_label),
        "success": bool(row["success"]),
    }


def audit(geometry, truth, results, costs, namespace, grid, prefix_length, suffix_limit, caps, fork_tol,
          nearest_arms, model=None, coverage_step=None, nearest_model=None, prefix_caps=None,
          legacy_mean=False):
    from itertools import combinations_with_replacement
    from return_sets import _nearest_indices

    bins = np.asarray(list(combinations_with_replacement(range(1 + 3 * grid), 3)))
    packet_numbers = {tuple(map(int, values)): i for i, values in enumerate(bins)}
    mapping = fine_to_coarse(grid)
    coordinates = None
    if model is not None:
        np.testing.assert_array_equal(model.bins, bins)
        coordinates = model.coordinates
    if nearest_arms and coordinates is None:
        from return_sets import _coarse_mapping

        _, coordinates = _coarse_mapping(grid)
    if nearest_arms:
        nearest_slot, nearest_packet = _nearest_indices(bins, coordinates)
    prefixes, tapes, by_arm = {}, {}, defaultdict(dict)
    discovery = {}
    diagnostics = defaultdict(list)
    total_steps, supported_steps = 0, 0
    for arm, row in records(results):
        index = row["index"]
        assert arm is not None and index not in by_arm[arm]
        if index not in tapes:
            tapes[index] = uniform_tape(namespace, index, (len(geometry.clock_s), 5, 7))
        tape = tapes[index]
        prefix = np.asarray(row["prefix"], dtype=np.int64)
        assert prefix.shape == (prefix_length, 2)
        assert row["prefix_hash"] == hashlib.sha256(prefix.tobytes()).hexdigest()
        if index in prefixes:
            np.testing.assert_array_equal(prefixes[index], prefix)
        else:
            prefixes[index] = prefix
        if model is not None:
            belief = np.full((1, len(model.labels)), 1. / len(model.labels))
        at = int(geometry.start)
        paid = np.zeros(3)
        for t, (node, packet) in enumerate(prefix, start=1):
            expected, _ = observe(truth, index, t, node, geometry.sites[node], tape, mapping, packet_numbers)
            assert packet == expected
            if model is not None:
                remaining = None if prefix_caps is None else (prefix_caps - paid)[None]
                command = coverage_step(belief, np.array([at], dtype=np.int64), t - 1,
                                        model.likelihood, model.addresses, model.labels, geometry.sites, geometry.cost,
                                        remaining)
                assert node == int(command[0])
                probability = model.likelihood[t, model.addresses[node], packet]
                b = belief[0] * probability
                if b.sum() > 1e-100:
                    belief[0] = b / b.sum()
            paid += costs[(at, node)][:3]
            at = node
        if prefix_caps is not None:
            assert np.all(paid <= prefix_caps + 1e-8)
        np.testing.assert_allclose(row["prefix_cost"], paid, atol=1e-8, rtol=0.0)
        if model is not None and index not in discovery:
            mass = np.bincount(model.labels, weights=belief[0], minlength=5)
            family_mass = np.bincount(model.families, weights=belief[0], minlength=4)
            true = int(truth.labels[index])
            discovery[index] = {
                "positive": true > 0,
                "binary_presence_correct": bool((mass[1:].sum() >= .5) == (true > 0)),
                "label_map_correct": int(np.argmax(mass)) == true,
                "positive_label_map_correct": bool(true > 0 and int(np.argmax(mass[1:])) + 1 == true),
                "family_map_correct": int(np.argmax(family_mass)) == int(truth.families[index]),
                "family": int(truth.families[index]),
                "true_label_mass": float(mass[true]),
            }
        suffix = np.zeros(3)
        trace = {}
        full_trace = {}
        pairs = [None, None]
        last = None
        suffix_model = nearest_model if arm in nearest_arms else model
        for expected_t, (t, node, packet, private, supported) in enumerate(row["trace"], start=prefix_length + 1):
            assert t == expected_t
            expected, kinds = observe(truth, index, t, node, geometry.sites[node], tape, mapping, packet_numbers)
            full_trace[t] = node, expected, kinds, supported
            if arm in nearest_arms:
                slot = nearest_slot[expected]
                kinds = [kinds[slot] if slot >= 0 else -1, -1, -1]
                expected = nearest_packet[expected]
            assert packet == expected and private == kinds, (index, arm, t, packet, expected, private, kinds)
            trace[t] = node, packet, kinds, supported
            if suffix_model is not None:
                likelihood = suffix_model.likelihood[t, suffix_model.addresses[node], packet]
                updated = belief[0] * likelihood
                expected_support = updated.sum() > 1e-100
                assert supported == expected_support
                if expected_support:
                    belief[0] = updated / updated.sum()
            if supported:
                supported_steps += 1
            if supported and packet > 0:
                now = (t, node, packet)
                for group in (0, 1):
                    if last is not None and geometry.sites[last[1]] == 4 + group and geometry.sites[node] == 3 + group:
                        pairs[group] = (last, now)
                last = now
            else:
                last = None
            suffix += costs[(at, node)][:3]
            assert np.all(suffix <= caps + 1e-8)
            at = node
            total_steps += 1
        assert len(trace) <= suffix_limit
        np.testing.assert_allclose(row["suffix_cost"], suffix, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose([row[name] for name in ("tcp_mm", "turn_deg", "rail_mm")],
                                   paid + suffix, atol=1e-8, rtol=0.0)
        capture = len(trace) + prefix_length
        assert row["captures"] == capture
        assert np.isclose(row["seconds"], geometry.clock_s[capture], atol=1e-10, rtol=0.0)
        endpoint(row, truth, trace, geometry.sites, prefix_length, fork_tol)
        if suffix_model is not None:
            check_posterior_endpoint(row, suffix_model, belief[0], pairs, fork_tol, legacy_mean)
        diagnostic = suffix_diagnostic(row, trace, pairs, geometry.sites, truth, index, fork_tol, full_trace)
        if diagnostic is not None:
            diagnostics[arm].append(diagnostic)
        by_arm[arm][index] = row
    assert by_arm
    expected_indices = set(range(len(truth.labels)))
    assert all(set(rows) == expected_indices for rows in by_arm.values())
    summary = {"prefixes": len(prefixes), "suffix_packets": total_steps,
               "supported_packets": supported_steps, "model_replayed": model is not None}
    if discovery:
        summary["distinct_prefix_node_sequences"] = len({tuple(row[:, 0]) for row in prefixes.values()})
        summary["distinct_prefix_view_sequences"] = len({tuple(geometry.views[row[:, 0]]) for row in prefixes.values()})
        summary["prescan_posterior_readout"] = {
            key: sum(row[key] for row in discovery.values()) for key in (
                "positive", "binary_presence_correct", "label_map_correct", "positive_label_map_correct", "family_map_correct",
            )
        }
        summary["prescan_posterior_readout"]["by_family"] = {
            family: {
                "n": sum(row["family"] == family for row in discovery.values()),
                "family_map_correct": sum(row["family"] == family and row["family_map_correct"] for row in discovery.values()),
                "mean_true_label_mass": mean(row["true_label_mass"] for row in discovery.values() if row["family"] == family),
            }
            for family in np.unique(truth.families).tolist()
        }
    summary["suffix_evidence_diagnostics_positive"] = {}
    for arm, rows in diagnostics.items():
        counts = defaultdict(int)
        for row in rows:
            counts[row["first_failure"]] += 1
        summary["suffix_evidence_diagnostics_positive"][arm] = {
            "n": len(rows),
            "disjoint_first_failure": dict(counts),
            **{key: sum(row[key] for row in rows) for key in rows[0] if key not in ("index", "first_failure")},
        }
    return by_arm, summary


def summarize(by_arm, baseline):
    from scipy.stats import beta

    def metrics(rows):
        return {
            "n": len(rows),
            "success": sum(row["success"] for row in rows),
            "correct_declaration": sum(row["correct_declaration"] for row in rows),
            "correct_images": sum(row["correct_images"] for row in rows),
            "wrong": sum(row["wrong"] for row in rows),
            "affirmative_error": sum(row["affirmative_error"] for row in rows),
            "abstain": sum(row["prediction"] == -1 for row in rows),
            "mean_cost": {name: mean(row[name] for row in rows)
                          for name in ("tcp_mm", "turn_deg", "rail_mm", "captures", "seconds")},
        }

    def contrast(rows, reference, cost_names):
        d = np.asarray([r["success"] - s["success"] for r, s in zip(rows, reference, strict=True)])
        wrong = np.asarray([r["affirmative_error"] - s["affirmative_error"]
                            for r, s in zip(rows, reference, strict=True)])
        cost = np.asarray([[r[name] - s[name] for name in cost_names]
                           for r, s in zip(rows, reference, strict=True)])
        rng = np.random.default_rng(723191)
        resample = rng.integers(0, len(rows), (16000, len(rows)))
        harmful = int(np.sum(wrong == 1))
        margins = np.array([2., 2., 5., .1])
        cost_ci = np.quantile(cost[resample].mean(axis=1),
                              [.05 / (2 * len(cost_names)), 1 - .05 / (2 * len(cost_names))], axis=0)
        return {
            "success_delta_pp": float(d.mean() * 100),
            "success_ci95_pp": np.quantile(d[resample].mean(axis=1) * 100, [.025, .975]).tolist(),
            "success_wins_losses": [int(np.sum(d == 1)), int(np.sum(d == -1))],
            "affirmative_harm_benefit": [harmful, int(np.sum(wrong == -1))],
            "harmful_one_sided_975_upper": float(beta.ppf(.975, harmful + 1, len(rows) - harmful))
            if harmful < len(rows) else 1.0,
            "cost_mean_delta": dict(zip(cost_names, cost.mean(axis=0).tolist())),
            "cost_ci95_simultaneous_bonferroni": dict(zip(cost_names, cost_ci.T.tolist())),
            "strict_cost_equivalence": bool(np.all(cost_ci[0] >= -margins) and np.all(cost_ci[1] <= margins)),
            "no_worse_upper_cost_margin": bool(np.all(cost_ci[1] <= margins)),
        }

    result = {}
    for arm, rows in by_arm.items():
        ordered = [rows[index] for index in sorted(rows)]
        table = {"all": metrics(ordered), "by_family": {
            str(family): metrics([row for row in ordered if row["family"] == family])
            for family in sorted({row["family"] for row in ordered})
        }}
        if baseline and arm != baseline:
            reference = [by_arm[baseline][index] for index in sorted(rows)]
            cost_names = ("tcp_mm", "turn_deg", "rail_mm", "seconds")
            table["contrast"] = {
                name: contrast([r for r in ordered if keep(r)], [r for r in reference if keep(r)], cost_names)
                for name, keep in (("all", lambda r: True), ("positive", lambda r: r["positive"]))
            }
        result[arm] = table
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "repo", "results", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("truth-seed", "count", "tape-namespace"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--grid", type=int, choices=(3, 5), default=5)
    parser.add_argument("--baseline")
    parser.add_argument("--nearest-arm", nargs="*", default=[])
    parser.add_argument("--prior-seed", type=int)
    parser.add_argument("--prior-size", type=int)
    parser.add_argument("--prior-split", default="train")
    parser.add_argument("--legacy-prescan", action="store_true")
    parser.add_argument("--legacy-terminal", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    sys.path.insert(0, str(args.repo.resolve() / "NMI_idea/experiments/ab_relational_b_20261113"))
    import core
    import world

    geometry = world.load(args.repo)
    truth = world.cases(geometry, args.truth_seed, args.count, args.split)
    _, cost, bank_report = recheck(args.repo / "NMI_idea/experiments")
    model, coverage_step, nearest_model, prefix_caps = None, None, None, None
    if args.prior_seed is not None:
        if args.legacy_prescan:
            from plan import coverage_step
        else:
            from prescan import coverage_step

            prefix_caps = core.PRESCAN_CAPS

        prior = world.cases(geometry, args.prior_seed, args.prior_size, args.prior_split)
        model, table = core.make_model(args.repo, geometry, prior, grid=args.grid)
        if args.nearest_arm:
            nearest_model = core.nearest_model(model, table)
    by_arm, info = audit(
        geometry, truth, args.results, cost, args.tape_namespace, args.grid,
        core.PREFIX, core.SUFFIX, core.CAPS, core.FORK_TOLERANCE, args.nearest_arm, model, coverage_step,
        nearest_model, prefix_caps, args.legacy_terminal,
    )
    result = {"reconstruction": info, "metrics": summarize(by_arm, args.baseline), "motion": bank_report}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
