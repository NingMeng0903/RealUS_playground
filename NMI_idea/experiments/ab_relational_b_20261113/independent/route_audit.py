"""Read-only replay of the frozen relational continuation benchmark."""

from collections import Counter
from functools import lru_cache
import hashlib
import json

import numpy as np
from scipy.stats import beta

import sensor_audit as reference

STUDY = reference.stage
TASKS = ("daughter", "trunk")
CONTEXTS = ("checkpoint", "probe")
POLICIES = ("MPC", "fixed", "nearest", "relations")
FIELDS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s",
          "rail_mm", "captures", "combined_s")
SAMPLED_PATH_CASES = {0, 7, 23, 47, 63}
SAMPLED_REPEAT_CASES = {0, 47}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_cohort(name):
    p = STUDY / "data" / name
    settings = json.loads((p / "settings.json").read_text())
    raw = reference.read_raw(p)
    rows = [json.loads(line) for line in raw.splitlines()]
    return settings, rows, raw


def seed_for(namespace, index):
    return int(np.random.SeedSequence([namespace, index]).generate_state(1)[0])


def draw_tape(seed, size):
    return np.random.default_rng(np.random.SeedSequence([seed, 68401])).random((size, 7))


def continue_tape(seed, old_tape, through):
    tape = draw_tape(seed, len(old_tape))
    tape[:through + 1] = old_tape[:through + 1]
    return tape


def design(task, route, public, desc, variant):
    keys = ("common", "nearest", "relations") if variant == "relations" else ("common", "nearest")
    x = {f"public.{k}.{name}": float(value) for k in keys for name, value in public[k].items()}
    x.update({f"route_evidence.{route}.{k}": value for k, value in tuple(x.items())})
    x.update({f"action.{k}": float(v) for k, v in desc["features"].items()})
    x.update({f"task.{task}": 1., f"action.{route}": 1., f"task_action.{task}.{route}": 1.})
    return x


def check_fit(fit, train):
    outputs = {}
    for ctx in CONTEXTS:
        fixed = {}
        cases = [(r["task"], r["contexts"][ctx]) for r in train]
        for task in TASKS:
            group = [c for t, c in cases if t == task]
            scores = {}
            for route in ("advance", "view"):
                scores[route] = float(np.mean([
                    np.mean(np.asarray(c["repeat"].get(route, c["repeat"]["MPC"]))[:, 0]
                            - np.asarray(c["repeat"]["MPC"])[:, 0]) for c in group]))
            best = max(("MPC", "advance", "view"), key=lambda k: scores.get(k, 0.))
            assert fit["contexts"][ctx]["fixed"][task] == best
            fixed[task] = scores | {"selected": best}
        outputs[ctx] = {"fixed_all_case_training_gains": fixed, "rows": {}}
        for variant in ("nearest", "relations"):
            frozen = fit["contexts"][ctx]["variants"][variant]
            samples = []
            for task, c in cases:
                options = sorted(set(c["actions"]) - {"MPC"})
                base = np.asarray(c["repeat"]["MPC"])[:, 0]
                for k in options:
                    outcomes = np.asarray(c["repeat"][k])[:, 0]
                    samples.append((design(task, k, c["public"], c["actions"][k], variant),
                                    (outcomes - base).mean(), 1 / len(options)))
            names = sorted(set.union(*(set(x) for x, _, _ in samples)))
            assert names == frozen["features"] and len(samples) == frozen["n_examples"]
            assert not any("eval_only" in k or "seed" in k or "repeat" in k for k in names)
            assert not any("public.relations" in k for k in names) if variant == "nearest" else True
            assert frozen["ridge"] == 4.
            x = np.asarray([[d.get(k, 0.) for k in names] for d, _, _ in samples])
            y = np.asarray([v for _, v, _ in samples])
            w = np.asarray([v for _, _, v in samples])
            w /= w.sum()
            center = w @ x
            scale = np.maximum(np.sqrt(w @ ((x - center) ** 2)), .05)
            z = (x - center) / scale
            slopes = np.asarray(frozen["coef"]) * scale
            residual = z.T @ (w * (y - z @ slopes)) - 4. * slopes
            assert np.linalg.norm(residual, ord=np.inf) < 1e-12
            np.testing.assert_allclose(center, frozen["center"], atol=1e-12, rtol=1e-13)
            np.testing.assert_allclose(scale, frozen["scale"], atol=1e-12, rtol=1e-13)
            np.testing.assert_allclose(w @ y - center @ frozen["coef"], frozen["intercept"], atol=1e-12)
            outputs[ctx]["rows"][variant] = len(samples)
    return outputs


def check_scores(task, context_name, c, fit, stats):
    f = fit["contexts"][context_name]
    options = c["actions"]
    public = c["public"]
    fixed = f["fixed"][task]
    assert c["fresh"]["fixed"]["route"] == (fixed if fixed in options else "MPC")
    assert c["fresh"]["MPC"]["route"] == "MPC"
    for variant in ("nearest", "relations"):
        model = f["variants"][variant]
        scores = {"MPC": 0.}
        best = "MPC"
        for k in sorted(set(options) - {"MPC"}):
            x = design(task, k, public, options[k], variant)
            scores[k] = float(model["intercept"] + sum(v * x.get(n, 0.)
                              for n, v in zip(model["features"], model["coef"])))
            if scores[k] > scores[best] + 1e-12:
                best = k
        assert c["fresh"][variant]["route"] == best
        assert scores.keys() == c["forecasts"][variant].keys()
        for k, score in scores.items():
            np.testing.assert_allclose(score, c["forecasts"][variant][k], atol=1e-12)
            stats["frozen scores"] += 1
    for p in POLICIES:
        picked = c["fresh"][p]
        assert picked["goal"] == options[picked["route"]]["goal"]
        stats["frozen choices"] += 1


def moment_features(b, m):
    return {"displacement": float(b @ m), "log_ess": float(-np.log(b @ b)),
            "displacement_positive": float(b[m >= 0].sum())}


def snapshot(data, geom, core, before, after, step):
    s = data["scene"]
    b, good, reserve, at, slot, tick, done, _ = after
    prior = before[0]
    common = {"progress": done, "slot": slot, "clock": tick, "site": s.site[at],
              "tilt": s.tilt[at], "rho": s.rho[at], "before_good": before[1].sum(),
              "target_offset": s.targets[done] - s.site[at] if done < len(s.targets) else 0}
    common.update({f"resource{i}": v for i, v in enumerate(reserve)})
    common.update({f"before_{k}": v for k, v in moment_features(prior, s.material_coordinate).items()})
    if step is None:
        return {"common": common, "nearest": {}, "relations": {}}
    m, packets, _, _, comp = geom
    packet = int(step[5])
    grid = int(m.max() // 3)
    px = np.array([(reference.X[(m.reshape(7, 7)[:, 0] - 1) // 3 == i].mean(),
                    reference.Z[(m.reshape(7, 7)[0] - 1) % 3 == j].mean())
                   for i in range(grid) for j in range(3)])
    obs = comp[packet]
    near = px[(obs - 1) % (grid * 3)] if obs else [0., 0.]
    _, _, l, j = core(tick, s.to_address[at])
    close_b, _, _, confidence, _ = reference.update(prior, before[1], l[obs], j[obs, None], False)
    nearest = {"present": int(obs > 0), "multiple": int(obs > grid * 3),
               "nearest_x": near[0], "nearest_z": near[1], "nearest_confidence": confidence}
    nearest.update({f"nearest_{k}": v for k, v in moment_features(close_b, s.material_coordinate).items()})
    visible = [value for value in packets[packet] if value]
    coords = px[np.asarray(visible, dtype=int) - 1]
    spread = np.ptp(coords, axis=0) if len(visible) else [0., 0.]
    pairwise = [np.linalg.norm(a - b) for a in coords for b in coords]
    relations = {"return_count": len(visible), "span_x": spread[0], "span_z": spread[1],
                 "max_pair_distance": max(pairwise, default=0.), "set_confidence": step[11]}
    relations.update({f"bin{i}": visible.count(i) for i in range(1, grid * 3 + 1)})
    relations.update({f"set_{k}": v for k, v in moment_features(b, s.material_coordinate).items()})
    return {"common": common, "nearest": nearest, "relations": relations}


def witness(s, at, slots, done, budget):
    if done == len(s.targets):
        return ()
    if slots < len(s.targets) - done:
        return None
    feasible = np.flatnonzero(np.all(s.edge_costs[at] <= budget + 1e-8, axis=1))
    for goal in feasible:
        hits = int(s.site[goal] == s.targets[done])
        if goal == at and not hits:
            continue
        suffix = witness(s, goal, slots - 1, done + hits, budget - s.edge_costs[at, goal])
        if suffix is not None:
            return (goal, *suffix)
    return None


def expected_probe(s, state, eligible):
    _, _, r, at, slot, _, done, _ = state
    if not eligible:
        return -1
    candidates = []
    for j in np.flatnonzero(s.site == s.site[at]):
        if j != at and s.tilt[j] == s.tilt[at]:
            continue
        expense = s.edge_costs[at, j]
        if np.all(expense <= r + 1e-8) and witness(s, j, 5 - slot, done, r - expense) is not None:
            candidates.append(j)
    changed = [j for j in candidates if j != at]
    return min(changed, key=lambda j: (*s.edge_costs[at, j], j)) if changed else at if at in candidates else -1


def expected_options(s, opt, state, eligible):
    result = {"MPC": {"goal": -1, "features": {}, "witness": None}}
    b, good, r, at, slot, t, done, _ = state
    if not eligible or done == len(s.targets) or slot == 6:
        return result
    incumbent = opt(at, slot, t, done, r, b, good)
    options = {}
    for j in np.flatnonzero(s.site == s.targets[done]):
        cost = s.edge_costs[at, j]
        if np.all(cost <= r + 1e-8):
            suffix = witness(s, j, 5 - slot, done + 1, r - cost)
            if suffix is not None:
                options[j] = [j, *suffix]
    center = [j for j in options if s.tilt[j] == 0]
    tilt = [j for j in options if s.tilt[j] != 0]
    opposite = [j for j in tilt if s.tilt[j] == -s.tilt[at]]
    for key, goals in (("advance", center), ("view", opposite or tilt)):
        if not goals:
            continue
        j = min(goals, key=lambda v: (*s.edge_costs[at, v], v))
        if j != incumbent:
            travel, turn, rail, ticks = s.edge_costs[at, j]
            result[key] = {"goal": j, "witness": options[j],
                           "features": {"travel": travel, "turn": turn, "rail": rail,
                                        "duration": ticks * s.clock_unit, "tilt": s.tilt[j],
                                        "rho_change": int(s.rho[j] != s.rho[at]),
                                        "same_site": int(s.site[j] == s.site[at])}}
    return result


def step_forward(s, geom, core, d, c, tape, state, goal, forced, recorded=None):
    b, good, r, at, slot, t, done, wrong = state
    cost = s.edge_costs[at, goal]
    assert np.all(cost <= r + 1e-8)
    assert slot < 6 and done < len(s.targets)
    t += int(cost[3])
    a = s.to_address[goal]
    packet, kinds, nearest, _, _ = reference.draw(d, c, tape, t, a, geom)
    l, j, _, _ = core(t, a)
    b, good, chosen, conf, claim = reference.update(b, good, l[packet], j[packet],
                                                     s.site[goal] == s.targets[done])
    if recorded is not None and int(recorded[6]) != chosen:
        slots = geom[1][packet]
        assert conf < .87 and slots[int(recorded[6])] == slots[chosen]
        chosen = int(recorded[6])
    kind = kinds[chosen] if chosen >= 0 else -1
    row = (at, goal, slot, t, packet, packet, chosen, kind, nearest, 0 in kinds,
           int(claim), conf, *cost[:3], int(forced))
    if recorded is not None:
        np.testing.assert_allclose(row, recorded, atol=1e-12, rtol=0.)
    return (b, good, r - cost, goal, slot + 1, t, done + int(claim),
            wrong or (claim and kind != 0)), row


def measurements(data, state):
    s = data["scene"]
    _, _, r, _, slot, t, done, wrong = state
    cost = s.reserve - r
    return (int(done == len(s.targets) and not wrong), int(wrong), *cost[:2],
            data["clocks"][t], cost[2], slot)


def replay_row(data, geom, core, row, settings, fit, stats):
    s = data["scene"]
    seed = seed_for(settings["namespace"], row["index"])
    assert row["case_seed"] == seed
    case = reference.acoustics.make_scenario(data["geometry"].world, seed)
    _, _, d, c = reference.acoustics.probabilities(data["geometry"], case, data["clocks"], "joint")
    original = draw_tape(seed, len(data["clocks"]))
    verify_actions = row["index"] in SAMPLED_PATH_CASES
    verify_repeats = row["index"] in SAMPLED_REPEAT_CASES
    task = row["task"]

    @lru_cache(None)
    def cached_action(at, slot, t, done, r, b, good):
        return reference.action(s, lambda t, a: core(t, a)[:2], at, slot, t, done,
                                np.frombuffer(r), np.frombuffer(b), np.frombuffer(good))

    def opt(at, slot, t, done, r, b, good):
        return cached_action(at, slot, t, done, r.tobytes(), b.tobytes(), good.tobytes())

    def current(st):
        return opt(st[3], st[4], st[5], st[6], st[2], st[0], st[1])

    def verify(st, tape, goal, logged):
        tail = logged["tail"]
        for i, edge in enumerate(tail):
            forced = i == 0 and goal >= 0
            j = int(edge[1])
            if forced:
                assert j == goal
            elif verify_actions:
                assert j == current(st)
                stats["independent fresh MPC actions"] += 1
            st, _ = step_forward(s, geom, core, d, c, tape, st, j, forced, edge)
            stats["fresh packet transitions"] += 1
        if verify_actions:
            assert current(st) == -1
        np.testing.assert_allclose(measurements(data, st), logged["measured"][:7], atol=1e-12, rtol=0.)
        stats["fresh terminal paths"] += 1

    def repeat(st, tape, forced, recorded):
        first = True
        while st[4] < 6 and st[6] < len(s.targets):
            is_forced = first and forced >= 0
            goal = forced if is_forced else current(st)
            if goal < 0:
                break
            st, _ = step_forward(s, geom, core, d, c, tape, st, goal, is_forced)
            first = False
        np.testing.assert_allclose(measurements(data, st), recorded[:7], atol=1e-12, rtol=0.)
        stats["sampled repeat terminal paths"] += 1

    addr = s.to_address[s.start]
    packet, *_ = reference.draw(d, c, original, 0, addr, geom)
    assert packet == row["anchor"]
    l, j, _, _ = core(0, addr)
    prior = np.full(s.likelihood.shape[-1], 1. / s.likelihood.shape[-1])
    b, g, _, _, _ = reference.update(prior, prior, l[packet], j[packet], False)
    before = state = (b, g, s.reserve.copy(), s.start, 0, 0, 0, False)
    last = None
    for edge in row["prefix"]:
        before = state
        goal = int(edge[1])
        if verify_actions:
            assert goal == current(state)
            stats["independent prefix MPC actions"] += 1
        state, last = step_forward(s, geom, core, d, c, original, state, goal, False, edge)
        stats["common prefix packet transitions"] += 1
    eligible = state[4] == 2 and state[6] < len(s.targets)
    assert expected_probe(s, state, eligible) == row["probe_goal"]
    probe = row["probe_goal"]
    has_probe = probe >= 0
    if has_probe:
        pseed = seed_for(seed, 300000)
        ptape = continue_tape(pseed, original, state[5])
        assert len(row["probe_trace"]) == 1
        exit_state, probe_edge = step_forward(s, geom, core, d, c, ptape, state, probe, True,
                                              row["probe_trace"][0])
        stats["probe packet transitions"] += 1
    else:
        pseed, ptape, exit_state, probe_edge = seed, original, state, None
        assert not row["probe_trace"]
    assert pseed == row["probe_prefix_seed"]
    inputs = (("checkpoint", before, state, last if state[4] == 2 else None, original, eligible, 210000),
              ("probe", state, exit_state, probe_edge, ptape,
               has_probe and exit_state[6] == state[6], 310000))
    for name, prior_state, now, edge, tape, can_override, base_index in inputs:
        stored = row["contexts"][name]
        expected = snapshot(data, geom, core, prior_state, now, edge)
        for field, values in expected.items():
            assert values.keys() == stored["public"][field].keys()
            for key, v in values.items():
                np.testing.assert_allclose(v, stored["public"][field][key], atol=1e-12, rtol=0.)
        stats["public feature packets"] += 1
        options = expected_options(s, opt, now, can_override)
        assert options == stored["actions"]
        stats["public action libraries"] += 1
        stats["individual full witnesses"] += len(options) - 1
        suffix = [seed_for(seed, base_index + i) for i in range(settings["n_repeats"])]
        assert suffix == stored["repeat_seeds"]
        assert len(set(suffix)) == settings["n_repeats"] and not ({seed, pseed} & set(suffix))
        stats["suffix tape keys"] += len(suffix)
        if verify_repeats:
            for i, rseed in enumerate(suffix):
                repeat_tape = continue_tape(rseed, tape, now[5])
                for k, desc in options.items():
                    repeat(now, repeat_tape, desc["goal"], stored["repeat"][k][i])
                if name == "probe":
                    repeat(state, repeat_tape, -1, stored["paired_no_probe"][i])
        if "fresh" in stored:
            fseed = seed_for(seed, base_index + 1000000)
            assert fseed not in suffix and fseed not in (seed, pseed)
            assert fseed == stored["fresh_seed"]
            ftape = continue_tape(fseed, tape, now[5])
            assert digest(ftape.tobytes()) == stored["fresh_tape_sha256"]
            stats["fresh tape hashes"] += 1
            check_scores(task, name, stored, fit, stats)
            for p in POLICIES:
                logged = stored["fresh"][p]
                verify(now, ftape, logged["goal"], logged)
            verify(state, ftape, -1, stored["fresh_no_probe"])
            if name == "probe":
                free_goal = row["contexts"]["checkpoint"]["fresh"]["relations"]["goal"]
                verify(state, ftape, free_goal, stored["fresh_free_relations"])
    cached_action.cache_clear()


def compare_stats(candidate, baseline, reported, boot_indices):
    delta = np.asarray(candidate) - np.asarray(baseline)
    if delta.ndim == 3:
        delta = delta.mean(axis=1)
        repeated = True
    else:
        repeated = False
    ci = np.quantile(delta[boot_indices].mean(axis=1), [.025, .975], axis=0)
    simultaneous = np.quantile(delta[boot_indices].mean(axis=1),
                               [.05 / 6, 1 - .05 / 6], axis=0)
    for i, k in enumerate(FIELDS):
        np.testing.assert_allclose(delta[:, i].mean(), reported["delta"][k], atol=1e-12)
        np.testing.assert_allclose(ci[:, i], reported["delta_ci95_case_clustered"][k], atol=1e-12)
        if k in ("tcp_mm", "turn_deg", "motion_s"):
            np.testing.assert_allclose(simultaneous[:, i], reported["cost_bonferroni_ci95"][k], atol=1e-12)
    for i, field in ((0, "success_wins_losses_ties"), (1, "wrong_harm_benefit_ties_cases")):
        counts = [(delta[:, i] > 0).sum(), (delta[:, i] < 0).sum(), (delta[:, i] == 0).sum()]
        assert counts == reported[field]
    if not repeated:
        harmed = int((delta[:, 1] > 0).sum())
        upper = beta.ppf(.975, harmed + 1, len(delta) - harmed) if harmed < len(delta) else 1.
        np.testing.assert_allclose(upper, reported["wrong_harm_975_upper"], atol=1e-12)


def check_summary(rows, report, stats):
    for task in TASKS:
        for name in CONTEXTS:
            cases = [r["contexts"][name] for r in rows if r["task"] == task]
            count = len(cases)
            out = report[task][name]
            boot = np.random.default_rng(57216).integers(0, count, size=(20_000, count))
            repeats = {p: np.array([r["repeat"][r["fresh"][p]["route"]] for r in cases])
                       for p in POLICIES}
            fresh = {p: np.array([r["fresh"][p]["measured"] for r in cases]) for p in POLICIES}
            raw_no = [c["paired_no_probe"] for c in cases]
            fresh_no = [c["fresh_no_probe"]["measured"] for c in cases]
            checks = [(repeats["relations"], repeats["nearest"], out["relations_vs_same_packet_nearest"]),
                      (repeats["relations"], repeats["fixed"], out["relations_vs_train_fixed"])]
            for p in POLICIES:
                if p != "MPC":
                    checks += [(repeats[p], repeats["MPC"], out["repeated_vs_exit_mpc"][p]),
                               (fresh[p], fresh["MPC"], out["fresh_vs_exit_mpc"][p])]
                checks += [(repeats[p], raw_no, out["repeated_vs_no_probe"][p]),
                           (fresh[p], fresh_no, out["fresh_vs_no_probe"][p])]
            if name == "probe":
                free = [c["fresh_free_relations"]["measured"] for c in cases]
                checks.append((fresh["relations"], free, out["paid_vs_free_relations_same_tape"]))
            oracle = []
            picks = Counter()
            for case in cases:
                values = {k: np.asarray(v) for k, v in case["repeat"].items()}
                rank_by_first = max(values, key=lambda k: values[k][:8, 0].sum())
                rank_by_second = max(values, key=lambda k: values[k][8:, 0].sum())
                picks.update([rank_by_first, rank_by_second])
                oracle.append(np.concatenate([values[rank_by_second][:8], values[rank_by_first][8:]]))
            assert picks == out["ranking_half_choices"]
            checks.append((oracle, repeats["MPC"], out["opportunity"]))
            for a, b, summary in checks:
                compare_stats(a, b, summary, boot)
                stats["verified paired contrast summaries"] += 1


def main():
    freeze = json.loads((STUDY / "FREEZE.json").read_bytes())
    for name, sha in freeze["sha256"].items():
        assert digest((STUDY / name).read_bytes()) == sha, name
    fit = json.loads((STUDY / "route_fit.json").read_bytes())
    a_settings, train, raw = read_cohort("train")
    b_settings, heldout, _ = read_cohort("heldout")
    assert digest(raw) == freeze["training_raw_sha256"]
    assert a_settings["namespace"] == freeze["training_namespace"]
    assert b_settings["namespace"] == freeze["namespace"] and b_settings["heldout_frozen"]
    assert len(train) == 2 * freeze["n_training"] and len(heldout) == 2 * freeze["n_heldout"]
    assert {r["case_seed"] for r in train}.isdisjoint(r["case_seed"] for r in heldout)
    assert a_settings["n_repeats"] == b_settings["n_repeats"] == freeze["repeats"]
    fit_audit = check_fit(fit, train)
    print(json.dumps({"freeze_sha256": digest((STUDY / "FREEZE.json").read_bytes()),
                      "fit": fit_audit}, sort_keys=True), flush=True)
    reference.initialize(a_settings)
    geom = reference.geometry(freeze["grid"])
    stats = Counter()
    for task, data in reference.pinned.DATA.items():
        s = data["scene"]
        @lru_cache(None)
        def core(clock, address):
            return reference.kernels(s.detect[clock, address], s.cdf[clock, address], freeze["grid"], geom)
        for cases, settings in ((train, a_settings), (heldout, b_settings)):
            for row in (r for r in cases if r["task"] == task):
                replay_row(data, geom, core, row, settings, fit, stats)
        core.cache_clear()
        print(f"Replayed {task} public inputs and outcomes", flush=True)
    report = json.loads((STUDY / "data/heldout/summary.json").read_bytes())
    check_summary(heldout, report, stats)
    print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
