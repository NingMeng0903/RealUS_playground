"""Independent acoustic kernel, sensor packet, and MPC replay audit."""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from pathlib import Path
import hashlib, json, lzma, sys
from tempfile import TemporaryDirectory
import numpy as np

stage = Path(__file__).resolve().parents[1]
repo = stage.parents[2]
full = repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"
sys.path[:0] = [str(stage), str(full)]
import acoustics
import return_sets
import run as pinned

X = np.asarray([-10., -6., -2.75, 0., 2.75, 6., 10.])
Z = np.asarray([-7., -3.5, -1.35, 0., 1.35, 3.5, 7.])

def geometry(grid):
    x = np.searchsorted([-1.5, 1.5] if grid == 3 else [-4., -1.5, 1.5, 4.], X)
    z = np.searchsorted([-.7, .7], Z)
    m = (3 * x[:, None] + z[None] + 1).ravel()
    xy = np.array([(X[x == i].mean(), Z[z == j].mean()) for i in range(grid) for j in range(3)])
    dist = np.square(xy / [4., 3.]).sum(axis=1)
    packets = list(combinations_with_replacement(range(3 * grid + 1), 3))
    nearest = []
    compressed = []
    for p in packets:
        visible = [(dist[v - 1], v, k) for k, v in enumerate(p) if v]
        if visible:
            _, v, k = min(visible)
            nearest.append(k)
            compressed.append(v + 3 * grid * (len(visible) > 1))
        else:
            nearest.append(-1)
            compressed.append(0)
    return m, packets, {p: i for i, p in enumerate(packets)}, np.array(nearest), np.array(compressed)

def kernels(d, c, grid, geom):
    m, packets, _, nearest, comp = geom
    pmf = np.diff(np.concatenate([np.zeros((len(d), 3, 1)), c], axis=2), axis=2)
    f = np.empty((len(d), 3, 3 * grid + 1))
    f[..., 0] = 1. - d
    for i in range(1, f.shape[-1]):
        f[..., i] = d * pmf[..., m == i].sum(axis=2)
    like = np.zeros((len(packets), len(d)))
    assoc = np.zeros((len(packets), 3, len(d)))
    for i, p in enumerate(packets):
        for assignment in sorted(set(permutations(p))):
            weight = f[:, 0, assignment[0]] * f[:, 1, assignment[1]] * f[:, 2, assignment[2]]
            like[i] += weight
            if assignment[0]:
                slots = [j for j, value in enumerate(p) if value == assignment[0]]
                assoc[i, slots] += weight / len(slots)
    near_like = np.zeros((1 + 6 * grid, len(d)))
    near_assoc = np.zeros_like(near_like)
    np.add.at(near_like, comp, like)
    v = np.flatnonzero(nearest >= 0)
    np.add.at(near_assoc, comp[v], assoc[v, nearest[v]])
    return like, assoc, near_like, near_assoc

def draw(d, c, tape, t, a, geom):
    m, _, ids, nearest, _ = geom
    px = {k: min(48, int(np.searchsorted(c[t, a, k], tape[t, k + 3])))
          for k in range(3) if tape[t, k] < d[t, a, k]}
    values = tuple(sorted([m[i] for i in px.values()] + [0] * (3 - len(px))))
    packet = ids[values]
    orders = sorted({q for q in permutations(list(px) + [-1] * (3 - len(px)))
                     if tuple(m[px[k]] if k >= 0 else 0 for k in q) == values})
    u = tape[t, 6]
    kinds = orders[min(len(orders) - 1, int(u * len(orders)))]
    near_kind = kinds[nearest[packet]] if nearest[packet] >= 0 else -1
    if px:
        ds = {k: (X[p // 7] / 4.)**2 + (Z[p % 7] / 3.)**2 for k, p in px.items()}
        choices = [k for k, v in ds.items() if abs(v - min(ds.values())) <= 1.e-12]
        old_kind = choices[int(u * len(choices))]
        old_packet = 1 + px[old_kind] + 49 * (len(px) > 1)
    else:
        old_kind, old_packet = -1, 0
    return packet, kinds, near_kind, old_packet, old_kind

def update(b, good, like, assoc, eligible):
    prob = b @ like
    if prob <= 1.e-260:
        return b.copy(), good.copy(), -1, 0., False
    scores = assoc @ b
    j = int(np.argmax(scores)) if np.any(scores > 0) else -1
    conf = float(scores[j] / prob) if j >= 0 else 0.
    claim = bool(eligible and conf >= .87 and j >= 0)
    return b * like / prob, good * (assoc[j] if claim else like) / prob, j, conf, claim

def expectation(b, good, l, j):
    mask = np.any((j >= (.87 - 1e-12) * l[:, None]) & (l[:, None] > 1e-260), axis=(1, 2))
    l, j = l[mask], j[mask]
    p = b @ l.T
    flat = j.reshape(j.shape[0] * j.shape[1], j.shape[2])
    projections = (b @ flat.T).reshape(b.shape[0], j.shape[0], j.shape[1])
    slot = projections.argmax(axis=2)
    identity = np.take_along_axis(projections, slot[..., None], axis=2)[..., 0]
    claim = (p > 1e-260) & (identity >= .87 * p)
    right = (good @ flat.T).reshape(b.shape[0], j.shape[0], j.shape[1])
    right = np.take_along_axis(right, slot[..., None], axis=2)[..., 0]
    wrong = np.maximum(0, good @ l.T - right)
    return (right * claim).sum(axis=1), (wrong * claim).sum(axis=1)

def action(s, source, at, slot, t, done, r, b, good):
    cap = len(s.potential) - 1
    if done == len(s.targets) or slot >= cap:
        return -1
    prices = np.array([.00035, .00015, .00015])
    goals = np.flatnonzero(np.all(s.edge_costs[at] <= r + 1e-8, axis=1))
    selected, best = -1, -np.inf
    def value(left, node, progress, res):
        return np.where(np.all(s.needs[left, node, progress] <= res[None] + 1e-8, axis=1),
                        s.potential[left, node, progress], 0.)
    def consider(score, goal):
        nonlocal best, selected
        if score > best + 1e-12:
            best, selected = score, goal
    for first in goals:
        cost = s.edge_costs[at, first]
        t1, a1 = t + int(cost[3]), s.to_address[first]
        l, j = source(t1, a1)
        price = cost[:3] @ prices
        if slot + 1 == cap:
            if s.site[first] == s.targets[done] and done + 1 == len(s.targets):
                yes, no = expectation(b[None], good[None], l, j)
                consider(float(yes[0] - no[0] - price), first)
            continue
        r1 = r - cost
        cumulative = np.cumsum(l @ b)
        cumulative[-1] = 1.
        chosen = np.searchsorted(cumulative, (np.arange(10) + .5) / 10.)
        next_b, next_good, lost, progress = [], [], [], []
        for p in chosen:
            nb, ng, _, _, claim = update(b, good, l[p], j[p], s.site[first] == s.targets[done])
            next_b.append(nb)
            next_good.append(ng)
            lost.append(max(0., ((good @ l[p]) / (b @ l[p]) if b @ l[p] > 1e-260 else good.sum()) - ng.sum()))
            progress.append(done + int(claim))
        nb, ng, loss, progress = map(np.asarray, (next_b, next_good, lost, progress))
        gmass = ng.sum(axis=1)
        consider(float((gmass * value(cap - slot - 1, first, progress, r1) - loss).mean() - price), first)
        live = progress < len(s.targets)
        for second in np.flatnonzero(np.all(s.edge_costs[first] <= r1 + 1e-8, axis=1)):
            cost2 = s.edge_costs[first, second]
            r2 = r1 - cost2
            v = gmass.copy() - loss
            v[live] = gmass[live] * value(cap - slot - 2, second, progress[live], r2) - loss[live] - cost2[:3] @ prices
            matching = live & (s.site[second] == s.targets[np.minimum(progress, len(s.targets) - 1)])
            if np.any(matching):
                l2, j2 = source(t1 + int(cost2[3]), s.to_address[second])
                yes, no = expectation(nb[matching], ng[matching], l2, j2)
                same = value(cap - slot - 2, second, progress[matching], r2)
                changed = value(cap - slot - 2, second, progress[matching] + 1, r2)
                v[matching] += yes * changed - (yes + no) * same - no
            consider(float(v.mean() - price), first)
    return int(selected) if best > 0. else -1

def initialize(settings):
    archive = lzma.decompress((full / "motion/BANK_MAIN.json.xz").read_bytes())
    assert hashlib.sha256(archive).hexdigest() == settings["bank_sha256"]
    assert all(hashlib.sha256((full / f).read_bytes()).hexdigest() == h for f, h in settings["sources"].items())
    with TemporaryDirectory() as directory:
        bank = Path(directory) / "BANK_MAIN.json"
        bank.write_bytes(archive)
        pinned.initialize(repo, bank, settings["model_size"], 6, "joint")


def read_raw(folder):
    return lzma.decompress((folder / "cases.jsonl.xz").read_bytes())


def main(sensor):
    settings = json.loads((sensor / "settings.json").read_text())
    rows = [json.loads(line) for line in read_raw(sensor).splitlines()]
    initialize(settings)
    counts = Counter()
    g = geometry(settings["grid"])
    m, packets, ids, nearest, comp = g
    for task, data in pinned.DATA.items():
        s = data["scene"]
        t = np.array([0, 3, 6])
        a = np.linspace(0, s.detect.shape[1] - 1, 5, dtype=int)
        h = np.linspace(0, s.detect.shape[2] - 1, 19, dtype=int)
        d = np.ascontiguousarray(s.detect[np.ix_(t, a, h)])
        c = np.ascontiguousarray(s.cdf[np.ix_(t, a, h)])
        for grid in (3, 5):
            geom = geometry(grid)
            actual = return_sets.build_tables(d, c, grid)
            np.testing.assert_array_equal(actual.mapping, geom[0])
            np.testing.assert_array_equal(actual.bins, geom[1])
            np.testing.assert_array_equal(actual.nearest_slot, geom[3])
            np.testing.assert_array_equal(actual.nearest_packet, geom[4])
            for i in range(len(t)):
                for j in range(len(a)):
                    expected = kernels(d[i, j], c[i, j], grid, geom)
                    actual_rows = (actual.total[i, j], actual.joint[i, j],
                                   actual.nearest_total[i, j], actual.nearest_correct[i, j])
                    for x, y in zip(actual_rows, expected):
                        np.testing.assert_allclose(x, y, atol=4e-15, rtol=1e-13)
                    np.testing.assert_allclose(expected[0].sum(axis=0), 1., atol=2e-14, rtol=0.)
                    np.testing.assert_allclose(expected[1].sum(axis=(0, 1)), d[i, j, :, 0], atol=2e-14, rtol=0.)
                    assert np.all(expected[1].sum(axis=1) <= expected[0] + 4e-15)
                    counts["model clock/address/atom slices"] += len(h)
                    counts["model packet and slot mass cells"] += len(h) * len(geom[1]) * 4
    for pixels in ((24, 24, 24), (24, 24, 48), (24, 48, 48), (0, 24, 48)):
        values = tuple(sorted(m[list(pixels)]))
        orders = {q for q in permutations(range(3)) if tuple(m[pixels[k]] for k in q) == values}
        d, c = np.ones((1, 1, 1, 3)), np.zeros((1, 1, 1, 3, 49))
        for k, p in enumerate(pixels):
            c[0, 0, 0, k, p:] = 1.
        table = return_sets.build_tables(d, c, settings["grid"])
        seen = Counter()
        for u in (np.arange(12) + .5) / 12.:
            p, k = return_sets.draw(d, c, 0, 0, 0, np.array([[.3] * 6 + [u]]), m, table.lookup)
            assert p == ids[values] and tuple(k) in orders
            seen[tuple(k)] += 1
        assert set(seen) == orders and len(set(seen.values())) == 1
        counts["tie-order tape tests"] += 12

    for task, data in pinned.DATA.items():
        s = data["scene"]
        @lru_cache(None)
        def rich(clock, address):
            return kernels(s.detect[clock, address], s.cdf[clock, address], settings["grid"], g)
        def model(mode, t, a):
            if mode == 0:
                return s.likelihood[t, a], s.correct[t, a][:, None]
            l, j, nl, nj = rich(t, a)
            return (nl, nj[:, None]) if mode == 1 else (l, j)
        for row in (r for r in rows if r["task"] == task):
            seed = int(np.random.SeedSequence([settings["namespace"], row["index"]]).generate_state(1)[0])
            assert seed == row["case_seed"]
            case = acoustics.make_scenario(data["geometry"].world, seed)
            _, _, d, c = acoustics.probabilities(data["geometry"], case, data["clocks"], "joint")
            tape = np.random.default_rng(np.random.SeedSequence([seed, 68401])).random((len(data["clocks"]), 7))
            assert hashlib.sha256(tape.tobytes()).hexdigest() == row["tape_sha256"]
            counts["paired tapes"] += 1
            for mode, arm in enumerate(settings["arms"]):
                log = row["arms"][arm]
                at, t, done, bad = s.start, 0, 0, False
                r = s.reserve.copy()
                b = good = np.full(s.likelihood.shape[-1], 1. / s.likelihood.shape[-1])
                a = s.to_address[at]
                p, _, _, op, _ = draw(d, c, tape, 0, a, g)
                assert p == log["anchor"]
                obs = [op, comp[p], p][mode]
                l, j = model(mode, 0, a)
                b, good, _, _, _ = update(b, good, l[obs], j[obs], False)
                counts["logged anchor packets"] += 1
                for slot, edge in enumerate(log["trace"]):
                    start, goal, ix, tick, obs, rich_obs, chosen, kind, nearest_kind, present, claim = map(int, edge[:11])
                    assert ix == slot and start == at and edge[15] == 0.
                    if sensor.name == "smoke":
                        a0 = action(s, lambda t, a: model(mode, t, a), at, slot, t, done, r, b, good)
                        assert a0 == goal, (row["index"], task, arm, slot, a0, goal)
                        counts["independent MPC actions"] += 1
                    expense = s.edge_costs[at, goal]
                    assert np.all(expense <= r + 1e-8)
                    np.testing.assert_allclose(edge[12:15], expense[:3], atol=1e-13, rtol=0.)
                    t += int(expense[3])
                    a = s.to_address[goal]
                    p, kinds, nk, op, ok = draw(d, c, tape, t, a, g)
                    assert (t, p, nk, 0 in kinds) == (tick, rich_obs, nearest_kind, present)
                    assert (op, comp[p], p)[mode] == obs
                    l, j = model(mode, t, a)
                    b, good, selected, conf, made = update(b, good, l[obs], j[obs], s.site[goal] == s.targets[done])
                    if (-1, nearest[p], selected)[mode] != chosen:
                        assert mode == 2 and conf < .87 and packets[p][chosen] == packets[p][selected]
                        selected = chosen
                        counts["nonclaim tied-slot arithmetic differences"] += 1
                    assert (ok, nk, kinds[selected] if selected >= 0 else -1)[mode] == kind
                    np.testing.assert_allclose(conf, edge[11], atol=1e-13, rtol=0.)
                    assert made == claim
                    if mode == 2 and s.site[goal] == s.targets[done]:
                        counts[f"{task} rich current-target captures"] += 1
                        counts[f"{task} rich current-target present"] += int(0 in kinds)
                        counts[f"{task} rich true target omitted by legacy nearest"] += int(0 in kinds and ok != 0)
                        counts[f"{task} rich correct claims where legacy nearest wrong"] += int(made and kind == 0 and ok != 0)
                        counts[f"{task} rich correct claims where coarse nearest wrong"] += int(made and kind == 0 and nk != 0)
                    done += int(made)
                    bad |= made and kind != 0
                    r -= expense
                    at = goal
                    counts["logged capture packets"] += 1
                n = len(log["trace"])
                if sensor.name == "smoke":
                    assert action(s, lambda t, a: model(mode, t, a), at, n, t, done, r, b, good) < 0
                    counts["independent MPC stops"] += 1
                costs = s.reserve - r
                measure = [int(done == len(s.targets) and not bad), int(bad), *costs[:2], data["clocks"][t], costs[2], n]
                np.testing.assert_allclose(measure, log["measured"][:-1], atol=1e-12, rtol=0.)
                counts["terminal paths"] += 1
        rich.cache_clear()
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    cohort = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    main(stage / "data" / cohort)
