"""Independent sequence-prediction and complementary-view checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import acoustics
import control
import run


NAMESPACE = 716905


def path_pair(scene):
    focus = scene.targets[0]
    best = None
    for at in range(len(scene.site)):
        if scene.site[at] not in (scene.anchor, focus) or scene.tilt[at] != 0:
            continue
        for first in range(len(scene.site)):
            if scene.site[first] != scene.site[at] or scene.tilt[first] == 0:
                continue
            if not control.allowed(scene.edge_costs, at, first, scene.reserve):
                continue
            remaining = scene.reserve - scene.edge_costs[at, first]
            if not control.allowed(scene.edge_costs, first, first, remaining):
                continue
            for end in range(len(scene.site)):
                if scene.site[end] != scene.site[first] or scene.tilt[end] != -scene.tilt[first]:
                    continue
                if not control.allowed(scene.edge_costs, first, end, remaining):
                    continue
                rank = (scene.site[at] != focus, scene.edge_costs[first, end, 2],
                        scene.edge_costs[at, first, 0] + scene.edge_costs[first, end, 0],
                        at, first, end)
                if best is None or rank < best:
                    best = rank
    return None if best is None else best[-3:]


def check(data, n):
    scene, geometry = data["scene"], data["geometry"]
    pair = path_pair(scene)
    if pair is None:
        return {"status": "no checked opposite-tilt sequence"}
    center, first, opposite = pair
    t1 = int(scene.edge_costs[center, first, 3])
    # Keep matched acquisition clocks for this sensor-only comparison.
    # If the tilt change takes longer, the repeated view is evaluated at that
    # same time. These are not matched-effort policy runs.
    t2 = t1 + int(max(scene.edge_costs[first, first, 3], scene.edge_costs[first, opposite, 3]))
    t3 = t2 + int(max(scene.edge_costs[first, first, 3], scene.edge_costs[opposite, opposite, 3]))
    prior = np.full(scene.likelihood.shape[-1], 1. / scene.likelihood.shape[-1])
    results = {"same": [], "opposite": []}
    sites = [scene.site[node] for node in pair]
    for j in range(n):
        seed = acoustics.seed_for(NAMESPACE, j)
        case = acoustics.make_scenario(geometry.world, seed, data["mode"])
        true_sign = case.body.w_subject_mm[5] >= 0.
        _, _, detection, cdf = acoustics.probabilities(geometry, case, data["clocks"], data["mode"])
        detection = np.ascontiguousarray(detection[:, :, None])
        cdf = np.ascontiguousarray(cdf[:, :, None])
        noise = acoustics.tape(seed, len(scene.likelihood))
        for condition, other in (("same", first), ("opposite", opposite)):
            posterior = prior.copy()
            ll, ll_marginal = 0., 0.
            event = []
            for k, node in zip((t1, t2, t3), (first, other, other)):
                address = scene.to_address[node]
                obs, kind = control.capture_draw(detection, cdf, k, address, 0, noise)
                p = scene.likelihood[k, address, obs]
                ll += float(np.log(max(1.e-12, posterior @ p)))
                ll_marginal += float(np.log(max(1.e-12, prior @ p)))
                posterior, conf = control.posterior(
                    posterior, scene.likelihood, scene.correct, k, address, obs)
                sign = float(posterior[scene.material_coordinate >= 0].sum())
                event.append(dict(obs=int(obs), target=(kind == 0), conf=float(conf), positive=sign,
                                  sign_brier=(sign - true_sign) ** 2,
                                  ess=float(1. / (posterior @ posterior))))
            results[condition].append(dict(case=j, ll=ll, ll_marginal=ll_marginal, packets=event))
    summary = {}
    for condition, rows in results.items():
        captures = [e for r in rows for e in r["packets"] if e["obs"]]
        accepted = [e for e in captures if e["conf"] >= control.CONFIDENCE]
        summary[condition] = {
            "cases": n,
            "joint_log_score_per_image": float(np.mean([r["ll"] / 3. for r in rows])),
            "independent_marginal_log_score_per_image": float(np.mean([r["ll_marginal"] / 3. for r in rows])),
            "sign_brier_each_step": np.array([[e["sign_brier"] for e in r["packets"]]
                                              for r in rows]).mean(axis=0).tolist(),
            "median_ess_at_three": float(np.median([r["packets"][-1]["ess"] for r in rows])),
            "identity_brier_nonempty": float(np.mean([(e["conf"] - e["target"]) ** 2 for e in captures])),
            "accepted_correct": sum(e["target"] for e in accepted),
            "accepted_wrong": sum(not e["target"] for e in accepted),
        }
    difference = np.array([r["packets"][1]["sign_brier"] for r in results["opposite"]]) - np.array(
        [r["packets"][1]["sign_brier"] for r in results["same"]])
    rng = np.random.default_rng(372681)
    ci = np.quantile(difference[rng.integers(n, size=(10000, n))].mean(axis=1), [.025, .975])
    return dict(status="checked", nodes=list(pair), sites=list(map(int, sites)),
                clocks_s=[float(data["clocks"][t]) for t in (t1, t2, t3)], summary=summary,
                opposite_minus_same_second_sign_brier=float(difference.mean()), paired_ci95=ci.tolist(),
                data=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=run.HERE.parent / "RealUS_playground")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 256, 768])
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()
    report = {}
    for size in args.sizes:
        run.initialize(args.repo, args.motion, size, 3, "joint")
        report[str(size)] = {task: check(data, args.n) for task, data in run.DATA.items()}
    args.out.write_text(json.dumps(dict(seed_namespace=NAMESPACE, calibration=report), indent=2,
                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
