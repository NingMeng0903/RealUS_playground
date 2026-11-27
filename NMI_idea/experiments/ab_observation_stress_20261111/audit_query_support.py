"""Measure task-query support and replay fixed public histories with public prior designs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from planner import Planner, initial_belief
from sensor import env, channel


def oracle_support(model, audit, correct):
    allowed = model["available"][:, None, :]
    physical = allowed & audit["target_qualified"][None, :, :]
    plausible = np.any(correct > 0., axis=(0, 1, 3))
    supported = physical & plausible[None, None, :]
    return {
        "contexts": [
            {
                "context": context,
                "geometrically_qualified_objects": int(a.any(axis=1).sum()),
                "qualified_objects_with_nonzero_public_correct_query": int(b.any(axis=1).sum()),
                "geometry_qualified_but_query_zero_objects": np.flatnonzero(
                    a.any(axis=1) & ~b.any(axis=1)
                ).tolist(),
            }
            for context, a, b in zip(model["context_ids"], physical, supported)
        ],
        "supported_pose_count": int(plausible.sum()),
        "scope": "Necessary positive support of a correct-stop query in either mode. Threshold, current belief, true public return, timing and execution may further reduce completion.",
    }


def broad_bank(base):
    # Independent smooth offsets in the public body chart. Their support is
    # fixed before this replay; no test geometry or observed label fits them.
    rng = np.random.default_rng(721121)
    offsets = rng.uniform([-.012, -.20], [.012, .20], size=(len(base), 2))
    worlds = []
    for w, (offset, slope) in zip(base, offsets):
        r = w.registration + np.array([0., offset])
        y = w.y_controls + slope * env.CONTROL_X[None, :]
        points = env._points(r, w.modes, y, w.z_controls, w.depths)
        worlds.append(replace(w, registration=r, y_controls=y, points=points))
    return worlds


def fixed_replay(rows, model, tables, threshold):
    result = {}
    for index, regime in enumerate(model["mode_model_ids"].tolist()):
        planner = Planner(
            tables["likelihood"], tables["correct_joint"], tables["joint"],
            model["public_gate"], model["mode_transitions"][index], threshold,
        )
        obs_nll, correctness, probability, stops = [], [], [], []
        audited_correct_public = 0
        accepted_audited_correct = 0
        for row in rows:
            if row["regime"] != regime or row["method"] != "A_task":
                continue
            belief = initial_belief(planner.world_count, model["stationary_modes"][index])
            first = None
            for block in row["blocks"]:
                public, audit = block["public"], block["audit"]
                action, outcome = public["action_index"], public["outcome_index"]
                a = planner.assess(belief, action)
                m = float(a.probability[outcome])
                if m <= 0:
                    raise ValueError("fixed trajectory outcome has zero public model evidence")
                obs_nll.append(-np.log(m))
                if public["public_gate"]:
                    p = float(a.correct_mass[outcome] / m)
                    truth = bool(audit["correct_if_stopped"])
                    correctness.append(truth)
                    probability.append(p)
                    audited_correct_public += int(truth)
                    accepted_audited_correct += int(truth and a.accepted[outcome])
                    if a.accepted[outcome] and first is None:
                        first = truth
                belief = planner.advance(belief, action, outcome)
            stops.append(first)
        if not stops:
            continue
        result[regime] = {
            "trials": len(stops),
            "public_gates": len(correctness),
            "brier_public_gate": float(np.mean((np.asarray(probability) - correctness) ** 2))
            if correctness else None,
            "mean_public_observation_nll": float(np.mean(obs_nll)) if obs_nll else None,
            "audit_correct_public_gates": audited_correct_public,
            "accepted_audit_correct_public_gates": accepted_audited_correct,
            "counterfactual_first_correct": sum(s is True for s in stops),
            "counterfactual_first_false": sum(s is False for s in stops),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-replay", action="store_true", help="Also replay fixed A_task histories with denser and wider public banks")
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    with np.load(args.results / "model_inputs.npz") as data:
        model = {key: data[key] for key in data.files}
    with np.load(args.results / "renderer_audit.npz") as data:
        audit = {"target_qualified": data["target_qualified"]}
    result = {"original": oracle_support(model, audit, model["correct_joint"])}
    if args.prior_replay:
        protocol = manifest["protocol"]
        seed = protocol["geometry"]["public_prior_seed"]
        base = env.sample_worlds(seed, 384)
        rows = [json.loads(line) for line in (args.results / "rows.jsonl").read_text().splitlines()]
        result["fixed_history_scope"] = (
            "Post-run replay of public A_task histories already collected, with alternate public priors "
            "and unmodified likelihood/stop gate. Earliest stops are counterfactual on existing prefixes. "
            "Histories truncate at the original policy stop; this is not a new rollout."
        )
        threshold = protocol["task"]["stop_probability"]
        with threadpool_limits(limits=1, user_api="blas"):
            result["original"]["fixed_history"] = fixed_replay(rows, model, model, threshold)
            for name, worlds in [("denser_same_prior_384", base), ("wider_smooth_offsets_384", broad_bank(base))]:
                c = channel(worlds, model["poses"])
                tables = {"joint": c.joint, "likelihood": c.likelihood, "correct_joint": c.correct_joint}
                entry = oracle_support(model, audit, c.correct_joint)
                entry["fixed_history"] = fixed_replay(rows, model, tables, threshold)
                result[name] = entry
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
