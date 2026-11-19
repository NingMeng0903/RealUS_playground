"""Replay deterministic public policy decisions and compare saved arm rows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def model_from_checkpoint(path, env, arm):
    from learner import ActorCritic, QNetwork

    data = torch.load(path, map_location="cpu", weights_only=True)
    if arm == "cql":
        model = QNetwork(env.feature_dim, env.action_count)
    else:
        model = ActorCritic(env.feature_dim, env.action_count, recurrent=arm in ("ppo", "bc"))
    model.load_state_dict(data["model"], strict=True)
    return model.eval()


def run(records, env, arm, model, indices):
    from core import ABSTAIN
    from learner import ActorCritic

    episode = env.reset(indices)
    hidden = None
    n = len(indices)
    if isinstance(model, ActorCritic):
        prefix = torch.from_numpy(episode.prefix_features())
        mask = torch.ones((*prefix.shape[:2], env.action_count), dtype=torch.bool)
        with torch.no_grad():
            _, _, hidden = model.sequence(prefix, mask, h=None)
    decisions = 0
    while not episode.done.all():
        active = np.flatnonzero(~episode.done)
        if model is None:
            actions = episode.actions("mpc" if arm == "nearest" else arm)
        else:
            values, legal = episode.features()
            x = torch.from_numpy(values)
            mask = torch.from_numpy(legal)
            with torch.no_grad():
                if isinstance(model, ActorCritic):
                    scores, _, hidden = model.step(x, mask, h=hidden)
                else:
                    scores = model(x)
                actions = scores.masked_fill(~mask, -torch.inf).argmax(dim=-1).numpy()
        actions = np.where(episode.done, ABSTAIN, actions)
        for pos in active:
            row = records[int(indices[pos])]
            t = int(episode.clock[pos] - len(row["prefix"]))
            if t < len(row["trace"]):
                expected = row["trace"][t][1]
            else:
                expected = env.action_count - 1 if row["prediction"] < 0 else 30 + row["prediction"]
            assert actions[pos] == expected, (arm, row["index"], t, actions[pos], expected)
        episode.step(actions)
        decisions += len(active)
    assert len(episode.outcomes()) == n
    for row in episode.outcomes():
        original = records[row["index"]]
        assert row == {key: original[key] for key in row}
    return {"cases": n, "commands_including_declaration": decisions}


def audit(stage, repo, cohorts, every):
    import core
    from run_topology import checkpoint_path, NEURAL
    import world

    torch.set_num_threads(1)
    geometry = world.load(repo)
    summary = {}
    model = None
    for cohort in cohorts:
        settings = json.loads((stage / cohort / "settings.json").read_text())
        seed, namespace, split = settings["cohort"]
        if model is None:
            prior = world.cases(geometry, settings["prior_seed"], settings["prior_size"], "train")
            model, table = core.make_model(repo, geometry, prior, grid=settings["grid"])
        batch = world.cases(geometry, seed, settings["cases"], split)
        env = core.BatchEnv(model, batch, namespace)
        env.prepare_prefix()
        indices = np.arange(0, env.n_cases, every)
        by_arm = {}
        for arm in settings["arms"]:
            path = stage / cohort / f"{arm}.jsonl"
            with path.open() as stream:
                records = {row["index"]: row for row in map(json.loads, stream)}
            other = env
            if arm == "nearest":
                other = core.BatchEnv(core.nearest_model(model, table), batch, namespace, full_model=model)
                other.prescan = env.prescan
            weights = None
            if arm in NEURAL:
                directory = stage / "fitted" / f"seed{settings['seed']}"
                weights = model_from_checkpoint(checkpoint_path(directory, arm), env, arm)
            by_arm[arm] = run(records, other, arm, weights, indices)
        summary[cohort] = by_arm
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "repo", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--every", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    report = audit(args.stage, args.repo.resolve(), args.cohorts, args.every)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
