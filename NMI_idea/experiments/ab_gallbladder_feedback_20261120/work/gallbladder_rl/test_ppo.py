"""Run the synthetic PPO checks with python -m work.gallbladder_rl.test_ppo."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json

import numpy as np
import torch

from . import env, train_rl
from .evaluate_rl import reject_training_overlap
from .recurrent_ppo import MOVE, SETTLE_CONFIRM, RecurrentActorCritic, update_ppo


class ScriptedEnv:
    observation_dim = env.OBSERVATION_DIM
    privileged_dim = env.PRIVILEGED_DIM
    episode_steps = 12

    def __init__(self, *, planner_features=False, freeze_belief=False,
                 decision_frames=3, reset_mode="prefix"):
        self.reset_mode = reset_mode

    def reset(self, *, seed, condition, stream):
        self.step_index = 0
        self.result = train_rl.OUTCOMES[stream % 3]
        return np.zeros(self.observation_dim), {
            "privileged": np.zeros(self.privileged_dim),
            "warmup": np.ones((env.HISTORY_FRAMES, self.observation_dim)),
        }

    def step(self, action):
        assert set(action) == {"mode", "latent", "delta"}
        np.testing.assert_allclose(np.tanh(action["latent"]), action["delta"], atol=1e-7)
        if action["mode"] == SETTLE_CONFIRM:
            assert not action["delta"].any()
        self.step_index += 1
        terminal = self.step_index == self.episode_steps
        info = {"privileged": np.ones(self.privileged_dim)}
        if terminal:
            info["outcome"] = {
                **{name: name == self.result for name in train_rl.OUTCOMES},
                "seconds": 3.6, "travel": 4.0, "ever": True,
                "strict_at_cine": self.result == "correct",
                "reset_kind": self.reset_mode,
            }
        return np.full(self.observation_dim, self.step_index / self.episode_steps), float(terminal), terminal, info


def check_network():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = RecurrentActorCritic(env.OBSERVATION_DIM, env.PRIVILEGED_DIM)
    episode, _ = train_rl.collect_episode(
        ScriptedEnv(), model, seed=0, condition="wide", stream=2
    )
    assert set(episode.modes) == {MOVE, SETTLE_CONFIRM}
    log_probs, values, _ = model.evaluate_sequence(
        episode.observations, episode.privileged, episode.modes,
        episode.latents, episode.warmup,
    )
    replay_errors = {
        "log_prob": float(np.max(np.abs(log_probs.detach().numpy() - episode.log_probs))),
        "value": float(np.max(np.abs(values.detach().numpy() - episode.values))),
    }
    assert max(replay_errors.values()) < 1e-5
    hidden = model.burn_in(episode.warmup)
    action_a, lp_a, _, hidden_a = model.act(
        episode.observations[0], hidden, np.zeros(env.PRIVILEGED_DIM), True
    )
    action_b, lp_b, _, hidden_b = model.act(
        episode.observations[0], hidden, np.ones(env.PRIVILEGED_DIM), True
    )
    assert action_a["mode"] == action_b["mode"] and lp_a == lp_b
    np.testing.assert_array_equal(action_a["delta"], action_b["delta"])
    torch.testing.assert_close(hidden_a, hidden_b, atol=0, rtol=0)
    values.sum().backward()
    weights_with_gradient = {name for name, p in model.named_parameters() if p.grad is not None}
    assert weights_with_gradient == {f"value_head.{name}" for name, _ in model.value_head.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.mode_head.weight.detach().clone()
    stats = update_ppo(model, optimizer, [episode, episode], epochs=2)
    assert not torch.equal(before, model.mode_head.weight.detach())
    assert np.isfinite(list(stats.values())).all()
    return replay_errors


def check_training_cli():
    flags = ["--episodes", "4", "--batch-episodes", "2", "--count", "2",
             "--epochs", "2", "--condition", "mixed"]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        missing = root / "missing"
        no_assets = FileNotFoundError("Mock missing anatomy assets.")
        with patch.object(env, "require_assets", side_effect=no_assets), redirect_stderr(StringIO()) as stderr:
            try:
                train_rl.main(flags + ["--output", str(missing)])
            except SystemExit as exc:
                assert exc.code == 1 and str(no_assets) in stderr.getvalue()
            else:
                raise AssertionError("Expected an asset preflight error")
        assert not missing.exists()
        fingerprints = {"synthetic_mock": {"sha256": None}}
        with (
            patch.object(env, "GallbladderRLEnv", ScriptedEnv),
            patch.object(env, "require_assets") as preflight,
            patch.object(env, "asset_fingerprints", return_value=fingerprints),
            redirect_stdout(StringIO()),
        ):
            first = root / "first"
            second = root / "second"
            initial = first / "checkpoint.pt"
            train_rl.main(flags + ["--output", str(first), "--curriculum", "near_success"])
            train_rl.main(flags + ["--output", str(second), "--seed-start", "58710008",
                                   "--initial-checkpoint", str(initial)])
            assert preflight.call_count == 2
            with redirect_stderr(StringIO()) as stderr:
                try:
                    train_rl.main(flags + ["--output", str(root / "mismatch"),
                                           "--initial-checkpoint", str(initial), "--planner-features"])
                except SystemExit as exc:
                    assert exc.code == 2 and "incompatible settings: planner_features" in stderr.getvalue()
                else:
                    raise AssertionError("Expected incompatible checkpoint to be rejected")

        rows = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
        episodes = [row for row in rows if row["type"] == "episode"]
        assert [row["condition"] for row in episodes] == ["wide", "stress", "stress", "wide"]
        assert [row["stream"] for row in episodes] == list(range(20000, 20004))
        assert {row["outcome"] for row in episodes} == set(train_rl.OUTCOMES)
        for row in episodes:
            assert row["endpoint"][row["outcome"]]
            assert row["endpoint"]["seconds"] == 3.6 and row["endpoint"]["travel"] == 4.0
            assert row["endpoint"]["ever"]
        checkpoint = torch.load(second / "checkpoint.pt", weights_only=True)
        assert checkpoint["schema_version"] == train_rl.CHECKPOINT_VERSION
        assert checkpoint["training_seed_ranges"] == [
            {"first": first_seed, "last_used": first_seed + 1,
             "first_stream": 20000, "last_stream": 20003}
            for first_seed in (58710000, 58710008)
        ]
        reject_training_overlap(checkpoint, [58710002, 58710010])
        for seed in (58710001, 58710009):
            try:
                reject_training_overlap(checkpoint, [seed])
            except ValueError:
                pass
            else:
                raise AssertionError("Expected overlap with inherited training to be rejected")
        assert checkpoint["interactions"] == {
            "episodes": 4, "policy_steps": 4 * ScriptedEnv.episode_steps,
            "warmup_steps": 4 * env.HISTORY_FRAMES,
        }
        assert checkpoint["source_info"]["initial_checkpoint_sha256"] == hashlib.sha256(initial.read_bytes()).hexdigest()
        assert checkpoint["source_info"]["assets"] == fingerprints


def main():
    replay_errors = check_network()
    check_training_cli()
    source = Path(__file__).resolve().parent
    print(json.dumps({
        "type": "synthetic_algorithm_check",
        "no_medical_success_rate": True,
        "status": "passed",
        "torch_version": str(torch.__version__),
        "test_environment": "synthetic vectors, scripted endpoint dictionaries, mocked asset preflight and fingerprints",
        "shape": {"public": env.OBSERVATION_DIM, "privileged": env.PRIVILEGED_DIM,
                  "hidden": 128, "warmup": env.HISTORY_FRAMES, "episode": ScriptedEnv.episode_steps},
        "maximum_absolute_replay_error": replay_errors,
        "checks": [
            "hybrid actions and zero settle delta", "GRU log probability and value replay",
            "public actor and detached critic", "actual policy update with finite losses",
            "asset preflight fails before creating outputs", "checkpoint records asset fingerprints",
            "terminal outcomes retain endpoint diagnostics", "mixed conditions toggle for each seed",
            "compatible curriculum transfer and incompatible checkpoint rejection",
            "inherited and current training seed overlap rejection",
        ],
        "file_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("env.py", "evaluate_rl.py", "recurrent_ppo.py", "train_rl.py", "test_ppo.py")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
