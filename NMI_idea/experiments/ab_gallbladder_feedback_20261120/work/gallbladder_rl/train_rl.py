"""Train the public-observation recurrent policy on simulator episodes."""

import argparse
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path


CHECKPOINT_VERSION = 1
OUTCOMES = ("correct", "wrong", "timeout")
CHECKPOINT_COMPATIBILITY_KEYS = (
    "observation_dim", "privileged_dim", "hidden",
    "planner_features", "freeze_belief", "decision_frames",
)


def collect_episode(env, model, *, seed, condition, stream):
    import numpy as np

    from .recurrent_ppo import Episode, SETTLE_CONFIRM

    obs, info = env.reset(seed=seed, condition=condition, stream=stream)
    warmup = np.array(info["warmup"], dtype=np.float32, copy=True)
    hidden = model.burn_in(warmup)
    observations, privileged, modes, latents = [], [], [], []
    log_probs, values, rewards = [], [], []
    while True:
        observations.append(np.array(obs, dtype=np.float32, copy=True))
        privileged.append(np.array(info["privileged"], dtype=np.float32, copy=True))
        action, log_prob, value, hidden = model.act(
            observations[-1], hidden, privileged[-1]
        )
        modes.append(action["mode"])
        latents.append(action["latent"].copy())
        log_probs.append(log_prob)
        values.append(value)
        obs, reward, terminated, info = env.step(action)
        rewards.append(float(reward))
        if terminated:
            break

    endpoint = info["outcome"]
    outcome = next((name for name in OUTCOMES if endpoint[name]), None)
    if outcome is None or sum(bool(endpoint[name]) for name in OUTCOMES) != 1:
        raise ValueError("Expected exactly one terminal outcome: correct, wrong, or timeout")
    episode = Episode(
        warmup=warmup,
        observations=np.asarray(observations, dtype=np.float32),
        privileged=np.asarray(privileged, dtype=np.float32),
        modes=np.asarray(modes, dtype=np.int64),
        latents=np.asarray(latents, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )
    metrics = {
        "seed": seed,
        "condition": condition,
        "stream": stream,
        "outcome": outcome,
        "endpoint": endpoint,
        **{name: int(outcome == name) for name in OUTCOMES},
        "steps": len(rewards),
        "warmup_steps": len(warmup),
        "settle_confirm_steps": modes.count(SETTLE_CONFIRM),
        "return": sum(rewards),
    }
    return episode, metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=58710000)
    parser.add_argument(
        "--count", type=int, help="Size of the episode seed pool (default: --episodes)"
    )
    parser.add_argument("--condition", choices=("wide", "stress", "mixed"), default="wide")
    parser.add_argument(
        "--stream", type=int, default=20000,
        help="First random stream; incremented for each episode",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for metrics and checkpoint")
    parser.add_argument("--curriculum", choices=("prefix", "near_success"), default="prefix")
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help="Initialize model weights from a compatible training checkpoint",
    )
    parser.add_argument("--planner-features", action="store_true")
    parser.add_argument("--freeze-belief", action="store_true")
    parser.add_argument("--decision-frames", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--policy-seed", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count is None:
        args.count = args.episodes
    for name in ("episodes", "batch_episodes", "count", "decision_frames", "hidden", "epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("gamma", "gae_lambda"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.stream < 0 or args.seed_start < 0:
        parser.error("--stream and --seed-start must be nonnegative")
    if not args.learning_rate > 0:
        parser.error("--learning-rate must be positive")
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"Output path is not a directory: {args.output}")
    for name in ("metrics.jsonl", "checkpoint.pt", "checkpoint.tmp"):
        if (args.output / name).exists():
            parser.error(f"Output file already exists: {args.output / name}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            parser.exit(1, "Training requires the CPU PyTorch package.\n")
        raise
    from .env import GallbladderRLEnv, asset_fingerprints, require_assets
    from .recurrent_ppo import RecurrentActorCritic, update_ppo

    try:
        require_assets()
    except FileNotFoundError as exc:
        parser.exit(1, f"{exc}\n")

    torch.set_num_threads(1)
    torch.manual_seed(args.policy_seed)
    env = GallbladderRLEnv(
        planner_features=args.planner_features,
        freeze_belief=args.freeze_belief,
        decision_frames=args.decision_frames,
        reset_mode=args.curriculum,
    )
    model = RecurrentActorCritic(env.observation_dim, env.privileged_dim, args.hidden)
    config = {
        **vars(args),
        "output": str(args.output),
        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "observation_dim": env.observation_dim,
        "privileged_dim": env.privileged_dim,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    }
    source_info = {
        "torch_version": str(torch.__version__),
        "assets": asset_fingerprints(),
        "file_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("env.py", "recurrent_ppo.py", "train_rl.py")
        },
    }
    previous_seed_ranges = []
    if args.initial_checkpoint is not None:
        checkpoint_bytes = args.initial_checkpoint.read_bytes()
        initial = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        if initial.get("schema_version") != CHECKPOINT_VERSION:
            parser.error("Initial checkpoint has an unsupported schema version")
        mismatched = [
            name for name in CHECKPOINT_COMPATIBILITY_KEYS
            if name not in initial["config"] or config[name] != initial["config"][name]
        ]
        if mismatched:
            parser.error("Initial checkpoint has incompatible settings: " + ", ".join(mismatched))
        model.load_state_dict(initial["model_state_dict"], strict=True)
        previous_seed_ranges = initial.get("training_seed_ranges")
        if previous_seed_ranges is None:
            previous_seed_ranges = [initial["training_seed_range"]]
        source_info["initial_checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    interactions = {"episodes": 0, "policy_steps": 0, "warmup_steps": 0}

    with (args.output / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
        for batch_start in range(0, args.episodes, args.batch_episodes):
            episodes = []
            outcomes = Counter()
            end = min(batch_start + args.batch_episodes, args.episodes)
            for index in range(batch_start, end):
                cycle, seed_offset = divmod(index, args.count)
                condition = args.condition
                if condition == "mixed":
                    condition = ("wide", "stress")[(cycle + seed_offset) % 2]
                episode, metrics = collect_episode(
                    env,
                    model,
                    seed=args.seed_start + seed_offset,
                    condition=condition,
                    stream=args.stream + index,
                )
                episodes.append(episode)
                outcomes[metrics["outcome"]] += 1
                interactions["episodes"] += 1
                interactions["policy_steps"] += metrics["steps"]
                interactions["warmup_steps"] += metrics["warmup_steps"]
                row = {"type": "episode", "episode": index + 1, **metrics}
                metrics_file.write(json.dumps(row) + "\n")

            stats = update_ppo(
                model,
                optimizer,
                episodes,
                epochs=args.epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_ratio=config["clip_ratio"],
                value_coef=config["value_coef"],
                entropy_coef=config["entropy_coef"],
            )
            row = {
                "type": "batch",
                "batch": batch_start // args.batch_episodes + 1,
                **interactions,
                "batch_episodes": len(episodes),
                **{name: outcomes[name] for name in OUTCOMES},
                **stats,
            }
            serialized = json.dumps(row, allow_nan=False)
            metrics_file.write(serialized + "\n")
            metrics_file.flush()
            temp_checkpoint = args.output / "checkpoint.tmp"
            current_seed_range = {
                "first": args.seed_start,
                "last_used": args.seed_start + min(args.count, end) - 1,
                "first_stream": args.stream,
                "last_stream": args.stream + end - 1,
            }
            torch.save(
                {
                    "schema_version": CHECKPOINT_VERSION,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "source_info": source_info,
                    "training_seed_range": current_seed_range,
                    "training_seed_ranges": [*previous_seed_ranges, current_seed_range],
                    "interactions": interactions,
                },
                temp_checkpoint,
            )
            temp_checkpoint.replace(args.output / "checkpoint.pt")
            print(serialized, flush=True)


if __name__ == "__main__":
    main()
