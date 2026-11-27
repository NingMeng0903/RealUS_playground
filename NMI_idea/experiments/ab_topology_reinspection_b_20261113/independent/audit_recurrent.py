"""Exercise recurrent collection, warm-up, optimization and inference on XOR."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


class MemoryBatch:
    """A balanced first bit in prescan and a second in the first suffix input."""

    action_count = 4
    feature_dim = 4

    def __init__(self, n_cases, seed):
        self.n_cases = n_cases
        self.bits = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (n_cases // 4, 1))
        np.random.default_rng(seed).shuffle(self.bits)

    def reset(self, indices):
        return MemoryEpisode(self.bits[indices])


class MemoryEpisode:
    def __init__(self, bits):
        self.bits = bits
        self.done = np.zeros(len(bits), dtype=bool)
        self.time = 0
        self.guess = None

    def prefix_features(self):
        frames = np.zeros((len(self.bits), 5, 4), dtype=np.float32)
        frames[:, 0, 0] = 2 * self.bits[:, 0] - 1
        frames[:, 0, 2] = 1.0
        return frames

    def features(self):
        x = np.zeros((len(self.bits), 4), dtype=np.float32)
        legal = np.zeros((len(self.bits), 4), dtype=bool)
        if self.time == 0:
            x[:, 1] = 2 * self.bits[:, 1] - 1
            x[:, 2] = 1.0
            legal[:, 2] = True
        elif self.time == 1:
            x[:, 3] = 1.0
            legal[:, :2] = True
        else:
            legal[:, 3] = True
        return x, legal

    def step(self, actions):
        _, legal = self.features()
        assert np.all(legal[np.arange(len(self.bits)), actions])
        self.time += 1
        if self.time == 1:
            reward = np.zeros(len(self.bits), dtype=np.float32)
        elif self.time == 2:
            self.guess = actions.copy()
            self.done[:] = True
            reward = np.where(self.guess == (self.bits[:, 0] ^ self.bits[:, 1]), 1.0, -1.0)
        else:
            raise RuntimeError("episode already ended")
        return reward, self.done.copy()

    def outcomes(self):
        assert np.all(self.done)
        right = self.guess == (self.bits[:, 0] ^ self.bits[:, 1])
        return [{"success": bool(correct), "wrong": not bool(correct)} for correct in right]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interactions", type=int, default=24000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    from train import evaluate, train_ppo

    result = {}
    train = MemoryBatch(256, 92601)
    development = MemoryBatch(128, 92602)
    test = MemoryBatch(128, 92603)
    for recurrent, name in ((True, "recurrent"), (False, "feedforward")):
        result[name] = {}
        for seed in (7, 29, 61):
            model, summary = train_ppo(
                train, development, seed=seed, interactions=args.interactions,
                output=args.out / name / str(seed), recurrent=recurrent,
            )
            rows = evaluate(model, test)
            result[name][str(seed)] = {
                "correct": sum(row["success"] for row in rows),
                "n": len(rows),
                "interactions": summary["actual_interactions"],
                "updates": len(summary["train_curve"]),
            }
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
