"""Verify frozen implementation, bank, protocol and learned checkpoint bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def audit(stage, archive):
    original = {
        "base_bank": archive / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz",
        "diagonal_bank": archive / "ab_inverse_acquisition_b_20261113/motion/BANK.json.xz",
        "atlas": archive / "ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz",
        "association": archive / "ab_relational_b_20261113/return_sets.py",
    }
    results = {}
    for manifest in sorted(stage.glob("FREEZE_*.json")):
        expected = load(manifest)
        for name, digest in expected["source_sha256"].items():
            path = original.get(name, stage / name)
            assert sha(path) == digest, (manifest.name, name)
        assert sha(stage / "PROTOCOL.md") == expected["protocol_sha256"]
        checkpoint_root = stage / "fitted" / f"seed{expected['seed']}"
        for arm, digest in expected["checkpoint_sha256"].items():
            filename = ("clone_policy.pt" if arm in ("bc", "bc_mlp")
                        else "cql_final.pt" if arm == "cql" else "ppo_final.pt")
            assert sha(checkpoint_root / arm / filename) == digest, (manifest.name, arm)
        run = stage / manifest.stem.removeprefix("FREEZE_")
        settings = run / "settings.json"
        if settings.exists():
            observed = load(settings)
            assert all(observed[key] == value for key, value in expected.items() if key != "protocol_sha256"), manifest.name
        results[manifest.name] = {
            "sha256": sha(manifest),
            "source_inputs": len(expected["source_sha256"]),
            "checkpoints": len(expected["checkpoint_sha256"]),
            "runtime_settings_present": settings.exists(),
        }
    assert len(results) == 6
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "archive", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.stage, args.archive), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
