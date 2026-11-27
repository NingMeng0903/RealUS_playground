"""Verify frozen learned-policy receipts and compare the logged development curves."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(fitted):
    result = {}
    for run in sorted(fitted.glob("seed*")):
        config = load(run / "fit_settings.json")
        reports = {}
        for arm in config["arms"]:
            directory = run / arm
            receipt = load(directory / "fit_receipt.json")
            if arm in ("bc", "bc_mlp"):
                summary = load(directory / "imitation_summary.json")
                checkpoint = directory / "clone_policy.pt"
                collection = summary["train_collection"]
                assert collection["invalid_teacher"] == collection["truncated"] == 0
                assert collection["complete"] == collection["requested"]
                reports[arm] = {
                    "training_episodes": collection["complete"],
                    "training_actions": collection["valid_frames"],
                    "teacher_agreement_on_development_actions": summary["dev_quality"]["accuracy"],
                    "development_actions": summary["dev_collection"]["valid_frames"],
                    "supervised_epochs": len(summary["train_curve"]),
                    "demonstrator_tapes_per_case": summary["repeats"],
                }
            elif arm in ("ppo", "mlp"):
                summary = load(directory / "ppo_summary.json")
                checkpoint = directory / "ppo_final.pt"
                assert summary["init_loaded"]
                assert summary["actual_interactions"] == config["interactions"]
                objective = summary["best_dev_objective"]
                best = [entry for entry in summary["dev_curve"] if entry["objective"] == objective][-1]
                reports[arm] = {
                    "actual_interactions": summary["actual_interactions"],
                    "updates": len(summary["train_curve"]),
                    "initial_development": summary["dev_curve"][0],
                    "selected_development": best,
                    "last_development": summary["dev_curve"][-1],
                }
            else:
                summary = load(directory / "cql_summary.json")
                checkpoint = directory / "cql_final.pt"
                assert summary["fixed_mixed_replay"]
                assert summary["replay_size"] == config["interactions"]
                reports[arm] = {
                    "replay_size": summary["replay_size"],
                    "updates": summary["updates"],
                    "development": summary["dev"],
                }
            assert sha(checkpoint) == receipt["checkpoint_sha256"]
            reports[arm]["checkpoint_sha256"] = receipt["checkpoint_sha256"]
        assert reports["bc"]["training_actions"] == reports["bc_mlp"]["training_actions"]
        assert reports["bc"]["development_actions"] == reports["bc_mlp"]["development_actions"]
        result[run.name] = {"configuration": config, "checks": reports}
    assert result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.fitted), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
