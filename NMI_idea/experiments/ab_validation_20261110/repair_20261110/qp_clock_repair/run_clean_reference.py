"""Repeat frozen reference groups after fixing pre-publication model time."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "qp_reference_followup"
sys.path.insert(0, str(OLD))
import run_reference_protocol as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    from rm75_control.control.joint_admittance_8dof.wbc_rt import client
    from rm75_control.control.joint_admittance_8dof import reference
    binary = client.find_wbc_rt_binary()
    manifest = {
        "protocol": sha(HERE / "PROTOCOL.txt"),
        "runner": sha(__file__),
        "original_frozen_runner": sha(OLD / "run_reference_protocol.py"),
        "input": sha(runner.INPUT),
        "native_client": {"path": client.__file__, "sha256": sha(client.__file__)},
        "qp_replay": {"path": runner.replay.__file__, "sha256": sha(runner.replay.__file__)},
        "config": {"path": str(runner.replay.CONFIG), "sha256": sha(runner.replay.CONFIG)},
        "native_binary": {"path": str(binary), "sha256": sha(binary)},
        "reference": {"path": reference.__file__, "sha256": sha(reference.__file__)},
        "clock_change": "model timestamp is finalized before seq publication; wall deadline unchanged",
        "old_results_preserved": str(OLD / "RESULTS.json"),
    }
    (HERE / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    runner.HERE = HERE
    runner.main()
