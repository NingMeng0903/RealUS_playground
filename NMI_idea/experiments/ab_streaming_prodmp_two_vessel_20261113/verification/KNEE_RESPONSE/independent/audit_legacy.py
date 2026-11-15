"""Compare the original and edited atlas entry on the same moving scan."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np

from audit_filter import configure_experiment
from audit_pilot import EXT, REPAIR


ORIGINAL_REVISION = "ce4cf9967155740fb1f42fb2dc6d9495ab82476a"


def original_scan_class(experiment, revision):
    repo = experiment.parents[2]
    file = experiment / "atlas_scan.py"
    payload = subprocess.check_output(["git", "show", revision + ":" + str(file.relative_to(repo))], cwd=repo)
    old = ModuleType("pre_response_atlas")
    old.__file__ = str(file)
    sys.modules[old.__name__] = old
    exec(compile(payload, str(file), "exec"), old.__dict__)
    return old.Scan


def check_legacy(experiment, among, repair, revision):
    from atlas_scan import DYNAMIC_SEED, FRAME_COUNT, FRAME_DT_S, SEED, Scan, load_witness
    from atlas_geometry import ScanTemplate
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator
    from residual import motion_tape

    template = ScanTemplate(among, repair)
    world = template.instantiate((0., 0.), 15.)
    cap = CapabilityEvaluator(device="cpu")
    _, witness = load_witness(experiment.parent / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz")
    with np.load(experiment / "verification/DEMONSTRATIONS.npz", allow_pickle=False) as frozen:
        regression = frozen["shape_regression"]
    old = original_scan_class(experiment, revision)(world, cap, witness, regression)
    new = Scan(world, cap, witness, regression)
    scene = old.sensor.scene(np.random.default_rng(SEED), FRAME_COUNT)
    motion = motion_tape(np.random.default_rng(DYNAMIC_SEED), FRAME_COUNT, FRAME_DT_S, "ou")
    old_report, old_path = old.execute(scene, "mpc_ou", motion)
    new_report, new_path = new.execute(scene, "mpc_ou", motion)
    assert old_report == new_report
    for a, b in zip(old_path, new_path, strict=True):
        np.testing.assert_array_equal(a, b)
    return dict(original_revision=revision, exposures=len(new_report["frames"]), robot_samples=len(new_path[0]),
                identical_case_report=True, identical_joint_fk_and_chart_states=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--among-us", type=Path, default=EXT / "Among_US")
    parser.add_argument("--repair", type=Path, default=REPAIR)
    parser.add_argument("--legacy-revision", default=ORIGINAL_REVISION)
    args = parser.parse_args()
    experiment = configure_experiment(args.experiment)
    print(json.dumps(check_legacy(experiment, args.among_us, args.repair, args.legacy_revision), indent=2,
                     allow_nan=False))
