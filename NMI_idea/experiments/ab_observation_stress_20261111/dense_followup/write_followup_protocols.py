"""Write arm-local protocols while preserving the frozen stress definitions."""
from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STRESS = HERE.parent
PARENT = STRESS / "PROTOCOL.json"


def main() -> None:
    parent = json.loads(PARENT.read_text())
    design_motivation = {
        "purpose": "Test whether shared candidate resolution, rather than a changed model or threshold, explains the post-hoc coverage gap.",
        "prior_heldout_seed": 721119,
        "prior_heldout_objects": 32,
        "prior_coarse_pose_count": 31,
        "prior_coarse_oracle_objects_with_qualified_pose": 6,
        "prior_coarse_oracle_statement": "31-pose held-out oracle qualified at least one candidate for 6/32 objects.",
        "posthoc_geometric_recompute_pose_count": 305,
        "posthoc_dense_oracle_objects_with_qualified_pose": 26,
        "posthoc_dense_oracle_statement": "Pure geometry-only post-hoc recomputation on the same 32 worlds found a qualified candidate for 26/32 at 2 mm y spacing.",
        "interpretation": "Design motivation only; these 721119 post-hoc values are not formal follow-up outcomes and were not supplied to a policy.",
        "formal_world_seed": 721129,
    }
    for arm in ("coarse", "dense"):
        protocol = copy.deepcopy(parent)
        protocol["study"] = "Shared-candidate-resolution follow-up of the paired observation stress experiment"
        protocol["scope"] = (
            "Independent continuous synthetic curves in the existing local scan patch, "
            "static robot endpoint witnesses, and the frozen local ProDMP reference timing. "
            "This is a declared categorical observation stress model; it is not a phantom, "
            "acoustic, or native closed-loop experiment."
        )
        protocol["resolution_arm"] = arm
        protocol["design_motivation"] = design_motivation
        geometry = protocol["geometry"]
        geometry["development_seed"] = 721111
        geometry["development_objects"] = 3
        geometry["test_seed"] = 721129
        geometry["test_objects"] = 32
        geometry["contexts"] = [
            "prior_only_layout",
            "prior_only_layout_partial_endpoint_witnesses",
        ]
        geometry["capability"] = (
            "Finite static FK, joint limits and configured self-collision at the two original "
            "fixed-rail layouts. Coarse reuses the frozen endpoint witnesses; dense adds only "
            "the declared shared candidate lattice and runs finite _static_pose searches seeded "
            "by each layout center q. A failed search is not an infeasibility certificate."
        )
        geometry["pose_library"] = (
            "Coarse: the frozen common 31-pose library. Dense: x=0, y=-.032:.002:.032, "
            "yaw=-.68:.17:.68 plus the original 31, rounded to 9 decimals and uniqued (305)."
        )
        protocol["observation"]["regimes"] = ["bursty"]
        protocol["methods"] = ["A_task", "B_two", "B_fixed", "B_no_future_cap"]
        protocol["comparisons"] = {
            name: protocol["comparisons"][name]
            for name in protocol["methods"]
        }
        protocol["task"]["max_actions"] = 6
        protocol["task"]["deadline_s"] = 60.0
        protocol["task"]["stop_probability"] = 0.65
        protocol["formal_followup"] = {
            "same_public_prior_seed": 721110,
            "same_public_prior_count": 96,
            "same_observation_channel": True,
            "same_thresholds": {"center_m": 0.002, "out_of_plane_angle_deg": 10.0},
            "same_prodmp_timing": True,
            "same_world_seed_and_crn_between_arms": True,
            "world_seed": 721129,
            "independent_objects": 32,
            "regimes": ["bursty"],
            "methods": protocol["methods"],
        }
        (HERE / arm / "PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2, allow_nan=False) + "\n"
        )
        print(f"wrote {HERE / arm / 'PROTOCOL.json'}", flush=True)


if __name__ == "__main__":
    main()
