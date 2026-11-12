"""Check legacy parity and the synthetic projection at new tilt angles."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def angles(site):
    old = {-12: 7, 0: 3, 12: 8}
    return [{"site": site, "view": old.get(deg, 100 + deg), "tilt_deg": deg}
            for deg in range(-12, 13, 3)]


def check(repo, bank):
    study = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(study), str(repo / "NMI_idea/experiments/ab_full_b_rollout_20261113")]
    import acoustics
    from world import ANGLES, Geometry

    world = acoustics.setup_world(repo)
    nodes = json.loads(bank.read_text())["nodes"] if bank else [
        node for site in range(3, 8) for node in angles(site)
    ]
    alias = [dict(node, view=node["view"] + 10000) if node["view"] not in ANGLES else node
             for node in nodes]
    clocks = np.arange(7) * 1.5
    counts = {"legacy_view_clock_seed_checks": 0, "novel_view_clock_seed_checks": 0,
              "max_new_world_mean_error_mm": 0.}
    separation = {"daughter": [], "NONBRANCH": []}
    for name, label, sites in (("daughter", 1, (3, 4, 5)), ("NONBRANCH", 0, (5, 6, 7))):
        geometry = Geometry(acoustics, world, nodes, label)
        alias_geometry = Geometry(acoustics, world, alias, label)
        legacy = acoustics.Geometry(world, [{"site": site, "view": view}
                                          for site in sites for view in ANGLES], label)
        for index in range(16):
            seed = int(np.random.SeedSequence([634993, index]).generate_state(1)[0])
            scenario = acoustics.make_scenario(world, seed)
            body = scenario.body
            p, q, detect, cdf = acoustics.probabilities(geometry, scenario, clocks, "joint")
            p_old, q_old, d_old, cdf_old = acoustics.probabilities(legacy, scenario, clocks, "joint")
            p_alias, q_alias, d_alias, cdf_alias = acoustics.probabilities(alias_geometry, scenario, clocks, "joint")
            for observed, renamed in zip((p, q, detect, cdf), (p_alias, q_alias, d_alias, cdf_alias)):
                np.testing.assert_array_equal(observed, renamed)
            np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1.e-13, rtol=0)
            assert np.all((q >= 0) & (q <= p + 1.e-13))
            means, _, _ = geometry.parameters(scenario, clocks, "joint")
            for k, (site, view) in enumerate(geometry.addresses):
                if site not in sites:
                    continue
                if view in ANGLES:
                    j = legacy.index[site, view]
                    for observed, expected in zip((p, q, detect, cdf), (p_old, q_old, d_old, cdf_old)):
                        np.testing.assert_array_equal(observed[:, k], expected[:, j])
                    counts["legacy_view_clock_seed_checks"] += len(clocks)
                    continue
                theta = np.deg2rad(geometry.angles[site, view])
                center = world.scan.view_frames[2, site, 3]
                first, second = center[:, 1], center[:, 2]
                image_axes = np.array([np.cos(theta) * first + np.sin(theta) * second,
                                       -np.sin(theta) * first + np.cos(theta) * second])
                origin = world.scan.views[2, site, 3] * 1000.
                labels = (label, 1 - label) if world.identity_site(site) else (0,)
                for kind, lumen in enumerate(labels):
                    base = world.base[2, site, lumen]
                    design = world.design[2, site, lumen]
                    expected = np.vstack([(base + design @ np.array([body.w_subject_mm[site],
                                                                      body.branch_offset_mm,
                                                                      body.z_mm(t), body.w_local]) - origin)
                                           @ image_axes.T + [0., scenario.calibration_depth] for t in clocks])
                    error = float(np.max(np.abs(expected - means[:, k, kind])))
                    assert error < 1.e-5, (site, view, error)
                    counts["max_new_world_mean_error_mm"] = max(counts["max_new_world_mean_error_mm"], error)
                counts["novel_view_clock_seed_checks"] += len(clocks)
                closest = min(ANGLES, key=lambda v: abs(ANGLES[v] - geometry.angles[site, view]))
                sep = 0.5 * np.abs(p[:, k] - p_old[:, legacy.index[site, closest]]).sum(axis=-1)
                separation[name].extend(sep.tolist())
    counts["new_kernel_total_variation_nearest_legacy"] = {
        key: {"mean": float(np.mean(values)), "range": [float(np.min(values)), float(np.max(values))]}
        for key, values in separation.items() if values
    }
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.repo.resolve(), args.bank), indent=2))
