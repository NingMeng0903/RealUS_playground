"""Check the projection symmetry and joint motion on the frozen leg asset."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ab_knee_latent_rl_20261112"))
from knee_world import World


def axes_at(world, angle, site, view):
    frame = world._blend(world.scan.view_frames[:, site, view], angle)
    lateral = frame[:, 1] / np.linalg.norm(frame[:, 1])
    inward = frame[:, 2] - lateral * (lateral @ frame[:, 2])
    inward /= np.linalg.norm(inward)
    return np.stack((lateral, inward), axis=1)


def main():
    world = World()
    rng = np.random.default_rng(7411529)
    rotations = Rotation.random(12, random_state=rng).as_matrix()
    translations = rng.normal(0., 700., (12, 3))
    max_invariant_error = 0.
    max_model_error = 0.
    checks = 0
    for angle in world.knot_angles:
        for site, view in np.argwhere(world.allowed):
            site, view = int(site), int(view)
            labels = (0, 1) if world.identity_site(site) else (0,)
            base, jac = world.world_design(float(angle), [(site, label) for label in labels])
            states = rng.normal(0., 2., 3)
            centerline = base + jac @ states
            probe = world.probe_world(angle, site, view)
            axes = axes_at(world, angle, site, view)
            direct = (centerline - probe) @ axes
            b, h = world.projection(angle, site, view)
            max_model_error = max(max_model_error, float(np.max(np.abs(direct - (b + h @ states)))))
            for rot, trans in zip(rotations, translations):
                transported = ((centerline @ rot.T + trans) - (rot @ probe + trans)) @ (rot @ axes)
                max_invariant_error = max(max_invariant_error, float(np.max(np.abs(transported - direct))))
                checks += 1

    sites = [(i, 0) for i in range(world.n)]
    p, _ = world.world_design(11., sites)
    q, _ = world.world_design(19., sites)
    p0, q0 = p - p.mean(axis=0), q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p0.T @ q0)
    orientation = np.eye(3)
    orientation[-1, -1] = np.linalg.det(u @ vt)
    fitted = p0 @ (u @ orientation @ vt)
    pair_p = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    pair_q = np.linalg.norm(q[:, None] - q[None, :], axis=-1)
    output = {
        "world_to_image": {
            "cases": checks,
            "max_projection_error_mm_after_common_se3": max_invariant_error,
            "max_difference_from_frozen_world_projection_mm": max_model_error,
        },
        "knee_change_11_to_19_degrees": {
            "same_material_addresses": len(p),
            "best_single_global_rigid_fit_rmse_mm": float(np.sqrt(np.mean(np.sum((q0 - fitted) ** 2, axis=1)))),
            "max_change_of_inter_address_distance_mm": float(np.max(np.abs(pair_p - pair_q))),
        },
    }
    assert max_invariant_error < 1.e-9 and max_model_error < 1.e-9
    assert output["knee_change_11_to_19_degrees"]["max_change_of_inter_address_distance_mm"] > .1
    (HERE / "GEOMETRY_CHECK.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
