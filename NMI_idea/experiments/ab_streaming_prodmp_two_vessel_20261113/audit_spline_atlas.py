"""Audit geometry outside the policy loop, including folded/overlapping controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from atlas_scan import LATERAL_BOUNDS_MM, ScanTemplate
from spline_atlas import SplineTube, TAU
from spline_certificate import certify_tube


def label_error(before, after):
    delta = after - before
    delta[:, 0] = np.angle(np.exp(1j * delta[:, 0]))
    return np.abs(delta).max(axis=0)


def derivative_jump(fun, samples, axis):
    eps = 1.e-7
    shift = np.eye(3)[axis] * eps
    center = fun(samples)
    left = (3. * center - 4. * fun(samples - shift) + fun(samples - 2. * shift)) / (2. * eps)
    right = (-3. * center + 4. * fun(samples + shift) - fun(samples + 2. * shift)) / (2. * eps)
    return float(np.max(np.abs(left - right)))


def smoothness(tube, depth, rng):
    pairs = np.column_stack((rng.uniform(tube.h[0], tube.h[-1], 512), rng.uniform(0., TAU, 512),
                             rng.uniform(tube.r[0], tube.r[-1], 512)))
    knots = (tube.h[1:-1], np.r_[0., tube.angles[1:]], tube.r[1:-1])
    report = {}
    for axis, name in enumerate(("h_knots", "periodic_theta_knots", "r_knots")):
        nodes = pairs.copy()
        nodes[:, axis] = rng.choice(knots[axis], len(nodes))
        if axis == 1:
            nodes[:32, 1] = 0.
        position = derivative_jump(tube.forward, nodes, axis)
        scalar_depth = derivative_jump(lambda x: depth.evaluate(*x.T), nodes, axis)
        assert position <= 2.e-6 and scalar_depth <= 5.e-6, (name, position, scalar_depth)
        report[name] = {"position_derivative_max_jump_m": position, "depth_derivative_max_jump": scalar_depth}
    return report


def fixed_queries(tube, depth):
    rng = np.random.default_rng(115)
    values = rng.uniform((tube.h[0], 0., tube.r[0]), (tube.h[-1], TAU, 1.), (4480, 3))
    # Cover both shells, both longitudinal ends and the circular seam.
    for i, (axis, value) in enumerate(((2, tube.r[0]), (2, 1.), (0, tube.h[0]), (0, tube.h[-1]), (1, 0.))):
        values[3200 + 256*i:3200 + 256*(i+1), axis] = value
    h, theta, r = values.T
    return np.column_stack((theta, h, depth.evaluate(h, theta, r)))


def inspect_geometry(base, smooth):
    tube, depth = smooth.volume_refiner.tube, smooth.volume_refiner.depth
    params = np.stack(np.meshgrid(tube.h, tube.angles, tube.r, indexing="ij"), axis=-1)
    nodes = tube.forward(params).transpose(0, 2, 1, 3)
    node_error = np.linalg.norm(nodes - tube.grid, axis=-1)
    graph_depth = smooth.template.asset.depth.reshape(tube.grid.shape[:-1])
    depths = depth.evaluate(*np.moveaxis(params, -1, 0)).transpose(0, 2, 1)
    depth_error = float(np.max(np.abs(depths - graph_depth)))
    assert node_error.max() <= 2.e-7 and depth_error <= 2.e-14
    query = fixed_queries(tube, depth)
    x = smooth.xi_to_xyz(query)
    recovered = smooth.xyz_to_xi(x)
    labels = label_error(query, recovered)
    reconstruction = float(np.linalg.norm(smooth.xi_to_xyz(recovered) - x, axis=1).max())
    assert np.all(labels <= [6.e-6, 2.e-7, 2.e-6]) and reconstruction < 4.e-9
    try:
        smooth.xyz_to_xi(x[0] + [1., 0., 0.])
    except ValueError:
        pass
    else:
        raise AssertionError("remote outside point was accepted")
    s, xy = np.meshgrid(np.linspace(smooth.s_mm[0], smooth.s_mm[-1], 101),
                        np.linspace(*LATERAL_BOUNDS_MM, 31))
    original, revised = base.pose(s, xy), smooth.pose(s, xy)
    origin = 1000. * np.linalg.norm(revised[..., :3, 3] - original[..., :3, 3], axis=-1)
    rot = np.swapaxes(original[..., :3, :3], -1, -2) @ revised[..., :3, :3]
    angles = np.rad2deg(Rotation.from_matrix(rot.reshape(-1, 3, 3)).magnitude())
    return {
        "beta": smooth.metadata["beta"], "knee_deg": smooth.metadata["knee_deg"],
        "certificate": smooth.metadata["spline_gate"], "lattice_node_count": int(graph_depth.size),
        "maximum_lattice_position_error_m": float(node_error.max()),
        "maximum_graph_depth_knot_error": depth_error,
        "inverse_query_count": len(query), "max_inverse_label_error_rad_h_d": labels.tolist(),
        "max_inverse_roundtrip_m": reconstruction,
        "vessel_binding": smooth.metadata.get("spline_vessel_binding", "legacy"),
        "max_vessel_prior_shift_vs_tet_mm": float(1000. * np.linalg.norm(
            smooth.vessels_m - base.vessels_m, axis=-1).max()),
        "probe_pose_comparison": {"queries": int(s.size), "max_tcp_shift_mm": float(origin.max()),
                                  "max_rotation_deg": float(angles.max())},
        "finite_difference_join_checks": smoothness(tube, depth, rng=np.random.default_rng(116)),
    }


def sample_det(tube):
    h = np.linspace(tube.h[0], tube.h[-1], 257)
    theta = np.linspace(0., TAU, 541, endpoint=False)
    H, T = np.meshgrid(h, theta, indexing="ij")
    minimum = float("inf")
    # The determinant after dividing by r is affine along a radial ray.
    for r in (tube.r[0], 1.):
        params = np.stack((H, T, np.full_like(H, r)), axis=-1)
        _, jacobian = tube.forward(params, derivative=True)
        # Convert the order (h, theta, r) to the oriented solid (h, u, v).
        minimum = min(minimum, float((-np.linalg.det(jacobian) / r).min()))
    return minimum


def rejection_controls(template):
    h, theta, radii = template.chart.levels, template.chart.bearings, template.asset.radial_fractions
    volume, _, _, _ = template._transport.deform("right", (2., 2.), 90.)
    folded = SplineTube(h, theta, radii, volume.astype(np.float32))
    result = {}
    gate = certify_tube(folded.skin.coeff, folded.core_coeff, h, folded.skin.theta)
    minimum = sample_det(folded)
    assert minimum < 0. and not gate["certificate_pass"]
    result["previous_extreme_beta_high_flexion"] = {"minimum_sampled_disk_determinant": minimum, "gate": gate}
    H, R, T = np.meshgrid(h, radii, theta, indexing="ij")
    double = SplineTube(h, theta, radii, np.stack((H, .06 * R * np.cos(2. * T), .06 * R * np.sin(2. * T)), axis=-1))
    points = double.forward([[h[25], 0., .7], [h[25], np.pi, .7]])
    gap = float(np.linalg.norm(points[0] - points[1]))
    minimum = sample_det(double)
    gate = certify_tube(double.skin.coeff, double.core_coeff, h, double.skin.theta)
    assert gap < 2.e-12 and minimum > 0. and not gate["certificate_pass"]
    result["two_to_one_positive_local_jacobian"] = {
        "minimum_sampled_disk_determinant": minimum, "duplicate_position_gap_m": gap, "gate": gate,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spline-binding", choices=("legacy", "registered"), default="registered")
    args = parser.parse_args()
    template = ScanTemplate(args.among_us, args.repair)
    beta2 = template.reference_beta.copy()
    beta2[2] += .5
    shapes = {
        "reference": ((0., 0.), 15., None), "minus-plus": ((-1., 1.), 15., None),
        "plus-minus": ((1., -1.), 15., None), "beta2": ((0., 0.), 15., beta2),
        "pose25": ((0., 0.), 25., None),
    }
    data = {"shapes": {}}
    for key, (delta, knee, beta) in shapes.items():
        world = template.instantiate(delta, knee, beta=beta)
        smooth = template.with_spline(world, args.spline_binding)
        data["shapes"][key] = inspect_geometry(world, smooth)
        print(key, "geometry and inverse passed", flush=True)
    data["rejection_controls"] = rejection_controls(template)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
