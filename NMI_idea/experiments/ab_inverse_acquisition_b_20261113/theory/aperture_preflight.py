"""Prior-only aperture/geometry screen; no outcome cohorts or policy execution.

This isolates an off-center-contact mechanism with a canonical world-fixed
ghost. It is not a new benchmark renderer and does not certify robot motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
from scipy.spatial.transform import Rotation


TASKS = {"daughter": ((4, 3), 1), "trunk": ((6, 7), 0)}
VIEWS = (3, 4, 5, 6)
CLOCKS = np.array([1.5, 3.0, 4.5])


def aperture(a, means):
    lower = np.array([-a.WIDTH, a.DEPTH_MIN])
    upper = np.array([a.WIDTH, a.DEPTH_MAX])
    out = np.zeros(means.shape[:-1])
    for sd, weight in ((a.IMAGE_SD, 1. - a.OUTLIER), (a.OUTLIER_SD, a.OUTLIER)):
        out += weight * np.prod(ndtr((upper-means)/sd) - ndtr((lower-means)/sd), axis=-1)
    return out


def unbounded_bins(a, means, nominal):
    relative = means - nominal[None, :, None]
    x = np.r_[-np.inf, a.X_BINS, np.inf]
    z = np.r_[-np.inf, a.Z_BINS, np.inf]
    out = np.zeros((*means.shape[:-1], a.NX, a.NZ))
    for sd, weight in ((a.IMAGE_SD, 1. - a.OUTLIER), (a.OUTLIER_SD, a.OUTLIER)):
        px = np.diff(ndtr((x-relative[..., 0, None])/sd[0]), axis=-1)
        pz = np.diff(ndtr((z-relative[..., 1, None])/sd[1]), axis=-1)
        out += weight*px[..., :, None]*pz[..., None, :]
    return out.reshape(*means.shape[:-1], a.PIXELS)


def nearest_kernel(a, bins, detect):
    far, tie = (bins @ a.FAR.T)*detect[..., None], (bins @ a.TIE.T)*detect[..., None]
    absent = (1.-detect)[..., None]
    present = bins*detect[..., None]
    single, multiple = np.empty_like(bins), np.empty_like(bins)
    for k in range(a.MAX_KINDS):
        i, j = [i for i in range(a.MAX_KINDS) if i != k]
        f, g, t, u = far[..., i, :], far[..., j, :], tie[..., i, :], tie[..., j, :]
        x, y = absent[..., i, :], absent[..., j, :]
        single[..., k, :] = present[..., k, :]*x*y
        multiple[..., k, :] = present[..., k, :]*(
            (f+.5*t)*y+(g+.5*u)*x+f*g+.5*(t*g+u*f)+t*u/3.)
    joint = np.concatenate((single, multiple), axis=-1)
    total = np.concatenate((np.prod(1.-detect, axis=-1)[..., None], joint.sum(axis=-2)), axis=-1)
    correct = np.concatenate((np.zeros_like(total[..., :1]), joint[..., 0, :]), axis=-1)
    return total, correct


def canonical_parameters(a, geometry, scenario):
    """Keep each site's center ghost as one world anchor across all offsets."""
    means, detect, nominal = geometry.parameters(scenario, CLOCKS, "joint")
    mass = aperture(a, means)
    raw = np.divide(detect, mass, out=np.zeros_like(detect), where=mass > 1e-250)
    world = geometry.world
    bias = np.array([0., scenario.calibration_depth])
    for site in sorted({s for s, _ in geometry.addresses}):
        center = geometry.index[site, 3]
        origin = world.probe_world(15., site, 3)
        frame = world._blend(world.scan.view_frames[:, site, 3], 15.)
        labels = (0, 1) if world.identity_site(site) else (0,)
        label = min(scenario.body.ghost_label, len(labels)-1)
        base = world._blend(world.base[:, site, label], 15.)
        projected_base = (base-origin) @ frame[:, 1:3]
        ghost_world = base[None] + (means[:, center, 2]-bias-projected_base) @ frame[:, 1:3].T
        center_raw = raw[:, center].copy()
        for view in VIEWS:
            index = geometry.index[site, view]
            origin_view = world.probe_world(15., site, view)
            frame_view = world._blend(world.scan.view_frames[:, site, view], 15.)
            means[:, index, 2] = (ghost_world-origin_view) @ frame_view[:, 1:3] + bias
            raw[:, index] = center_raw
    mass = aperture(a, means)
    return means, raw*mass, nominal, raw, mass


def geometry_report(z, targets):
    result = []
    start = z["target_robot"][2, 5, 3]
    routes = []
    for v0 in VIEWS:
        for v1 in VIEWS:
            poses = (start, z["target_robot"][2, targets[0], v0], z["target_robot"][2, targets[1], v1])
            travel = sum(np.linalg.norm(b[:3, 3]-a[:3, 3])*1000. for a, b in zip(poses, poses[1:]))
            turn = sum(np.rad2deg(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())
                       for a, b in zip(poses, poses[1:]))
            routes.append(dict(views=[v0, v1], chord_tcp_mm=float(travel),
                               endpoint_turn_lower_bound_deg=float(turn)))
    for position, site in enumerate(targets):
        for view in VIEWS:
            center, target = z["target_robot"][2, site, 3], z["target_robot"][2, site, view]
            supporting = [r for r in routes if r["views"][position] == view]
            shortest = min(supporting, key=lambda r:r["chord_tcp_mm"])
            result.append(dict(site=site, view=view, offset_mm=float(z["view_offsets_mm"][view]),
                actual_contact_shift_mm=float(np.linalg.norm(target[:3,3]-center[:3,3])*1000.),
                frame_rotation_from_center_deg=float(np.rad2deg(Rotation.from_matrix(
                    center[:3,:3].T@target[:3,:3]).magnitude())),
                projection_feasible=bool(z["view_projection_feasible"][site,view]),
                projection_residual_mm=float(z["view_projection_residual_m"][site,view]*1000.),
                archived_static_found=bool(z["static_found"][2,site,view]),
                archived_dird_margin=float(z["dird_margin"][2,site,view]),
                shortest_two_capture_chord_route=shortest,
                necessary_endpoint_bounds_permit_65mm_90deg=any(
                    r["chord_tcp_mm"]<=65. and r["endpoint_turn_lower_bound_deg"]<=90. for r in supporting)))
    return result, routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    legacy = args.repo/"NMI_idea/experiments/ab_full_b_rollout_20261113"
    sys.path.insert(0, str(legacy))
    import acoustics as a
    world = a.setup_world(args.repo)
    geometry_path = args.repo/"NMI_idea/experiments/ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
    z = np.load(geometry_path, allow_pickle=False)
    report = dict(kind="prior_only_geometry_and_aperture_preflight", model_size=args.size,
        prior_seed=int(a.PRIOR_SEED), clocks_s=CLOCKS.tolist(), views=list(VIEWS),
        offsets_mm=[float(z["view_offsets_mm"][v]) for v in VIEWS],
        geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        scenario_source="public model particles only; no screen/training/confirmation cases or outcome access",
        sensor_control="one center-view ghost world anchor per model particle/site, transported rigidly to offsets; center raw detection probabilities retained",
        constraints="geometry-only endpoint lower bounds and archived endpoint metadata; no new checked bank or motion execution",
        tasks={})
    for task, (targets, label) in TASKS.items():
        nodes = [dict(site=s, view=v) for s in targets for v in VIEWS]
        geometry = a.Geometry(world, nodes, label)
        all_p, all_c, all_ap, all_raw, all_detect, all_unbounded = [], [], [], [], [], []
        translation_error = 0.
        for h in range(args.size):
            scenario = a.make_scenario(world, a.seed_for(a.PRIOR_SEED,h))
            mu, detect, nominal, raw, survival = canonical_parameters(a, geometry, scenario)
            bins = a.component_probabilities(mu, nominal)
            p,c = nearest_kernel(a,bins,detect)
            pu,_ = nearest_kernel(a,unbounded_bins(a,mu,nominal),raw)
            for site in targets:
                center=geometry.index[site,3]
                cmu,cnom=mu[:,center:center+1],nominal[center:center+1]
                original=unbounded_bins(a,cmu,cnom)
                for view in VIEWS:
                    shift=(world.probe_world(15.,site,view)-world.probe_world(15.,site,3)) @ world._blend(
                        world.scan.view_frames[:,site,3],15.)[:,1:3]
                    translated=unbounded_bins(a,cmu-shift,cnom-shift)
                    translation_error=max(translation_error,float(np.max(np.abs(original-translated))))
            if not np.allclose(p.sum(axis=-1),1.,atol=2e-12) or np.any(c>p+1e-13):
                raise AssertionError("invalid finite observation kernel")
            all_p.append(p);all_c.append(c);all_ap.append(survival)
            all_raw.append(raw);all_detect.append(detect);all_unbounded.append(pu)
        p,c,ap,raw,det,pu=map(np.asarray,(all_p,all_c,all_ap,all_raw,all_detect,all_unbounded))
        marginal_p,marginal_c=p.mean(axis=0),c.mean(axis=0)
        rows=[]
        for site in targets:
            center=geometry.index[site,3]
            for view in VIEWS:
                j=geometry.index[site,view]
                for tick,clock in enumerate(CLOCKS):
                    accepted=(marginal_c[tick,j]>=.87*marginal_p[tick,j]) & (marginal_p[tick,j]>1e-260)
                    accepted[0]=False
                    target=float(det[:,tick,j,0].mean())
                    non_target=float(det[:,tick,j,1:].sum(axis=-1).mean())
                    base_target=float(det[:,tick,center,0].mean())
                    base_non_target=float(det[:,tick,center,1:].sum(axis=-1).mean())
                    rows.append(dict(site=site,view=view,offset_mm=float(z["view_offsets_mm"][view]),clock_s=float(clock),
                        aperture_survival_by_kind=ap[:,tick,j].mean(axis=0).tolist(),
                        presence_after_aperture_by_kind=det[:,tick,j].mean(axis=0).tolist(),
                        target_detection_delta_from_center=target-base_target,
                        non_target_expected_count_delta_from_center=non_target-base_non_target,
                        target_fraction_of_expected_detected_returns=target/(target+non_target) if target+non_target else None,
                        selected_target_probability=float(marginal_c[tick,j].sum()),
                        coarse_prior_correct_accept_probability=float(marginal_c[tick,j,accepted].sum()),
                        coarse_prior_wrong_accept_probability=float((marginal_p[tick,j,accepted]-marginal_c[tick,j,accepted]).sum()),
                        prior_predictive_tv_from_center=float(.5*np.abs(marginal_p[tick,j]-marginal_p[tick,center]).sum()),
                        mean_particle_tv_from_center=float(.5*np.abs(p[:,tick,j]-p[:,tick,center]).sum(axis=-1).mean()),
                        without_aperture_mean_particle_tv_from_center=float(.5*np.abs(pu[:,tick,j]-pu[:,tick,center]).sum(axis=-1).mean()),
                        no_target_fraction=float((ap[:,tick,j,0]<.05).mean()),
                        target_favored_aperture_fraction=float(((ap[:,tick,j,0]>.8)&(ap[:,tick,j,2]<.5)).mean())))
        geometry_rows,routes=geometry_report(z,targets)
        report["tasks"][task]=dict(prior_only_acoustic_rows=rows,geometry=geometry_rows,
            endpoint_lower_bound_routes=routes,
            pure_translation_unbounded_bins_max_abs_error=translation_error)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({task:{"translation_error":data["pure_translation_unbounded_bins_max_abs_error"],
                           "minimum_chord_mm":min(r["chord_tcp_mm"] for r in data["endpoint_lower_bound_routes"])}
                      for task,data in report["tasks"].items()}))


if __name__=="__main__":
    main()
