"""Audit public atlas reference paths under one fixed virtual RM75 placement.

Run with the existing genesis Python environment. This is a separate kinematic
gate, not a robot execution result or part of the Cartesian feature pilot.
Only LegWorld.reference_at is used to construct task poses; true_points and
the observation generator are never consulted for candidates or placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .smplx_scene import SmplxAtlasLeg as AtlasLeg
from .simulation import LegWorld, Spec
from .dird_probe import DirdProbe, convert_image_poses, dependency_audit


def reference_poses(world: LegWorld, progress: np.ndarray, mode: int) -> np.ndarray:
    """Extract only the declared nominal public reference, including attitude."""
    poses = np.repeat(np.eye(4)[None], len(progress), axis=0)
    for i, s in enumerate(progress):
        p, r = world.reference_at(float(s), mode=mode)
        poses[i, :3, 3], poses[i, :3, :3] = p, r
    return poses


def _pose_from_fk(kin, q):
    placement = kin.fk_placement(np.asarray(q, dtype=float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = placement.rotation, placement.translation
    return T


def run_audit(output: Path, *, checkpoint: Path, asset_root: str = "/media/camp/EXT_DRIVE/Among_US", allow_stale: bool = False) -> dict:
    started = time.perf_counter()
    atlas = AtlasLeg.load(asset_root)
    probe = DirdProbe(checkpoint=checkpoint,allow_stale_audit=allow_stale, placement_label="single_virtual_placement_canonical_reference_to_fk")
    kin = probe.kinematics()
    progress = np.linspace(.62, .76, 9)
    rails = np.array([.3, .4, .5])
    # Fixed controller nominal arm pose, with a declared mid-stroke raw rail.
    # This is an illustrative virtual entry, not an observed robot configuration.
    q_seed = np.r_[.4, np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])]
    from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
    collision = CollisionModel(kin.model, collision_urdf=probe.spec.collision_urdf,
                               pair_config=probe.spec.collision_pairs)
    collision.update(q_seed)
    seed_clearance = float(collision.min_distance())
    if not np.isfinite(seed_clearance) or seed_clearance < probe.spec.collision_security_margin_m:
        raise RuntimeError("Declared FK seed failed current configured arm self-collision audit")
    if np.any(q_seed < kin.q_lower) or np.any(q_seed > kin.q_upper):
        raise RuntimeError("Declared FK seed exceeds current URDF joint limits")

    T_image_tcp = np.eye(4)
    T_image_tcp[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    # The baseline reference (mode=0) is independent of all hidden deviations.
    # u/v are required by the world fixture but are never read by this script.
    baseline = LegWorld(atlas, Spec("dird_public_reference", 61108, 1, -1))
    first_image = reference_poses(baseline, progress[:1], mode=0)
    first_tcp = convert_image_poses(first_image, T_image_tcp)[0]
    seed_tcp = _pose_from_fk(kin, q_seed)
    # This is the only placement choice; it is NOT refitted for another mode,
    # knee pose or body shape, regardless of the subsequent audit outcomes.
    base_world = first_tcp @ np.linalg.inv(seed_tcp)
    placement_hash = hashlib.sha256(np.ascontiguousarray(base_world).tobytes()).hexdigest()
    anchor_error = np.linalg.norm((base_world @ seed_tcp) - first_tcp)

    cases, summary = [], []
    for pose in (0., 60.):
        for width, length in ((1., 1.), (1.18, .88)):
            spec = Spec("dird_public_reference", 61108, 1, -1, knee_deg=pose,
                        width_scale=width, length_scale=length)
            world = LegWorld(atlas, spec)
            for reference_mode in (-1, 1):
                T_image_world = reference_poses(world, progress, reference_mode)
                T_tcp_world = convert_image_poses(T_image_world, T_image_tcp)
                report = probe.audit_ik_paths(T_tcp_world, base_world, rails, q_seed, top_k=2)
                case_id = f"pose_{pose:g}_width_{width:g}_length_{length:g}_reference_{reference_mode:+d}"
                case = {
                    "case_id": case_id, "knee_deg": pose, "width_scale": width,
                    "length_scale": length, "reference_mode": reference_mode,
                    "placement_sha256": placement_hash,
                    "public_reference_T_image_world": T_image_world.tolist(),
                    "public_reference_T_tcp_world": T_tcp_world.tolist(),
                    "audit": report,
                }
                cases.append(case)
                for mode in report["mode_audits"]:
                    audit = mode["audit"]
                    row = {
                        "case_id": case_id, "knee_deg": pose, "width_scale": width,
                        "length_scale": length, "reference_mode": reference_mode,
                        "rail_m": mode["rail_m"][0], "dird_min_score": mode["dird_min_score"],
                        "lift_valid": audit.get("lift_valid", False),
                        "kinematic_audit_passed": audit.get("kinematic_audit_passed", False),
                        "retiming_passed": audit.get("retiming_passed", False),
                        "nominal_duration_s": audit.get("nominal_duration_s"),
                        "minimum_self_collision_distance_m": audit.get("minimum_self_collision_distance_m"),
                        "entry_to_first_q_delta": audit.get("entry_to_first_q_delta"),
                        "failure": audit.get("failure", ""), "ready_for_execution": False,
                    }
                    summary.append(row)
                print(json.dumps({"case_id": case_id,
                                  "passed": sum(x["audit"].get("kinematic_audit_passed", False)
                                                and x["audit"].get("retiming_passed", False)
                                                for x in report["mode_audits"]),
                                  "audited": len(report["mode_audits"])}), flush=True)

    report = {
        "schema": "atlas_public_reference_dird_audit_v1",
        "physical_execution": False, "ready_for_execution": False,
        "included_in_cartesian_pilot_dynamics": False,
        "checkpoint_metadata": probe.metadata(), "dependencies": dependency_audit(),
        "source_atlas": str(atlas.atlas_path), "source_vessel_reference": str(atlas.vessel_path),
        "candidate_source": "LegWorld.reference_at public nominal material route only",
        "hidden_true_curve_used_for_candidates": False,
        "progress": progress.tolist(), "rail_candidates_m": rails.tolist(), "top_k_audited": 2,
        "q_seed": q_seed.tolist(),
        "q_seed_source": "fixed controller nominal arm angles; virtual raw rail_y=0.4m",
        "seed_min_configured_arm_self_collision_distance_m": seed_clearance,
        "base_world": base_world.tolist(), "placement_sha256": placement_hash,
        "placement_rule": "Canonical 0deg shape(1,1) public mode0 first waypoint mapped to fixed seed FK; frozen for every case",
        "placement_anchor_matrix_error": float(anchor_error),
        "T_image_tcp": T_image_tcp.tolist(),
        "image_tcp_transform_status": "illustrative proper Rx(pi), zero origin offset; NOT physically calibrated",
        "geometry_model": "native SMPL-X55 with preserved material chart and fixed canonical interior attachment",
        "limitations": ["No leg/bed/rail environment collision validation",
                        "No transition from actual entry to q_ref[0] has been planned or timed",
                        "No contact, force, acquired image, or hardware validation",
                        "Timing is existing model retiming of scan path only, not measured execution duration",
                        "SRS search stays in the supplied seed branch; failure does not prove global unreachability"],
        "cases": cases, "summary": summary,
        "mode_audits": len(summary),
        "passed_mode_audits": sum(r["kinematic_audit_passed"] and r["retiming_passed"] for r in summary),
        "wall_seconds": time.perf_counter() - started,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--allow-stale-audit", action="store_true")
    parser.add_argument("--asset-root", default="/media/camp/EXT_DRIVE/Among_US")
    args = parser.parse_args()
    report = run_audit(args.output, checkpoint=args.checkpoint, asset_root=args.asset_root, allow_stale=args.allow_stale_audit)
    print(json.dumps({"output": str(args.output), "passed": report["passed_mode_audits"],
                      "audited": report["mode_audits"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
