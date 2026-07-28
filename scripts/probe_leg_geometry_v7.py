"""Probe posed leg geometry for one V7 subject/pose cell.

Reports the quantities the knee/hip/patellofemoral gates key on, recomputed
straight from the final posed vertex array and posed bone matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    apply_subject_pose,
    load_subject_asset,
)


def _load_pose(spec: str) -> np.ndarray:
    if spec == "zero":
        return np.zeros((55, 3), dtype=np.float64)
    data = np.load(spec)
    return np.asarray(data["pose_axis_angle"], dtype=np.float64).reshape(55, 3)


def _domain(domains: dict, key: str) -> np.ndarray:
    return np.asarray(domains[key], dtype=np.int64).reshape(-1)


def _pair_gap(vertices: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.spatial import cKDTree

    tree = cKDTree(vertices[b])
    distances, _ = tree.query(vertices[a], k=1)
    return float(np.min(distances)), float(np.max(distances))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--domains", required=True)
    args = parser.parse_args()

    subject = load_subject_asset(args.subject)
    pose = _load_pose(args.pose)
    domains = json.loads(Path(args.domains).read_text())
    domain_map = domains.get("domains", domains)

    vertices = apply_subject_pose(
        subject, pose_axis_angle=pose, transl=None, validate=False
    ).astype(np.float64)

    report: dict[str, object] = {}
    for side in ("left", "right"):
        prefix = f"{side}/"
        entry: dict[str, object] = {}
        for name, (a_key, b_key) in {
            "medial": (
                f"{prefix}femoral_condyle_medial",
                f"{prefix}tibial_plateau_medial",
            ),
            "lateral": (
                f"{prefix}femoral_condyle_lateral",
                f"{prefix}tibial_plateau_lateral",
            ),
            "patellofemoral": (f"{prefix}patella_articular", f"{prefix}trochlea"),
            "patella_vs_femur": (f"{prefix}patella_articular", f"{prefix}femur"),
            "hip": (f"{prefix}femoral_head", f"{prefix}acetabulum"),
        }.items():
            if a_key not in domain_map or b_key not in domain_map:
                entry[name] = "missing domain"
                continue
            gap_min, gap_max = _pair_gap(
                vertices, _domain(domain_map, a_key), _domain(domain_map, b_key)
            )
            entry[name] = {"min_mm": gap_min * 1000.0, "max_mm": gap_max * 1000.0}
        report[side] = entry

    report["bones"] = _bone_attitudes(subject, pose)
    report["solve"] = _solve_trace(subject, pose)
    print(json.dumps(report, indent=2, default=float))
    return 0


def _solve_trace(subject, pose: np.ndarray) -> dict[str, object]:
    """Why the hinge IK needs the hip twist it asks for."""
    from scipy.spatial.transform import Rotation

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        _validate_leg_hinge_solve_entry_v1,
        build_source_driver_coupling,
        joint_global_transforms,
        solve_leg_hinge_v1,
        source_bone_driver_frames,
    )

    asset = subject.rigged_asset
    names = list(asset.source_bone_names)
    rest = np.asarray(asset.source_rest_global, dtype=np.float64)
    drivers = source_bone_driver_frames(asset, pose)
    coupling = np.asarray(build_source_driver_coupling(asset), dtype=np.float64)
    target = joint_global_transforms(
        pose_axis_angle=pose,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    points = target[:, :3, 3]

    def unit(v):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return v / max(float(np.linalg.norm(v)), 1.0e-12)

    def acute(a, b):
        return float(
            np.degrees(np.arccos(min(abs(float(np.dot(unit(a), unit(b)))), 1.0)))
        )

    out: dict[str, object] = {}
    leg_solve = (asset.metadata or {}).get("source_leg_hinge_solve_v1") or {}
    for side, raw in leg_solve.items():
        entry = _validate_leg_hinge_solve_entry_v1(
            raw, side=str(side), bone_count=len(names), joint_count=len(points)
        )
        femur = int(entry["femur_bone"])
        hip_j, knee_j, ankle_j = (
            int(entry["smplx_hip"]),
            int(entry["smplx_knee"]),
            int(entry["smplx_ankle"]),
        )
        H = points[hip_j]
        K = H + (points[knee_j] - points[hip_j])
        A = K + (points[knee_j + 0] * 0 + points[ankle_j] - points[knee_j])
        R_driver = (drivers[femur] @ coupling[femur])[:3, :3]
        R_bind = rest[femur, :3, :3]
        h0_local = np.asarray(entry["hinge_axis_femur_local"], dtype=np.float64)

        d = unit(K - H)
        drive_normal = np.cross(K - H, A - K)
        h_driver = unit(R_driver @ h0_local)
        R_femur, theta, _raw, hinge_world = solve_leg_hinge_v1(
            hip=H,
            knee=K,
            ankle=A,
            bind_hip=rest[femur, :3, 3],
            bind_knee=rest[int(entry["knee_bone"]), :3, 3],
            bind_ankle=rest[int(entry["ankle_bone"]), :3, 3],
            bind_femur_rotation=R_bind,
            hinge_axis_femur_local=h0_local,
            driver_femur_rotation=R_driver,
            blend_lo_deg=float(entry["blend_lo_deg"]),
            blend_hi_deg=float(entry["blend_hi_deg"]),
        )
        rotvec = Rotation.from_matrix(R_driver.T @ R_femur).as_rotvec()
        out[str(side)] = {
            "drive_flexion_deg": float(
                np.degrees(
                    np.arccos(
                        float(np.clip(np.dot(d, unit(A - K)), -1.0, 1.0))
                    )
                )
            ),
            "theta_deg": float(np.degrees(theta)),
            "driver_hinge_vs_drive_normal_deg": acute(h_driver, drive_normal),
            "posed_hinge_vs_drive_normal_deg": acute(hinge_world, drive_normal),
            "hip_twist_about_long_axis_deg": float(
                np.degrees(float(np.dot(rotvec, d)))
            ),
            "hip_offaxis_deg": float(
                np.degrees(
                    float(np.linalg.norm(rotvec - float(np.dot(rotvec, d)) * d))
                )
            ),
        }
    return out


def _bone_attitudes(subject, pose: np.ndarray) -> dict[str, object]:
    """Femur twist vs its SMPL-X driver and the applied knee flexion."""
    from scipy.spatial.transform import Rotation

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        source_bone_driver_frames,
        source_bone_posed_global,
    )

    asset = subject.rigged_asset
    names = list(asset.source_bone_names)
    posed = source_bone_posed_global(asset, pose)
    drivers = source_bone_driver_frames(asset, pose)
    rest = np.asarray(asset.source_rest_global, dtype=np.float64)
    out: dict[str, object] = {}
    for side in ("L", "R"):
        for bone in (f"Femur_Rot_{side}", f"Knee_Rotate_{side}", f"Tibia_Bone_{side}"):
            if bone not in names:
                continue
            index = names.index(bone)
            relative = drivers[index][:3, :3].T @ posed[index][:3, :3]
            local_delta = np.linalg.inv(rest[index]) @ posed[index]
            out[bone] = {
                "twist_vs_driver_deg": float(
                    np.degrees(Rotation.from_matrix(relative).magnitude())
                ),
                "rotation_vs_rest_deg": float(
                    np.degrees(
                        Rotation.from_matrix(local_delta[:3, :3]).magnitude()
                    )
                ),
            }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
