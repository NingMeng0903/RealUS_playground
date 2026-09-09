"""Offline-only leg articulation fitting with fixed hip-head anchors.

This optimizer is not called by the runtime. It changes two joint rotations,
then counter-rotates the ankle locally to preserve the authored foot direction.
Every descendant and every original weighted receiver uses the same resulting
controller transforms. No foot translation, bone-length change or rebinding is
introduced at pose time.
"""
from __future__ import annotations

import numpy as np
import igl
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .chain_rest_fit_v1 import _global_to_local
from .pose_map_v1 import _fk
from .sparse_lbs_v14 import SparseLBSV14


def articulated_leg_globals(base, pose55, twists, head_centers_local):
    """Apply per-side hip/knee local rotations and an ankle counter-rotation."""
    global0=base.globals_from_source(base.source_globals(pose55))
    local=_global_to_local(global0,base.parents)
    names=list(base.source_asset.source_bone_names)
    for side,values,center in zip(('L','R'),np.asarray(twists).reshape(2,2,3),head_centers_local):
        hip=names.index(f'Femur_Rot_{side}'); knee=names.index(f'Knee_Rotate_{side}')
        for index,rotation_vector in ((hip,values[0]),(knee,values[1])):
            delta=np.eye(4); delta[:3,:3]=Rotation.from_rotvec(rotation_vector).as_matrix()
            if index==hip:
                delta[:3,3]=center-delta[:3,:3]@center
            local[index]=local[index]@delta
    result=_fk(local,base.parents)
    for side in ('L','R'):
        ankle=names.index(f'Ankle_Rot_{side}'); parent=int(base.parents[ankle])
        local[ankle,:3,:3]=result[parent,:3,:3].T@global0[ankle,:3,:3]
    return _fk(local,base.parents)


def head_centers_local_v17(base,calibration):
    names=list(base.source_asset.source_bone_names)
    result=[]
    for side,suffix in (('left','L'),('right','R')):
        ids=np.asarray(calibration.domains[f'{side}/femoral_head.fit'],dtype=np.int64)
        center=np.mean(base.target_rest[ids],axis=0)
        bind=base.target_bind[names.index(f'Femur_Rot_{suffix}')]
        result.append(bind[:3,:3].T@(center-bind[:3,3]))
    return np.asarray(result)


_SKIN_CONSTRAINED_BONE_QUOTAS_V17 = {
    "Femur": 250,
    "Tibia": 100,
    "Fibula": 80,
    "Patella": 80,
}


def _deterministic_mesh_sample_v17(base, mesh_name, quota):
    """Return evenly spaced, stable source vertex IDs for one named mesh."""
    names = list(base.source_asset.source_mesh_names)
    if mesh_name not in names:
        raise ValueError(f"missing required lower-chain mesh {mesh_name!r}")
    mesh_index = names.index(mesh_name)
    start, stop = (int(value) for value in base.source_asset.source_vertex_ranges[mesh_index])
    count = stop - start
    if count <= 0:
        raise ValueError(f"lower-chain mesh {mesh_name!r} is empty")
    sample_count = min(int(quota), count)
    if sample_count == count:
        return np.arange(start, stop, dtype=np.int64)
    # ``floor`` over an inclusive range gives exactly sample_count distinct
    # rows when count > sample_count and is independent of mesh iteration order.
    offsets = np.floor(np.linspace(0.0, float(count - 1), sample_count)).astype(np.int64)
    return start + offsets


def _skin_constrained_bone_samples_v17(base, suffix):
    """Build fixed sparse-LBS samplers for one side's four lower bones."""
    samples = []
    for stem, quota in _SKIN_CONSTRAINED_BONE_QUOTAS_V17.items():
        name = f"{stem}_{suffix}"
        vertex_ids = _deterministic_mesh_sample_v17(base, name, quota)
        samples.append(
            {
                "mesh": name,
                "vertex_ids": vertex_ids,
                "lbs": SparseLBSV14(
                    base.target_rest[vertex_ids],
                    base.indices[vertex_ids],
                    base.weights[vertex_ids],
                ),
            }
        )
    return samples


def _skin_surface_stats_v17(samples, signed_m):
    """Compact signed-distance stats for each sampled lower-bone mesh."""
    signed_m = np.asarray(signed_m, dtype=np.float64).reshape(-1)
    result = {}
    offset = 0
    for sample in samples:
        count = len(sample["vertex_ids"])
        values = signed_m[offset:offset + count] * 1000.0
        offset += count
        result[sample["mesh"]] = {
            "sample_count": int(count),
            "signed_min_mm": float(np.min(values)),
            "signed_max_mm": float(np.max(values)),
            "max_outside_mm": float(np.max(np.maximum(values, 0.0))),
            "outside_count": int(np.count_nonzero(values > 0.0)),
            "gt5_count": int(np.count_nonzero(values > 5.0)),
            "gt10_count": int(np.count_nonzero(values > 10.0)),
        }
    return result


def fit_leg_pose_v17(
    base,
    pose55,
    smplx_rest_to_pose,
    head_centers_local,
    max_evaluations=60,
    skin_geometry=None,
):
    """Fit bounded articulation, optionally adding fixed bone-skin planes.

    ``skin_geometry`` is an offline-only ``(vertices, faces)`` tuple.  When it
    is omitted the historical kinematic-only solve is retained exactly.  With
    geometry supplied, each side uses two outer SDF/plane rounds; the inner
    least-squares calls only evaluate the fixed planes and sparse authored LBS.
    """
    names=list(base.source_asset.source_bone_names)
    global0=base.globals_from_source(base.source_globals(pose55))
    local0=_global_to_local(global0,base.parents)
    output=np.zeros((2,2,3)); reports=[]
    skin_vertices = skin_faces = None
    if skin_geometry is not None:
        if not isinstance(skin_geometry, (tuple, list)) or len(skin_geometry) != 2:
            raise ValueError("skin_geometry must be a (vertices, faces) pair")
        skin_vertices = np.ascontiguousarray(np.asarray(skin_geometry[0], dtype=np.float64))
        skin_faces = np.ascontiguousarray(np.asarray(skin_geometry[1], dtype=np.int64))
        if (skin_vertices.ndim != 2 or skin_vertices.shape[1] != 3 or len(skin_vertices) == 0
                or not np.isfinite(skin_vertices).all()):
            raise ValueError("skin_geometry vertices must be finite [N,3]")
        if (skin_faces.ndim != 2 or skin_faces.shape[1] != 3 or len(skin_faces) == 0
                or np.any(skin_faces < 0) or np.any(skin_faces >= len(skin_vertices))):
            raise ValueError("skin_geometry faces must be valid integer triangles")
    for side_index,(suffix,knee_joint,ankle_joint) in enumerate((('L',4,7),('R',5,8))):
        hip=names.index(f'Femur_Rot_{suffix}'); knee=names.index(f'Knee_Rotate_{suffix}')
        ankle=names.index(f'Ankle_Rot_{suffix}')
        expected_knee=(smplx_rest_to_pose[knee_joint]@base.target_bind[knee])[:3,3]
        expected_ankle=(smplx_rest_to_pose[ankle_joint]@base.target_bind[ankle])[:3,3]
        center=head_centers_local[side_index]
        samples = _skin_constrained_bone_samples_v17(base, suffix) if skin_vertices is not None else None
        sample_count = sum(len(sample["vertex_ids"]) for sample in samples) if samples else 0

        def evaluate(values):
            local=local0.copy()
            for index,rv in ((hip,values[:3]),(knee,values[3:])):
                delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(rv).as_matrix()
                if index==hip:delta[:3,3]=center-delta[:3,:3]@center
                local[index]=local[index]@delta
            return _fk(local,base.parents)

        def evaluate_sample_points(global_transforms):
            return np.concatenate(
                [sample["lbs"](global_transforms @ base.target_inverse) for sample in samples]
            )

        def kinematic_residual(values):
            g=evaluate(values)
            return np.r_[(g[ankle,:3,3]-expected_ankle)*1000,
                         .15*(g[knee,:3,3]-expected_knee)*1000,
                         .2*np.rad2deg(values)]

        if samples is None:
            result=least_squares(kinematic_residual,np.zeros(6),bounds=(-np.deg2rad(12),np.deg2rad(12)),
                                 max_nfev=max_evaluations,ftol=1e-7,xtol=1e-7,gtol=1e-6)
            values=result.x
            after=evaluate(values)
            reports.append(dict(side=suffix,success=bool(result.success),evaluations=int(result.nfev),
                angles_deg=np.rad2deg(values).tolist(),
                ankle_error_before_mm=float(np.linalg.norm(global0[ankle,:3,3]-expected_ankle)*1000),
                ankle_error_after_mm=float(np.linalg.norm(after[ankle,:3,3]-expected_ankle)*1000),
                knee_error_after_mm=float(np.linalg.norm(after[knee,:3,3]-expected_knee)*1000)))
            output[side_index]=values.reshape(2,3)
            continue

        values=np.zeros(6, dtype=np.float64)
        outer_rounds=[]
        for outer_index in range(2):
            current_global=evaluate(values)
            current_points=evaluate_sample_points(current_global)
            signed, _face, closest, _normal = igl.signed_distance(
                np.ascontiguousarray(current_points, dtype=np.float64),
                skin_vertices,
                skin_faces,
                igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            signed=np.asarray(signed,dtype=np.float64).reshape(-1)
            closest=np.asarray(closest,dtype=np.float64).reshape(-1,3)
            delta=current_points-closest
            distance=np.linalg.norm(delta,axis=1)
            sign=np.where(signed>=0.0,1.0,-1.0)
            normals=np.zeros_like(delta)
            valid_distance=distance>1.0e-12
            normals[valid_distance]=(sign[valid_distance,None]*delta[valid_distance]
                                     /distance[valid_distance,None])
            active=signed>=-0.030
            plane_centers=closest[active].copy()
            plane_normals=normals[active].copy()

            def residual(values):
                g=evaluate(values)
                result=kinematic_residual(values)
                if len(plane_centers):
                    posed=evaluate_sample_points(g)[active]
                    plane_violation=np.maximum(
                        np.einsum('ij,ij->i',plane_normals,posed-plane_centers)-0.003,
                        0.0,
                    )*1000.0
                    result=np.r_[result,np.sqrt(20.0/sample_count)*plane_violation]
                return result

            result=least_squares(residual,values,bounds=(-np.deg2rad(12),np.deg2rad(12)),
                                 max_nfev=max_evaluations,ftol=1e-7,xtol=1e-7,gtol=1e-6)
            values=result.x
            after_global=evaluate(values)
            after_points=evaluate_sample_points(after_global)
            after_signed, _after_face, _after_closest, _after_normal = igl.signed_distance(
                np.ascontiguousarray(after_points,dtype=np.float64),skin_vertices,skin_faces,
                igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            after_signed=np.asarray(after_signed,dtype=np.float64).reshape(-1)
            outer_rounds.append({
                "round": int(outer_index+1),
                "active_point_count": int(np.count_nonzero(active)),
                "sample_count": int(sample_count),
                "plane_constraint_count": int(len(plane_centers)),
                "before_surface_signed_mm": _skin_surface_stats_v17(samples,signed),
                "after_surface_signed_mm": _skin_surface_stats_v17(samples,after_signed),
                "before_max_outside_mm": float(np.max(np.maximum(signed,0.0))*1000.0),
                "after_max_outside_mm": float(np.max(np.maximum(after_signed,0.0))*1000.0),
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "optimizer_message": str(result.message),
                "evaluations": int(result.nfev),
                "ankle_error_after_mm": float(np.linalg.norm(after_global[ankle,:3,3]-expected_ankle)*1000),
                "knee_error_after_mm": float(np.linalg.norm(after_global[knee,:3,3]-expected_knee)*1000),
            })
        output[side_index]=values.reshape(2,3)
        after=evaluate(values)
        reports.append(dict(side=suffix,success=bool(all(row["optimizer_success"] for row in outer_rounds)),
            evaluations=int(sum(row["evaluations"] for row in outer_rounds)),
            angles_deg=np.rad2deg(values).tolist(),
            ankle_error_before_mm=float(np.linalg.norm(global0[ankle,:3,3]-expected_ankle)*1000),
            ankle_error_after_mm=float(np.linalg.norm(after[ankle,:3,3]-expected_ankle)*1000),
            knee_error_after_mm=float(np.linalg.norm(after[knee,:3,3]-expected_knee)*1000),
            skin_geometry_constraint=True,
            sampled_bone_meshes=[dict(mesh=sample["mesh"],sample_count=int(len(sample["vertex_ids"]))) for sample in samples],
            outer_rounds=outer_rounds))
    return output,reports
