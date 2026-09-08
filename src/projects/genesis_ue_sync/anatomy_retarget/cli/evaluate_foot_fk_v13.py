"""Probe a foot-only parent-local FK ablation on the frozen V13 runtime.

The production V10 map keeps both hand and foot terminal subtrees on the
hybrid source-global policy.  This bounded experiment restores target
parent-local FK only for the two ``Ankle_Rot_*`` subtrees while leaving hands,
the target rest, sparse 14-slot weights, and the attachment compiler
untouched.  It consumes the already-published subject-213328 runtime and its
five real comparison cells; no source artifact is rebuilt.

The output is deliberately non-publishable.  ``source_vertices`` in each
comparison NPZ are the existing V10-hybrid runtime candidate and
``candidate_vertices`` are the foot-FK ablation, so the renderer shows the
actual before/after geometry under one camera contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    source_bone_posed_global,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    _global_to_local,
    _weighted_rest_correction,
)
from projects.genesis_ue_sync.anatomy_retarget.material_runtime_v13 import (
    load_material_runtime_v13,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    _se3,
    apply_pose_map_global_v10,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUNTIME = ROOT / (
    "outputs/anatomy_retarget/v13_material_20260908_001/"
    "subjects/subject_213328/runtime"
)
DEFAULT_COMPARISONS = ROOT / (
    "outputs/anatomy_retarget/v13_material_20260908_001/"
    "subjects/subject_213328/comparisons"
)
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_foot_fk_20260908_001"

POSE_NAMES = (
    "tpose",
    "pose_213328",
    "heldout_sitting",
    "heldout_kicking",
    "flex_knee_elbow_120",
)
FOOT_ROOTS = ("Ankle_Rot_L", "Ankle_Rot_R")
SMPLX_ANKLE_IDS = {"L": 7, "R": 8}
SIDE_ROOTS = {"L": "Ankle_Rot_L", "R": "Ankle_Rot_R"}
SIDE_LABELS = {"L": "left", "R": "right"}
SIDE_SHANK_CONTROLLERS = {"L": "Tibia_Bone_L", "R": "Tibia_Bone_R"}
NONREGRESSION_TOLERANCE_M = 1.0e-6
REST_TOLERANCE_M = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _descendants(root: int, parents: np.ndarray) -> set[int]:
    result: set[int] = set()
    for bone in range(len(parents)):
        current = bone
        while current >= 0:
            if current == int(root):
                result.add(bone)
                break
            current = int(parents[current])
    return result


def _hierarchy_depth(bone: int, parents: np.ndarray) -> int:
    depth = 0
    current = int(bone)
    while int(parents[current]) >= 0:
        depth += 1
        current = int(parents[current])
    return depth


def _foot_parent_local_globals(runtime: Any, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (V10 hybrid globals, foot-parent-local ablation globals)."""

    pose_map = runtime.pose_map
    asset = runtime.source_asset
    baseline = np.asarray(
        apply_pose_map_global_v10(
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
        ),
        dtype=np.float64,
    )
    parents = np.asarray(pose_map.bone_parents, dtype=np.int64)
    names = [str(name) for name in pose_map.bone_names.tolist()]
    source_global = np.asarray(source_bone_posed_global(asset, pose), dtype=np.float64)
    source_local = _global_to_local(source_global, parents)
    source_bind_local = np.asarray(pose_map.source_bind_local, dtype=np.float64)
    target_bind_local = np.asarray(pose_map.target_bind_local, dtype=np.float64)

    foot_ids: set[int] = set()
    for root_name in FOOT_ROOTS:
        if root_name not in names:
            raise ValueError(f"runtime pose map has no foot root {root_name!r}")
        foot_ids |= _descendants(names.index(root_name), parents)

    result = baseline.copy()
    for bone in sorted(foot_ids, key=lambda value: (_hierarchy_depth(value, parents), value)):
        # This is exactly the V10 parent-local residual composition, applied
        # only to the foot roots and their descendants.  The parent outside a
        # foot subtree remains the hybrid V10 global, while descendants use
        # the already rebuilt ablation parent.
        delta = np.linalg.inv(source_bind_local[bone]) @ source_local[bone]
        local_pose = target_bind_local[bone] @ _se3(delta[:3, :3], delta[:3, 3])
        parent = int(parents[bone])
        result[bone] = local_pose if parent < 0 else result[parent] @ local_pose

    if not np.all(np.isfinite(result)):
        raise ValueError("foot parent-local FK produced non-finite globals")
    return baseline, result


def _apply_foot_ablation(
    runtime: Any,
    pose: np.ndarray,
    *,
    soft_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the foot-only target globals with the unchanged runtime weights."""

    baseline_globals, ablated_globals = _foot_parent_local_globals(runtime, pose)
    asset = runtime.source_asset
    transforms = ablated_globals @ np.asarray(runtime.pose_map.target_inverse_bind, dtype=np.float64)
    result = _weighted_rest_correction(
        np.asarray(runtime.target_rest, dtype=np.float64),
        np.asarray(asset.driver_indices),
        np.asarray(asset.driver_weights),
        transforms,
    )
    source = np.asarray(runtime.apply_pose(pose, mode="source"), dtype=np.float64)
    if soft_mode == "attachments":
        if runtime.attachment is None:
            raise ValueError("--soft-mode attachments requested but runtime has no attachment map")
        soft_ids = np.asarray(runtime.soft_ids, dtype=np.int64)
        result[soft_ids] = runtime.attachment.transport(
            source[soft_ids], source, result
        )
    elif soft_mode != "weights":
        raise ValueError(f"unsupported soft mode {soft_mode!r}")
    if not np.all(np.isfinite(result)):
        raise ValueError("foot ablation produced non-finite vertices")
    return baseline_globals, ablated_globals, np.asarray(result, dtype=np.float32)


def _region_vertex_ids(runtime: Any) -> dict[str, np.ndarray]:
    asset = runtime.source_asset
    names = [str(name) for name in asset.source_bone_names]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = [str(value).strip().lower() for value in asset.source_tissues]
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64).reshape(-1)
    result: dict[str, np.ndarray] = {}

    for side, root_name in SIDE_ROOTS.items():
        root = names.index(root_name)
        allowed = _descendants(root, parents)
        chunks = [
            np.arange(int(start), int(stop), dtype=np.int64)
            for tissue, (start, stop), controller in zip(tissues, ranges.tolist(), controllers.tolist())
            if tissue == "bone" and int(controller) in allowed
        ]
        if not chunks:
            raise ValueError(f"no bone vertices found for {root_name}")
        result[f"{SIDE_LABELS[side]}_foot"] = np.concatenate(chunks)

    for side, controller_name in SIDE_SHANK_CONTROLLERS.items():
        controller = names.index(controller_name)
        chunks = [
            np.arange(int(start), int(stop), dtype=np.int64)
            for tissue, (start, stop), mesh_controller in zip(
                tissues, ranges.tolist(), controllers.tolist()
            )
            if tissue == "bone" and int(mesh_controller) == controller
        ]
        if not chunks:
            raise ValueError(f"no bone vertices found for {controller_name}")
        result[f"{SIDE_LABELS[side]}_shank"] = np.concatenate(chunks)
    return result


def _mesh_for_vertex(asset: Any, vertex: int) -> str:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    names = [str(value) for value in asset.source_mesh_names]
    for name, (start, stop) in zip(names, ranges.tolist()):
        if int(start) <= int(vertex) < int(stop):
            return name
    return "<unknown>"


def _signed_region_metrics(
    vertices: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    regions: Mapping[str, np.ndarray],
    *,
    asset: Any,
) -> dict[str, Any]:
    names = list(regions)
    ids = np.concatenate([np.asarray(regions[name], dtype=np.int64) for name in names])
    signed = np.asarray(_signed_distance(np.asarray(vertices, dtype=np.float64)[ids], skin, skin_faces))
    if signed.shape != (len(ids),) or not np.all(np.isfinite(signed)):
        raise ValueError("signed-distance region query returned invalid values")
    result: dict[str, Any] = {}
    start = 0
    for name in names:
        region_ids = np.asarray(regions[name], dtype=np.int64)
        values = signed[start : start + len(region_ids)]
        start += len(region_ids)
        local = int(np.argmax(values))
        maximum = float(values[local])
        outside = values > 0.0
        result[name] = {
            "vertex_count": int(len(values)),
            "outside_vertex_count": int(np.count_nonzero(outside)),
            "outside_vertex_fraction": float(np.mean(outside)),
            "max_signed_distance_m": maximum,
            "max_outside_m": max(0.0, maximum),
            "max_outside_mm": max(0.0, maximum * 1000.0),
            "outside_p95_m": float(np.quantile(values[outside], 0.95)) if np.any(outside) else 0.0,
            "max_vertex": int(region_ids[local]),
            "max_mesh": _mesh_for_vertex(asset, int(region_ids[local])),
        }
    return result


def _comparison(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "candidate_vertices",
            "skin_vertices",
            "skin_faces",
            "smplx_joints",
            "pose",
            "faces",
            "skin_faces",
            "vertex_tissue",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing comparison arrays {missing}")
        return {name: data[name].copy() for name in data.files}


def _save_comparison(path: Path, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        faces=np.asarray(arrays["faces"], dtype=np.int32),
        skin_faces=np.asarray(arrays["skin_faces"], dtype=np.int32),
        # Renderer source/candidate panels show existing V10 hybrid versus the
        # foot-FK ablation under exactly the same skin and camera joints.
        source_vertices=np.asarray(arrays["baseline_vertices"], dtype=np.float32),
        candidate_vertices=np.asarray(arrays["ablation_vertices"], dtype=np.float32),
        skin_vertices=np.asarray(arrays["skin_vertices"], dtype=np.float32),
        smplx_joints=np.asarray(arrays["smplx_joints"], dtype=np.float32),
        pose=np.asarray(arrays["pose"], dtype=np.float32),
        vertex_tissue=np.asarray(arrays["vertex_tissue"], dtype=np.int8),
        baseline_vertices=np.asarray(arrays["baseline_vertices"], dtype=np.float32),
        ablation_vertices=np.asarray(arrays["ablation_vertices"], dtype=np.float32),
        original_v10_vertices=np.asarray(arrays["original_v10_vertices"], dtype=np.float32),
        weights_vertices=np.asarray(arrays["weights_vertices"], dtype=np.float32),
        raw_vertices=np.asarray(arrays["raw_vertices"], dtype=np.float32),
        baseline_bone_globals=np.asarray(arrays["baseline_bone_globals"], dtype=np.float64),
        ablation_bone_globals=np.asarray(arrays["ablation_bone_globals"], dtype=np.float64),
        metadata_label=np.asarray("v10_hybrid_to_foot_parent_local_fk"),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )
    np.savez_compressed(path, **payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--soft-mode",
        choices=("attachments", "weights"),
        default="attachments",
        help="pose all non-bone material with fixed attachments or shared weights",
    )
    parser.add_argument(
        "--poses",
        default=",".join(POSE_NAMES),
        help="comma-separated subset of the five frozen comparison pose names",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_path = Path(args.runtime).expanduser().resolve()
    comparisons = Path(args.comparisons).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite foot-FK output: {output}")
    selected = tuple(name.strip() for name in str(args.poses).split(",") if name.strip())
    unknown = sorted(set(selected) - set(POSE_NAMES))
    if unknown or not selected:
        raise ValueError(f"--poses must select names from {POSE_NAMES}, got {unknown or selected}")

    runtime = load_material_runtime_v13(runtime_path)
    if np.asarray(runtime.source_asset.driver_indices).shape[1] != 14:
        raise ValueError("foot FK ablation requires the frozen 14-slot sparse driver weights")
    driver_indices_before = np.array(runtime.source_asset.driver_indices, copy=True)
    driver_weights_before = np.array(runtime.source_asset.driver_weights, copy=True)
    target_rest_before = np.array(runtime.target_rest, copy=True)
    regions = _region_vertex_ids(runtime)
    names = [str(name) for name in runtime.pose_map.bone_names.tolist()]
    parents = np.asarray(runtime.pose_map.bone_parents, dtype=np.int64)
    foot_ids = set()
    for root_name in FOOT_ROOTS:
        foot_ids |= _descendants(names.index(root_name), parents)
    hand_ids = set()
    for root_name in ("Wrist_Rotate_L", "Wrist_Rotate_R1"):
        hand_ids |= _descendants(names.index(root_name), parents)

    output.mkdir(parents=True, exist_ok=False)
    comparison_rows: dict[str, Any] = {}
    progress_path = output / "progress.json"
    input_files = {name: comparisons / f"{name}.npz" for name in selected}
    for name, path in input_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)

    for pose_name in selected:
        cell_started = time.perf_counter()
        comparison = _comparison(input_files[pose_name])
        pose = np.asarray(comparison["pose"], dtype=np.float32).reshape(55, 3)
        baseline_vertices = np.asarray(comparison["candidate_vertices"], dtype=np.float32)
        original_v10_vertices = np.asarray(comparison.get("source_vertices"), dtype=np.float32)
        if original_v10_vertices.shape != baseline_vertices.shape:
            raise ValueError(f"{pose_name}: source/candidate geometry shape mismatch")
        baseline_globals, ablation_globals, ablation_vertices = _apply_foot_ablation(
            runtime,
            pose,
            soft_mode=str(args.soft_mode),
        )
        if pose_name == "tpose":
            rest_vertex_delta = float(
                np.max(np.abs(ablation_vertices.astype(np.float64) - baseline_vertices.astype(np.float64)))
            )
        else:
            rest_vertex_delta = None
        skin = np.asarray(comparison["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(comparison["skin_faces"], dtype=np.int64)
        baseline_surface = _signed_region_metrics(
            baseline_vertices,
            skin,
            skin_faces,
            regions,
            asset=runtime.source_asset,
        )
        ablation_surface = _signed_region_metrics(
            ablation_vertices,
            skin,
            skin_faces,
            regions,
            asset=runtime.source_asset,
        )
        joints = np.asarray(comparison["smplx_joints"], dtype=np.float64).reshape(55, 3)
        ankle_rows: dict[str, Any] = {}
        for side, root_name in SIDE_ROOTS.items():
            bone = names.index(root_name)
            smplx_id = SMPLX_ANKLE_IDS[side]
            baseline_distance = float(np.linalg.norm(baseline_globals[bone, :3, 3] - joints[smplx_id]))
            ablation_distance = float(np.linalg.norm(ablation_globals[bone, :3, 3] - joints[smplx_id]))
            ankle_rows[f"{SIDE_LABELS[side]}_ankle"] = {
                "bone": root_name,
                "smplx_joint_id": smplx_id,
                "baseline_distance_m": baseline_distance,
                "baseline_distance_mm": baseline_distance * 1000.0,
                "ablation_distance_m": ablation_distance,
                "ablation_distance_mm": ablation_distance * 1000.0,
                "delta_m": ablation_distance - baseline_distance,
                "delta_mm": (ablation_distance - baseline_distance) * 1000.0,
                "nonregressed": ablation_distance <= baseline_distance + NONREGRESSION_TOLERANCE_M,
            }
        region_rows: dict[str, Any] = {}
        for region in regions:
            base = baseline_surface[region]
            after = ablation_surface[region]
            delta = float(after["max_outside_m"] - base["max_outside_m"])
            region_rows[region] = {
                "baseline": base,
                "ablation": after,
                "delta_m": delta,
                "delta_mm": delta * 1000.0,
                "nonregressed": delta <= NONREGRESSION_TOLERANCE_M,
            }
        hands_unchanged = float(
            np.max(np.abs(ablation_globals[sorted(hand_ids)] - baseline_globals[sorted(hand_ids)]))
        ) if hand_ids else 0.0
        foot_global_delta = float(
            np.max(np.abs(ablation_globals[sorted(foot_ids)] - baseline_globals[sorted(foot_ids)]))
        ) if foot_ids else 0.0
        row = {
            "status": "complete",
            "pose": pose_name,
            "soft_mode": str(args.soft_mode),
            "regions": region_rows,
            "ankle_joint_distances": ankle_rows,
            "foot_global_max_abs_delta_m": foot_global_delta,
            "hands_hybrid_max_abs_delta_m": hands_unchanged,
            "rest_vertex_delta_m": rest_vertex_delta,
            "elapsed_seconds": float(time.perf_counter() - cell_started),
        }
        comparison_rows[pose_name] = row
        _save_comparison(
            output / "comparisons" / f"{pose_name}.npz",
            {
                "faces": comparison["faces"],
                "skin_faces": comparison["skin_faces"],
                "baseline_vertices": baseline_vertices,
                "ablation_vertices": ablation_vertices,
                "original_v10_vertices": original_v10_vertices,
                "weights_vertices": comparison.get("weights_vertices", baseline_vertices),
                "raw_vertices": comparison.get("raw_vertices", original_v10_vertices),
                "skin_vertices": comparison["skin_vertices"],
                "smplx_joints": comparison["smplx_joints"],
                "pose": pose,
                "vertex_tissue": comparison["vertex_tissue"],
                "baseline_bone_globals": baseline_globals,
                "ablation_bone_globals": ablation_globals,
            },
            {
                "schema_version": 13,
                "artifact_kind": "FootFKComparisonV13",
                "subject": "213328",
                "pose": pose_name,
                "baseline_label": "V10hybrid_foot_source_global_runtime_attachment",
                "ablation_label": "foot_subtree_target_parent_local_FK_runtime_attachment",
                "hands_policy": "unchanged_V10hybrid",
                "rest_policy": "target_rest_unchanged",
                "driver_weight_policy": "source_asset_14_slot_indices_and_weights_unchanged",
                "soft_material_policy": str(args.soft_mode),
                "baseline_input": str(input_files[pose_name]),
                "baseline_input_sha256": _sha256(input_files[pose_name]),
                "metrics": row,
                "publishable": False,
            },
        )
        _write_json(
            progress_path,
            {
                "schema_version": 13,
                "artifact_kind": "FootFKProbeProgressV13",
                "subject": "213328",
                "status": "running",
                "publishable": False,
                "poses": comparison_rows,
            },
        )
        print(
            f"pose={pose_name} foot_fk=complete "
            f"left_foot_delta_mm={region_rows['left_foot']['delta_mm']:.3f} "
            f"right_foot_delta_mm={region_rows['right_foot']['delta_mm']:.3f}",
            flush=True,
        )

    # The ablation must leave the frozen runtime data itself untouched.
    rest_unchanged = bool(np.array_equal(target_rest_before, runtime.target_rest))
    weights_unchanged = bool(
        np.array_equal(driver_indices_before, runtime.source_asset.driver_indices)
        and np.array_equal(driver_weights_before, runtime.source_asset.driver_weights)
    )
    tpose_rest_delta = comparison_rows.get("tpose", {}).get("rest_vertex_delta_m")
    rest_invariant = bool(rest_unchanged and weights_unchanged and (
        tpose_rest_delta is not None and float(tpose_rest_delta) <= REST_TOLERANCE_M
    ))
    rejection_reasons: list[str] = []
    if not rest_invariant:
        rejection_reasons.append("target rest or frozen 14-slot weights changed")
    for pose_name, row in comparison_rows.items():
        if row["hands_hybrid_max_abs_delta_m"] > REST_TOLERANCE_M:
            rejection_reasons.append(
                f"{pose_name}/hands: hybrid hand globals changed by "
                f"{row['hands_hybrid_max_abs_delta_m'] * 1000.0:.6f} mm"
            )
        for region, metrics in row["regions"].items():
            if not metrics["nonregressed"]:
                rejection_reasons.append(
                    f"{pose_name}/{region}: max outside worsened by {metrics['delta_mm']:.3f} mm"
                )
        for ankle, metrics in row["ankle_joint_distances"].items():
            if not metrics["nonregressed"]:
                rejection_reasons.append(
                    f"{pose_name}/{ankle}: SMPL-X ankle distance worsened by {metrics['delta_mm']:.3f} mm"
                )
    report = {
        "schema_version": 13,
        "artifact_kind": "FootFKProbeV13",
        "subject": "213328",
        "runtime_root": str(runtime_path),
        "runtime_manifest_sha256": _sha256(runtime_path / "manifest.json"),
        "comparison_root": str(comparisons),
        "pose_names": list(selected),
        "hypothesis": "restore target parent-local FK only for Ankle_Rot_L/R subtrees",
        "baseline_policy": "existing V10hybrid source-global terminal foot runtime candidate",
        "hands_policy": "V10hybrid unchanged",
        "foot_policy": "target parent-local residual FK; parent outside subtree remains V10hybrid",
        "rest_policy": "target_rest and original runtime source rest unchanged",
        "driver_weight_policy": "14-slot source driver indices/weights unchanged",
        "soft_material_policy": str(args.soft_mode),
        "region_definition": {
            "foot": "bone mesh ranges whose source mesh controller is in the Ankle_Rot side subtree",
            "shank": "bone mesh ranges controlled by Tibia_Bone_L/R (Tibia and Fibula)",
            "signed_distance": "positive outside SMPL-X skin; maximum reported as protrusion",
        },
        "nonregression_tolerance_m": NONREGRESSION_TOLERANCE_M,
        "rest_tolerance_m": REST_TOLERANCE_M,
        "rest_invariant": rest_invariant,
        "weights_unchanged": weights_unchanged,
        "target_rest_unchanged": rest_unchanged,
        "tpose_rest_vertex_delta_m": tpose_rest_delta,
        "poses": comparison_rows,
        "accepted": bool(rest_invariant and not rejection_reasons),
        "rejection_reasons": rejection_reasons,
        "publishable": False,
    }
    _write_json(output / "report.json", report)
    _write_json(output / "progress.json", {**report, "status": "complete"})
    np.savez_compressed(
        output / "region_ids.npz",
        **{name: np.asarray(ids, dtype=np.int64) for name, ids in regions.items()},
    )
    print(
        f"foot_fk_probe subject=213328 accepted={report['accepted']} "
        f"rejections={len(rejection_reasons)} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
