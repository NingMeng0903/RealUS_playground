"""Independent lower-chain surface measurement for V17 geometry exports.

The command only reads exported geometry, an optional compiled runtime, and
the frozen calibration domains.  It does not fit, rebind, or alter any input.
The signed distance is a point-to-triangle distance with libigl's generalized
winding-number sign.  The report records skin topology because the SMPL-X
surface used by the captures has a mouth boundary and therefore is not a
strict closed containment surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import igl
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_CALIBRATION = (
    _REPO_ROOT
    / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006"
    / "anatomical_calibration_v1/anatomical_calibration_v1.npz"
)

_MAJOR_BASES = ("Femur", "Tibia", "Fibula", "Patella", "Talus", "Calcaneus")
_MAJOR_NAMES = frozenset(
    f"{base}_{side}" for base in _MAJOR_BASES for side in ("L", "R")
)

# These tokens cover the authored lower chain, including every toe segment.
# Ilium/Sacrum are included because they are the proximal pelvic members of
# the lower-chain contact graph.
_LOWER_NAME_TOKENS = (
    "ilium",
    "sacrum",
    "femur",
    "patella",
    "tibia",
    "fibula",
    "talus",
    "calcaneus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
)

_CAP_DOMAIN_SUFFIXES = (".fit",)
_CAP_DOMAIN_TOKENS = (
    "femoral_head",
    "femoral_condyle_medial",
    "femoral_condyle_lateral",
    "tibial_plateau_medial",
    "tibial_plateau_lateral",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_counts(faces: np.ndarray) -> tuple[int, int, int]:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return (
        int(np.count_nonzero(counts == 1)),
        int(np.count_nonzero(counts > 2)),
        int(len(counts)),
    )


def _mesh_ids(ranges: np.ndarray, mesh_index: int) -> np.ndarray:
    start, stop = (int(value) for value in ranges[mesh_index])
    return np.arange(start, stop, dtype=np.int64)


def _stat(signed_mm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(signed_mm, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    values = values[finite]
    count = int(len(values))
    outside = values > 0.0
    gt5 = values > 5.0
    gt10 = values > 10.0
    return {
        "point_count": count,
        "finite_count": int(np.count_nonzero(finite)),
        "signed_min_mm": float(np.min(values)) if count else None,
        "signed_max_mm": float(np.max(values)) if count else None,
        "max_outside_mm": float(np.max(np.maximum(values, 0.0))) if count else None,
        "outside_count": int(np.count_nonzero(outside)),
        "outside_fraction": float(np.mean(outside)) if count else None,
        "gt5_count": int(np.count_nonzero(gt5)),
        "gt5_fraction": float(np.mean(gt5)) if count else None,
        "gt10_count": int(np.count_nonzero(gt10)),
        "gt10_fraction": float(np.mean(gt10)) if count else None,
        "signed_p50_mm": float(np.percentile(values, 50)) if count else None,
        "signed_p95_mm": float(np.percentile(values, 95)) if count else None,
        "signed_p99_mm": float(np.percentile(values, 99)) if count else None,
    }


def _signed_distance(
    points: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
) -> np.ndarray:
    """Return point-to-triangle signed distances in millimetres.

    ``WINDING_NUMBER`` is deliberately selected instead of a closest-face
    normal or a proxy sign.  The skin may still be open, which is reported by
    the caller and keeps this measurement from being mistaken for a strict
    containment certificate.
    """

    signed, _face, _closest, _normal = igl.signed_distance(
        np.ascontiguousarray(np.asarray(points, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(skin_vertices, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(skin_faces, dtype=np.int64)),
        igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
    )
    return np.asarray(signed, dtype=np.float64) * 1000.0


def _kabsch_residual(before: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    x = np.asarray(before, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 3:
        return {
            "available": False,
            "point_count": int(len(x)),
            "reason": "insufficient_or_mismatched_points",
        }
    cx = np.mean(x, axis=0)
    cy = np.mean(y, axis=0)
    u, _singular, vt = np.linalg.svd((x - cx).T @ (y - cy), full_matrices=False)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    fitted = (x - cx) @ rotation.T + cy
    residual = np.linalg.norm(y - fitted, axis=1) * 1000.0
    displacement = np.linalg.norm(y - x, axis=1) * 1000.0
    return {
        "available": True,
        "point_count": int(len(x)),
        "rms_mm": float(np.sqrt(np.mean(residual * residual))),
        "p95_mm": float(np.percentile(residual, 95)),
        "max_mm": float(np.max(residual)),
        "mean_mm": float(np.mean(residual)),
        "raw_displacement_max_mm": float(np.max(displacement)),
        "raw_displacement_mean_mm": float(np.mean(displacement)),
        "det_rotation": float(np.linalg.det(rotation)),
    }


def _runtime_candidates(geometry: Path) -> list[Path]:
    parent = geometry.resolve().parent
    return [
        parent / "compiled_authenticated/runtime",
        parent / "compiled/runtime",
        parent / "compiled",
    ]


def _load_runtime(
    geometry: Path, runtime: Path | None
) -> tuple[Any | None, Path | None, dict[str, Any]]:
    """Load a neighboring compiled runtime when available.

    The CLI remains usable with geometry-only exports.  Runtime loading is
    only used to verify mesh/controller identity and choose lower controller
    members; it never evaluates or mutates the runtime.
    """

    try:
        from ..consistent_runtime_v14 import load_compiled_subject
    except ImportError:
        try:
            # Support both ``python -m projects...`` and direct execution
            # after the repository's ``src`` has been placed on PYTHONPATH.
            from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
                load_compiled_subject,
            )
        except ImportError:
            load_compiled_subject = None
    if load_compiled_subject is None:
        return None, None, {
            "loaded": False,
            "reason": "compiled runtime loader import unavailable",
            "attempts": [],
        }
    candidates = [runtime] if runtime is not None else _runtime_candidates(geometry)
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            attempts.append(
                {
                    "path": str(candidate) if candidate is not None else None,
                    "exists": False,
                    "loaded": False,
                    "reason": "path_missing",
                }
            )
            continue
        try:
            loaded = load_compiled_subject(candidate)
            attempts.append(
                {
                    "path": str(candidate),
                    "exists": True,
                    "loaded": True,
                }
            )
            return loaded, candidate.resolve(), {
                "loaded": True,
                "attempts": attempts,
            }
        except Exception as exc:
            # A damaged neighboring runtime must be visible in the report.
            # Geometry-only measurement may continue by name, but it is then
            # explicitly marked as lacking 235-controller identity evidence.
            attempts.append(
                {
                    "path": str(candidate),
                    "exists": True,
                    "loaded": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            continue
    return None, None, {
        "loaded": False,
        "attempts": attempts,
        "reason": "all runtime candidates failed or were missing",
    }


def _select_meshes(
    names: list[str],
    tissues: list[str],
    controller_ids: np.ndarray | None,
    controller_names: list[str] | None,
    controller_parents: np.ndarray | None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    major = [
        i for i, (name, tissue) in enumerate(zip(names, tissues))
        if tissue == "bone" and name in _MAJOR_NAMES
    ]
    artery = [
        i for i, (name, tissue) in enumerate(zip(names, tissues))
        if tissue == "vessel" and name == "Artery"
    ]
    vein = [
        i for i, (name, tissue) in enumerate(zip(names, tissues))
        if tissue == "vessel" and name == "Vein"
    ]
    nerves = [i for i, tissue in enumerate(tissues) if tissue == "nerve"]

    lower_by_name = [
        i
        for i, (name, tissue) in enumerate(zip(names, tissues))
        if tissue == "bone"
        and any(token in name.lower() for token in _LOWER_NAME_TOKENS)
    ]
    lower_by_controller: list[int] = []
    lower_controller_names: list[str] = []
    if controller_ids is not None and controller_names is not None and controller_parents is not None:
        lower_roots = {
            name
            for name in (
                "Femur_Rot_L",
                "Knee_Rotate_L",
                "Tibia_Bone_L",
                "Tibia_Twist_L",
                "Ankle_Rot_L",
                "Patella_Rotate_L",
                "Femur_Rot_R",
                "Knee_Rotate_R",
                "Tibia_Bone_R",
                "Tibia_Twist_R",
                "Ankle_Rot_R",
                "Patella_Rotate_R",
            )
            if name in controller_names
        }
        roots = {controller_names.index(name) for name in lower_roots}
        lower_controller_ids: set[int] = set(roots)
        for bone in range(len(controller_names)):
            current = bone
            for _ in range(len(controller_names) + 1):
                if current in roots:
                    lower_controller_ids.add(bone)
                    break
                if current < 0:
                    break
                current = int(controller_parents[current])
        lower_controller_names = [controller_names[i] for i in sorted(lower_controller_ids)]
        lower_by_controller = [
            i for i, controller in enumerate(controller_ids.tolist())
            if int(controller) in lower_controller_ids and tissues[i] == "bone"
        ]

    lower = sorted(set(lower_by_name) | set(lower_by_controller))
    missing_major = sorted(_MAJOR_NAMES - {names[i] for i in major})
    required_toe = [
        i for i, (name, tissue) in enumerate(zip(names, tissues))
        if tissue == "bone" and "phalanx_foot" in name.lower()
    ]
    omitted_toe = sorted(set(required_toe) - set(lower))
    if missing_major:
        raise ValueError(f"geometry is missing required major bone meshes: {missing_major}")
    if omitted_toe:
        raise ValueError(
            "lower-bone selection omitted toe phalanges: "
            + repr([names[i] for i in omitted_toe])
        )
    selection = {
        "major_bones": major,
        "Artery": artery,
        "Vein": vein,
        "all_nerves": nerves,
        "all_lower_bones": lower,
    }
    provenance = {
        "name_token_set": list(_LOWER_NAME_TOKENS),
        "selection_rule": "bone mesh name tokens union runtime lower-controller descendants",
        "runtime_lower_controller_names": lower_controller_names,
        "runtime_lower_controller_available": bool(lower_controller_names),
        "major_bone_mesh_names": [names[i] for i in major],
        "artery_mesh_names": [names[i] for i in artery],
        "vein_mesh_names": [names[i] for i in vein],
        "nerve_mesh_names": [names[i] for i in nerves],
        "toe_mesh_names": [names[i] for i in required_toe],
        "selected_lower_bone_mesh_names": [names[i] for i in lower],
    }
    return selection, provenance


def _domain_lookup(calibration: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    values = np.load(calibration, allow_pickle=False)
    names = [str(value) for value in values["domain_names"].tolist()]
    offsets = np.asarray(values["domain_offsets"], dtype=np.int64)
    vertex_ids = np.asarray(values["domain_vertex_ids"], dtype=np.int64)
    domains = {
        name: vertex_ids[int(offsets[index]) : int(offsets[index + 1])]
        for index, name in enumerate(names)
    }
    wanted = [
        name
        for name in names
        if name.endswith(_CAP_DOMAIN_SUFFIXES)
        and any(token in name for token in _CAP_DOMAIN_TOKENS)
    ]
    if not wanted:
        raise ValueError(f"calibration has no femoral-head/condyle/tibial-plateau fit domains: {calibration}")
    return {name: np.asarray(domains[name], dtype=np.int64) for name in wanted}, wanted


def _read_compiled_report(geometry: Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = geometry.resolve().parent / "report.json"
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), path
    except (OSError, json.JSONDecodeError):
        return None, path


def evaluate(
    geometry: Path,
    output: Path,
    calibration: Path,
    runtime: Path | None = None,
) -> dict[str, Any]:
    geometry = geometry.resolve()
    output = output.resolve()
    calibration = calibration.resolve()
    if not geometry.is_dir():
        raise NotADirectoryError(geometry)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    scene_paths = sorted(path for path in geometry.glob("*.npz") if path.is_file())
    if not scene_paths:
        raise FileNotFoundError(f"no geometry NPZ files in {geometry}")
    output.mkdir(parents=True, exist_ok=True)

    compiled, runtime_path, runtime_load = _load_runtime(geometry, runtime)
    runtime_asset = getattr(compiled, "source_asset", None)
    calibration_domains, calibration_domain_names = _domain_lookup(calibration)
    compiled_report, compiled_report_path = _read_compiled_report(geometry)

    scenes: dict[str, Any] = {}
    selection_provenance: dict[str, Any] | None = None
    first_target_beta: list[float] | None = None
    for scene_path in scene_paths:
        data = np.load(scene_path, allow_pickle=False)
        required = {
            "before_vertices",
            "candidate_vertices",
            "skin_vertices",
            "skin_faces",
            "mesh_names",
            "mesh_tissues",
            "mesh_ranges",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{scene_path} is missing geometry fields: {missing}")
        names = [str(value) for value in data["mesh_names"].tolist()]
        tissues = [str(value).strip().lower() for value in data["mesh_tissues"].tolist()]
        ranges = np.asarray(data["mesh_ranges"], dtype=np.int64)
        before_vertices = np.asarray(data["before_vertices"], dtype=np.float64)
        candidate_vertices = np.asarray(data["candidate_vertices"], dtype=np.float64)
        if before_vertices.shape != candidate_vertices.shape or before_vertices.ndim != 2:
            raise ValueError(f"before/candidate shape mismatch in {scene_path}")
        if ranges.shape != (len(names), 2) or len(tissues) != len(names):
            raise ValueError(f"mesh metadata shape mismatch in {scene_path}")
        skin_vertices = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int64)
        boundary_edges, nonmanifold_edges, unique_edges = _topology_counts(skin_faces)

        controller_ids = None
        controller_names = None
        controller_parents = None
        if runtime_asset is not None:
            runtime_names = [str(value) for value in runtime_asset.source_mesh_names]
            if runtime_names != names:
                raise ValueError(f"runtime and geometry mesh order differs in {scene_path}")
            runtime_ranges = np.asarray(runtime_asset.source_vertex_ranges, dtype=np.int64)
            if not np.array_equal(runtime_ranges, ranges):
                raise ValueError(f"runtime and geometry mesh ranges differ in {scene_path}")
            controller_ids = np.asarray(runtime_asset.source_mesh_controller_bones, dtype=np.int64)
            controller_names = [str(value) for value in runtime_asset.source_bone_names]
            controller_parents = np.asarray(runtime_asset.source_bone_parents, dtype=np.int64)
        selection, scene_selection_provenance = _select_meshes(
            names, tissues, controller_ids, controller_names, controller_parents
        )
        if selection_provenance is None:
            selection_provenance = scene_selection_provenance
        elif scene_selection_provenance["selected_lower_bone_mesh_names"] != selection_provenance["selected_lower_bone_mesh_names"]:
            raise ValueError("lower mesh selection differs between geometry scenes")

        target_beta = None
        if "target_betas" in data.files:
            target_beta = np.asarray(data["target_betas"], dtype=np.float64).reshape(-1).tolist()
            if first_target_beta is None:
                first_target_beta = target_beta
        elif first_target_beta is not None:
            target_beta = first_target_beta

        def ids_for(mesh_indices: Iterable[int]) -> np.ndarray:
            indices = list(mesh_indices)
            if not indices:
                return np.empty(0, dtype=np.int64)
            return np.concatenate([_mesh_ids(ranges, index) for index in indices])

        # One exact surface query per aggregate category. Per-mesh lower stats
        # are sliced from the same result, avoiding a different query rule.
        aggregate_signed: dict[str, np.ndarray] = {}
        aggregate_ids: dict[str, np.ndarray] = {}
        for category in ("major_bones", "Artery", "Vein", "all_nerves", "all_lower_bones"):
            ids = ids_for(selection[category])
            aggregate_ids[category] = ids
            aggregate_signed[category] = _signed_distance(
                candidate_vertices[ids], skin_vertices, skin_faces
            ) if len(ids) else np.empty(0, dtype=np.float64)
        before_aggregate_signed: dict[str, np.ndarray] = {}
        for category, ids in aggregate_ids.items():
            before_aggregate_signed[category] = _signed_distance(
                before_vertices[ids], skin_vertices, skin_faces
            ) if len(ids) else np.empty(0, dtype=np.float64)

        categories = {}
        for category, ids in aggregate_ids.items():
            categories[category] = {
                "mesh_names": [names[index] for index in selection[category]],
                "before": _stat(before_aggregate_signed[category]),
                "candidate": _stat(aggregate_signed[category]),
            }

        lower_meshes: dict[str, Any] = {}
        lower_ids = aggregate_ids["all_lower_bones"]
        lower_signed = aggregate_signed["all_lower_bones"]
        lower_before_signed = before_aggregate_signed["all_lower_bones"]
        # Map contiguous mesh ranges into the aggregate query by a direct
        # offset map. This preserves identical point ordering and signs.
        lower_offset = 0
        lower_index_map: dict[int, tuple[int, int]] = {}
        for index in selection["all_lower_bones"]:
            count = int(len(_mesh_ids(ranges, index)))
            lower_index_map[index] = (lower_offset, lower_offset + count)
            lower_offset += count
        for index in selection["all_lower_bones"]:
            start, stop = lower_index_map[index]
            lower_meshes[names[index]] = {
                "mesh_index": int(index),
                "vertex_count": int(stop - start),
                "before": _stat(lower_before_signed[start:stop]),
                "candidate": _stat(lower_signed[start:stop]),
            }

        cap_stats: dict[str, Any] = {}
        for domain_name, domain_ids in calibration_domains.items():
            valid = domain_ids[(domain_ids >= 0) & (domain_ids < len(before_vertices))]
            cap_stats[domain_name] = _kabsch_residual(
                before_vertices[valid], candidate_vertices[valid]
            )
        displacement = np.linalg.norm(candidate_vertices - before_vertices, axis=1) * 1000.0
        scene_row: dict[str, Any] = {
            "path": str(scene_path),
            "path_sha256": _sha256(scene_path),
            "target_betas": target_beta,
            "vertex_count": int(len(before_vertices)),
            "skin_vertex_count": int(len(skin_vertices)),
            "skin_face_count": int(len(skin_faces)),
            "skin_boundary_edge_count": boundary_edges,
            "skin_nonmanifold_edge_count": nonmanifold_edges,
            "skin_unique_edge_count": unique_edges,
            "categories": categories,
            "lower_bone_meshes": lower_meshes,
            "cap_rigid_procrustes": {
                "calibration_domains_are_fit_ids": True,
                "domains": cap_stats,
            },
            "candidate_global_displacement": {
                "max_mm": float(np.max(displacement)),
                "mean_mm": float(np.mean(displacement)),
                "p95_mm": float(np.percentile(displacement, 95)),
            },
        }
        scenes[scene_path.stem] = scene_row

    worst_regions: list[dict[str, Any]] = []
    for scene, row in scenes.items():
        for category, stats in row["categories"].items():
            candidate = stats["candidate"]
            worst_regions.append(
                {
                    "scene": scene,
                    "region": category,
                    "max_outside_mm": candidate["max_outside_mm"],
                    "gt10_count": candidate["gt10_count"],
                    "outside_fraction": candidate["outside_fraction"],
                }
            )
        for mesh, stats in row["lower_bone_meshes"].items():
            candidate = stats["candidate"]
            worst_regions.append(
                {
                    "scene": scene,
                    "region": mesh,
                    "max_outside_mm": candidate["max_outside_mm"],
                    "gt10_count": candidate["gt10_count"],
                    "outside_fraction": candidate["outside_fraction"],
                }
            )
    worst_regions.sort(
        key=lambda value: (
            value["max_outside_mm"] is not None,
            value["max_outside_mm"] if value["max_outside_mm"] is not None else -1.0,
        ),
        reverse=True,
    )

    report: dict[str, Any] = {
        "schema": "v17_lower_geometry_surface_measurement_cli_v1",
        "scope": "read-only geometry measurement; scene count is discovered from geometry directory",
        "geometry_directory": str(geometry),
        "output_directory": str(output),
        "runtime_path": str(runtime_path) if runtime_path is not None else None,
        "runtime_load": runtime_load,
        "runtime_manifest_sha256": (
            _sha256(runtime_path / "manifest.json")
            if runtime_path is not None and (runtime_path / "manifest.json").is_file()
            else None
        ),
        "compiled_report_path": str(compiled_report_path) if compiled_report_path else None,
        "compiled_report_sha256": (
            _sha256(compiled_report_path) if compiled_report_path and compiled_report_path.is_file() else None
        ),
        "compiled_report_summary": (
            {
                key: compiled_report.get(key)
                for key in (
                    "method",
                    "target_beta",
                    "source_reference_beta",
                    "subject_specific_branches",
                    "capture_or_amass_frames_used_for_fit",
                    "anatomical_passed",
                )
                if key in compiled_report
            }
            if compiled_report is not None
            else None
        ),
        "calibration": {
            "path": str(calibration),
            "sha256": _sha256(calibration),
            "cap_domain_names": calibration_domain_names,
        },
        "selection": selection_provenance,
        "measurement_method": {
            "point_surface_distance": "libigl signed_distance point-to-triangle",
            "sign_type": "WINDING_NUMBER generalized winding",
            "algorithm_change_note": (
                "Earlier exploratory reports may use FAST_WINDING_NUMBER; this CLI uses "
                "WINDING_NUMBER for the more exact generalized-winding evaluation. "
                "Values from the two sign modes must not be compared as an improvement "
                "without rerunning both with the same mode."
            ),
            "units": "xyz metres; distances millimetres",
            "outside_convention": "positive signed distance",
            "thresholds_mm": [0.0, 5.0, 10.0],
            "outside_fraction": "outside_count / point_count; no area weighting",
            "cap_shape": "Kabsch best rigid Procrustes on frozen calibration .fit vertex IDs",
            "strict_containment_certificate": False,
            "triangle_intersection_tested": False,
            "skin_boundary_policy": "record boundary topology; do not claim closed-surface containment",
        },
        "scene_count": len(scenes),
        "scenes": scenes,
        "worst_candidate_regions": worst_regions[:40],
    }
    report_path = output / "lower_geometry_measurement_v17.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_markdown(report, report_path.with_suffix(".md"))
    summary = _summary(report)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    _write_summary_markdown(summary, output / "summary.md")
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# V17 下肢独立表面量测",
        "",
        f"输入目录：`{report['geometry_directory']}`；自动发现 `{report['scene_count']}` 个场景。",
        "",
        "本报告只读取几何、相邻编译 runtime 和 calibration，不拟合、不重绑定、不修改输入。",
        "",
        "## 证据边界",
        "",
        "- 使用 libigl point-to-triangle `signed_distance`，符号为 `WINDING_NUMBER`；正值表示点在广义 winding 分类下位于皮肤外侧。",
        "- 早期 exploratory 报告可能使用 `FAST_WINDING_NUMBER`；本 CLI 切换到 `WINDING_NUMBER` 是为了更精确的 generalized-winding 评估，两种符号算法的数值不能直接当作改进比较。",
        "- 每个场景记录皮肤边界边和非流形边数量。存在边界时，winding 只作近似分类，不能作为严格闭合 containment 证明。",
        "- 统计包括 12 个重点骨、全部下肢骨（含所有脚趾）、Artery、Vein 和全部 `mesh_tissues == nerve` mesh；outside fraction 按点数计算。",
        "- cap 形状用 calibration 的冻结 `.fit` 顶点 ID 做最佳刚体 Procrustes；这只判断形状是否被拉扯，不证明站点或关节正确。",
        "",
        "## 场景类别汇总",
        "",
        "|场景|类别|before 最大皮外 mm|candidate 最大皮外 mm|candidate 出皮/总点|>5 mm|>10 mm|outside fraction|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, row in report["scenes"].items():
        for category in ("major_bones", "all_lower_bones", "Artery", "Vein", "all_nerves"):
            stats = row["categories"][category]
            before = stats["before"]
            candidate = stats["candidate"]
            lines.append(
                f"|{scene}|{category}|{before['max_outside_mm']:.2f}|"
                f"{candidate['max_outside_mm']:.2f}|{candidate['outside_count']}/{candidate['point_count']}|"
                f"{candidate['gt5_count']}|{candidate['gt10_count']}|{candidate['outside_fraction'] * 100:.2f}%|"
            )
        lines.append("")
    lines += [
        "## 每场景最大下肢骨 mesh",
        "",
        "|场景|mesh|candidate 最大皮外 mm|出皮/总点|>5 mm|>10 mm|outside fraction|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scene, row in report["scenes"].items():
        ranked = sorted(
            row["lower_bone_meshes"].items(),
            key=lambda item: item[1]["candidate"]["max_outside_mm"] or -1.0,
            reverse=True,
        )
        for mesh, stats in ranked[:6]:
            candidate = stats["candidate"]
            lines.append(
                f"|{scene}|{mesh}|{candidate['max_outside_mm']:.2f}|"
                f"{candidate['outside_count']}/{candidate['point_count']}|{candidate['gt5_count']}|"
                f"{candidate['gt10_count']}|{candidate['outside_fraction'] * 100:.2f}%|"
            )
        lines.append("")
    lines += [
        "## cap 刚体残差",
        "",
        "|calibration .fit 域|跨场景最大 RMS mm|跨场景最大 max mm|跨场景最大原始位移 mm|",
        "|---|---:|---:|---:|",
    ]
    cap_names = report["calibration"]["cap_domain_names"]
    for name in cap_names:
        values = [row["cap_rigid_procrustes"]["domains"][name] for row in report["scenes"].values()]
        lines.append(
            f"|{name}|{max(value['rms_mm'] for value in values):.5f}|"
            f"{max(value['max_mm'] for value in values):.5f}|"
            f"{max(value['raw_displacement_max_mm'] for value in values):.2f}|"
        )
    lines += [
        "",
        "## 运行选择与限制",
        "",
        f"相邻 compiled runtime 加载：`{report['runtime_load']['loaded']}`；若为 false，必须查看 JSON 的 `runtime_load.attempts` 和 `reason`，此时下肢集合只能按 mesh 名称选择，不能作为 235 控制器身份验证。",
        "",
        f"下肢 mesh 选择：`{report['selection']['selection_rule']}`。脚趾 mesh 共 `{len(report['selection']['toe_mesh_names'])}` 个，均被纳入。",
        "",
        "本文件只提供几何量测，未测试 triangle-triangle crossing，也不根据这些距离单独推断 theta 响应或宣布解剖通过。",
        "",
        f"完整 JSON：`{path.with_suffix('.json')}`。",
        "",
    ]
    path.write_text("\n".join(lines))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    worst = list(report["worst_candidate_regions"][:10])
    scene_summary: dict[str, Any] = {}
    for scene, row in report["scenes"].items():
        scene_summary[scene] = {
            "skin_boundary_edge_count": row["skin_boundary_edge_count"],
            "skin_nonmanifold_edge_count": row["skin_nonmanifold_edge_count"],
            "categories": {
                category: {
                    "before_max_outside_mm": row["categories"][category]["before"]["max_outside_mm"],
                    "candidate_max_outside_mm": row["categories"][category]["candidate"]["max_outside_mm"],
                    "candidate_outside_count": row["categories"][category]["candidate"]["outside_count"],
                    "candidate_point_count": row["categories"][category]["candidate"]["point_count"],
                    "candidate_gt5_count": row["categories"][category]["candidate"]["gt5_count"],
                    "candidate_gt10_count": row["categories"][category]["candidate"]["gt10_count"],
                    "candidate_outside_fraction": row["categories"][category]["candidate"]["outside_fraction"],
                }
                for category in ("major_bones", "all_lower_bones", "Artery", "Vein", "all_nerves")
            },
        }
    return {
        "schema": "v17_lower_geometry_surface_measurement_summary_v1",
        "geometry_directory": report["geometry_directory"],
        "scene_count": report["scene_count"],
        "measurement_json": str(Path(report["output_directory"]) / "lower_geometry_measurement_v17.json"),
        "sign_type": report["measurement_method"]["sign_type"],
        "algorithm_change_note": report["measurement_method"]["algorithm_change_note"],
        "runtime_load": report["runtime_load"],
        "selection_counts": {
            "major_bones": len(report["selection"]["major_bone_mesh_names"]),
            "all_lower_bones": len(report["selection"]["selected_lower_bone_mesh_names"]),
            "toe_meshes": len(report["selection"]["toe_mesh_names"]),
            "Artery": len(report["selection"]["artery_mesh_names"]),
            "Vein": len(report["selection"]["vein_mesh_names"]),
            "all_nerves": len(report["selection"]["nerve_mesh_names"]),
        },
        "strict_containment_certificate": False,
        "worst_candidate_regions": worst,
        "scenes": scene_summary,
    }


def _write_summary_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# V17 下肢量测快速摘要",
        "",
        f"场景数：`{summary['scene_count']}`；符号方法：`{summary['sign_type']}`。",
        f"运行包加载：`{summary['runtime_load']['loaded']}`；选择计数：`{summary['selection_counts']}`。",
        "早期 exploratory 报告可能使用 `FAST_WINDING_NUMBER`；本 CLI 使用 `WINDING_NUMBER`，两种算法的数值不可直接当作改进比较。",
        "皮肤存在边界时，结果是广义 winding 近似，不是严格闭合 containment 证书。",
        "",
        "## 候选最坏区域",
        "",
        "|场景|区域|最大皮外 mm|>10 mm 点数|outside fraction|",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["worst_candidate_regions"]:
        fraction = row["outside_fraction"]
        lines.append(
            f"|{row['scene']}|{row['region']}|{row['max_outside_mm']:.2f}|"
            f"{row['gt10_count']}|{fraction * 100:.2f}%|"
        )
    lines += [
        "",
        "完整统计见 `lower_geometry_measurement_v17.json`；本摘要不判断 triangle crossing 或解剖通过。",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True, help="directory containing exported geometry NPZ files")
    parser.add_argument("--output", type=Path, required=True, help="directory for measurement JSON and Markdown")
    parser.add_argument("--calibration", type=Path, default=_DEFAULT_CALIBRATION, help="frozen calibration NPZ")
    parser.add_argument("--runtime", type=Path, default=None, help="optional compiled runtime; otherwise discover neighboring V17 runtime")
    args = parser.parse_args(argv)
    report = evaluate(args.geometry, args.output, args.calibration, args.runtime)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scene_count": report["scene_count"],
                "worst_candidate_regions": report["worst_candidate_regions"][:5],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
