"""Fail-closed V7 acceptance matrix over subjects × poses.

The previous matrix let a candidate compare against a spline it had just
fitted and copied `pass` / measured values out of its own build metadata.
That made a broken hinge able to redefine the probes and the verdict.  This
runner rebuilds every gate from final posed vertices, frozen domain ids, and
an on-disk patella oracle, keeps vessel/compound hooks pending until their
modules exist, and refuses to publish while any required evidence is missing.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    source_bone_posed_global,
)
from projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7 import (
    fk_reference_from_patella_oracle_v7,
    observations_report_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
    JointContactThresholdsV7,
    diagnose_joint_contact_geometry_v7,
    patellofemoral_trajectory_metrics_v7,
    rigid_edge_metrics_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
    patella_bind_frames_v7,
    patella_oracle_sweep_v7,
    patella_world_transform_v7,
    solve_patella_contact_corrections_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    SubjectAssetV7,
    apply_subject_pose,
    load_subject_asset,
)


ACCEPTANCE_MATRIX_SCHEMA_VERSION = 7
_SIDES = ("left", "right")
_DETERMINISM_LIMIT_M = 1.0e-6
_KNEE_JOINT_INDICES = (4, 5)


@dataclass(frozen=True)
class MatrixPoseSpecV7:
    label: str
    pose_axis_angle: np.ndarray
    transl: np.ndarray
    source: str


@dataclass(frozen=True)
class MatrixSubjectSpecV7:
    label: str
    path: Path
    subject: Any


def parse_label_value_pair(raw: str) -> tuple[str, str]:
    text = str(raw)
    if "=" not in text:
        raise ValueError(f"expected LABEL=VALUE, got {raw!r}")
    label, value = text.split("=", 1)
    if not label or not value:
        raise ValueError(f"expected LABEL=VALUE, got {raw!r}")
    return label, value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix_pose_v7(label: str, path_or_zero: str | Path) -> MatrixPoseSpecV7:
    name = str(label).strip()
    if not name:
        raise ValueError("pose label must be non-empty")
    if str(path_or_zero) == "zero":
        return MatrixPoseSpecV7(
            label=name,
            pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
            transl=np.zeros(3, dtype=np.float32),
            source="synthetic",
        )
    path = Path(path_or_zero).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"pose NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "pose_axis_angle" not in data.files:
            raise ValueError(f"{path} must contain pose_axis_angle")
        pose = np.asarray(data["pose_axis_angle"], dtype=np.float32)
        transl = (
            np.asarray(data["transl"], dtype=np.float32).reshape(3)
            if "transl" in data.files
            else np.zeros(3, dtype=np.float32)
        )
    try:
        pose = pose.reshape(55, 3)
    except ValueError as exc:
        raise ValueError(f"pose_axis_angle must be [55,3], got {pose.shape}") from exc
    if transl.shape != (3,):
        raise ValueError(f"transl must be [3], got {transl.shape}")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(transl)):
        raise ValueError(f"pose NPZ contains non-finite values: {path}")
    return MatrixPoseSpecV7(
        label=name,
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
        transl=np.asarray(transl, dtype=np.float32),
        source=str(path),
    )


def synthetic_knee_sweep_poses_v7(
    *,
    count: int = 13,
    maximum_deg: float = 120.0,
) -> list[MatrixPoseSpecV7]:
    if int(count) < 2:
        raise ValueError("sweep count must be at least 2")
    degrees = np.linspace(0.0, float(maximum_deg), int(count), dtype=np.float64)
    poses: list[MatrixPoseSpecV7] = []
    for index, theta_deg in enumerate(degrees):
        pose = np.zeros((55, 3), dtype=np.float32)
        theta = float(np.radians(theta_deg))
        if index == 0:
            theta = 0.0
            theta_deg = 0.0
        for joint in _KNEE_JOINT_INDICES:
            pose[joint] = np.asarray((theta, 0.0, 0.0), dtype=np.float32)
        label = f"knee_sweep_{index:02d}_{theta_deg:.0f}deg"
        poses.append(
            MatrixPoseSpecV7(
                label=label,
                pose_axis_angle=pose,
                transl=np.zeros(3, dtype=np.float32),
                source="synthetic",
            )
        )
    return poses


def capture_interpolated_sweep_poses_v7(
    *,
    capture: MatrixPoseSpecV7,
    count: int = 13,
) -> list[MatrixPoseSpecV7]:
    """Sweep from rest into a captured pose, keeping the capture's whole attitude.

    ``synthetic_knee_sweep_poses_v7`` writes the knee joints and leaves every
    other joint, the hips included, at exactly zero.  No capture presents that
    configuration, and it is the one that trips the leg solve, so a trajectory
    measured there says little about the retarget on real drive.  Scaling the
    captured axis-angle by ``t`` instead keeps every joint in the capture's own
    proportion at every sample -- and because each joint rotates about a fixed
    axis, ``t * axis_angle`` is the slerp from identity to that joint's captured
    rotation, not an approximation of it.

    Flexion coverage is therefore bounded by whatever the capture reached.
    """
    if int(count) < 2:
        raise ValueError("sweep count must be at least 2")
    target = np.asarray(capture.pose_axis_angle, dtype=np.float64).reshape(55, 3)
    target_transl = np.asarray(capture.transl, dtype=np.float64).reshape(3)
    poses: list[MatrixPoseSpecV7] = []
    for index, t in enumerate(
        np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    ):
        scaled = (target * float(t)).astype(np.float32)
        poses.append(
            MatrixPoseSpecV7(
                label=f"capture_sweep_{index:02d}_{float(t) * 100.0:.0f}pct",
                pose_axis_angle=scaled,
                transl=(target_transl * float(t)).astype(np.float32),
                source=f"interpolated:{capture.label}",
            )
        )
    return poses


def body_surface_for_cell_v7(
    *,
    subject: SubjectAssetV7,
    pose_spec: MatrixPoseSpecV7,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict[str, Any]]:
    """Build the SMPL-X skin for exactly this beta and pose.

    Containment is only evidence when the skin belongs to the same subject and
    pose as the anatomy: comparing a zero-posed anatomy against a capture-posed
    skin reported the whole body as 0.99 m outside itself.
    """
    try:
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
            _default_smplx_model_path,
        )
        from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
            load_smplx_model_v7,
            smplx_body_surface_v7,
        )
    except ImportError as exc:
        return None, {"available": False, "reason": f"smplx surface unavailable: {exc}"}
    try:
        model_path = Path(_default_smplx_model_path(str(subject.gender)))
        model = _cached_smplx_model(load_smplx_model_v7, model_path)
        vertices, faces = smplx_body_surface_v7(
            model,
            betas=np.asarray(subject.betas, dtype=np.float64).reshape(-1),
            pose_axis_angle=pose_spec.pose_axis_angle,
            transl=pose_spec.transl,
        )
    except Exception as exc:
        return None, {"available": False, "reason": str(exc)}
    return (vertices, faces), {
        "available": True,
        "source": "numpy_smplx_forward",
        "model": model_path.name,
        "vertex_count": int(len(vertices)),
    }


_SMPLX_MODEL_CACHE: dict[str, Any] = {}


def _cached_smplx_model(loader: Any, model_path: Path) -> Any:
    key = str(model_path)
    if key not in _SMPLX_MODEL_CACHE:
        _SMPLX_MODEL_CACHE[key] = loader(model_path)
    return _SMPLX_MODEL_CACHE[key]


def _evaluate_vessel_gates_hook(
    *,
    asset: Any,
    posed_vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    runtime_coefficients: Mapping[str, np.ndarray],
    body_surface: tuple[np.ndarray, np.ndarray] | None,
    reference_faces_digest: str | None,
) -> dict[str, Any]:
    try:
        from projects.genesis_ue_sync.anatomy_retarget import vessel_gates_v7
    except ImportError:
        return {"available": False, "reason": "module not present", "pass": False}
    try:
        return vessel_gates_v7.evaluate_vessel_gates_v7(
            asset=asset,
            posed_vertices=posed_vertices,
            domains=domains,
            runtime_coefficients=dict(runtime_coefficients),
            body_surface=body_surface,
            reference_faces_digest=reference_faces_digest,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "pass": False}


def _evaluate_compound_gates_hook(
    *,
    asset: Any,
    posed_vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    runtime_coefficients: Mapping[str, np.ndarray],
    body_surface: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    try:
        from projects.genesis_ue_sync.anatomy_retarget import compound_gates_v7
    except ImportError:
        return {"available": False, "reason": "module not present", "pass": False}
    try:
        return compound_gates_v7.evaluate_compound_gates_v7(
            asset=asset,
            posed_vertices=posed_vertices,
            domains=domains,
            runtime_coefficients=dict(runtime_coefficients),
            body_surface=body_surface,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "pass": False}


def _contact_tables_for_subject(
    law: Any,
    *,
    subject: SubjectAssetV7,
    domains: FrozenJointMaterialDomainsV7,
) -> dict[str, np.ndarray]:
    rest = np.asarray(subject.rigged_asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(subject.rigged_asset.faces)
    tables: dict[str, np.ndarray] = {}
    for side in _SIDES:
        table, _report = solve_patella_contact_corrections_v7(
            law,
            vertices=rest,
            faces=faces,
            domains=domains,
            asset=subject.rigged_asset,
            side=side,
            knee_axis_local=law.axis_knee_local[side],
        )
        tables[side] = np.asarray(table, dtype=np.float64)
    return tables


def _interp_contact_row(
    table: np.ndarray,
    knots_deg: np.ndarray,
    flexion_deg: float,
) -> np.ndarray:
    knots = np.asarray(knots_deg, dtype=np.float64).reshape(-1)
    values = np.asarray(table, dtype=np.float64)
    angle = float(abs(float(flexion_deg)))
    return np.asarray(
        [np.interp(angle, knots, values[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _oracle_frame_for_cell(
    law: Any,
    *,
    asset: Any,
    domains: FrozenJointMaterialDomainsV7,
    rest_vertices: np.ndarray,
    contact_tables: Mapping[str, np.ndarray],
    controller_observations: Mapping[str, Mapping[str, Any]],
    posed_bone_global: np.ndarray,
    posed_vertices: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray | None:
    """Predict this cell's patella from the oracle riding the candidate's femur.

    Everything except the patella is the candidate's own posed surface, and the
    oracle chain is anchored on the candidate's actual posed femur.  Both
    matter: anchoring on the bind femur measured the whole leg's motion as
    patella error, and leaving the trochlea at rest measured the global drive
    translation as patella error.  What is left is the coupling under test.
    """
    rest = np.asarray(rest_vertices, dtype=np.float64)
    oracle = np.asarray(posed_vertices, dtype=np.float64).copy()
    drive = np.asarray(transl, dtype=np.float64).reshape(3)
    knots = np.asarray(law.knots_deg, dtype=np.float64).reshape(-1)
    bone_global = np.asarray(posed_bone_global, dtype=np.float64)
    for side in _SIDES:
        observation = controller_observations.get(f"knee_{side}", {})
        flexion_deg = observation.get("flexion_deg")
        if flexion_deg is None or not np.isfinite(float(flexion_deg)):
            return None
        flexion_deg = float(flexion_deg)
        contact = _interp_contact_row(
            contact_tables[side], knots, flexion_deg
        )
        frames = patella_bind_frames_v7(asset, side=side)
        frames = dict(frames)
        frames["femur_bind"] = bone_global[int(frames["femur"])]
        world = patella_world_transform_v7(
            law,
            frames=frames,
            side=side,
            flexion_rad=float(np.radians(flexion_deg)),
            knee_axis_local=law.axis_knee_local[side],
            contact_translation_parent_local_m=contact,
        )
        patella_ids = np.asarray(domains.require(f"{side}/patella"), dtype=np.int64)
        bind = rest[patella_ids]
        oracle[patella_ids] = (world[:3, :3] @ bind.T).T + world[:3, 3] + drive
    return oracle


def _gate_failure_names(prefix: str, gate: Mapping[str, Any]) -> list[str]:
    failures = gate.get("failures")
    if isinstance(failures, list):
        return [f"{prefix}.{name}" for name in failures]
    return [] if gate.get("pass") else [prefix]


def _min_pairwise_gap(a: np.ndarray, b: np.ndarray) -> float:
    points = np.asarray(a, dtype=np.float64)
    target = np.asarray(b, dtype=np.float64)
    if not len(points) or not len(target):
        return float("inf")
    forward = float(np.min(cKDTree(target).query(points, k=1)[0]))
    backward = float(np.min(cKDTree(points).query(target, k=1)[0]))
    return float(min(forward, backward))


def _knee_interface_gaps(
    domains: FrozenJointMaterialDomainsV7,
    vertices: np.ndarray,
) -> dict[str, Any]:
    frame = np.asarray(vertices, dtype=np.float64)
    interfaces: dict[str, Any] = {}
    for side in _SIDES:
        for compartment in ("medial", "lateral"):
            condyle = domains.require(f"{side}/femoral_condyle_{compartment}")
            plateau = domains.require(f"{side}/tibial_plateau_{compartment}")
            gap = _min_pairwise_gap(frame[condyle], frame[plateau])
            interfaces[f"{side}/{compartment}"] = float(gap)
    return interfaces


def _collect_subject_meta(
    subject_spec: MatrixSubjectSpecV7,
    *,
    law: Any,
) -> dict[str, Any]:
    subject = subject_spec.subject
    build_report = dict(subject.build_report or {})
    recorded = build_report.get("patella_oracle_digest")
    law_digest = str(law.content_digest())
    return {
        "path": str(subject_spec.path),
        "content_digest": str(subject.content_digest()),
        "cache_key": str(subject.cache_key),
        "operator_digest": str(subject.operator_digest),
        "betas": np.asarray(subject.betas, dtype=np.float64).reshape(-1).tolist(),
        "gender": str(subject.gender),
        "build_report_patella_oracle_digest": (
            None if recorded is None else str(recorded)
        ),
        "oracle_digest_matches": bool(
            recorded is not None and str(recorded) == law_digest
        ),
        "materialize_publishable": False,
    }


def _run_cell(
    *,
    subject_spec: MatrixSubjectSpecV7,
    pose_spec: MatrixPoseSpecV7,
    domains: FrozenJointMaterialDomainsV7,
    law: Any,
    thresholds: JointContactThresholdsV7,
    contact_tables: Mapping[str, np.ndarray],
    fk_reference: Any,
    vertices_dir: Path | None,
    source_topology_digest: str | None,
) -> dict[str, Any]:
    subject = subject_spec.subject
    asset = subject.rigged_asset
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(asset.faces)
    started = time.perf_counter()
    vertices = apply_subject_pose(
        subject,
        pose_axis_angle=pose_spec.pose_axis_angle,
        transl=pose_spec.transl,
        validate=False,
    )
    pose_seconds = float(time.perf_counter() - started)
    vertices = np.asarray(vertices, dtype=np.float64)

    fk_report = observations_report_v7(
        asset,
        pose_axis_angle=pose_spec.pose_axis_angle,
        posed_vertices=vertices,
        domains=domains,
        reference=fk_reference,
    )
    controller = fk_report["controller"]
    local_fk = fk_report["local_fk"]
    local_fk_arms = fk_report["local_fk_arms"]
    observations = fk_report["observations"]

    oracle_frame = _oracle_frame_for_cell(
        law,
        asset=asset,
        domains=domains,
        rest_vertices=rest,
        contact_tables=contact_tables,
        controller_observations=observations["controller_observations"],
        posed_bone_global=source_bone_posed_global(
            asset, pose_spec.pose_axis_angle
        ),
        posed_vertices=vertices,
        transl=pose_spec.transl,
    )
    if oracle_frame is None:
        trajectory_vertices = None
        oracle_trajectory_vertices = None
    else:
        trajectory_vertices = np.stack([rest, vertices], axis=0)
        oracle_trajectory_vertices = np.stack([rest, oracle_frame], axis=0)

    geometry = diagnose_joint_contact_geometry_v7(
        domains,
        reference_vertices=rest,
        final_vertices=vertices,
        faces=faces,
        trajectory_vertices=trajectory_vertices,
        oracle_trajectory_vertices=oracle_trajectory_vertices,
        thresholds=thresholds,
    )

    body_surface, body_provenance = body_surface_for_cell_v7(
        subject=subject, pose_spec=pose_spec
    )
    vessel = _evaluate_vessel_gates_hook(
        asset=asset,
        posed_vertices=vertices,
        domains=domains,
        runtime_coefficients=subject.runtime_coefficients,
        body_surface=body_surface,
        reference_faces_digest=source_topology_digest,
    )
    compound = _evaluate_compound_gates_hook(
        asset=asset,
        posed_vertices=vertices,
        domains=domains,
        runtime_coefficients=subject.runtime_coefficients,
        body_surface=body_surface,
    )

    failures = (
        _gate_failure_names("controller", controller)
        + _gate_failure_names("local_fk", local_fk)
        + _gate_failure_names("local_fk_arms", local_fk_arms)
        + _gate_failure_names("geometry", geometry)
        + _gate_failure_names("vessel", vessel)
        + _gate_failure_names("compound", compound)
    )
    passed = bool(
        controller.get("pass")
        and local_fk.get("pass")
        # The arm links were measured and returned but left out of the verdict,
        # so a candidate could report passed=true with every elbow link
        # unavailable.
        and local_fk_arms.get("pass")
        and geometry.get("pass")
        and vessel.get("pass")
        and compound.get("pass")
    )

    if vertices_dir is not None:
        vertices_dir.mkdir(parents=True, exist_ok=True)
        out = vertices_dir / f"{subject_spec.label}_{pose_spec.label}.npz"
        np.savez(
            out,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            pose_axis_angle=np.asarray(pose_spec.pose_axis_angle, dtype=np.float32),
            transl=np.asarray(pose_spec.transl, dtype=np.float32),
            subject_digest=np.asarray(str(subject.content_digest())),
            pose_digest=np.asarray(
                smplx_pose_hash(pose_spec.pose_axis_angle, pose_spec.transl)
            ),
        )

    return {
        "subject": subject_spec.label,
        "pose": pose_spec.label,
        "pose_source": pose_spec.source,
        "pose_seconds": pose_seconds,
        "controller": controller,
        "local_fk": local_fk,
        "local_fk_arms": local_fk_arms,
        "geometry": geometry,
        "observations": observations,
        "body_surface": body_provenance,
        "vessel": vessel,
        "compound": compound,
        "failures": failures,
        "passed": passed,
    }


def _run_subject_sweep(
    *,
    subject_spec: MatrixSubjectSpecV7,
    domains: FrozenJointMaterialDomainsV7,
    law: Any,
    thresholds: JointContactThresholdsV7,
    contact_tables: Mapping[str, np.ndarray],
    sweep_count: int,
    capture: MatrixPoseSpecV7 | None = None,
) -> dict[str, Any]:
    subject = subject_spec.subject
    asset = subject.rigged_asset
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(asset.faces)
    if capture is None:
        sweep_poses = synthetic_knee_sweep_poses_v7(count=int(sweep_count))
    else:
        sweep_poses = capture_interpolated_sweep_poses_v7(
            capture=capture, count=int(sweep_count)
        )
    # Flexion is the magnitude of the left knee rotation, not its X component:
    # a captured knee does not rotate about the SMPL-X X axis alone.
    flexion_rad = np.asarray(
        [
            float(np.linalg.norm(np.asarray(pose.pose_axis_angle)[4]))
            for pose in sweep_poses
        ],
        dtype=np.float64,
    )
    candidate_frames: list[np.ndarray] = []
    sample_reports: list[dict[str, Any]] = []
    for pose_spec in sweep_poses:
        vertices = apply_subject_pose(
            subject,
            pose_axis_angle=pose_spec.pose_axis_angle,
            transl=pose_spec.transl,
            validate=False,
        )
        frame = np.asarray(vertices, dtype=np.float32)
        candidate_frames.append(frame)
        gaps = _knee_interface_gaps(domains, frame)
        rigidity: dict[str, Any] = {}
        for side in _SIDES:
            for structure in ("femur", "tibia", "patella"):
                key = f"{side}/{structure}"
                rigidity[key] = rigid_edge_metrics_v7(
                    reference_vertices=rest,
                    final_vertices=np.asarray(frame, dtype=np.float64),
                    faces=faces,
                    indices=domains.require(key),
                    thresholds=thresholds,
                )
        sample_reports.append(
            {
                "label": pose_spec.label,
                "flexion_rad": float(pose_spec.pose_axis_angle[4, 0]),
                "knee_interface_gaps_m": gaps,
                "rigidity": rigidity,
            }
        )

    candidate_sweep = np.stack(candidate_frames, axis=0).astype(np.float32)
    del candidate_frames
    oracle_sweep, _oracle_report = patella_oracle_sweep_v7(
        law,
        asset=asset,
        domains=domains,
        flexion_rad=flexion_rad,
        side_contact_translations=contact_tables,
        knee_axis_local={side: law.axis_knee_local[side] for side in _SIDES},
    )
    trajectory: dict[str, Any] = {}
    for side in _SIDES:
        trajectory[side] = patellofemoral_trajectory_metrics_v7(
            domains,
            posed_vertices=candidate_sweep,
            oracle_vertices=oracle_sweep,
            faces=faces,
            side=side,
            thresholds=thresholds,
        )
    del candidate_sweep
    del oracle_sweep

    failures: list[str] = []
    for side in _SIDES:
        if not trajectory[side].get("pass"):
            failures.append(f"trajectory/{side}")

    gap_summary: dict[str, Any] = {}
    for key in sample_reports[0]["knee_interface_gaps_m"]:
        values = [
            float(sample["knee_interface_gaps_m"][key]) for sample in sample_reports
        ]
        violations = []
        for value in values:
            if value < thresholds.knee_gap_min_m:
                violations.append(thresholds.knee_gap_min_m - value)
            elif value > thresholds.knee_gap_max_m:
                violations.append(value - thresholds.knee_gap_max_m)
            else:
                violations.append(0.0)
        worst_index = int(np.argmax(np.asarray(violations, dtype=np.float64)))
        gap_summary[key] = {
            "min_m": float(np.min(values)),
            "max_m": float(np.max(values)),
            "worst_sample": sample_reports[worst_index]["label"],
            "pass": bool(
                all(
                    thresholds.knee_gap_min_m - 1.0e-12
                    <= value
                    <= thresholds.knee_gap_max_m
                    for value in values
                )
            ),
        }
        if not gap_summary[key]["pass"]:
            failures.append(f"knee_gap/{key}")

    rigidity_summary: dict[str, Any] = {}
    for key in sample_reports[0]["rigidity"]:
        ratios_min = [float(sample["rigidity"][key].get("ratio_min", np.inf)) for sample in sample_reports]
        ratios_max = [float(sample["rigidity"][key].get("ratio_max", -np.inf)) for sample in sample_reports]
        passed_flags = [bool(sample["rigidity"][key].get("pass")) for sample in sample_reports]
        worst_index = int(np.argmin(np.asarray(passed_flags, dtype=np.int32)))
        if all(passed_flags):
            # Still report extremal ratios over the sweep.
            worst_index = int(
                np.argmax(
                    np.maximum(
                        np.abs(np.asarray(ratios_min) - 1.0),
                        np.abs(np.asarray(ratios_max) - 1.0),
                    )
                )
            )
        rigidity_summary[key] = {
            "ratio_min": float(np.min(ratios_min)),
            "ratio_max": float(np.max(ratios_max)),
            "worst_sample": sample_reports[worst_index]["label"],
            "pass": bool(all(passed_flags)),
        }
        if not rigidity_summary[key]["pass"]:
            failures.append(f"rigidity/{key}")

    return {
        "pose_count": int(len(sweep_poses)),
        "flexion_rad": flexion_rad.tolist(),
        "trajectory": trajectory,
        "knee_interface_gaps": gap_summary,
        "rigidity": rigidity_summary,
        "samples": sample_reports,
        "failures": failures,
        "passed": not failures,
    }


def _default_determinism_cell(
    subjects: Sequence[MatrixSubjectSpecV7],
    poses: Sequence[MatrixPoseSpecV7],
) -> tuple[str, str]:
    if not subjects or not poses:
        raise ValueError("determinism cell requires at least one subject and pose")
    non_zero = [
        pose
        for pose in poses
        if not (
            np.allclose(pose.pose_axis_angle, 0.0) and np.allclose(pose.transl, 0.0)
        )
    ]
    pose = non_zero[0] if non_zero else poses[0]
    return subjects[0].label, pose.label


def run_acceptance_matrix_v7(
    *,
    subjects: Sequence[MatrixSubjectSpecV7],
    poses: Sequence[MatrixPoseSpecV7],
    domains: FrozenJointMaterialDomainsV7,
    law: Any,
    thresholds: JointContactThresholdsV7 | None = None,
    action_oracle_path: str | Path | None = None,
    operator_digest: str = "",
    source_topology_digest: str | None = None,
    sweep_count: int = 13,
    vertices_dir: str | Path | None = None,
    determinism_cell: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if not subjects:
        raise ValueError("at least one subject is required")
    if not poses:
        raise ValueError("at least one pose is required")
    limits = thresholds or JointContactThresholdsV7()
    vertices_path = (
        None if vertices_dir is None else Path(vertices_dir).expanduser().resolve()
    )

    action_path = (
        None
        if action_oracle_path is None
        else Path(action_oracle_path).expanduser().resolve()
    )
    action_file_digest = ""
    if action_path is not None:
        if not action_path.is_file():
            raise ValueError(f"action oracle does not exist: {action_path}")
        action_file_digest = _file_sha256(action_path)
    action_digest_verified = bool(
        action_file_digest
        and action_file_digest == str(law.action_source_digest)
    )

    subject_by_label = {spec.label: spec for spec in subjects}
    if len(subject_by_label) != len(subjects):
        raise ValueError("subject labels must be unique")
    pose_by_label = {spec.label: spec for spec in poses}
    if len(pose_by_label) != len(poses):
        raise ValueError("pose labels must be unique")

    subject_meta = {
        spec.label: _collect_subject_meta(spec, law=law) for spec in subjects
    }
    operator_digests = {meta["operator_digest"] for meta in subject_meta.values()}
    shared_operator = next(iter(operator_digests)) if len(operator_digests) == 1 else ""
    operators_consistent = len(operator_digests) == 1 and (
        not operator_digest or str(operator_digest) == shared_operator
    )
    recorded_operator = (
        str(operator_digest) if operator_digest else shared_operator
    )

    cells: dict[str, Any] = {}
    sweeps: dict[str, Any] = {}
    contact_cache: dict[str, dict[str, np.ndarray]] = {}
    reference_cache: dict[str, Any] = {}

    for subject_spec in subjects:
        tables = _contact_tables_for_subject(
            law, subject=subject_spec.subject, domains=domains
        )
        contact_cache[subject_spec.label] = tables
        reference_cache[subject_spec.label] = fk_reference_from_patella_oracle_v7(
            law, contact_translations=tables
        )
        for pose_spec in poses:
            cell_key = f"{subject_spec.label}/{pose_spec.label}"
            cells[cell_key] = _run_cell(
                subject_spec=subject_spec,
                pose_spec=pose_spec,
                domains=domains,
                law=law,
                thresholds=limits,
                contact_tables=tables,
                fk_reference=reference_cache[subject_spec.label],
                vertices_dir=vertices_path,
                source_topology_digest=source_topology_digest,
            )
        # One sweep per captured pose, so the trajectory is judged on the drive
        # the captures actually present rather than on a knee-only pose.
        captures = [pose for pose in poses if pose.source != "synthetic"]
        if not captures:
            # Fail closed rather than substituting the knee-only synthetic sweep,
            # which never presents the drive a capture does.
            sweeps[f"{subject_spec.label}/no_capture"] = {
                "available": False,
                "passed": False,
                "failures": ["no_captured_pose"],
                "reason": (
                    "no captured pose was supplied, so the trajectory sweep has "
                    "no real drive to interpolate"
                ),
            }
        for capture in captures:
            sweeps[f"{subject_spec.label}/{capture.label}"] = _run_subject_sweep(
                subject_spec=subject_spec,
                domains=domains,
                law=law,
                thresholds=limits,
                contact_tables=tables,
                sweep_count=int(sweep_count),
                capture=capture,
            )
        del tables

    if determinism_cell is None:
        det_subject, det_pose = _default_determinism_cell(subjects, poses)
    else:
        det_subject, det_pose = determinism_cell
    if det_subject not in subject_by_label or det_pose not in pose_by_label:
        raise ValueError(f"unknown determinism cell {det_subject!r}/{det_pose!r}")
    det_subject_spec = subject_by_label[det_subject]
    det_pose_spec = pose_by_label[det_pose]
    first = apply_subject_pose(
        det_subject_spec.subject,
        pose_axis_angle=det_pose_spec.pose_axis_angle,
        transl=det_pose_spec.transl,
        validate=False,
    )
    second = apply_subject_pose(
        det_subject_spec.subject,
        pose_axis_angle=det_pose_spec.pose_axis_angle,
        transl=det_pose_spec.transl,
        validate=False,
    )
    max_vertex_delta_m = float(
        np.max(
            np.abs(
                np.asarray(first, dtype=np.float64)
                - np.asarray(second, dtype=np.float64)
            )
        )
    )
    determinism = {
        "cell": f"{det_subject}/{det_pose}",
        "max_vertex_delta_m": max_vertex_delta_m,
        "pass": bool(max_vertex_delta_m <= _DETERMINISM_LIMIT_M),
    }

    vessel_available = all(
        bool(cell.get("vessel", {}).get("available")) for cell in cells.values()
    )
    compound_available = all(
        bool(cell.get("compound", {}).get("available")) for cell in cells.values()
    )
    pending_gates = {"vessel": vessel_available, "compound": compound_available}

    failures: list[str] = []
    for cell_key, cell in cells.items():
        if not cell.get("passed"):
            failures.append(f"cell.{cell_key}")
            failures.extend(f"cell.{cell_key}.{name}" for name in cell.get("failures", []))
    for subject_label, sweep in sweeps.items():
        if not sweep.get("passed"):
            failures.append(f"sweep.{subject_label}")
            failures.extend(
                f"sweep.{subject_label}.{name}" for name in sweep.get("failures", [])
            )
    if not determinism["pass"]:
        failures.append("determinism")
    if not action_digest_verified:
        failures.append("action_digest_verified")
    for subject_label, meta in subject_meta.items():
        if not meta["oracle_digest_matches"]:
            failures.append(f"oracle_digest_matches.{subject_label}")
    if not operators_consistent:
        failures.append("operator_digest_consistent")
    if not vessel_available:
        failures.append("pending_gates.vessel")
    if not compound_available:
        failures.append("pending_gates.compound")

    cells_passed = all(bool(cell.get("passed")) for cell in cells.values())
    sweeps_passed = all(bool(sweep.get("passed")) for sweep in sweeps.values())
    oracles_match = all(
        bool(meta["oracle_digest_matches"]) for meta in subject_meta.values()
    )
    passed = bool(
        cells_passed
        and sweeps_passed
        and determinism["pass"]
        and action_digest_verified
        and oracles_match
        and operators_consistent
        and vessel_available
        and compound_available
    )
    if passed:
        reason = "all acceptance gates passed; publishable remains false pending evidence pack"
    else:
        reason = "acceptance matrix failed: " + ", ".join(failures[:32])
        if len(failures) > 32:
            reason += f" (+{len(failures) - 32} more)"

    return {
        "schema_version": ACCEPTANCE_MATRIX_SCHEMA_VERSION,
        "spec": "MD/v7_acceptance_spec.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operator_digest": recorded_operator,
        "domain_topology_digest": str(domains.topology_digest),
        "patella_oracle": {
            "content_digest": str(law.content_digest()),
            "action_source_digest": str(law.action_source_digest),
            "action_file_digest": action_file_digest,
            "action_digest_verified": action_digest_verified,
        },
        "thresholds": limits.to_dict(),
        "subjects": subject_meta,
        "cells": cells,
        "sweeps": sweeps,
        "determinism": determinism,
        "pending_gates": pending_gates,
        "failures": failures,
        "passed": passed,
        "publishable": False,
        "reason": reason,
    }


def load_matrix_subject_v7(label: str, path: str | Path) -> MatrixSubjectSpecV7:
    name = str(label).strip()
    if not name:
        raise ValueError("subject label must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"subject asset does not exist: {resolved}")
    return MatrixSubjectSpecV7(
        label=name,
        path=resolved,
        subject=load_subject_asset(resolved),
    )


__all__ = [
    "ACCEPTANCE_MATRIX_SCHEMA_VERSION",
    "MatrixPoseSpecV7",
    "MatrixSubjectSpecV7",
    "_json_ready",
    "load_matrix_pose_v7",
    "load_matrix_subject_v7",
    "parse_label_value_pair",
    "run_acceptance_matrix_v7",
    "synthetic_knee_sweep_poses_v7",
]
