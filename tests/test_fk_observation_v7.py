from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7 import (
    FkReferenceV7,
    default_fk_reference_v7,
    observe_fk_v7,
    observations_report_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
    REQUIRED_LOCAL_FK_LINKS,
    evaluate_controller_gate_v7,
    evaluate_local_fk_gate_v7,
)


_ASSET = Path(
    "outputs/anatomy_retarget/v7_candidates/rebuild_002/corrected_213328.npz"
)
_DOMAINS = Path(
    "outputs/anatomy_retarget/v7_candidates/joint_rebuild_001/fixed_joint_domains_v7.json"
)
_POSE = Path(
    "outputs/anatomy_retarget/v7_candidates/rebuild_002/input_pose_213328.npz"
)

_OCTA_FACES = np.asarray(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ],
    dtype=np.int64,
)


def _octahedron(center: tuple[float, float, float], radius: float) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    return center_array + radius * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def _eye(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _synthetic_fixture():
    names = [
        "Root",
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Tibia_Bone_L",
        "Patella_Rotate_L",
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Tibia_Bone_R",
        "Patella_Rotate_R",
    ]
    parents = np.asarray([-1, 0, 1, 2, 3, 0, 5, 6, 7], dtype=np.int64)
    bind = np.stack(
        [
            _eye((0.0, 0.0, 0.0)),
            _eye((-0.1, 0.0, 0.0)),
            _eye((-0.1, -0.4, 0.0)),
            _eye((-0.1, -0.45, 0.0)),
            _eye((-0.1, -0.42, 0.05)),
            _eye((0.1, 0.0, 0.0)),
            _eye((0.1, -0.4, 0.0)),
            _eye((0.1, -0.45, 0.0)),
            _eye((0.1, -0.42, 0.05)),
        ],
        axis=0,
    )

    chunks: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    memberships: dict[str, np.ndarray] = {}

    def add(name: str, center: tuple[float, float, float], radius: float) -> np.ndarray:
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(_octahedron(center, radius))
        faces.append(_OCTA_FACES + start)
        indices = np.arange(start, start + 6, dtype=np.int64)
        memberships[name] = indices
        return indices

    for side, x in (("left", -0.10), ("right", 0.10)):
        head = add(f"{side}/femoral_head", (x, 0.0, 0.0), 0.010)
        add(f"{side}/acetabulum", (x, 0.0, 0.0), 0.012)
        medial = add(
            f"{side}/femoral_condyle_medial",
            (x + (-0.006 if side == "left" else 0.006), -0.40, 0.0),
            0.004,
        )
        lateral = add(
            f"{side}/femoral_condyle_lateral",
            (x + (0.006 if side == "left" else -0.006), -0.40, 0.0),
            0.004,
        )
        add(f"{side}/tibial_plateau_medial", (x, -0.41, 0.0), 0.004)
        add(f"{side}/tibial_plateau_lateral", (x, -0.41, 0.0), 0.004)
        trochlea = add(f"{side}/trochlea", (x, -0.40, 0.003), 0.003)
        patella = add(f"{side}/patella", (x, -0.40, 0.010), 0.003)
        memberships[f"{side}/patella_articular"] = patella.copy()
        memberships[f"{side}/pelvis"] = memberships[f"{side}/acetabulum"].copy()
        memberships[f"{side}/femur"] = np.concatenate(
            (head, medial, lateral, trochlea)
        )
        memberships[f"{side}/tibia"] = np.concatenate(
            (
                memberships[f"{side}/tibial_plateau_medial"],
                memberships[f"{side}/tibial_plateau_lateral"],
            )
        )

    vertices = np.concatenate(chunks, axis=0)
    triangles = np.concatenate(faces, axis=0)
    domains = FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=vertices,
        faces=triangles,
        domains=memberships,
    )
    smplx_a = np.zeros(len(names), dtype=np.int64)
    smplx_b = np.zeros(len(names), dtype=np.int64)
    smplx_a[1] = 1
    smplx_b[1] = 4
    smplx_a[5] = 2
    smplx_b[5] = 5
    rest_joints = np.zeros((55, 3), dtype=np.float64)
    rest_joints[1] = (-0.1, 0.0, 0.0)
    rest_joints[4] = (-0.1, -0.4, 0.0)
    rest_joints[2] = (0.1, 0.0, 0.0)
    rest_joints[5] = (0.1, -0.4, 0.0)
    # Minimal SMPL-X parent topology for the hip/knee joints used above.
    smplx_parents = np.full(55, -1, dtype=np.int32)
    smplx_parents[1] = 0
    smplx_parents[2] = 0
    smplx_parents[4] = 1
    smplx_parents[5] = 2
    asset = SimpleNamespace(
        source_bone_names=names,
        source_bone_parents=parents,
        source_bone_smplx_a=smplx_a,
        source_bone_smplx_b=smplx_b,
        target_bind_global=bind,
        vertices_rest=vertices.copy(),
        faces=triangles,
        rest_joints=rest_joints,
        parents=smplx_parents,
    )
    return asset, domains, bind


def _pose_from_bind(
    bind: np.ndarray,
    *,
    knee_l_angle_rad: float = 0.0,
    knee_l_axis_local: np.ndarray | None = None,
    knee_l_twist_rad: float = 0.0,
    knee_l_translation_parent_local_m: np.ndarray | None = None,
) -> np.ndarray:
    posed = bind.copy()
    femur_l = 1
    knee_l = 2
    tibia_l = 3
    patella_l = 4
    axis = np.asarray(
        knee_l_axis_local if knee_l_axis_local is not None else (0.0, 0.0, 1.0),
        dtype=np.float64,
    )
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    bind_local = np.linalg.inv(bind[femur_l]) @ bind[knee_l]
    auth = Rotation.from_rotvec(axis * float(knee_l_angle_rad)).as_matrix()
    twist_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    twist = Rotation.from_rotvec(twist_axis * float(knee_l_twist_rad)).as_matrix()
    posed_local = bind_local.copy()
    posed_local[:3, :3] = bind_local[:3, :3] @ auth @ twist
    if knee_l_translation_parent_local_m is not None:
        posed_local[:3, 3] = bind_local[:3, 3] + np.asarray(
            knee_l_translation_parent_local_m, dtype=np.float64
        )
    posed[knee_l] = posed[femur_l] @ posed_local
    # Rigid bind_follow children.
    for child, parent in ((tibia_l, knee_l), (patella_l, tibia_l)):
        child_local = np.linalg.inv(bind[parent]) @ bind[child]
        posed[child] = posed[parent] @ child_local
    # Right chain stays at bind.
    return posed


def _skin_rigid(asset, posed_global: np.ndarray, bind: np.ndarray) -> np.ndarray:
    """Apply femur (and mirrored) rigid deltas to domain vertices for hip Procrustes."""
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    # Approximate: move all vertices with identity (rest==posed) except we will
    # leave them equal so hip rotation compares bone delta to identity Procrustes
    # when posed_global == bind. For flexed knee, femur stays at bind so hip
    # remains consistent.
    del posed_global, bind
    return vertices


def test_authorized_hinge_and_injected_errors(monkeypatch):
    asset, domains, bind = _synthetic_fixture()
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    flexion = np.radians(40.0)

    def make_reference() -> FkReferenceV7:
        return FkReferenceV7(
            patella_response_rad=lambda side, flexion_rad: 0.0,
            patella_axis_local={"left": axis.copy(), "right": axis.copy()},
            knee_axis_local={"left": axis.copy(), "right": axis.copy()},
            source="synthetic",
        )

    # Pure authorized hinge.
    posed = _pose_from_bind(bind, knee_l_angle_rad=flexion, knee_l_axis_local=axis)
    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7.source_bone_posed_global",
        lambda _asset, _pose: posed,
    )
    posed_vertices = _skin_rigid(asset, posed, bind)
    result = observe_fk_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=posed_vertices,
        domains=domains,
        reference=make_reference(),
    )
    knee = result["controller_observations"]["knee_left"]
    link = result["local_fk_observations"]["left/Femur_Rot>Knee_Rotate"]
    assert knee["available"] is True
    assert abs(float(knee["rotation_error_deg"])) < 1.0e-9
    assert abs(float(knee["translation_error_m"])) < 1.0e-12
    assert float(knee["flexion_deg"]) == pytest.approx(40.0, abs=1.0e-9)
    # The knee link has no authorized angle outside the candidate, so it reports
    # unavailable rather than scoring itself against its own measurement. The
    # measurements it can still make honestly stay in the record.
    assert link["available"] is False
    assert "no independent knee flexion reference" in link["reason"]
    assert link["rotation_error_deg"] is None
    assert link["response_error_deg"] is None
    assert link["authorized_angle_deg"] is None
    assert float(link["flexion_deg"]) == pytest.approx(40.0, abs=1.0e-9)
    assert abs(float(link["off_axis_residual_deg"])) < 1.0e-9
    assert abs(float(link["translation_error_m"])) < 1.0e-12

    # 3 degree pure off-axis twist (no on-hinge flexion, so residual == twist).
    posed_twist = _pose_from_bind(
        bind,
        knee_l_angle_rad=0.0,
        knee_l_axis_local=axis,
        knee_l_twist_rad=np.radians(3.0),
    )
    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7.source_bone_posed_global",
        lambda _asset, _pose: posed_twist,
    )
    twisted = observe_fk_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=posed_vertices,
        domains=domains,
        reference=make_reference(),
    )
    knee_twist = twisted["controller_observations"]["knee_left"]
    link_twist = twisted["local_fk_observations"]["left/Femur_Rot>Knee_Rotate"]
    assert float(knee_twist["rotation_error_deg"]) == pytest.approx(3.0, abs=1.0e-6)
    # Off-axis twist is still caught on the link, as the residual rather than as a
    # rotation error against a self-derived reference.
    assert float(link_twist["off_axis_residual_deg"]) == pytest.approx(3.0, abs=1.0e-6)
    assert abs(float(knee_twist["translation_error_m"])) < 1.0e-12

    # 5 mm parent-local translation.
    posed_shift = _pose_from_bind(
        bind,
        knee_l_angle_rad=flexion,
        knee_l_axis_local=axis,
        knee_l_translation_parent_local_m=np.asarray((0.005, 0.0, 0.0)),
    )
    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7.source_bone_posed_global",
        lambda _asset, _pose: posed_shift,
    )
    shifted = observe_fk_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=posed_vertices,
        domains=domains,
        reference=make_reference(),
    )
    knee_shift = shifted["controller_observations"]["knee_left"]
    link_shift = shifted["local_fk_observations"]["left/Femur_Rot>Knee_Rotate"]
    assert float(knee_shift["translation_error_m"]) == pytest.approx(0.005, abs=1.0e-12)
    assert float(link_shift["translation_error_m"]) == pytest.approx(0.005, abs=1.0e-12)
    assert abs(float(knee_shift["rotation_error_deg"])) < 1.0e-9


def test_hip_direction_and_axial_twist_fields(monkeypatch):
    asset, domains, bind = _synthetic_fixture()
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    posed = bind.copy()
    # Twist the left femur 12 deg about its long axis (Y) while keeping the
    # head→knee direction identical to SMPL-X (rest joints match bind).
    femur_l = 1
    knee_l = 2
    direction = bind[knee_l, :3, 3] - bind[femur_l, :3, 3]
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    twist = Rotation.from_rotvec(direction * np.radians(12.0)).as_matrix()
    posed[femur_l] = posed[femur_l].copy()
    posed[femur_l, :3, :3] = twist @ bind[femur_l, :3, :3]
    posed[knee_l] = posed[femur_l] @ (np.linalg.inv(bind[femur_l]) @ bind[knee_l])
    for child, parent in ((3, 2), (4, 3)):
        posed[child] = posed[parent] @ (np.linalg.inv(bind[parent]) @ bind[child])

    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7.source_bone_posed_global",
        lambda _asset, _pose: posed,
    )
    result = observe_fk_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
        domains=domains,
        reference=FkReferenceV7(
            patella_response_rad=lambda side, flexion_rad: 0.0,
            patella_axis_local={"left": axis.copy(), "right": axis.copy()},
            knee_axis_local={"left": axis.copy(), "right": axis.copy()},
            source="synthetic",
        ),
    )
    hip = result["controller_observations"]["hip_left"]
    assert hip["available"] is True
    assert hip["direction_error_deg"] is not None
    assert hip["axial_twist_deg"] is not None
    assert float(hip["direction_error_deg"]) == pytest.approx(0.0, abs=1.0e-6)
    assert abs(float(hip["axial_twist_deg"])) == pytest.approx(12.0, abs=1.0e-4)
    assert "rotation_error_deg" in hip

    gate = evaluate_controller_gate_v7(result["controller_observations"])
    assert gate["items"]["hip_left"]["direction_error_deg"] == pytest.approx(
        0.0, abs=1.0e-6
    )
    # Gate must use direction_error_deg, so pure axial twist still passes.
    assert gate["items"]["hip_left"]["pass"] is True


def test_default_reference_fail_closed(monkeypatch):
    asset, domains, bind = _synthetic_fixture()
    posed = bind.copy()
    monkeypatch.setattr(
        "projects.genesis_ue_sync.anatomy_retarget.fk_observation_v7.source_bone_posed_global",
        lambda _asset, _pose: posed,
    )
    observations = observe_fk_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
        domains=domains,
        reference=default_fk_reference_v7(),
    )
    for side in ("left", "right"):
        assert observations["controller_observations"][f"knee_{side}"]["available"] is False
        assert (
            observations["local_fk_observations"][f"{side}/Femur_Rot>Knee_Rotate"][
                "available"
            ]
            is False
        )
        assert (
            observations["local_fk_observations"][f"{side}/Tibia_Bone>Patella_Rotate"][
                "available"
            ]
            is False
        )
        assert observations["controller_observations"][f"knee_{side}"][
            "rotation_error_deg"
        ] is None

    controller = evaluate_controller_gate_v7(observations["controller_observations"])
    local_fk = evaluate_local_fk_gate_v7(
        observations["local_fk_observations"],
        required=REQUIRED_LOCAL_FK_LINKS,
    )
    assert controller["pass"] is False
    assert local_fk["pass"] is False

    report = observations_report_v7(
        asset,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
        domains=domains,
        reference=default_fk_reference_v7(),
    )
    assert report["controller"]["pass"] is False
    assert report["local_fk"]["pass"] is False
    assert "local_fk_arms" in report


@pytest.mark.skipif(
    not (_ASSET.is_file() and _DOMAINS.is_file() and _POSE.is_file()),
    reason="V7 rebuild_002 asset / domains / pose missing",
)
def test_real_asset_measurements():
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset

    asset = load_rigged_asset(str(_ASSET), validate=False)
    domains = FrozenJointMaterialDomainsV7.load_json(_DOMAINS)
    pose = np.asarray(
        np.load(str(_POSE))["pose_axis_angle"], dtype=np.float64
    )
    posed_vertices = np.asarray(
        skin_vertices(asset, pose, validate=False), dtype=np.float64
    )
    observations = observe_fk_v7(
        asset,
        pose_axis_angle=pose,
        posed_vertices=posed_vertices,
        domains=domains,
        reference=default_fk_reference_v7(),
    )

    print("\n=== real-asset controller / local-FK measurements ===")
    for name, obs in observations["controller_observations"].items():
        t_mm = (
            None
            if obs["translation_error_m"] is None
            else 1000.0 * float(obs["translation_error_m"])
        )
        r_deg = obs["rotation_error_deg"]
        print(
            f"controller {name}: translation_mm={t_mm} "
            f"rotation_deg={r_deg} available={obs['available']} reason={obs['reason']!r}"
        )
        if name.startswith("hip_"):
            assert obs["translation_error_m"] is not None
            assert np.isfinite(obs["translation_error_m"])
            assert obs["rotation_error_deg"] is not None
            assert np.isfinite(obs["rotation_error_deg"])
        else:
            assert obs["translation_error_m"] is not None
            assert np.isfinite(obs["translation_error_m"])

    for key in REQUIRED_LOCAL_FK_LINKS:
        obs = observations["local_fk_observations"][key]
        t_mm = (
            None
            if obs["translation_error_m"] is None
            else 1000.0 * float(obs["translation_error_m"])
        )
        print(
            f"local_fk {key}: translation_mm={t_mm} "
            f"rotation_deg={obs['rotation_error_deg']} "
            f"available={obs['available']} reason={obs['reason']!r}"
        )
        assert obs["translation_error_m"] is not None
        assert np.isfinite(obs["translation_error_m"])
        if key.endswith("Knee_Rotate>Tibia_Bone"):
            assert obs["rotation_error_deg"] is not None
            assert np.isfinite(obs["rotation_error_deg"])
