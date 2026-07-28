from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
    extract_patella_law_v7,
    load_patella_oracle_v7,
    patella_bind_frames_v7,
    patella_oracle_sweep_v7,
    patella_world_transform_v7,
    save_patella_oracle_v7,
    solve_patella_contact_corrections_v7,
)


_REAL_ACTION = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "anatomy_retarget"
    / "v7_source_bake_001"
    / "blender_action_oracle_v7.npz"
)

_AXIS = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
_SLOPE = {"left": -0.20, "right": -0.21}
_TET_FACES = np.asarray(
    [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    dtype=np.int32,
)


def _rotation4(angle_rad: float, translation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(_AXIS * float(angle_rad)).as_matrix()
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _tet(center: tuple[float, float, float], scale: float = 0.01) -> np.ndarray:
    base = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(center, dtype=np.float64) + scale * base


def _build_synthetic_action(path: Path, *, frames: int = 80) -> Path:
    bone_names = [
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Tibia_Bone_L",
        "Patella_Rotate_L",
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Tibia_Bone_R",
        "Patella_Rotate_R",
    ]
    bone_parents = np.asarray([-1, 0, 1, 2, -1, 4, 5, 6], dtype=np.int32)
    rest = np.zeros((8, 4, 4), dtype=np.float32)
    for index in range(8):
        rest[index] = np.eye(4, dtype=np.float32)
        rest[index, :3, 3] = (0.0, -0.05 * (index % 4), 0.0)
    action = np.repeat(rest[None, :, :, :], frames, axis=0).astype(np.float32)

    # Keyed flexion samples plus intentional unkeyed frames that must be ignored.
    keyed_angles = np.linspace(10.0, 100.0, 40)
    unkeyed_angles = np.linspace(15.0, 90.0, 20)
    all_angles = np.concatenate((np.zeros(frames - 60), keyed_angles, unkeyed_angles))
    assert len(all_angles) == frames

    for frame, theta_deg in enumerate(all_angles):
        theta = float(np.radians(theta_deg))
        for side, knee_i, pat_i in (
            ("left", 1, 3),
            ("right", 5, 7),
        ):
            action[frame, knee_i] = _rotation4(theta, rest[knee_i, :3, 3]).astype(
                np.float32
            )
            if frame >= frames - 20:
                # Unkeyed: flexed knee but near-zero patella response.
                phi = float(np.radians(0.1 * np.sign(theta_deg or 1.0)))
            else:
                phi = float(_SLOPE[side]) * theta
            action[frame, pat_i] = _rotation4(phi, rest[pat_i, :3, 3]).astype(
                np.float32
            )

    payload: dict[str, np.ndarray] = {
        "bone_names": np.asarray(bone_names),
        "bone_parents": bone_parents,
        "bone_rest_local": rest,
        "bone_rest_global": rest.copy(),
        "bone_action_local": action,
        "bone_action_global": action.copy(),
    }
    for side, suffix, femur_c, pat_c in (
        ("left", "L", (-0.05, 0.0, 0.0), (-0.05, 0.0, 0.05)),
        ("right", "R", (0.05, 0.0, 0.0), (0.05, 0.0, 0.05)),
    ):
        femur = _tet(femur_c)
        patella = _tet(pat_c)
        payload[f"mesh__Femur_{suffix}__vertices"] = np.repeat(
            femur[None, :, :], frames, axis=0
        ).astype(np.float32)
        payload[f"mesh__Femur_{suffix}__faces"] = _TET_FACES.copy()
        payload[f"mesh__Patella_{suffix}__vertices"] = np.repeat(
            patella[None, :, :], frames, axis=0
        ).astype(np.float32)
        payload[f"mesh__Patella_{suffix}__faces"] = _TET_FACES.copy()
    np.savez(path, **payload)
    return path


def _synthetic_asset_and_domains():
    chunks: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    domains: dict[str, np.ndarray] = {}

    def add(name: str, center: tuple[float, float, float], scale: float = 0.01) -> None:
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(_tet(center, scale=scale))
        faces.append(_TET_FACES.astype(np.int64) + start)
        domains[name] = np.arange(start, start + 4, dtype=np.int64)

    # Separated femur / patella clouds so the deadband can keep zero translation.
    add("left/femur", (-0.05, 0.0, 0.0), 0.005)
    add("left/patella", (-0.05, 0.0, 0.004), 0.005)
    add("right/femur", (0.05, 0.0, 0.0), 0.005)
    add("right/patella", (0.05, 0.0, 0.004), 0.005)
    vertices = np.concatenate(chunks, axis=0)
    triangles = np.concatenate(faces, axis=0)
    frozen = FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=vertices,
        faces=triangles,
        domains=domains,
    )
    bone_names = [
        "Femur_Rot_L",
        "Knee_Rotate_L",
        "Tibia_Bone_L",
        "Patella_Rotate_L",
        "Femur_Rot_R",
        "Knee_Rotate_R",
        "Tibia_Bone_R",
        "Patella_Rotate_R",
    ]
    bind = np.zeros((8, 4, 4), dtype=np.float64)
    for index in range(8):
        bind[index] = np.eye(4, dtype=np.float64)
        bind[index, :3, 3] = (0.0, -0.05 * (index % 4), 0.0)
    asset = SimpleNamespace(
        source_bone_names=bone_names,
        target_bind_global=bind,
        vertices_rest=vertices.copy(),
    )
    return asset, frozen, vertices, triangles


def test_extract_recovers_injected_slope_and_excludes_unkeyed(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(action_path, knots_deg=np.linspace(0.0, 120.0, 25))
    for side in ("left", "right"):
        assert law.response_slope[side] == pytest.approx(_SLOPE[side], abs=1.0e-6)
        # 40 keyed frames only; the trailing 20 unkeyed flexed frames are dropped.
        assert law.keyed_frame_count[side] == 40


def test_save_load_round_trip_and_corruption(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(action_path)
    out = save_patella_oracle_v7(tmp_path / "law.npz", law)
    loaded = load_patella_oracle_v7(out)
    assert loaded.content_digest() == law.content_digest()

    corrupted = tmp_path / "corrupted.npz"
    data = dict(np.load(out, allow_pickle=False))
    # Keep response[0] == 0 and monotone so only the digest gate fires.
    response = np.asarray(data["response_deg__left"], dtype=np.float64).copy()
    response[1:] -= 0.25
    data["response_deg__left"] = response
    np.savez(corrupted, **data)
    with pytest.raises(ValueError, match="content_digest"):
        load_patella_oracle_v7(corrupted)


def test_response_rad_clamps_hyperextension_and_extrapolates(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(action_path)
    sample = np.asarray([0.0, 0.3, 0.6, 1.0, 2.5], dtype=np.float64)
    for side in ("left", "right"):
        positive = np.asarray(law.response_rad(side, sample), dtype=np.float64)
        negative = np.asarray(law.response_rad(side, -sample), dtype=np.float64)
        # Below-zero policy: clamp to the rest (0) response, not an odd extension.
        rest = float(law.response_rad(side, 0.0))
        assert negative == pytest.approx(rest, abs=1.0e-12)
        # Non-increasing under positive flexion for the synthetic monotone law.
        assert np.all(np.diff(positive) <= 1.0e-12)
        # Linear extrapolation past the last knot uses the final interval slope.
        knots = law.knots_deg
        resp = law.response_deg[side]
        final_slope = (resp[-1] - resp[-2]) / (knots[-1] - knots[-2])
        past = float(np.radians(knots[-1] + 15.0))
        expected = np.radians(resp[-1] + final_slope * 15.0)
        assert float(law.response_rad(side, past)) == pytest.approx(expected, abs=1.0e-12)


def test_world_transform_identity_at_zero_flexion(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(action_path)
    asset, _domains, _vertices, _faces = _synthetic_asset_and_domains()
    for side in ("left", "right"):
        frames = patella_bind_frames_v7(asset, side=side)
        world = patella_world_transform_v7(
            law, frames=frames, side=side, flexion_rad=0.0
        )
        assert world == pytest.approx(np.eye(4), abs=1.0e-12)


def test_contact_corrections_bound_and_deadband(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(
        action_path,
        corridor_min_m=0.0,
        corridor_max_m=0.10,
        corridor_target_m=0.003,
        max_contact_translation_m=0.005,
    )
    asset, domains, vertices, faces = _synthetic_asset_and_domains()
    # Gap ~4 mm sits inside the deadband
    # [corridor_min+0.2mm, corridor_max-1mm] so translations stay zero.
    translations, report = solve_patella_contact_corrections_v7(
        law,
        vertices=vertices,
        faces=faces,
        domains=domains,
        asset=asset,
        side="left",
        knots_deg=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
    )
    assert translations.shape == (3, 3)
    assert float(np.max(np.linalg.norm(translations, axis=1))) <= (
        float(law.max_contact_translation_m) + 1.0e-12
    )
    assert translations[0] == pytest.approx(np.zeros(3), abs=1.0e-12)
    deadband_lo = float(law.corridor_min_m) + 0.0002
    deadband_hi = float(law.corridor_max_m) - 0.001
    for row in report["per_knot"]:
        assert deadband_lo <= row["uncorrected_hard_min_m"] <= deadband_hi
        assert row["translation_norm_m"] == pytest.approx(0.0, abs=1.0e-12)

    # Overlapping clouds force a correction that must respect the translation bound.
    patella_ids = domains.require("left/patella")
    close = vertices.copy()
    close[patella_ids] = close[domains.require("left/femur")]
    asset.vertices_rest = close.copy()
    close_translations, _close_report = solve_patella_contact_corrections_v7(
        law,
        vertices=close,
        faces=faces,
        domains=domains,
        asset=asset,
        side="left",
        knots_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float64),
    )
    assert float(np.max(np.linalg.norm(close_translations, axis=1))) <= (
        float(law.max_contact_translation_m) + 1.0e-12
    )


def test_sweep_requires_zero_initial_flexion(tmp_path: Path) -> None:
    action_path = _build_synthetic_action(tmp_path / "action.npz")
    law = extract_patella_law_v7(action_path)
    asset, domains, _vertices, _faces = _synthetic_asset_and_domains()
    with pytest.raises(ValueError, match="flexion_rad\\[0\\]"):
        patella_oracle_sweep_v7(
            law,
            asset=asset,
            domains=domains,
            flexion_rad=np.asarray([0.1, 0.2], dtype=np.float64),
        )


@pytest.mark.skipif(not _REAL_ACTION.is_file(), reason="real V71 action oracle missing")
def test_real_action_law_matches_validated_slopes() -> None:
    law = extract_patella_law_v7(_REAL_ACTION)
    assert law.response_slope["right"] == pytest.approx(-0.210728, abs=1.0e-4)
    assert law.response_slope["left"] == pytest.approx(-0.195073, abs=5.0e-3)
    for side in ("left", "right"):
        assert 0.0005 <= law.penetration_envelope_m[side] <= 0.005
        assert law.keyed_frame_count[side] >= 30
