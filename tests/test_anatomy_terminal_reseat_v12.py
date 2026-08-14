"""Terminal re-seat: LBS apply, anchor ball, and solver/apply agreement."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    FOOT_ROOTS,
    HAND_ROOTS,
)
from projects.genesis_ue_sync.anatomy_retarget.terminal_reseat_v12 import (
    NON_REGRESSION_SLACK_M,
    TERMINAL_ROOTS,
    apply_terminal_reseat_v12,
    solve_terminal_reseat_v12,
)


def _cube(half: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    h = float(half)
    vertices = np.asarray(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _asset_two_bone() -> SimpleNamespace:
    """Tibia + ankle. One vertex is 50/50 so a hard cluster move is illegal."""

    names = ["Tibia_Bone_L", "Ankle_Rot_L"]
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        source_bone_names=names,
        source_tissues=np.asarray(["bone", "bone"]),
        source_mesh_controller_bones=np.asarray([0, 1], dtype=np.int64),
        source_vertex_ranges=np.asarray([[0, 1], [1, 3]], dtype=np.int64),
        driver_indices=np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.int64),
        driver_weights=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64
        ),
    )


def _translation(dx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(dx)
    return matrix


def test_apply_moves_a_mixed_weight_vertex_by_its_lbs_fraction() -> None:
    asset = _asset_two_bone()
    rest = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64
    )
    bind = np.stack([np.eye(4), np.eye(4)])
    bind[1, :3, 3] = (1.0, 0.0, 0.0)
    reseat = {
        "Ankle_Rot_L": {
            "transform": _translation(0.10),
            "controllers": [1],
            "vertex_ids": np.asarray([1, 2], dtype=np.int64),
            "translation_m": 0.10,
            "rotation_deg": 0.0,
            "root_origin_shift_m": 0.10,
            "max_outside_before_m": 0.0,
            "max_outside_after_m": 0.0,
            "outside_count_before": 0,
            "outside_count_after": 0,
        }
    }

    moved, matrices, _report = apply_terminal_reseat_v12(
        rest, bind, asset=asset, reseat=reseat
    )

    assert moved[0] == pytest.approx([0.0, 0.0, 0.0])
    assert moved[1] == pytest.approx([1.10, 0.0, 0.0])
    assert moved[2] == pytest.approx([0.55, 0.0, 0.0])
    assert matrices[1, 0, 3] == pytest.approx(1.10)


def test_identity_reseat_is_a_noop() -> None:
    asset = _asset_two_bone()
    rest = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64
    )
    bind = np.stack([np.eye(4), np.eye(4)])
    reseat = {
        "Ankle_Rot_L": {
            "transform": np.eye(4),
            "controllers": [1],
            "vertex_ids": np.asarray([1, 2], dtype=np.int64),
            "translation_m": 0.0,
            "rotation_deg": 0.0,
            "root_origin_shift_m": 0.0,
            "max_outside_before_m": 0.0,
            "max_outside_after_m": 0.0,
            "outside_count_before": 0,
            "outside_count_after": 0,
        }
    }

    moved, matrices, _report = apply_terminal_reseat_v12(
        rest, bind, asset=asset, reseat=reseat
    )

    assert moved == pytest.approx(rest)
    assert matrices == pytest.approx(bind)


def _solver_asset() -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    """One bone mesh per terminal root, all fully weighted to that root."""

    names = list(TERMINAL_ROOTS)
    n = len(names)
    vertices = np.zeros((n, 3), dtype=np.float64)
    for index, name in enumerate(names):
        vertices[index] = (0.2 * index, 0.0, 1.05)
    ranges = np.stack([np.arange(n), np.arange(n) + 1], axis=1)
    asset = SimpleNamespace(
        source_bone_names=names,
        source_tissues=np.asarray(["bone"] * n),
        source_mesh_controller_bones=np.arange(n, dtype=np.int64),
        source_vertex_ranges=ranges.astype(np.int64),
        driver_indices=np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int64),
        driver_weights=np.stack([np.ones(n), np.zeros(n)], axis=1),
    )
    bind = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    bind[:, :3, 3] = vertices
    parents = np.full(n, -1, dtype=np.int64)
    return asset, bind, parents


def test_solver_stays_inside_the_anatomical_ball() -> None:
    asset, bind, parents = _solver_asset()
    rest = bind[:, :3, 3].copy()
    skin, faces = _cube(0.5)
    target = rest[0].copy()
    budget = 0.02

    result = solve_terminal_reseat_v12(
        rest,
        asset=asset,
        skin=skin,
        skin_faces=faces,
        bone_parents=parents,
        bind=bind,
        anchor_targets={TERMINAL_ROOTS[0]: target},
        anchor_budget_m={TERMINAL_ROOTS[0]: budget},
        max_translation_m=0.20,
        max_rotation_deg=1.0,
        samples=4,
    )

    transform = result[TERMINAL_ROOTS[0]]["transform"]
    moved_origin = transform[:3, :3] @ bind[0, :3, 3] + transform[:3, 3]
    assert float(np.linalg.norm(moved_origin - target)) <= budget + 1.0e-6


def test_solver_after_metric_matches_the_lbs_apply() -> None:
    asset, bind, parents = _solver_asset()
    rest = bind[:, :3, 3].copy()
    skin, faces = _cube(2.0)

    solved = solve_terminal_reseat_v12(
        rest,
        asset=asset,
        skin=skin,
        skin_faces=faces,
        bone_parents=parents,
        bind=bind,
        max_translation_m=0.0,
        max_rotation_deg=0.0,
        samples=4,
    )
    moved, _matrices, report = apply_terminal_reseat_v12(
        rest, bind, asset=asset, reseat=solved
    )

    assert moved == pytest.approx(rest)
    for name in TERMINAL_ROOTS:
        assert report[name]["max_outside_after_m"] == pytest.approx(
            solved[name]["max_outside_after_m"]
        )
        assert solved[name]["translation_m"] == pytest.approx(0.0)
        assert solved[name]["rotation_deg"] == pytest.approx(0.0)


def test_terminal_roots_cover_both_hands_and_feet() -> None:
    assert TERMINAL_ROOTS == (*HAND_ROOTS, *FOOT_ROOTS)
    assert NON_REGRESSION_SLACK_M == pytest.approx(0.001)


def test_solver_will_not_buy_a_rest_win_with_a_posed_regression() -> None:
    """A tiny posed skin around the unmoved cluster forbids any real T."""

    asset, bind, parents = _solver_asset()
    rest = bind[:, :3, 3].copy()
    rest_skin, rest_faces = _cube(0.5)
    # 2 mm posed pocket around the unmoved first terminal.  Any translation
    # beyond that plus the 1 mm slack is a posed regression and must be
    # refused even though rest wants a 550 mm inward move.
    pose_skin, pose_faces = _cube(0.002)
    pose_skin = pose_skin + rest[0]
    identity = np.tile(np.eye(4, dtype=np.float64), (len(TERMINAL_ROOTS), 1, 1))
    result = solve_terminal_reseat_v12(
        rest,
        asset=asset,
        skin=rest_skin,
        skin_faces=rest_faces,
        bone_parents=parents,
        bind=bind,
        pose_frames=(
            {"skin": pose_skin, "skin_faces": pose_faces, "source_transforms": identity},
        ),
        max_translation_m=0.20,
        max_rotation_deg=5.0,
        samples=4,
    )

    first = result[TERMINAL_ROOTS[0]]
    assert first["translation_m"] <= 0.002 + NON_REGRESSION_SLACK_M + 1.0e-3
