from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import (
    MAX_OUTSIDE_REGRESSION_M,
    WHOLE_BODY_GROUP,
    absolute_poke_metrics,
    bone_mesh_group_v12,
    compare_absolute_poke_v12,
    evaluate_absolute_posed_poke_v12,
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

_CUBE_VERTICES = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

# Outward-oriented so the winding number is positive strictly inside.
_CUBE_FACES = np.asarray(
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


def _octahedron(center: tuple[float, float, float], radius: float) -> np.ndarray:
    return np.asarray(center, dtype=np.float64) + radius * np.asarray(
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


def _asset(forearm_center: tuple[float, float, float]):
    """Two bones plus a vessel inside a unit cube skin.

    ``Femur_L`` sits well inside; ``Radius_L`` is placed by the caller so a
    test can move it through the skin.  ``Aorta`` is soft tissue and must be
    excluded from every bone statistic.
    """

    meshes = (
        ("Femur_L", "bone", _octahedron((0.0, 0.0, 0.0), 0.1)),
        ("Radius_L", "bone", _octahedron(forearm_center, 0.05)),
        ("Aorta", "vessel", _octahedron((0.0, 0.3, 0.0), 0.05)),
    )
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    names: list[str] = []
    tissues: list[str] = []
    cursor = 0
    for name, tissue, block in meshes:
        vertices.append(block)
        faces.append(_OCTA_FACES + cursor)
        ranges.append((cursor, cursor + len(block)))
        names.append(name)
        tissues.append(tissue)
        cursor += len(block)
    rest = np.concatenate(vertices)
    asset = SimpleNamespace(
        source_mesh_names=np.asarray(names),
        source_tissues=np.asarray(tissues),
        source_vertex_ranges=np.asarray(ranges, dtype=np.int64),
        faces=np.concatenate(faces),
        # Area weights are taken from here, never from the posed vertices.
        vertices_rest=rest,
    )
    return asset, rest


def _metrics(forearm_center: tuple[float, float, float]):
    asset, vertices = _asset(forearm_center)
    return asset, vertices, absolute_poke_metrics(
        vertices, asset=asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
    )


def test_metrics_separate_contained_and_poking_bones() -> None:
    _asset_obj, _vertices, metrics = _metrics((1.02, 0.0, 0.0))

    assert metrics["femur_L"]["max_outside_m"] == pytest.approx(0.0)
    assert metrics["femur_L"]["outside_area_fraction"] == pytest.approx(0.0)

    # +x tip sits 0.07 beyond the x=1 wall.
    assert metrics["forearm_L"]["max_outside_m"] == pytest.approx(0.07, abs=1.0e-9)
    assert metrics["forearm_L"]["outside_area_fraction"] > 0.0
    assert metrics["forearm_L"]["area_weighted_outside_depth_m"] > 0.0

    assert metrics[WHOLE_BODY_GROUP]["max_outside_m"] == pytest.approx(
        metrics["forearm_L"]["max_outside_m"]
    )


def test_soft_tissue_is_excluded_from_bone_statistics() -> None:
    _asset_obj, _vertices, metrics = _metrics((1.02, 0.0, 0.0))

    assert "vessel" not in metrics
    # 6 vertices per bone octahedron, two bones, no vessel.
    assert metrics[WHOLE_BODY_GROUP]["vertex_count"] == 12


def test_gate_fails_when_a_group_is_worse_than_the_reference() -> None:
    contained_asset, contained = _asset((0.5, 0.0, 0.0))
    reference = {
        "pose_a": absolute_poke_metrics(
            contained, asset=contained_asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
        )
    }
    poking_asset, poking = _asset((1.02, 0.0, 0.0))

    report = evaluate_absolute_posed_poke_v12(
        {"pose_a": poking},
        asset=poking_asset,
        skins={"pose_a": (_CUBE_VERTICES, _CUBE_FACES)},
        reference=reference,
    )

    assert report["passed"] is False
    reasons = {f["reason"] for f in report["failures"]}
    assert reasons == {"worse_than_reference"}
    # The aggregate group is reported but never judged, so it must not appear
    # as a second failure for the same vertex.
    assert {f["group"] for f in report["failures"]} == {"forearm_L"}
    assert WHOLE_BODY_GROUP in report["cells"]["pose_a"]["groups"]


@pytest.mark.parametrize(
    ("extra_m", "should_pass"),
    [(0.0005, True), (0.0011, False)],
)
def test_gate_brackets_the_one_millimetre_regression_limit(
    extra_m: float, should_pass: bool
) -> None:
    """A regression just under the limit passes; just over it fails."""

    base_x = 1.02
    reference_asset, reference_vertices = _asset((base_x, 0.0, 0.0))
    reference = {
        "pose_a": absolute_poke_metrics(
            reference_vertices,
            asset=reference_asset,
            skin=_CUBE_VERTICES,
            skin_faces=_CUBE_FACES,
        )
    }
    asset, vertices = _asset((base_x + extra_m, 0.0, 0.0))

    report = evaluate_absolute_posed_poke_v12(
        {"pose_a": vertices},
        asset=asset,
        skins={"pose_a": (_CUBE_VERTICES, _CUBE_FACES)},
        reference=reference,
    )

    regression = (
        report["cells"]["pose_a"]["groups"]["forearm_L"]["max_outside_m"]
        - reference["pose_a"]["forearm_L"]["max_outside_m"]
    )
    assert regression == pytest.approx(extra_m, abs=1.0e-9)
    assert (regression <= MAX_OUTSIDE_REGRESSION_M) is should_pass
    assert report["passed"] is should_pass


def test_groups_filter_restricts_what_is_judged() -> None:
    contained_asset, contained = _asset((0.5, 0.0, 0.0))
    reference = {
        "pose_a": absolute_poke_metrics(
            contained, asset=contained_asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
        )
    }
    asset, vertices = _asset((1.02, 0.0, 0.0))
    metrics = {
        "pose_a": absolute_poke_metrics(
            vertices, asset=asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
        )
    }

    assert compare_absolute_poke_v12(metrics, reference=reference)["passed"] is False
    # femur_L is contained in both, so judging only it must pass.
    scoped = compare_absolute_poke_v12(
        metrics, reference=reference, groups=["femur_L"]
    )
    assert scoped["passed"] is True


def test_target_tier_uses_the_all_vertex_p95_not_the_outside_only_p95() -> None:
    """One deep spike must not make p95 collapse onto max."""

    asset, vertices = _asset((1.02, 0.0, 0.0))
    metrics = absolute_poke_metrics(
        vertices, asset=asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
    )
    whole = metrics[WHOLE_BODY_GROUP]

    # Only the small forearm octahedron poke; the femur is fully inside.
    assert whole["outside_p95_m"] > whole["poke_p95_all_m"]
    assert whole["poke_p95_all_m"] < whole["max_outside_m"]

    reference = {"pose_a": metrics}
    report = compare_absolute_poke_v12({"pose_a": metrics}, reference=reference)
    # Nothing regressed against itself, but 70 mm blows the 15 mm ceiling.
    assert report["passed"] is True
    assert report["target_met"] is False
    assert {m["group"] for m in report["target_misses"]} == {"forearm_L"}
    assert all("poke_p95_all_m" in m for m in report["target_misses"])


def test_area_weights_ignore_the_posed_deformation() -> None:
    """Weighting on rest keeps the metric comparable across candidates."""

    asset, vertices = _asset((1.02, 0.0, 0.0))
    baseline = absolute_poke_metrics(
        vertices, asset=asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
    )
    # Same signed distances, but the femur is scaled so posed triangle areas
    # would shift the weighting if it were computed on the posed vertices.
    stretched = vertices.copy()
    stretched[0:6] = stretched[0:6] * 3.0
    stretched_metrics = absolute_poke_metrics(
        stretched, asset=asset, skin=_CUBE_VERTICES, skin_faces=_CUBE_FACES
    )

    assert stretched_metrics[WHOLE_BODY_GROUP][
        "outside_area_fraction"
    ] == pytest.approx(baseline[WHOLE_BODY_GROUP]["outside_area_fraction"])


def test_every_bone_mesh_gets_an_anatomical_group() -> None:
    """The v10 grouping drops bare vertebrae and teeth into 'other'."""

    for name in ("C3", "C7", "T1", "T12", "L5", "Molar_1st_4", "Canine_2"):
        group = bone_mesh_group_v12(name)
        assert not group.startswith("other"), name
    assert bone_mesh_group_v12("Femur_L") == "femur_L"
    assert bone_mesh_group_v12("C4") == "cervical"
    assert bone_mesh_group_v12("T7") == "thoracic"
    assert bone_mesh_group_v12("L3") == "lumbar"
    assert bone_mesh_group_v12("Premolar_2nd") == "teeth"

    with pytest.raises(ValueError):
        bone_mesh_group_v12("Nonexistent_Widget")


def test_gate_fails_closed_when_the_reference_is_missing() -> None:
    asset, vertices = _asset((1.02, 0.0, 0.0))

    with pytest.raises(KeyError):
        evaluate_absolute_posed_poke_v12(
            {"pose_a": vertices},
            asset=asset,
            skins={"pose_a": (_CUBE_VERTICES, _CUBE_FACES)},
            reference={},
        )
