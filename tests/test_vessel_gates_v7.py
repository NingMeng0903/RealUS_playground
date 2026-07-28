from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    _default_smplx_model_path,
    easymocap_drive_translation,
    load_easymocap_smplx_fit_drive,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    body_surface_for_cell_v7,
    load_smplx_model_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    apply_subject_pose,
    load_source_operator,
    load_subject_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.vessel_gates_v7 import (
    VesselGateThresholdsV7,
    evaluate_vessel_gates_v7,
    vessel_topology_digest_v7,
)


_REPO = Path(__file__).resolve().parents[1]
_CAPTURE_NPZ = (
    _REPO / "smplx_outputs" / "20260713_213328" / "moment_0000" / "smplx_result.npz"
)
_SUBJECT_NPZ = (
    _REPO
    / "outputs"
    / "anatomy_retarget"
    / "v7_candidates"
    / "rebuild_003"
    / "subject_213328.npz"
)

# Measured against the capture fit with load_easymocap_smplx_fit_drive
# (closed-mouth) + easymocap_drive_translation: max vertex error ~0.367 mm.
_SMPLX_FORWARD_MAX_ERROR_M = 5.0e-4


def _box_mesh(
    *,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.asarray(half_extents, dtype=np.float64).reshape(3)
    corners = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = c + corners * h[None, :]
    # Outward-facing winding so igl.signed_distance is negative inside.
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 5, 4],
            [0, 1, 5],
            [1, 6, 5],
            [1, 2, 6],
            [2, 7, 6],
            [2, 3, 7],
            [3, 4, 7],
            [3, 0, 4],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _straight_tube(*, count: int = 40, radius: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Build a coarse open tube along +Z with four vertices per ring."""
    rings = count // 4
    assert rings * 4 == count
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    ring = np.stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(4)),
        axis=1,
    )
    vertices = []
    for index in range(rings):
        z = 0.02 * index / max(rings - 1, 1)
        vertices.append(ring + np.asarray([0.0, 0.0, z], dtype=np.float64))
    points = np.concatenate(vertices, axis=0)
    faces = []
    for ring_index in range(rings - 1):
        base = 4 * ring_index
        for edge in range(4):
            a = base + edge
            b = base + ((edge + 1) % 4)
            c = base + 4 + ((edge + 1) % 4)
            d = base + 4 + edge
            faces.append([a, b, c])
            faces.append([a, c, d])
    return points, np.asarray(faces, dtype=np.int32)


def _append_ring_faces(faces: list[list[int]], ring_a: int, ring_b: int) -> None:
    for edge in range(4):
        a = ring_a + edge
        b = ring_a + ((edge + 1) % 4)
        c = ring_b + ((edge + 1) % 4)
        d = ring_b + edge
        faces.append([a, b, c])
        faces.append([a, c, d])


def _y_shaped_tube(
    *,
    stem_rings: int = 14,
    branch_rings: int = 8,
    radius: float = 0.002,
    spacing: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build a Y-tube: stem along +Z, then left/right branches in ±X/+Z.

    Stem is longer than each arm so the graph diameter prefers stem↔one tip and
    the opposite arm becomes a detectable side branch.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    ring0 = np.stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(4)),
        axis=1,
    )
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []

    def add_ring(center: np.ndarray) -> int:
        index = 4 * len(vertices)
        vertices.append(ring0 + np.asarray(center, dtype=np.float64).reshape(1, 3))
        return index

    stem_ids = [add_ring([0.0, 0.0, spacing * i]) for i in range(stem_rings)]
    for a, b in zip(stem_ids[:-1], stem_ids[1:]):
        _append_ring_faces(faces, a, b)
    junction = stem_ids[-1]
    junction_z = spacing * (stem_rings - 1)

    left_ids = [junction]
    for i in range(1, branch_rings):
        t = float(i)
        left_ids.append(
            add_ring([spacing * t, 0.0, junction_z + spacing * t])
        )
    for a, b in zip(left_ids[:-1], left_ids[1:]):
        _append_ring_faces(faces, a, b)

    right_ids = [junction]
    for i in range(1, branch_rings):
        t = float(i)
        right_ids.append(
            add_ring([-spacing * t, 0.0, junction_z + spacing * t])
        )
    for a, b in zip(right_ids[:-1], right_ids[1:]):
        _append_ring_faces(faces, a, b)

    points = np.concatenate(vertices, axis=0)
    left_only = np.asarray(
        [idx + k for idx in left_ids[1:] for k in range(4)], dtype=np.int64
    )
    right_only = np.asarray(
        [idx + k for idx in right_ids[1:] for k in range(4)], dtype=np.int64
    )
    return (
        points,
        np.asarray(faces, dtype=np.int32),
        {"left_arm": left_only, "right_arm": right_only},
    )


def _fan_shaped_tube(
    *,
    trunk_rings: int = 10,
    long_rings: int = 40,
    short_rings: int = 16,
    segments: int = 8,
    radius: float = 0.002,
    spacing: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Trunk that splits into two gently diverging strands of unequal length.

    Nerve roots fan out like this. Both strands stay nearly straight, so a
    per-strand centerline turns very little at rest; a centerline that averages
    both strands per geodesic bin instead lurches sideways where the short
    strand ends. Sampling matches real vessel components (hundreds of vertices)
    so the measurement is not resolution-limited.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring0 = np.stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(segments),
        ),
        axis=1,
    )
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []

    def add_ring(center: np.ndarray) -> int:
        index = segments * len(vertices)
        vertices.append(ring0 + np.asarray(center, dtype=np.float64).reshape(1, 3))
        return index

    def stitch(ring_a: int, ring_b: int) -> None:
        for edge in range(segments):
            a = ring_a + edge
            b = ring_a + ((edge + 1) % segments)
            c = ring_b + ((edge + 1) % segments)
            d = ring_b + edge
            faces.append([a, b, c])
            faces.append([a, c, d])

    trunk_ids = [add_ring([0.0, 0.0, spacing * i]) for i in range(trunk_rings)]
    for a, b in zip(trunk_ids[:-1], trunk_ids[1:]):
        stitch(a, b)
    junction = trunk_ids[-1]
    junction_z = spacing * (trunk_rings - 1)

    def add_strand(sign: float, ring_count: int) -> list[int]:
        direction = np.asarray([0.3 * sign, 0.0, 1.0], dtype=np.float64)
        direction /= float(np.linalg.norm(direction))
        ids = [junction]
        for i in range(1, ring_count):
            ids.append(
                add_ring(
                    np.asarray([0.0, 0.0, junction_z], dtype=np.float64)
                    + direction * spacing * float(i)
                )
            )
        for a, b in zip(ids[:-1], ids[1:]):
            stitch(a, b)
        return ids

    long_ids = add_strand(1.0, long_rings)
    short_ids = add_strand(-1.0, short_rings)
    points = np.concatenate(vertices, axis=0)
    return (
        points,
        np.asarray(faces, dtype=np.int32),
        {
            "long_strand": np.asarray(
                [idx + k for idx in long_ids[1:] for k in range(segments)],
                dtype=np.int64,
            ),
            "short_strand": np.asarray(
                [idx + k for idx in short_ids[1:] for k in range(segments)],
                dtype=np.int64,
            ),
        },
    )


def _fan_stub_asset() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    tube_vertices, tube_faces, strands = _fan_shaped_tube()
    return (
        SimpleNamespace(
            vertices_rest=tube_vertices.astype(np.float64),
            faces=tube_faces.astype(np.int32),
            source_mesh_names=["FanNerve"],
            source_vertex_ranges=np.asarray(
                [[0, len(tube_vertices)]], dtype=np.int64
            ),
            source_tissues=["nerve"],
        ),
        strands,
    )


def _y_stub_asset() -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    tube_vertices, tube_faces, arms = _y_shaped_tube()
    bone_vertices, bone_faces = _box_mesh(
        center=np.asarray([0.05, 0.0, 0.01], dtype=np.float64),
        half_extents=np.asarray([0.008, 0.008, 0.008], dtype=np.float64),
    )
    vertices = np.concatenate((tube_vertices, bone_vertices), axis=0)
    faces = np.concatenate((tube_faces, bone_faces + len(tube_vertices)), axis=0)
    asset = SimpleNamespace(
        vertices_rest=vertices.astype(np.float64),
        faces=faces.astype(np.int32),
        source_mesh_names=["YVessel", "BoneBox"],
        source_vertex_ranges=np.asarray(
            [[0, len(tube_vertices)], [len(tube_vertices), len(vertices)]],
            dtype=np.int64,
        ),
        source_tissues=["vessel", "bone"],
    )
    return asset, arms


def _stub_asset() -> SimpleNamespace:
    tube_vertices, tube_faces = _straight_tube(count=40)
    bone_vertices, bone_faces = _box_mesh(
        center=np.asarray([0.05, 0.0, 0.01], dtype=np.float64),
        half_extents=np.asarray([0.008, 0.008, 0.008], dtype=np.float64),
    )
    vertices = np.concatenate((tube_vertices, bone_vertices), axis=0)
    faces = np.concatenate((tube_faces, bone_faces + len(tube_vertices)), axis=0)
    return SimpleNamespace(
        vertices_rest=vertices.astype(np.float64),
        faces=faces.astype(np.int32),
        source_mesh_names=["TubeVessel", "BoneBox"],
        source_vertex_ranges=np.asarray(
            [[0, len(tube_vertices)], [len(tube_vertices), len(vertices)]],
            dtype=np.int64,
        ),
        source_tissues=["vessel", "bone"],
    )


def _enclosing_body(vertices: np.ndarray, *, margin: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    lo = vertices.min(axis=0) - margin
    hi = vertices.max(axis=0) + margin
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return _box_mesh(center=center, half_extents=half)


def test_synthetic_centerline_and_penetration_identity_pass() -> None:
    asset = _stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    body = _enclosing_body(posed)
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=body,
        runtime_coefficients=None,
        thresholds=VesselGateThresholdsV7(),
    )
    assert result["centerline"]["available"] is True
    assert result["centerline"]["pass"] is True
    assert result["bone_penetration"]["available"] is True
    assert result["bone_penetration"]["pass"] is True
    assert result["bone_penetration"]["added_penetration_m"] == pytest.approx(0.0, abs=1e-12)


def test_synthetic_sharp_bend_fails_centerline() -> None:
    asset = _stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    # Displace the middle rings hard in +X to create a sharp kink.
    posed[16:24, 0] += 0.03
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(np.asarray(asset.vertices_rest)),
        runtime_coefficients=None,
    )
    assert result["centerline"]["available"] is True
    assert result["centerline"]["pass"] is False
    assert "centerline" in result["failures"]


def test_synthetic_outside_body_fails_containment() -> None:
    asset = _stub_asset()
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    body = _enclosing_body(rest, margin=0.002)
    posed = rest.copy()
    posed[:40] += np.asarray([0.0, 0.0, 0.010], dtype=np.float64)
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=body,
        runtime_coefficients=None,
    )
    assert result["containment"]["available"] is True
    assert result["containment"]["pass"] is False
    assert result["containment"]["max_outside_m"] >= 0.005
    assert "containment" in result["failures"]


def test_synthetic_into_bone_fails_bone_penetration() -> None:
    asset = _stub_asset()
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    posed = rest.copy()
    # Translate the tube into the bone box centered at x=0.05.
    posed[:40, 0] += 0.048
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(rest, margin=0.10),
        runtime_coefficients=None,
    )
    assert result["bone_penetration"]["available"] is True
    assert result["bone_penetration"]["pass"] is False
    assert result["bone_penetration"]["added_penetration_m"] > 0.001
    assert "bone_penetration" in result["failures"]


def test_missing_body_surface_fails_closed() -> None:
    asset = _stub_asset()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
        body_surface=None,
        runtime_coefficients=None,
    )
    assert result["containment"]["available"] is False
    assert result["containment"]["pass"] is False
    assert result["pass"] is False
    assert "containment" in result["failures"]


def test_missing_tube_coefficients_fails_closed() -> None:
    asset = _stub_asset()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=np.asarray(asset.vertices_rest, dtype=np.float64),
        body_surface=_enclosing_body(np.asarray(asset.vertices_rest)),
        runtime_coefficients=None,
    )
    assert result["cross_section"]["available"] is False
    assert result["cross_section"]["pass"] is False
    assert result["pass"] is False
    assert "cross_section" in result["failures"]


@pytest.mark.skipif(
    not _CAPTURE_NPZ.is_file(),
    reason="capture SMPL-X fit NPZ missing",
)
def test_smplx_body_surface_round_trip_against_capture() -> None:
    try:
        model_path = _default_smplx_model_path("male")
    except FileNotFoundError:
        local = (
            _REPO
            / "ref_code_library"
            / "EasyMocap"
            / "data"
            / "smplx"
            / "smplx"
            / "SMPLX_MALE.pkl"
        )
        if not local.is_file():
            pytest.skip("SMPL-X male model pickle missing")
        model_path = local

    data = np.load(_CAPTURE_NPZ)
    gt = np.asarray(data["vertices"], dtype=np.float64)
    betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)
    pose55, th = load_easymocap_smplx_fit_drive(_CAPTURE_NPZ, model_path=model_path)
    pose = np.asarray(pose55, dtype=np.float64).reshape(55, 3)
    model = load_smplx_model_v7(model_path)
    v_shaped = model["v_template"] + np.einsum(
        "vks,s->vk",
        model["shapedirs"][:, :, :10],
        betas[:10],
    )
    pelvis = (model["J_regressor"] @ v_shaped)[0]
    transl = easymocap_drive_translation(pose[0], th, pelvis)
    vertices, faces = smplx_body_surface_v7(
        model,
        betas=betas,
        pose_axis_angle=pose,
        transl=transl,
    )
    assert faces.shape == np.asarray(data["faces"]).shape
    max_error = float(np.max(np.linalg.norm(vertices - gt, axis=1)))
    print(
        f"smplx_body_surface_v7 max vertex error vs capture: "
        f"{max_error * 1000.0:.4f} mm"
    )
    assert max_error <= _SMPLX_FORWARD_MAX_ERROR_M

    capture_vertices, capture_faces, provenance = body_surface_for_cell_v7(
        capture_result_path=_CAPTURE_NPZ
    )
    assert provenance["source"] == "capture_fit_vertices"
    assert capture_vertices.shape == gt.shape
    assert capture_faces.shape[1] == 3


@pytest.mark.skipif(
    not _SUBJECT_NPZ.is_file() or not _CAPTURE_NPZ.is_file(),
    reason="subject asset or capture NPZ missing",
)
def test_real_asset_smoke_subgates_available() -> None:
    subject = load_subject_asset(_SUBJECT_NPZ)
    asset = subject.rigged_asset
    posed = apply_subject_pose(
        subject,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        transl=None,
        validate=False,
    )
    body_vertices, body_faces, provenance = body_surface_for_cell_v7(
        capture_result_path=_CAPTURE_NPZ
    )
    assert provenance["source"] == "capture_fit_vertices"
    operator = load_source_operator(_SUBJECT_NPZ.parent / "source_operator_v7.npz")
    template_digest = vessel_topology_digest_v7(operator.template_asset)

    # Without a digest from outside the candidate the topology gate has nothing to
    # compare against and must fail closed instead of matching itself.
    unreferenced = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        runtime_coefficients=dict(subject.runtime_coefficients),
        body_surface=(body_vertices, body_faces),
    )
    assert unreferenced["topology"]["available"] is False
    assert unreferenced["topology"]["pass"] is False

    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        runtime_coefficients=dict(subject.runtime_coefficients),
        body_surface=(body_vertices, body_faces),
        reference_faces_digest=template_digest,
    )
    assert result["topology"]["reference_faces_digest"] == template_digest
    assert result["topology"]["pass"] is True
    for name in (
        "topology",
        "cross_section",
        "centerline",
        "containment",
        "bone_penetration",
    ):
        gate = result[name]
        assert gate["available"] is True, name
        for key, value in gate.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"{name}.{key}"

    print("vessel_gates_v7 real-asset smoke:")
    print(f"  cross_section.max_abs_change={result['cross_section']['radius_edge_ratio_max_abs_change']}")
    print(
        f"  centerline.worst_max_turn_increase_deg="
        f"{result['centerline']['worst_max_turn_increase_deg']} "
        f"worst={result['centerline']['worst_component']}"
    )
    print(
        f"  containment.inside_ratio={result['containment']['inside_ratio']} "
        f"max_outside_m={result['containment']['max_outside_m']}"
    )
    print(
        f"  bone_penetration.reference_max={result['bone_penetration']['reference_max_penetration_m']} "
        f"posed_max={result['bone_penetration']['posed_max_penetration_m']} "
        f"added={result['bone_penetration']['added_penetration_m']}"
    )
    print(f"  failures={result['failures']} pass={result['pass']}")


def test_y_tube_identity_passes_geodesic_centerline() -> None:
    """Branched Y identity must pass; single-axis slabbing would spuriously fail."""
    asset, _arms = _y_stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(posed),
        runtime_coefficients=None,
    )
    center = result["components"]["YVessel"]["centerline"]
    assert center["available"] is True
    assert center["skipped"] is False
    assert center["centerline_method"] == "geodesic_diameter_bins_v7"
    assert center["max_turn_increase_deg"] == pytest.approx(0.0, abs=1e-9)
    assert center["q99_turn_increase_deg"] == pytest.approx(0.0, abs=1e-9)
    assert result["centerline"]["pass"] is True
    assert "centerline" not in result["failures"]


def test_y_tube_kinked_branch_fails_and_names_branch() -> None:
    asset, arms = _y_stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    # Sharply kink a short mid-span of the left arm in +Y. With a longer stem
    # the diameter path prefers stem↔right tip, so the left arm is a named
    # side branch under the geodesic centerline method.
    left = arms["left_arm"]
    lo = max(4, (len(left) * 2) // 5)
    hi = min(len(left) - 4, (len(left) * 3) // 5)
    assert hi > lo
    posed[left[lo:hi], 1] += 0.05
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(np.asarray(asset.vertices_rest)),
        runtime_coefficients=None,
    )
    assert result["centerline"]["available"] is True
    assert result["centerline"]["pass"] is False
    assert "centerline" in result["failures"]
    center = result["components"]["YVessel"]["centerline"]
    assert center["pass"] is False
    failed_branches = [
        name
        for name, report in center["branches"].items()
        if not report.get("skipped", False) and not report.get("pass", True)
    ]
    assert failed_branches
    assert any("branch" in name for name in failed_branches)
    assert "branch" in str(center["worst_branch"])


def test_fan_identity_pose_reports_no_turn_increase() -> None:
    """A fork's blended rest centroid must cancel between rest and posed."""
    asset, _strands = _fan_stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(posed),
        runtime_coefficients=None,
    )
    center = result["components"]["FanNerve"]["centerline"]
    assert center["skipped"] is False
    assert center["max_turn_increase_deg"] == pytest.approx(0.0, abs=1e-9)
    assert center["q99_turn_increase_deg"] == pytest.approx(0.0, abs=1e-9)
    assert result["centerline"]["pass"] is True


def test_fan_kink_on_one_strand_is_not_diluted_by_its_neighbour() -> None:
    """A kink in one strand must be measured on that strand, not averaged out."""
    asset, strands = _fan_stub_asset()
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    long_strand = strands["long_strand"]
    lo = (len(long_strand) * 2) // 5
    hi = (len(long_strand) * 3) // 5
    assert hi > lo
    posed[long_strand[lo:hi], 1] += 0.02
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(np.asarray(asset.vertices_rest)),
        runtime_coefficients=None,
    )
    assert result["centerline"]["pass"] is False
    assert "centerline" in result["failures"]
    center = result["components"]["FanNerve"]["centerline"]
    kinked = [
        report
        for report in center["branches"].values()
        if not report.get("skipped", False)
        and float(report.get("max_turn_increase_deg", 0.0)) > 20.0
    ]
    assert kinked, "one-strand kink was averaged away"


def test_short_tube_branch_recorded_as_skipped() -> None:
    tube_vertices, tube_faces = _straight_tube(count=20)
    asset = SimpleNamespace(
        vertices_rest=tube_vertices.astype(np.float64),
        faces=tube_faces.astype(np.int32),
        source_mesh_names=["ShortTube"],
        source_vertex_ranges=np.asarray([[0, len(tube_vertices)]], dtype=np.int64),
        source_tissues=["vessel"],
    )
    posed = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        body_surface=_enclosing_body(posed),
        runtime_coefficients=None,
    )
    center = result["components"]["ShortTube"]["centerline"]
    assert center["skipped"] is True
    assert center["reason"] == "fewer_than_24_vertices"
    assert center["vertex_count"] == 20
    assert "ShortTube" in result["centerline"]["skipped_components"]


_INPUT_POSE_NPZ = (
    _REPO
    / "outputs"
    / "anatomy_retarget"
    / "v7_candidates"
    / "rebuild_002"
    / "input_pose_213328.npz"
)


@pytest.mark.skipif(
    not _SUBJECT_NPZ.is_file() or not _INPUT_POSE_NPZ.is_file(),
    reason="subject asset or capture input pose NPZ missing",
)
def test_real_asset_centerline_capture_pose_finite() -> None:
    import time

    subject = load_subject_asset(_SUBJECT_NPZ)
    asset = subject.rigged_asset
    pose = np.load(_INPUT_POSE_NPZ)
    posed = apply_subject_pose(
        subject,
        pose_axis_angle=np.asarray(pose["pose_axis_angle"], dtype=np.float32),
        transl=np.asarray(pose["transl"], dtype=np.float32),
        validate=False,
    )
    started = time.perf_counter()
    result = evaluate_vessel_gates_v7(
        asset=asset,
        posed_vertices=posed,
        runtime_coefficients=dict(subject.runtime_coefficients),
        body_surface=None,
        thresholds=VesselGateThresholdsV7(),
    )
    # Only the centerline sub-gate is under test here; isolation via direct call
    # would require exporting internals, so measure wall time of full evaluate
    # but print/assert on centerline fields only.
    elapsed = time.perf_counter() - started
    center = result["centerline"]
    assert center["available"] is True
    assert center["component_count"] >= 1
    assert center["evaluated_component_count"] + len(center["skipped_components"]) == (
        center["component_count"]
    )
    print(
        f"vessel_gates_v7 capture-pose centerline "
        f"(full evaluate wall {elapsed:.3f}s):"
    )
    print(
        f"  method={center.get('centerline_method')} "
        f"worst={center['worst_component']} "
        f"worst_max_turn_increase_deg={center['worst_max_turn_increase_deg']:.4f} "
        f"failed={center['failed_components']}"
    )
    for name in sorted(result["components"]):
        report = result["components"][name]["centerline"]
        assert report["available"] is True, name
        for key, value in report.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"{name}.{key}={value}"
        if report.get("skipped"):
            print(
                f"  {name}: skipped reason={report.get('reason')} "
                f"vertex_count={report.get('vertex_count')}"
            )
            continue
        print(
            f"  {name}: pass={report.get('pass')} "
            f"ref_max={report.get('reference_max_turn_deg'):.4f} "
            f"posed_max={report.get('posed_max_turn_deg'):.4f} "
            f"dmax={report.get('max_turn_increase_deg'):.4f} "
            f"dq99={report.get('q99_turn_increase_deg'):.4f} "
            f"worst_branch={report.get('worst_branch')} "
            f"branches={len(report.get('branches', {}))}"
        )
        for branch_name, branch in sorted(report.get("branches", {}).items()):
            for key, value in branch.items():
                if isinstance(value, float):
                    assert np.isfinite(value), f"{name}.{branch_name}.{key}"
            if branch.get("skipped"):
                print(
                    f"    {branch_name}: skipped "
                    f"vertex_count={branch.get('vertex_count')}"
                )
            else:
                print(
                    f"    {branch_name}: pass={branch.get('pass')} "
                    f"dmax={branch.get('max_turn_increase_deg', float('nan')):.4f} "
                    f"verts={branch.get('vertex_count')}"
                )
    # Re-time centerline-only path for the performance claim.
    from projects.genesis_ue_sync.anatomy_retarget.vessel_gates_v7 import (
        _evaluate_centerline,
        vessel_tissue_vertex_ids_v7,
    )

    ids = vessel_tissue_vertex_ids_v7(asset)
    reference = np.asarray(asset.vertices_rest, dtype=np.float64)
    t0 = time.perf_counter()
    summary, _components = _evaluate_centerline(
        posed_vertices=posed,
        reference_vertices=reference,
        vertex_ids_by_mesh=ids,
        faces=np.asarray(asset.faces, dtype=np.int64),
        thresholds=VesselGateThresholdsV7(),
    )
    centerline_elapsed = time.perf_counter() - t0
    print(f"  centerline-only wall time: {centerline_elapsed:.3f}s")
    assert np.isfinite(summary["worst_max_turn_increase_deg"])
    assert centerline_elapsed < 10.0
