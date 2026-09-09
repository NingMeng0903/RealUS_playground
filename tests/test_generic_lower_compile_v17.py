"""Bounded API tests for the generic lower-subject V17 envelope.

The optimizer and SMPL-X model are intentionally outside this file.  These
tests cover the finite ten-dimensional beta contract, the authenticated save
envelope, and the closest-face proxy's documented sign convention.
"""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import generic_lower_compile_v17 as generic


@pytest.mark.parametrize("edge", [3.1, -3.1])
def test_target_beta_accepts_any_finite_ten_vector_without_three_sigma_cap(edge):
    values = np.zeros(10, dtype=np.float64)
    values[0] = edge
    result = generic._beta(values)
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize(
    "bad",
    [np.zeros(9), np.zeros(11), np.array([np.inf] + [0.0] * 9)],
)
def test_target_beta_rejects_wrong_shape_or_nonfinite_values(bad):
    with pytest.raises(ValueError, match="ten finite"):
        generic._beta(bad)


class _FakeRuntime:
    """Minimal persisted-runtime surface for envelope tests."""

    def __init__(self, betas, digest):
        self.betas = np.asarray(betas, dtype=np.float64)
        self.source_pack = SimpleNamespace(operator_runtime_digest=str(digest))

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text("{}\n", encoding="utf-8")


def _fake_compiled(target=None):
    source = np.linspace(-0.2, 0.2, 10, dtype=np.float64)
    target = source.copy() if target is None else np.asarray(target, dtype=np.float64)
    digest = "d" * 64
    runtime = _FakeRuntime(source, digest)
    report = {
        "method": "generic_lower_station_fit_v17",
        "target_beta": target.tolist(),
        "source_reference_beta": source.tolist(),
        "source_operator_digest": digest,
        "anatomical_passed": False,
        "whole_body_beta_fit_completed": False,
        "subject_specific_branches": False,
    }
    return generic.CompiledLowerSubjectV17(target, runtime, report)


def test_save_load_roundtrip_authenticates_target_reference_and_runtime_manifest(tmp_path, monkeypatch):
    target = np.zeros(10, dtype=np.float64)
    target[0] = 3.1
    compiled = _fake_compiled(target)
    output = tmp_path / "compiled"
    compiled.save(output)

    loaded_runtime = _FakeRuntime(compiled.runtime.betas, "d" * 64)
    monkeypatch.setattr(generic, "load_compiled_subject", lambda path: loaded_runtime)
    loaded = generic.load_lower_subject(output)
    np.testing.assert_array_equal(loaded.target_betas, target)
    np.testing.assert_array_equal(loaded.runtime.betas, compiled.runtime.betas)
    assert loaded.report["anatomical_passed"] is False
    assert loaded.report["whole_body_beta_fit_completed"] is False

    # The runtime manifest is covered by the outer envelope hash.  A change is
    # rejected before any runtime loader can consume it.
    (output / "runtime" / "manifest.json").write_text("{\"tampered\":true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime manifest hash"):
        generic.load_lower_subject(output)


def test_load_rejects_target_beta_identity_tampering(tmp_path, monkeypatch):
    target = np.zeros(10, dtype=np.float64)
    target[1] = -3.1
    compiled = _fake_compiled(target)
    output = tmp_path / "compiled"
    compiled.save(output)
    monkeypatch.setattr(
        generic,
        "load_compiled_subject",
        lambda path: _FakeRuntime(compiled.runtime.betas, "d" * 64),
    )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actual_target_beta"][1] = 0.0
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target beta identity"):
        generic.load_lower_subject(output)


def test_skin_fit_proxy_signed_distance_matches_closed_cube_orientation():
    # Outward winding for all six cube faces.  SkinFitProxy defines positive as
    # the outward side of the selected closest face and negative as inside.
    vertices = np.asarray(
        [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 2, 1), (0, 3, 2),       # z = 0
            (4, 5, 6), (4, 6, 7),       # z = 1
            (0, 4, 7), (0, 7, 3),       # x = 0
            (1, 2, 6), (1, 6, 5),       # x = 1
            (0, 1, 5), (0, 5, 4),       # y = 0
            (3, 7, 6), (3, 6, 2),       # y = 1
        ],
        dtype=np.int64,
    )
    proxy = generic.SkinFitProxy(vertices, faces)
    values = proxy.signed(
        np.asarray(
            [(0.5, 0.5, 0.5), (1.2, 0.5, 0.5), (0.0, 0.5, 0.5)],
            dtype=np.float64,
        )
    )
    assert values[0] < 0.0
    np.testing.assert_allclose(values[0], -0.5, atol=1e-12)
    np.testing.assert_allclose(values[1], 0.2, atol=1e-12)
    np.testing.assert_allclose(values[2], 0.0, atol=1e-12)


KNOWN_NONCONVEX_POINT = Path(
    "/home/camp/anatomy_retarget_outputs/v17_lower_213328_20260909_002/geometry/tpose.npz"
)


@pytest.mark.skipif(
    not KNOWN_NONCONVEX_POINT.exists(),
    reason="the fixed V17 non-convex skin regression fixture is optional",
)
def test_skin_fit_proxy_uses_winding_for_a_known_nearest_face_counterexample():
    """A nearest face normal can point outward for an actually internal point.

    This is the fixed point used during the V17 fit audit.  Keeping the test
    tied to the saved geometry makes the old closest-normal implementation
    fail deterministically while avoiding a large model fixture in ordinary
    unit runs.
    """
    import igl

    with np.load(KNOWN_NONCONVEX_POINT, allow_pickle=False) as data:
        skin_vertices = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int64)
        point = np.asarray(data["candidate_vertices"][134807], dtype=np.float64)

    squared, face_ids, closest = _closest_face_query(skin_vertices, skin_faces, point)
    tri = skin_vertices[skin_faces[face_ids]]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    nearest_face_dot = float(np.einsum("ij,ij->i", point[None] - closest, normals)[0])
    winding = float(igl.fast_winding_number(skin_vertices, skin_faces, point[None])[0])

    # The closest triangle's normal gives the wrong outside sign here, while
    # whole-surface winding identifies the point as internal.
    assert nearest_face_dot > 0.0
    assert winding > 0.5
    np.testing.assert_allclose(
        generic.SkinFitProxy(skin_vertices, skin_faces).signed(point[None])[0],
        -np.sqrt(max(float(squared[0]), 0.0)),
        atol=1e-10,
    )


def _closest_face_query(vertices, faces, point):
    """Small test helper exposing libigl's closest face query."""
    import igl

    tree = igl.AABB()
    tree.init(vertices, faces)
    return tree.squared_distance(vertices, faces, np.asarray(point, dtype=np.float64)[None])


CANONICAL_REFERENCE = Path(
    "/home/camp/anatomy_retarget_outputs/v17_canonical_reference_20260909_001"
)


@pytest.mark.skipif(not CANONICAL_REFERENCE.exists(), reason="canonical V17 reference is optional")
def test_existing_canonical_reference_is_unfitted_and_beta_independent():
    reference = generic.load_compiled_subject(CANONICAL_REFERENCE)
    assert reference.provenance["method"] == "canonical_reference_v17"
    assert reference.provenance["target_beta_used"] is False
    assert reference.provenance["shape_reference_kind"] == "frozen_operator_template"
    assert "lower_chain_rest_v17" not in reference.provenance
