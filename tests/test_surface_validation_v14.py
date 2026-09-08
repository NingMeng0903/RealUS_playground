import numpy as np
import pytest

pytest.importorskip('vtk')
pytest.importorskip('igl')
import igl
import trimesh
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import audit_bone_pair, triangle_contacts


def test_crossed_thin_boxes_have_intersections_without_interior_vertices():
    a = trimesh.creation.box(extents=(2, .2, .2))
    b = trimesh.creation.box(extents=(.2, 2, .2))
    assert np.all(igl.signed_distance(a.vertices, b.vertices, b.faces)[0] > 0)
    assert np.all(igl.signed_distance(b.vertices, a.vertices, a.faces)[0] > 0)
    result = triangle_contacts(a.vertices, a.faces, b.vertices, b.faces)
    assert result['triangle_pair_count'] > 0
    assert not audit_bone_pair(a.vertices, a.faces, b.vertices, b.faces)['passed']


def test_containment_fails_even_without_surface_crossings():
    a = trimesh.creation.box(extents=(.2, .2, .2))
    b = trimesh.creation.box(extents=(2, 2, 2))
    result = audit_bone_pair(a.vertices, a.faces, b.vertices, b.faces)
    assert result['triangle_pair_count'] == 0
    assert not result['passed']
    assert result['signed_samples'][0]['max_depth_m'] > .5


def test_separated_pair_and_bad_winding_are_distinguished():
    a = trimesh.creation.box(extents=(.2, .2, .2))
    b = trimesh.creation.box(extents=(.2, .2, .2)); b.apply_translation((1, 0, 0))
    result = audit_bone_pair(a.vertices, a.faces, b.vertices, b.faces)
    assert result['passed']
    invalid = audit_bone_pair(a.vertices, a.faces[:, ::-1], b.vertices, b.faces)
    assert not invalid['passed']
    assert invalid['reason'] == 'invalid_closed_oriented_mesh'
