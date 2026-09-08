from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.material_attachment_v13 import compile_material_attachment_map_v13
from projects.genesis_ue_sync.anatomy_retarget.material_runtime_v13 import _validate_attachment_authority


def _fixture():
    source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [.2, .2, .05], [.5, .2, .1]])
    target = source + [.01, 0., 0.]
    faces = np.asarray([[0, 1, 2]])
    ids = np.asarray([3, 4])
    attachment = compile_material_attachment_map_v13(source, faces, target, faces, source[ids])
    return SimpleNamespace(vertices_rest=source), target, ids, attachment


def test_rejects_same_length_soft_vertex_permutation():
    asset, rest, ids, attachment = _fixture()
    _validate_attachment_authority(asset, rest, ids, attachment)
    with pytest.raises(ValueError, match="ordered soft"):
        _validate_attachment_authority(asset, rest, ids[::-1], attachment)


def test_rejects_attachment_from_another_target_body():
    asset, rest, ids, attachment = _fixture()
    with pytest.raises(ValueError, match="target rest"):
        _validate_attachment_authority(asset, rest + [.001, 0., 0.], ids, attachment)


def test_rejects_source_bone_change_even_when_soft_points_match():
    asset, rest, ids, attachment = _fixture()
    asset.vertices_rest[0, 0] += .001
    with pytest.raises(ValueError, match="source rest"):
        _validate_attachment_authority(asset, rest, ids, attachment)
