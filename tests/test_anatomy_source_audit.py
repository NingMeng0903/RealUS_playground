from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.source_audit import (
    aggregate_weight_stats,
    compress_runtime_influences,
    pack_source_influences,
    transform_audit,
)


def test_source_influences_preserve_every_nonzero_authored_value() -> None:
    rows = [
        [(0, 0.40), (1, 0.25), (2, 0.15), (3, 0.10), (4, 0.10), (5, 0.50)],
        [],
    ]
    packed = pack_source_influences(
        rows,
        group_names={
            0: "bone0",
            1: "bone1",
            2: "bone2",
            3: "bone3",
            4: "bone4",
            5: "helper",
        },
        source_bone_index={
            "bone0": 0,
            "bone1": 1,
            "bone2": 2,
            "bone3": 3,
            "bone4": 4,
        },
        driver_width=5,
    )

    np.testing.assert_array_equal(packed.source_offsets, [0, 6, 6])
    np.testing.assert_array_equal(packed.source_group_indices, [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(
        packed.source_values, [0.40, 0.25, 0.15, 0.10, 0.10, 0.50]
    )
    np.testing.assert_array_equal(packed.driver_indices[0], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(
        packed.driver_weights[0], [0.40, 0.25, 0.15, 0.10, 0.10]
    )
    np.testing.assert_array_equal(packed.empty_driver_vertices, [1])
    assert packed.stats["vertices_over_four_armature_influences"] == 1
    assert packed.stats["excluded_non_armature_groups"] == {"helper": 1}
    assert not packed.source_values.flags.writeable
    with pytest.raises(ValueError):
        packed.source_values[0] = 0.0


def test_top4_runtime_view_is_separate_and_reports_error() -> None:
    packed = pack_source_influences(
        [[(index, weight) for index, weight in enumerate((0.4, 0.25, 0.15, 0.1, 0.1))]],
        group_names={index: f"bone{index}" for index in range(5)},
        source_bone_index={f"bone{index}": index for index in range(5)},
        driver_width=5,
    )
    full_before = packed.driver_weights.copy()
    compressed = compress_runtime_influences(
        packed.driver_indices, packed.driver_weights, top_k=4
    )

    assert compressed.indices.shape == (1, 4)
    np.testing.assert_allclose(compressed.weights.sum(axis=1), 1.0)
    assert compressed.error["affected_vertex_count"] == 1
    assert compressed.error["omitted_mass_max"] == pytest.approx(0.1)
    assert compressed.error["l1_error_max"] == pytest.approx(0.2)
    np.testing.assert_array_equal(packed.driver_weights, full_before)


def test_transform_and_aggregate_audit_report_actual_values() -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.diag([-1.0, 2.0, 3.0])
    transform[:3, 3] = (4.0, 5.0, 6.0)
    audited = transform_audit(transform)
    assert audited["mirrored"] is True
    assert audited["mirror_determinant"] == pytest.approx(-6.0)
    assert audited["translation"] == [4.0, 5.0, 6.0]

    aggregate = aggregate_weight_stats(
        [
            {
                "vertex_count": 3,
                "source_influence_count": 9,
                "armature_influence_count": 8,
                "vertices_without_source_influences": 0,
                "vertices_without_armature_influences": 1,
                "vertices_over_four_armature_influences": 2,
                "armature_influences_per_vertex": {"max": 6},
                "excluded_non_armature_groups": {"helper": 1},
            },
            {
                "vertex_count": 2,
                "source_influence_count": 4,
                "armature_influence_count": 4,
                "vertices_without_source_influences": 0,
                "vertices_without_armature_influences": 0,
                "vertices_over_four_armature_influences": 0,
                "armature_influences_per_vertex": {"max": 3},
                "excluded_non_armature_groups": {},
            },
        ]
    )
    assert aggregate["mesh_count"] == 2
    assert aggregate["vertex_count"] == 5
    assert aggregate["source_influence_count"] == 13
    assert aggregate["maximum_armature_influences_per_vertex"] == 6
