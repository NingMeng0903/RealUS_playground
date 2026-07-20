from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.source_contact_fit import (
    _endpoint_caps,
    _mesh_indices,
    _orient_caps,
    fit_lower_leg_source_contacts,
)


class _Asset:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


def _column(x: float, y0: float, y1: float) -> np.ndarray:
    levels = np.linspace(y0, y1, 9)
    cross_section = np.asarray(
        ((-0.02, -0.01), (-0.02, 0.01), (0.02, -0.01), (0.02, 0.01))
    )
    return np.asarray(
        [
            (x + dx, y, dz)
            for y in levels
            for dx, dz in cross_section
        ],
        dtype=np.float64,
    )


def _fixture() -> _Asset:
    names: list[str] = []
    tissues: list[str] = []
    ranges: list[tuple[int, int]] = []
    source_parts: list[np.ndarray] = []
    mapped_parts: list[np.ndarray] = []
    offset = 0
    for side, sign in (("L", 1.0), ("R", -1.0)):
        parts = (
            (f"Femur_{side}", _column(0.25 * sign, 1.1, 2.0)),
            (f"Tibia_{side}", _column(0.25 * sign, 0.0, 1.0)),
            (f"Fibula_{side}", _column(0.39 * sign, 0.0, 1.0)),
            (f"Talus_{side}", _column(0.32 * sign, -0.2, -0.1)),
        )
        for name, points in parts:
            names.append(name)
            tissues.append("bone")
            source_parts.append(points)
            mapped = points.copy()
            if "Tibia" in name or "Fibula" in name:
                mapped[:, 1] = 0.2 + 0.6 * mapped[:, 1]
            mapped_parts.append(mapped)
            ranges.append((offset, offset + len(points)))
            offset += len(points)
    source = np.concatenate(source_parts)
    mapped = np.concatenate(mapped_parts)
    return _Asset(
        vertices_rest=mapped.astype(np.float32),
        registration_reference=source.astype(np.float32),
        source_mesh_names=names,
        source_vertex_ranges=np.asarray(ranges, dtype=np.int32),
        source_tissues=tissues,
        metadata={"fixture": True},
    )


def test_contact_fit_restores_source_clearances_without_moving_neighbours() -> None:
    asset = _fixture()
    before = np.asarray(asset.vertices_rest).copy()
    fitted, report = fit_lower_leg_source_contacts(asset)

    changed = np.zeros(len(before), dtype=bool)
    for side in ("left", "right"):
        for token in ("tibia", "fibula"):
            changed[_mesh_indices(asset, token=token, side=side)] = True
        for key, source_gap in report[side]["source_clearance_m"].items():
            mapped_gap = report[side]["mapped_clearance_m"][key]
            # A different surface pair may become closer after both shafts
            # move; the authored clearance is an upper bound on separation.
            assert mapped_gap <= source_gap + 1.0e-6
            assert mapped_gap < report[side]["mapped_clearance_before_m"][key]
        assert report[side]["policy"] == (
            "minimum_displacement_source_clearance_shaft_fit_v2"
        )

    np.testing.assert_array_equal(fitted.vertices_rest[~changed], before[~changed])
    assert np.any(fitted.vertices_rest[changed] != before[changed])
    assert fitted.metadata["fixture"] is True


def test_contact_fit_keeps_each_endpoint_cap_rigid() -> None:
    asset = _fixture()
    fitted, _report = fit_lower_leg_source_contacts(asset)
    source = np.asarray(asset.registration_reference, dtype=np.float64)

    for side in ("left", "right"):
        femur = _mesh_indices(asset, token="femur", side=side)
        talus = _mesh_indices(asset, token="talus", side=side)
        for token in ("tibia", "fibula"):
            indices = _mesh_indices(asset, token=token, side=side)
            caps = _endpoint_caps(indices, source)
            if token == "tibia":
                proximal_neighbour = femur
            else:
                proximal_neighbour = _endpoint_caps(
                    _mesh_indices(asset, token="tibia", side=side), source
                )[1]
            proximal, distal = _orient_caps(
                caps,
                proximal_neighbour=proximal_neighbour,
                distal_neighbour=talus,
                vertices=source,
            )
            for cap in (proximal, distal):
                before = np.linalg.norm(
                    asset.vertices_rest[cap[0]] - asset.vertices_rest[cap[1:]], axis=1
                )
                after = np.linalg.norm(
                    fitted.vertices_rest[cap[0]] - fitted.vertices_rest[cap[1:]], axis=1
                )
                np.testing.assert_allclose(after, before, atol=1.0e-6, rtol=0.0)
