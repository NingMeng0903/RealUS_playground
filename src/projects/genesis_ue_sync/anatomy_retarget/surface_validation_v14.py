"""Independent triangle contact and bidirectional volume diagnostics.

VTK's OBB-tree triangle contact query complements signed point distances:
two thin surfaces can cross without either mesh having an interior vertex.
Neither sampled depth nor a contact count establishes anatomical contact
quality. Closed, consistently wound targets are required for signed depths.
This module is offline validation only; compiled posing never imports it.
"""
from __future__ import annotations

import numpy as np


def _mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all():
        raise ValueError('mesh vertices must be finite [N,3]')
    if f.ndim != 2 or f.shape[1:] != (3,) or f.dtype.kind not in 'iu':
        raise ValueError('mesh faces must be integer triangles')
    f = np.asarray(f, dtype=np.int64)
    if len(v) == 0 or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
        raise ValueError('mesh faces have invalid indices or are empty')
    return np.ascontiguousarray(v), np.ascontiguousarray(f)


def triangle_contacts(vertices_a, faces_a, vertices_b, faces_b) -> dict:
    """Return intersecting triangle pairs, including crossings missed by vertices."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    a, fa = _mesh(vertices_a, faces_a)
    b, fb = _mesh(vertices_b, faces_b)
    models = []
    for v, f in ((a, fa), (b, fb)):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(v, deep=True))
        cells = vtk.vtkCellArray()
        offsets = np.arange(0, 3 * (len(f) + 1), 3, dtype=np.int64)
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(f.ravel(), deep=True))
        model = vtk.vtkPolyData(); model.SetPoints(points); model.SetPolys(cells)
        models.append(model)
    query = vtk.vtkCollisionDetectionFilter()
    transforms = []
    for i, model in enumerate(models):
        transform = vtk.vtkTransform(); transform.Identity(); transforms.append(transform)
        query.SetInputData(i, model); query.SetTransform(i, transform)
    query.SetCollisionModeToAllContacts()
    query.SetBoxTolerance(0.0); query.SetCellTolerance(0.0)
    query.SetNumberOfCellsPerNode(2); query.GenerateScalarsOff(); query.Update()
    count = int(query.GetNumberOfContacts())
    pairs = np.column_stack([vtk_to_numpy(query.GetContactCells(i)) for i in (0, 1)]) if count else np.empty((0, 2), dtype=np.int64)
    output = query.GetContactsOutput()
    contact_points = vtk_to_numpy(output.GetPoints().GetData()).astype(np.float64) if output.GetNumberOfPoints() else np.empty((0, 3))
    return dict(triangle_pair_count=count, triangle_pairs=pairs,
                contact_points=contact_points, method='vtkCollisionDetectionFilter/all_contacts',
                vtk_version=vtk.vtkVersion.GetVTKVersion())


def closed_mesh_quality(vertices, faces) -> dict:
    import trimesh
    v, f = _mesh(vertices, faces)
    mesh = trimesh.Trimesh(v, f, process=False)
    tri = v[f]
    double_area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    volume = float(mesh.volume)
    valid = bool(mesh.is_watertight and mesh.is_winding_consistent and volume > 1e-12 and np.all(double_area > 1e-14))
    return dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                signed_volume_m3=volume, degenerate_triangle_count=int(np.count_nonzero(double_area <= 1e-14)),
                signed_distance_valid=valid)


def audit_bone_pair(vertices_a, faces_a, vertices_b, faces_b, *, depth_tolerance_m=.0005) -> dict:
    """Conservative gate: crossing triangles are unresolved even at small depth.

    Positive signed samples alone do not overrule a triangle crossing.
    Complete containment is detected even when the boundaries do not cross.
    Max depth is explicitly a sampled lower bound, not a penetration theorem.
    """
    import igl
    a, fa = _mesh(vertices_a, faces_a); b, fb = _mesh(vertices_b, faces_b)
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
        raise ValueError('invalid depth tolerance')
    quality = [closed_mesh_quality(a, fa), closed_mesh_quality(b, fb)]
    contacts = triangle_contacts(a, fa, b, fb)
    result = dict(mesh_quality=quality, triangle_pair_count=contacts['triangle_pair_count'],
                  method=contacts['method'], vtk_version=contacts['vtk_version'],
                  depth_tolerance_m=float(depth_tolerance_m),
                  max_depth_is_sampled_lower_bound=True,
                  triangle_pairs=contacts['triangle_pairs'].tolist())
    if not all(q['signed_distance_valid'] for q in quality):
        result.update(passed=False, reason='invalid_closed_oriented_mesh', signed_samples=None)
        return result
    directions = []
    for v, f, target, target_faces in ((a, fa, b, fb), (b, fb, a, fa)):
        # Include face interiors as well as authored vertices. The triangle
        # intersection test above remains necessary for sub-face crossings.
        samples = np.concatenate((v, v[f].mean(axis=1)))
        signed = igl.signed_distance(samples, target, target_faces)[0]
        directions.append(dict(sample_count=len(samples),
                               max_depth_m=float(np.maximum(-signed, 0).max()),
                               above_tolerance_count=int(np.count_nonzero(signed < -depth_tolerance_m)),
                               min_absolute_distance_m=float(np.abs(signed).min())))
    depth_failed = any(d['above_tolerance_count'] > 0 for d in directions)
    unresolved_crossing = contacts['triangle_pair_count'] > 0
    result.update(signed_samples=directions, passed=not depth_failed and not unresolved_crossing,
                  reason=('penetration_exceeds_tolerance' if depth_failed else
                          'triangle_crossing_requires_contact_review' if unresolved_crossing else
                          'no_detected_crossing_or_excess_sampled_depth'))
    return result
