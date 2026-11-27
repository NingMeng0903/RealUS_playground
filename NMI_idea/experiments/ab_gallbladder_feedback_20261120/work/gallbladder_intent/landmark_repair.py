"""Versioned fix for a source-body landmark that was outside the lumen mesh."""
from pathlib import Path
import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt
import trimesh

VERSION = "gb-midpoint-interior-voxel-v2"
PITCH_MM = 1.25
MIN_WALL_CLEARANCE_MM = 1.5
_INSTALLED = False
_AUDIT = None
_SOURCE_HASH = None


def winding_number(vertices, faces, point):
    """Independent solid-angle inside test for a closed triangle surface."""
    tri = np.asarray(vertices, float)[np.asarray(faces, int)] - np.asarray(point, float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a,b,c))
    numerator = np.einsum('ij,ij->i', a, np.cross(b,c))
    denominator = (la*lb*lc + np.einsum('ij,ij->i',a,b)*lc
                   + np.einsum('ij,ij->i',b,c)*la
                   + np.einsum('ij,ij->i',c,a)*lb)
    return float(np.sum(2.*np.arctan2(numerator, denominator))/(4.*np.pi))


def nearest_surface_distance(vertices, faces, point):
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices,float), faces=np.asarray(faces,int), process=False)
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, np.asarray(point,float)[None,:])
    return float(distance[0])


def install(geometry_module):
    """Replace the global source body point before any Reference/Body is built."""
    global _INSTALLED, _AUDIT, _SOURCE_HASH
    if _INSTALLED:
        return _AUDIT
    original = geometry_module._landmark_regions
    atlas = Path(geometry_module.ATLAS_PATH)
    with np.load(atlas, allow_pickle=False) as data:
        source_vertices = np.asarray(data['gallbladder_vertices_mm'], float)
        faces = np.asarray(data['gallbladder_faces'], np.int32)
    mesh = trimesh.Trimesh(vertices=source_vertices, faces=faces, process=False)
    _SOURCE_HASH = hashlib.sha256(source_vertices.tobytes()).hexdigest()
    if not mesh.is_watertight:
        raise RuntimeError('source gallbladder mesh is not watertight; cannot repair interior label')
    voxels = mesh.voxelized(pitch=PITCH_MM).fill()
    occupancy = np.asarray(voxels.matrix, bool)
    clearance = distance_transform_edt(occupancy, sampling=PITCH_MM)
    ids = np.argwhere(occupancy)
    centers = trimesh.transform_points(ids.astype(float), voxels.transform)
    y = source_vertices[:, 1]
    lo, hi = np.quantile(y, [.40, .60])
    middle_surface = source_vertices[(y >= lo) & (y <= hi)].mean(axis=0)
    keep = (centers[:, 1] >= lo) & (centers[:, 1] <= hi)
    candidates = centers[keep]
    edt_candidates = clearance[tuple(ids[keep].T)]
    if not len(candidates):
        raise RuntimeError('no interior voxel remains in the source mid-body band')
    # The old surface mean is the deterministic reference. The voxel EDT only
    # orders candidates; acceptance uses an independent triangle-surface query.
    order = sorted(range(len(candidates)), key=lambda i:
                   (float(np.linalg.norm(candidates[i]-middle_surface)), -float(edt_candidates[i])))
    chosen = None
    for i in order:
        candidate = candidates[i]
        winding_i = winding_number(source_vertices, faces, candidate)
        if abs(winding_i) < .99:
            continue
        surface_i = nearest_surface_distance(source_vertices, faces, candidate)
        if surface_i >= MIN_WALL_CLEARANCE_MM:
            chosen = i
            winding = winding_i
            surface_distance = surface_i
            break
    if chosen is None:
        raise RuntimeError('no midband interior voxel meets the independent 1.5 mm wall-clearance rule')
    body_point = candidates[chosen]
    voxel_clearance = float(edt_candidates[chosen])
    original_winding = winding_number(source_vertices, faces, middle_surface)
    original_surface_distance = nearest_surface_distance(source_vertices, faces, middle_surface)
    if abs(winding) < .99:
        raise RuntimeError(f'repaired body point failed independent winding test: {winding}')
    base_landmarks, axis = original(source_vertices)
    fixed_landmarks = dict(base_landmarks)
    fixed_landmarks['body'] = body_point.copy()
    def versioned_landmarks(vertices):
        vertices = np.asarray(vertices, float)
        digest = hashlib.sha256(vertices.tobytes()).hexdigest()
        if digest != _SOURCE_HASH:
            return original(vertices)
        return {**fixed_landmarks, 'body': body_point.copy()}, axis.copy()
    geometry_module._landmark_regions = versioned_landmarks
    _AUDIT = {"version": VERSION, "pitch_mm": PITCH_MM,
              "min_wall_clearance_mm": MIN_WALL_CLEARANCE_MM,
              "body_point_source_mm": body_point.tolist(),
              "original_midband_surface_mean_mm": middle_surface.tolist(),
              "original_midband_surface_mean_winding": original_winding,
              "original_midband_surface_mean_distance_mm": original_surface_distance,
              "source_midband_quantile_y_mm": [float(lo), float(hi)],
              "candidate_voxel_edt_mm": voxel_clearance,
              "source_solid_angle_winding": winding,
              "source_nearest_surface_distance_mm": surface_distance,
              "source_winding_inside_pass": True,
              "source_vertex_sha256": _SOURCE_HASH,
              "mesh_watertight": bool(mesh.is_watertight)}
    _INSTALLED = True
    return _AUDIT


def audit():
    return _AUDIT


def validate_transported(anatomy, faces):
    winding = winding_number(anatomy.gallbladder, faces, anatomy.landmarks['body'])
    distance = nearest_surface_distance(anatomy.gallbladder, faces, anatomy.landmarks['body'])
    return {"winding": winding, "inside": bool(abs(winding) >= .99),
            "nearest_surface_distance_mm": distance}
