"""Compare section-based and joint bone/skin retargeting on the Open3DModel right leg."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from time import perf_counter

import igl
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from inputs import CLOUD, write_json
from align_cloud_sources import (
    INPUT,
    SMPL_DIR,
    bp_reference,
    open3d_to_bodyparts,
    read_mesh_pack,
    section_outline,
    target_model,
)
from containment_map import intersections, skin_center, subdivide
from geometry_audit import Slicer
from leg_proximal_vessel_geometry import VOLUME
from render_right_leg_bone_overlay import add_bones, signed_depth
from verify_containment import boundary_contacts
from render_cloud_views import BG, INK, MUTED, add_skin, orthographic
import matplotlib.pyplot as plt


def ray_hit(lines_xz, origin, direction):
    ab = lines_xz - origin
    normal = np.array([-direction[1], direction[0]])
    n = ab @ normal
    dv = n[:, 1] - n[:, 0]
    use = (n[:, 0] * n[:, 1] <= 0) & (np.abs(dv) > 1e-8)
    u = -n[use, 0] / dv[use]
    hit = (ab[use, 0] + u[:, None] * (ab[use, 1] - ab[use, 0])) @ direction
    hit = np.sort(hit[hit > 1e-6])
    if not len(hit):
        return None
    # BodyParts3D is a thin skin shell. Keep its outer wall and the first limb.
    return origin + np.max(hit[hit < hit[0] + 4.0]) * direction


def correspond_chart(skin):
    subject = trimesh.load_mesh(SMPL_DIR / "smpl_canonical_tpose.obj", process=False)
    subject.vertices *= 1000
    assert np.array_equal(subject.faces, skin.faces)
    with np.load(VOLUME) as a:
        points = a["skin_vertices"].astype(np.float64) * 1000.0
        h = a["skin_h"]
    sampled = np.arange(len(points)).reshape(len(np.unique(h)), -1)[::2, ::6].ravel()
    pts = points[sampled]
    sq, face, contact = igl.point_mesh_squared_distance(pts, subject.vertices, subject.faces)
    tri = subject.faces[face]
    weights = trimesh.triangles.points_to_barycentric(subject.vertices[tri], contact)
    target = np.einsum("ij,ijk->ik", weights, skin.vertices[tri])
    print("charted skin", len(pts), "projection error mm", np.sqrt(sq).max(), flush=True)
    return target


def make_base(bp_anchors, target_anchors):
    s = bp_anchors[:, 0].copy()
    t = target_anchors[:, 0].copy()
    # Near the sole the reference centers can move forward relative to the ankle.
    # Keep that correction for the field controls rather than introducing a large
    # vertical dependence in the bulk translation sagittally.
    s[0, [0, 2]] = s[1, [0, 2]]
    t[0, [0, 2]] = t[1, [0, 2]]
    y = PchipInterpolator(s[:, 1], t[:, 1], extrapolate=False)
    inverse = PchipInterpolator(
        y(np.linspace(s[0, 1], s[-1, 1], 1201)), np.linspace(s[0, 1], s[-1, 1], 1201)
    )
    c0 = PchipInterpolator(s[:, 1], s[:, [0, 2]], axis=0)
    c1 = PchipInterpolator(t[:, 1], t[:, [0, 2]], axis=0)

    def move(p):
        levels = p[:, 1]
        clipped = np.clip(levels, s[0, 1], s[-1, 1])
        # Linear exterior continuation from the atlas range.
        q_y = y(clipped) + (levels - clipped) * y.derivative()(clipped)
        xz = p[:, [0, 2]] + c1(y(clipped)) - c0(clipped)
        return np.column_stack((xz[:, 0], q_y, xz[:, 1]))

    return move, inverse, c0, c1


def skin_controls(skin, bp_skin, inv, c_src, c_tgt):
    chart = correspond_chart(skin)
    src_outline = section_outline(bp_skin)
    tgt_outline = section_outline(skin)
    pts = []
    for q in chart:
        level = q[1]
        y_src = inv(level)
        start = c_src(y_src)
        direct = q[[0, 2]] - c_tgt(level)
        direct /= np.linalg.norm(direct)
        aux = ray_hit(src_outline(y_src)[:, [0, 2]].reshape(-1, 2, 2), start, direct)
        if (
            aux is not None
            and max(aux[0], q[0]) < -2.0
            and np.linalg.norm(aux - start) < 1.65 * np.linalg.norm(q[[0, 2]] - c_tgt(level)) + 10
        ):
            pts.append((np.array([aux[0], y_src, aux[1]]), q, True))
    angles = np.linspace(-np.pi, np.pi, 26, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    for level in np.r_[
        np.arange(-1339.0, chart[:, 1].min() - 4, 13.0),
        np.arange(chart[:, 1].max() + 6, -355, 13.0),
    ]:
        y_src = inv(level)
        source_lines = src_outline(y_src)[:, [0, 2]].reshape(-1, 2, 2)
        target_lines = tgt_outline(level)[:, [0, 2]].reshape(-1, 2, 2)
        start_src = c_src(y_src)
        start_tgt = c_tgt(level)
        # The low foot sections extend forward from the ankle axis.
        if level < -1300.0:
            u = min(1.0, (-1300.0 - level) / 35.0)
            start_src = start_src + [0.0, u * 62.0]
            start_tgt = start_tgt + [0.0, u * 64.0]
        for direction in directions:
            src = ray_hit(source_lines, start_src, direction)
            dst = ray_hit(target_lines, start_tgt, direction)
            if src is None or dst is None or max(src[0], dst[0]) > -4.0:
                continue
            sl, tl = np.linalg.norm(src - start_src), np.linalg.norm(dst - start_tgt)
            if not (8.0 < sl < 130.0 and 8.0 < tl < 145.0 and 0.45 < tl / sl < 2.0):
                continue
            pts.append(
                (
                    np.array([src[0], y_src, src[1]]),
                    np.array([dst[0], level, dst[1]]),
                    False,
                )
            )
    a, b, c = zip(*pts)
    print("retained skin", int(np.sum(c)), len(c) - int(np.sum(c)), flush=True)
    return np.array(a), np.array(b), np.array(c)


def voxel_choose(p, spacing):
    # Keep a surface sample nearest to the voxel center.
    indices = np.floor(p / spacing)
    score = np.linalg.norm(p - (indices + 0.5) * spacing, axis=1)
    order = np.argsort(score)
    _, idx = np.unique(indices[order], axis=0, return_index=True)
    return np.sort(order[idx])


def compute_inputs():
    shape, shore = target_model()
    bp, landmarks = bp_reference()
    matrix, offset, _ = open3d_to_bodyparts(bp)
    path = INPUT / "open3d_right_leg"
    raw_bones = read_mesh_pack(path / "bones.npz")
    target_bones = read_mesh_pack(CLOUD / "right_leg_joint/right_leg_bones_aligned.npz")
    src_bones = {k: (v * 1000 @ matrix + offset, f) for k, (v, f) in raw_bones.items() if k in target_bones}
    bone_x = np.concatenate([v for v, f in src_bones.values()])
    bone_y = np.concatenate([target_bones[k][0] for k in src_bones])
    raw = read_mesh_pack(path / "vessels.npz")
    src_vessels = {k: (v * 1000 @ matrix + offset, f) for k, (v, f) in raw.items()}
    basemap, inv, c0, c1 = make_base(landmarks, shore)
    sx, sy, charted = skin_controls(shape, bp["skin"], inv, c0, c1)
    return (
        shape,
        src_bones,
        target_bones,
        src_vessels,
        basemap,
        bone_x,
        bone_y,
        sx,
        sy,
        charted,
    )


def kernel(distance, radius):
    r = np.clip(distance / radius, 0.0, 1.0)
    return (1.0 - r) ** 4 * (4.0 * r + 1.0)


class SharedField:
    def __init__(self, source, target, radius=68.0, alpha=0.004):
        self.source = source
        self.tree = cKDTree(source)
        self.radius = radius
        aa = self.tree.sparse_distance_matrix(self.tree, radius, output_type="coo_matrix")
        aa.data = kernel(aa.data, radius)
        self.weights = spsolve(aa.tocsc() + diags(np.full(len(source), alpha)), target - source)

    def __call__(self, q):
        move = []
        for sub in np.array_split(q, max(1, (len(q) + 14999) // 15000)):
            mat = self.tree.sparse_distance_matrix(cKDTree(sub), self.radius, output_type="coo_matrix")
            mat.data = kernel(mat.data, self.radius)
            move.append(sub + mat.T.tocsr() @ self.weights)
        return np.concatenate(move)


def overlaps(vertices, bones):
    inside = np.zeros(len(vertices), bool)
    deepest = 0.0
    hits = {}
    for key, (v, f) in bones.items():
        choose = np.all((vertices >= v.min(0)) & (vertices <= v.max(0)), axis=1)
        if not choose.any():
            continue
        d = signed_depth(v, f, vertices[choose])
        inside[choose] |= d > 0.5
        deepest = max(deepest, d.max())
        if np.any(d > 0.5):
            hits[key] = (int((d > 0.5).sum()), round(float(d.max()), 3))
    return inside, deepest, hits


class FlowStep(SharedField):
    def derivative(self, points):
        out = []
        for p in np.array_split(points, max(1, (len(points) + 3499) // 3500)):
            e = self.tree.sparse_distance_matrix(cKDTree(p), self.radius, output_type="coo_matrix")
            dr = -20.0 * (1.0 - e.data / self.radius) ** 3 / self.radius**2
            displacement = p[e.col] - self.source[e.row]
            w = self.weights[e.row] * dr[:, None]
            stack = np.column_stack(
                [
                    np.bincount(e.col, weights=w[:, i] * displacement[:, j], minlength=len(p))
                    for i, j in product(range(3), repeat=2)
                ]
            ).reshape(-1, 3, 3)
            out.append(stack)
        return np.concatenate(out)


def fields(anchors, targets, checks, *, radius=55.0, count=12, alpha=0.003):
    x = anchors.copy()
    q = checks.copy()
    det = np.ones(len(q))
    p = 0.0
    steps = []
    while p < 1.0 - 1e-8:
        advance = min(1 - p, 1 / count)
        for _ in range(7):
            next_pos = anchors + (p + advance) * (targets - anchors)
            local = FlowStep(x, next_pos, radius=radius, alpha=alpha)
            j = local.derivative(q)
            j0 = local.derivative(x)
            maxnorm = max(
                np.linalg.norm(j, axis=(1, 2)).max(),
                np.linalg.norm(j0, axis=(1, 2)).max(),
            )
            if maxnorm < 0.92:
                break
            advance *= 0.8 / maxnorm
        if maxnorm >= 0.98:
            raise RuntimeError(("singular progress", maxnorm, p))
        p += advance
        x = local(x)
        q = local(q)
        det *= np.linalg.det(j + np.eye(3))
        steps.append(local)
        if len(steps) >= 50:
            raise RuntimeError("excess subdivisions")
    return steps, float(det.min())


class SkinBarrier:
    duration = 5.0
    step = 0.03125

    def __init__(self, skin, bones):
        lo, hi = [-252.0, -1390.0, -175.0], [88.0, -284.0, 178.0]
        self.pitch = 2.7
        self.origin = np.array(lo)
        axes = tuple(np.arange(a, b + self.pitch, self.pitch) for a, b in zip(lo, hi, strict=True))
        x, y, z = axes
        grid0 = np.empty(tuple(map(len, axes)), np.float32)
        gridb = np.empty_like(grid0)
        vv = np.concatenate([v for v, _ in bones.values()])
        vo = np.r_[0, np.cumsum([len(v) for v, _ in bones.values()])]
        ff = np.concatenate([f + a for (_, f), a in zip(bones.values(), vo[:-1], strict=True)])
        for i in range(0, len(y), 15):
            slab = y[i : i + 15]
            q = np.stack(np.meshgrid(x, slab, z, indexing="ij"), axis=-1).reshape(-1, 3)
            grid0[:, i : i + len(slab), :] = signed_depth(skin.vertices, skin.faces, q).reshape(
                len(x), len(slab), len(z)
            )
            gridb[:, i : i + len(slab), :] = -signed_depth(vv, ff, q).reshape(len(x), len(slab), len(z))
        self.skin = spline_filter(grid0, order=3, output=np.float32)
        self.bone = spline_filter(gridb, order=3, output=np.float32)
        self.gradient = np.stack(
            [
                spline_filter(g, order=3, output=np.float32)
                for g in np.gradient(grid0, self.pitch, edge_order=2)
            ]
        )
        _, joints = target_model()
        right = joints[:, 0]
        self.inward_axis = skin_center(skin, right, np.arange(-725.0, -278.0, 3.0), -1)
        self.radial_heights = np.arange(-685.0, -280.0, 2.0)
        self.angle_count = 144
        theta = np.linspace(-np.pi, np.pi, self.angle_count, endpoint=False)
        section_centers = self.inward_axis(self.radial_heights)
        origins = np.repeat(
            np.column_stack((section_centers[:, 0], self.radial_heights, section_centers[:, 1])),
            self.angle_count,
            axis=0,
        )
        directions = np.tile(
            np.column_stack((np.cos(theta), np.zeros_like(theta), np.sin(theta))),
            (len(self.radial_heights), 1),
        )
        boundary = intersections(skin, origins, directions)
        medial = directions[:, 0] > 1.0e-6
        boundary[medial] = np.minimum(boundary[medial], -origins[medial, 0] / directions[medial, 0])
        self.radial_limit = spline_filter(
            np.pad(
                boundary.reshape(len(self.radial_heights), self.angle_count), ((0, 0), (4, 4)), mode="wrap"
            ),
            order=3,
        )

    def sample(self, a, idx):
        return map_coordinates(a, idx, order=3, mode="nearest", prefilter=False, output=np.float64)

    def boundary_state(self, points, x):
        d = self.sample(self.skin, x)
        # In the groin the two faces of the skin crease can pull a tube toward
        # different heights. Use the right-side atlas section's interior as the
        # target for the proximal correction, keeping its longitudinal station.
        s = np.clip((points[:, 1] + 680.0) / 85.0, 0.0, 1.0)
        blend = s**3 * (10.0 + s * (-15.0 + 6.0 * s))
        guide = self.inward_axis(np.clip(points[:, 1], -720.0, -285.0))
        offset = points[:, [0, 2]] - guide
        angle = (np.arctan2(offset[:, 1], offset[:, 0]) + np.pi) % (2.0 * np.pi)
        radial_idx = np.stack(
            (
                np.clip((points[:, 1] - self.radial_heights[0]) / 2.0, 0.0, len(self.radial_heights) - 1.0),
                4.0 + angle * self.angle_count / (2.0 * np.pi),
            )
        )
        radius = self.sample(self.radial_limit, radial_idx)
        radial_depth = radius - np.linalg.norm(offset, axis=1)
        margin = 1.85 * (1.0 - blend) + 3.3 * blend
        deficit = margin - ((1.0 - blend) * d + blend * radial_depth)
        return deficit, offset, blend[:, None]

    def delta(self, points):
        x = (points - self.origin).T / self.pitch
        deficit, offset, blend = self.boundary_state(points, x)
        r = np.maximum(deficit, 0.0)
        bone_gap = self.sample(self.bone, x)
        u = np.clip((bone_gap - 1.1) / 3.0, 0.0, 1.0)
        scale = (u**3 * (10.0 + u * (-15.0 + 6.0 * u))) * r * r / (r + 0.4)
        gradient = np.column_stack([self.sample(field, x) for field in self.gradient])
        gradient /= np.sqrt(np.square(gradient).sum(axis=1, keepdims=True) + 0.01)
        inward = np.zeros_like(points)
        inward[:, [0, 2]] = -offset
        inward /= np.sqrt(np.square(inward).sum(axis=1, keepdims=True) + 0.01)
        return scale[:, None] * ((1.0 - blend) * gradient + blend * inward)

    def transport(self, points, *, measure_jacobian=False):
        # The velocity vanishes at the skin margin and in a halo around the
        # target bones. Exported rigid bone meshes therefore remain fixed.
        x = (points - self.origin).T / self.pitch
        active = self.boundary_state(points, x)[0] > 0.0
        q = points[active].copy()
        dt = self.step
        determinant = np.ones(len(q))
        if len(q):
            for _ in range(round(self.duration / dt)):
                mid = q + 0.5 * dt * self.delta(q)
                if measure_jacobian:
                    j0 = self.velocity_jacobian(q)
                    j1 = self.velocity_jacobian(mid)
                    determinant *= np.linalg.det(np.eye(3) + dt * j1 @ (np.eye(3) + 0.5 * dt * j0))
                q += dt * self.delta(mid)
        result = points.copy()
        result[active] = q
        determinants = np.ones(len(points))
        determinants[active] = determinant
        return result, determinants

    def velocity_jacobian(self, points):
        offset = np.eye(3) * 0.02
        return np.stack(
            [(self.delta(points + d) - self.delta(points - d)) / 0.04 for d in offset],
            axis=-1,
        )


JOIN_PAIRS = (
    ("Femoral_vein.r", "Great_saphenous_vein.r"),
    ("Popliteal_vein.r", "Small_saphenous_vein.r"),
    ("Femoral_artery.r", "Popliteal_artery.r"),
    ("Popliteal_artery.r", "Anterior_tibial_artery.r"),
    ("Popliteal_artery.r", "Posterior_tibial_artery.r"),
    ("Popliteal_vein.r", "Posterior_tibial_vein.r"),
    ("Femoral_vein.r", "Popliteal_vein.r"),
)
FOCUS = {
    "Femoral_artery.r": ("股动脉", "#bc5147"),
    "Femoral_vein.r": ("股静脉", "#3768a5"),
    "Great_saphenous_vein.r": ("大隐静脉", "#148370"),
    "Popliteal_artery.r": ("腘动脉", "#c07424"),
    "Popliteal_vein.r": ("腘静脉", "#255d9e"),
    "Anterior_tibial_artery.r": ("胫前动脉", "#8f3b82"),
    "Posterior_tibial_artery.r": ("胫后动脉", "#b7734e"),
    "Small_saphenous_vein.r": ("小隐静脉", "#328498"),
}


def transport_field(points, steps, barrier, *, measure_jacobian=False):
    result = points
    determinant = np.ones(len(points))
    for step in steps:
        if measure_jacobian:
            determinant *= np.linalg.det(np.eye(3) + step.derivative(result))
        result = step(result)
    result, skin_determinant = barrier.transport(result, measure_jacobian=measure_jacobian)
    return result, determinant * skin_determinant


def skin_audit(skin, old, new, native, map_native):
    """Refine long facets in the transported mesh wherever skin is crossed."""
    skin_faces = skin.vertices[skin.faces]
    centers = skin_faces.mean(axis=1)
    tree = cKDTree(centers)
    radius = np.linalg.norm(skin_faces - centers[:, None], axis=2).max()
    rows = {}
    for name, (vertices, faces) in new.items():
        src = native[name][0]
        old_vertices, old_faces = old[name]
        old_depth = signed_depth(skin.vertices, skin.faces, old_vertices)
        old_contacts = boundary_contacts(old_vertices, old_faces, old_depth, skin_faces, tree, radius)
        for refinement in range(6):
            depth = signed_depth(skin.vertices, skin.faces, vertices)
            crossed = boundary_contacts(vertices, faces, depth, skin_faces, tree, radius)
            if not crossed or refinement == 5:
                break
            selected = np.isin(np.arange(len(faces)), list(crossed))
            src, faces = subdivide(src, faces, selected)
            extra = map_native(src[len(vertices) :])
            vertices = np.concatenate((vertices, extra))
        new[name] = (vertices, faces)
        rows[name] = {
            "native_vertices": len(native[name][0]),
            "added_mesh_refinement_vertices": len(vertices) - len(native[name][0]),
            "legacy_mesh_skin_crossing_triangles": len(old_contacts),
            "joint_field_mesh_skin_crossing_triangles": len(crossed),
            "legacy_mesh_outside_vertices": int((old_depth < 0.0).sum()),
            "joint_field_mesh_outside_vertices": int((depth < 0.0).sum()),
            "legacy_min_skin_depth_mm": float(old_depth.min()),
            "joint_field_min_skin_depth_mm": float(depth.min()),
        }
    return rows


def surface_gap(first, second):
    v, f = first
    w, g = second
    return float(
        min(
            np.sqrt(igl.point_mesh_squared_distance(v, w, g)[0].min()),
            np.sqrt(igl.point_mesh_squared_distance(w, v, f)[0].min()),
        )
    )


def section_area_ratios(native, legacy_vertices, field_vertices):
    slicer = Slicer(*native)
    y = native[0][:, 1]
    ratios = [[], []]
    for level in np.arange(np.ceil(y.min() / 5.0) * 5.0, y.max(), 5.0) + 0.37:
        rings = slicer.at(level)
        if not rings:
            continue
        ring = max(rings, key=lambda item: item.area)
        for vertices, values in zip((legacy_vertices, field_vertices), ratios, strict=True):
            polygon = ring.transport_polygon(vertices)
            centered = polygon - polygon.mean(axis=0)
            area_vector = 0.5 * np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
            values.append(float(np.linalg.norm(area_vector) / ring.area))
    return len(ratios[0]), [np.quantile(values, (0.1, 0.5, 0.9)).tolist() for values in ratios]


def bone_metrics(native, old, new, src_bones, target_bones, base_vertices):
    old_vertices = np.concatenate([old[key][0][: len(v)] for key, (v, _) in native.items()])
    new_vertices = np.concatenate([new[key][0][: len(v)] for key, (v, _) in native.items()])
    native_vertices = np.concatenate([v for v, _ in native.values()])
    hit0, max0, _ = overlaps(native_vertices, src_bones)
    hit1, max1, _ = overlaps(old_vertices, target_bones)
    hit2, max2, _ = overlaps(new_vertices, target_bones)
    h = base_vertices[:, 1]
    in_atlas = (h >= -1210.0) & (h <= -560.0)
    report = {
        "penetration_threshold_mm": 0.5,
        "matched_original_vessel_vertices": len(h),
        "native_bone_vertices": int(hit0.sum()),
        "legacy_bone_vertices": int(hit1.sum()),
        "joint_field_bone_vertices": int(hit2.sum()),
        "legacy_new_bone_vertices": int((hit1 & ~hit0).sum()),
        "joint_field_new_bone_vertices": int((hit2 & ~hit0).sum()),
        "legacy_max_penetration_mm": max1,
        "joint_field_max_penetration_mm": max2,
        "native_max_penetration_mm": max0,
        "shaft_y_span_mm": [-1210.0, -560.0],
        "shaft_matched_vertices": int(in_atlas.sum()),
        "shaft_legacy_new_bone_vertices": int((hit1 & ~hit0 & in_atlas).sum()),
        "shaft_joint_field_new_bone_vertices": int((hit2 & ~hit0 & in_atlas).sum()),
    }
    bounds = np.r_[0, np.cumsum([len(v) for v, _ in native.values()])]
    per_mesh = {}
    for i, key in enumerate(native):
        rows = slice(bounds[i], bounds[i + 1])
        per_mesh[key] = {
            method: int(hits[rows].sum())
            for method, hits in zip(("native", "sections", "joint_field"), (hit0, hit1, hit2), strict=True)
        }
    return report, per_mesh, (old_vertices, new_vertices), (hit1 & ~hit0, hit2 & ~hit0)


def save_mesh_pack(path, meshes):
    vertices, faces = zip(*meshes.values())
    np.savez_compressed(
        path,
        names=np.array(list(meshes)),
        vertices=np.concatenate(vertices),
        triangles=np.concatenate(faces),
        vertex_offsets=np.r_[0, np.cumsum([len(v) for v in vertices])],
        triangle_offsets=np.r_[0, np.cumsum([len(f) for f in faces])],
    )


def plot_comparison(out, skin, bones, native, points, new_hits, results):
    # Highlight newly embedded points so that a 2D projection does not stand
    # in for the solid intersection test.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(11.1, 8.2), facecolor=BG)
    fig.text(
        0.052,
        0.947,
        "右腿血管重定向对照",
        fontsize=24,
        color=INK,
        weight="bold",
        va="top",
    )
    fig.text(
        0.055,
        0.875,
        "同一套 Open3DModel 血管 · 同一 SMPL-X 体表与目标骨骼",
        fontsize=11,
        color=MUTED,
    )
    axes = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.965,
        top=0.772,
        bottom=0.10,
        wspace=0.13,
        width_ratios=(1.0, 1.0, 1.02),
    ).subplots()
    yaw = 1.42
    bounds = (-159.0, 148.0, -1368.0, -322.0)
    skin_faces = skin.faces[skin.vertices[skin.faces, 0].mean(axis=1) < 0.0]
    edges = np.r_[0, np.cumsum([len(v) for v, _ in native.values()])]
    entries = list(native)
    labels = ("逐截面映射", "骨面约束的共享三维形变")
    for i, (ax, vertices, flags) in enumerate(zip(axes[:2], points, new_hits, strict=True)):
        add_skin(ax, skin.vertices, skin_faces, yaw, bounds)
        add_bones(ax, bones, yaw)
        for key, (_, color) in FOCUS.items():
            j = entries.index(key)
            xyz = orthographic(vertices[edges[j] : edges[j + 1]], yaw)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                s=1.75,
                c=color,
                alpha=0.66,
                linewidths=0,
                zorder=4,
            )
        xyz = orthographic(vertices[flags], yaw)
        ax.scatter(xyz[:, 0], xyz[:, 1], s=2.3, c="#cc2832", alpha=0.66, linewidths=0, zorder=5)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal")
        ax.set_axis_off()
        ax.set_title(labels[i], fontsize=12, color=INK, pad=13)
    panel = axes[2]
    panel.axis("off")
    panel.text(0.01, 0.98, "新增穿骨点", fontsize=14, color=INK, va="top", weight="medium")
    for label, key, y in zip(
        labels,
        ("legacy_new_bone_vertices", "joint_field_new_bone_vertices"),
        (0.875, 0.733),
        strict=True,
    ):
        panel.text(0.01, y, label, fontsize=10.4, color=MUTED)
        panel.text(0.01, y - 0.062, f"{results[key]:,}", fontsize=26, weight="bold", color=INK)
    handles = [
        Patch(facecolor="#979c9f", label="目标骨骼"),
        Line2D([], [], marker="o", linestyle="", color="#cc2832", label="新增穿骨 > 0.5 mm"),
    ]
    handles.extend(Line2D([], [], color=color, lw=2.0, label=label) for label, color in FOCUS.values())
    panel.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.035, 0.58),
        fontsize=9.8,
        borderpad=0.55,
        labelspacing=0.78,
    )
    fig.text(
        0.055,
        0.043,
        "血管主体用原始网格点表示。红点按三维骨体积判定，包含未着色展示的细支。",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(out / "right_leg_retarget_comparison.png", dpi=177, facecolor=BG)
    plt.close(fig)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()
    skin, source_bones, target_bones, native, base, bx, by, sx, sy, charted = compute_inputs()
    old_pack = read_mesh_pack(CLOUD / "aligned_vessel_surfaces.npz")
    old = {key: old_pack[f"Open3DModel|{key}"] for key in native}
    sk = voxel_choose(sx, 10.0)
    b0 = base(bx)
    x0 = base(sx[sk])
    separated = cKDTree(b0).query(x0)[0] > 3.0
    sy0, x0 = sy[sk][separated], x0[separated]
    bone_ids = voxel_choose(b0, 8.0)
    vertices = np.concatenate([v for v, _ in native.values()])
    base_vertices = base(vertices)
    rng = np.random.default_rng(997)
    sample = base_vertices[rng.choice(len(vertices), 3500, replace=False)]
    checks = np.r_[sample, sample + rng.uniform(-7.0, 7.0, sample.shape), b0[::12]]
    steps, min_training_det = fields(
        np.r_[x0, b0[bone_ids]],
        np.r_[sy0, by[bone_ids]],
        checks,
        radius=51.0,
        alpha=0.012,
    )
    print("Fitted", len(steps), "shared deformation steps", flush=True)
    barrier = SkinBarrier(skin, target_bones)

    def map_native(p):
        return transport_field(base(p), steps, barrier)[0]

    result, _ = transport_field(np.r_[base_vertices, b0, x0], steps, barrier)
    mapped = result[: len(vertices)]
    after_bone_fit = result[len(vertices) : len(vertices) + len(by)]
    fit_error = np.linalg.norm(after_bone_fit - by, axis=1)
    skin_fit = np.linalg.norm(result[-len(sy0) :] - sy0, axis=1)
    fixed_bones, _ = barrier.transport(by)
    bone_shift = float(np.linalg.norm(fixed_bones - by, axis=1).max())
    assert bone_shift < 1e-6
    sizes = np.r_[0, np.cumsum([len(v) for v, _ in native.values()])]
    new = {key: (mapped[sizes[i] : sizes[i + 1]], f) for i, (key, (_, f)) in enumerate(native.items())}
    skin_rows = skin_audit(skin, old, new, native, map_native)
    print("Checked and refined mesh facets", flush=True)
    info, bone_rows, paired_points, hits = bone_metrics(
        native, old, new, source_bones, target_bones, base_vertices
    )
    joints = {
        f"{a}|{b}": {
            method: surface_gap(meshes[a], meshes[b])
            for method, meshes in zip(
                ("native", "sections", "joint_field"), (native, old, new), strict=True
            )
        }
        for a, b in JOIN_PAIRS
    }
    local_details = {}
    for i, key in enumerate(native):
        if key not in FOCUS:
            continue
        q1 = paired_points[0][sizes[i] : sizes[i + 1]]
        q2 = paired_points[1][sizes[i] : sizes[i + 1]]
        section_count, areas = section_area_ratios(native[key], q1, q2)
        local_details[key] = {
            "section_samples": section_count,
            "area_ratio_p10_p50_p90": dict(zip(("sections", "joint_field"), areas, strict=True)),
            "skin_depth_mm_p10_p50_p90": {
                method: np.quantile(
                    signed_depth(skin.vertices, skin.faces, points), (0.1, 0.5, 0.9)
                ).tolist()
                for method, points in zip(("sections", "joint_field"), (q1, q2), strict=True)
            },
        }
    independent = base_vertices[np.random.default_rng(72).choice(len(vertices), 8192, replace=False)]
    test = np.r_[
        independent,
        independent + np.random.default_rng(28).normal(0.0, 4.0, independent.shape),
    ]
    _, determinant = transport_field(test, steps, barrier, measure_jacobian=True)
    import json

    audit = {
        "design": "Matched original Open3DModel vertices. Both arms use the same target bones and SMPL-X skin.",
        "skin_anchor_source": "Target harmonic chart skin faces mapped to neutral SMPL-X plus end-region surface pairs; BodyParts3D outer shell is the source skin proxy.",
        "mesh_groups": len(new),
        "skin_support": [
            int(charted[sk][separated].sum()),
            int((~charted[sk][separated]).sum()),
        ],
        "bone_control_points": len(bone_ids),
        "contour_fit_error_mm_p50_p95_max": np.quantile(skin_fit, (0.5, 0.95, 1.0)).tolist(),
        "bone_fit_error_mm_p50_p95_max": np.quantile(fit_error, (0.5, 0.95, 1.0)).tolist(),
        "target_bone_displacement_by_skin_margin_mm": bone_shift,
        "min_training_jacobian_determinant_before_skin_flow": min_training_det,
        "independent_jacobian_samples": len(test),
        "independent_nonpositive_determinants": int((determinant <= 0.0).sum()),
        "independent_jacobian_min_p01_p50": np.quantile(determinant, (0.0, 0.01, 0.5)).tolist(),
        "mapping_steps": len(steps),
        "bone_summary": info,
        "bone_counts_by_vessel": bone_rows,
        "skin_checks_by_vessel": skin_rows,
        "junction_sampled_vertex_to_surface_gap_mm": joints,
        "local_vessel_details": local_details,
    }
    write_json(out / "comparison.json", audit)
    save_mesh_pack(out / "right_leg_open3dmodel_shared_field.npz", new)
    plot_comparison(out, skin, target_bones, native, paired_points, hits, info)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    for key in (
        "legacy_mesh_skin_crossing_triangles",
        "joint_field_mesh_skin_crossing_triangles",
        "legacy_mesh_outside_vertices",
        "joint_field_mesh_outside_vertices",
    ):
        print(key, sum(row[key] for row in skin_rows.values()), flush=True)
    print(
        "independent nonpositive:",
        audit["independent_nonpositive_determinants"],
        "jacobian",
        audit["independent_jacobian_min_p01_p50"],
        "seconds",
        round(perf_counter() - tick, 2),
        flush=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CLOUD / "right_leg_joint" / "shared_field_test")
    run(parser.parse_args().out)
