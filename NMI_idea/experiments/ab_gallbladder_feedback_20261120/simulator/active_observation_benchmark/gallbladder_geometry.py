"""Native liver/gallbladder geometry for the active observation benchmark.

The cardiac benchmark in :mod:`outputs.thorax_continuous.model` already owns
the torso material map, SMPL-X shape transport, and rib ray caster.  This
module deliberately reuses those objects.  The abdominal structures are kept
as their registered triangle meshes; a measurement is an intersection of the
mesh with the actual probe plane, followed by the probe fan and rib shadow
clips.  In particular, no hidden target is consulted by ``view`` or
``measure``.

Coordinates are the atlas millimetres (``+x`` left, ``+y`` superior,
``+z`` anterior).  The public chart is ``(y_mm, theta_deg, roll_deg,
rock_deg)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

# heart_outflow is the stable public import point used by the existing
# benchmark.  Importing it also establishes the continuous torso model path
# and therefore keeps SMPL-X/SOURCE paths in one place.
HERE = Path(__file__).resolve()
CONTINUOUS = HERE.parents[1] / "outputs" / "thorax_continuous"
if str(CONTINUOUS) not in sys.path:
    sys.path.insert(0, str(CONTINUOUS))
try:  # Package import when called by a controller.
    from .heart_outflow import (  # type: ignore  # noqa: E402
        BEAM_ANGLE,
        DEPTH_MM,
        Bones,
        Body as _MaterialBody,
        Reference as _MaterialReference,
        READOUT,
    )
except ImportError:  # Direct ``python gallbladder_geometry.py`` / legacy path.
    from heart_outflow import (  # noqa: E402
        BEAM_ANGLE,
        DEPTH_MM,
        Bones,
        Body as _MaterialBody,
        Reference as _MaterialReference,
        READOUT,
    )
from model import View as _ModelView  # noqa: E402


PREVIEW = HERE.parents[1] / "outputs" / "thorax_preview"
from gallbladder_runtime import ATLAS_ARCHIVE

ATLAS_PATH = ATLAS_ARCHIVE

try:  # trimesh/shapely are already dependencies of the atlas generator.
    import trimesh
    from shapely.geometry import LineString, MultiLineString, Point, Polygon
    from shapely.ops import polygonize, unary_union
except Exception:  # pragma: no cover - a deterministic fallback is provided
    trimesh = None
    LineString = MultiLineString = Point = Polygon = None
    polygonize = unary_union = None


def _unit(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return a finite unit vector, with a deterministic fallback."""

    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm > 1.e-10:
        return value / norm
    if fallback is None:
        fallback = np.array([0., 1., 0.])
    return np.asarray(fallback, dtype=float) / np.linalg.norm(fallback)


def _wrap_pi(angle: float) -> float:
    """Wrap an orientation whose period is pi to [-pi/2, pi/2)."""

    return float((angle + .5 * np.pi) % np.pi - .5 * np.pi)


def _landmark_regions(vertices: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Find neck/fundus/body from the source mesh, not a fitted primitive.

    The registered GB is superior-inferior elongated.  Small quantile bands
    are used instead of one extreme vertex so that the landmarks remain
    stable under a material warp and mesh resampling.  ``axis`` points from
    fundus to neck, matching the anatomical direction used by ``aim``.
    """

    vertices = np.asarray(vertices, dtype=float)
    y = vertices[:, 1]
    # Distinct source regions: the superior tip is the neck at the liver
    # fossa, while the inferior tip is the fundus.  Five percent is large
    # enough to avoid selecting a single triangulation corner.
    low = y <= np.quantile(y, .05)
    high = y >= np.quantile(y, .95)
    fundus = vertices[low].mean(axis=0)
    neck = vertices[high].mean(axis=0)
    axis = _unit(neck - fundus, np.array([0., 1., 0.]))
    # A body landmark is kept distinct from both tips and is useful for the
    # long-axis visual-support check.
    middle = (y >= np.quantile(y, .40)) & (y <= np.quantile(y, .60))
    body = vertices[middle].mean(axis=0)
    return {"neck": neck, "body": body, "fundus": fundus}, axis


@dataclass(frozen=True)
class Anatomy:
    """One transported abdominal anatomy instance.

    ``liver_vertices`` and ``gallbladder_vertices`` are the complete native
    triangle-mesh vertices in millimetres.  Triangle topology is constant on
    the owning :class:`Reference` and exposed there as ``*_faces``.  The
    landmark dictionary contains source-mesh ``neck``, ``body`` and
    ``fundus`` points after the requested pose.
    """

    liver_vertices: np.ndarray
    gallbladder_vertices: np.ndarray
    landmarks: Mapping[str, np.ndarray]
    axis: np.ndarray
    center: np.ndarray

    # Friendly aliases make the dataclass convenient for controllers while
    # keeping the explicit ``*_vertices`` names unambiguous for tests.
    @property
    def liver(self) -> np.ndarray:
        return self.liver_vertices

    @property
    def gallbladder(self) -> np.ndarray:
        return self.gallbladder_vertices

    @property
    def neck(self) -> np.ndarray:
        return np.asarray(self.landmarks["neck"])

    @property
    def body(self) -> np.ndarray:
        return np.asarray(self.landmarks["body"])

    @property
    def fundus(self) -> np.ndarray:
        return np.asarray(self.landmarks["fundus"])


@dataclass(frozen=True)
class Slice:
    """Ray-clipped image features for one probe pose.

    Feature columns are ``[area_mm2, centroid_x_mm, centroid_depth_mm,
    long_span_mm, signed_angle_rad]``.  ``visible`` is the visible GB area
    fraction relative to its raw section, or ``None`` when no GB tissue lies
    in the fan.  ``visible_liver`` is provided for diagnostics without
    changing the compact public return tuple expected by the controller.
    """

    gall: np.ndarray
    liver: np.ndarray
    coverage: float
    visible: float | None
    qualified: bool
    visible_liver: float | None = None
    raw_gall: np.ndarray | None = None
    raw_liver: np.ndarray | None = None

    # A couple of descriptive aliases avoid coupling callers to the short
    # field names while retaining the requested ``gall``/``liver`` API.
    @property
    def gallbladder(self) -> np.ndarray:
        return self.gall


class Reference(_MaterialReference):
    """Loaded torso material plus native abdominal meshes."""

    def __init__(self):
        super().__init__()
        with np.load(ATLAS_PATH, allow_pickle=False) as data:
            # Copies detach the reusable reference from the npz mmap and make
            # accidental mutation by a controller harmless.
            self.liver_vertices = np.asarray(data["liver_vertices_mm"], dtype=float).copy()
            self.liver_faces = np.asarray(data["liver_faces"], dtype=np.int32).copy()
            self.gallbladder_vertices = np.asarray(data["gallbladder_vertices_mm"], dtype=float).copy()
            self.gallbladder_faces = np.asarray(data["gallbladder_faces"], dtype=np.int32).copy()
            self.organ_shift_mm = np.asarray(data.get("liver_gallbladder_shift_mm", np.zeros(3)), dtype=float).copy()
            self.gallbladder_center_mm = np.asarray(data["gallbladder_center_mm"], dtype=float).copy()
            self.bone_names = np.asarray(data.get("bone_names", []), dtype=str).copy()
        self.liver_mesh = (self.liver_vertices, self.liver_faces)
        self.gallbladder_mesh = (self.gallbladder_vertices, self.gallbladder_faces)
        self.gallbladder_landmarks, self.gallbladder_axis = _landmark_regions(self.gallbladder_vertices)


class Body(_MaterialBody):
    """SMPL-X transported torso with native liver/gallbladder geometry."""

    def __init__(self, reference: Reference, beta):
        # MaterialBody accepts the two-dimensional beta deltas used by the
        # benchmark.  Pad/truncate only for friendly standalone use; the
        # inherited warp remains the source of all torso/rib deformation.
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if not np.isfinite(beta).all():
            raise ValueError("beta must be finite")
        if beta.size == 0:
            beta = np.zeros(2)
        if beta.size != 2:
            beta = np.resize(beta, 2)
        super().__init__(reference, beta)
        self.reference = reference
        self.liver_faces = reference.liver_faces
        self.gallbladder_faces = reference.gallbladder_faces
        # Warp meshes once per body; each trial anatomy then only applies a
        # rigid/anisotropic GB transform and a shared translation.
        self._liver_base = np.asarray(self.warp(reference.liver_vertices), dtype=float)
        self._gall_base = np.asarray(self.warp(reference.gallbladder_vertices), dtype=float)
        source_landmarks, source_axis = _landmark_regions(reference.gallbladder_vertices)
        # Transport the source landmarks through the same material map before
        # deriving the body-local frame.
        self._landmarks_base = {k: np.asarray(self.warp(v), dtype=float) for k, v in source_landmarks.items()}
        self._axis_base = _unit(self._landmarks_base["neck"] - self._landmarks_base["fundus"], source_axis)
        self._center_base = np.asarray(self.warp(reference.gallbladder_center_mm), dtype=float)
        # The atlas center lies near the GB body; use the mesh centroid only as
        # a fallback for unusual meshes lacking a finite registered center.
        if not np.isfinite(self._center_base).all():
            self._center_base = self._gall_base.mean(axis=0)
        self.gallbladder_center = self._center_base.copy()
        self.gallbladder_axis = self._axis_base.copy()
        # A stable transverse frame for width scaling around the neck.
        normal_seed = np.array([0., 0., 1.])
        if abs(float(normal_seed @ self._axis_base)) > .9:
            normal_seed = np.array([1., 0., 0.])
        self._width_a = _unit(np.cross(self._axis_base, normal_seed), np.array([1., 0., 0.]))
        self._width_b = _unit(np.cross(self._axis_base, self._width_a), np.array([0., 0., 1.]))
        # Native topology is reused by sectioning and collision checks.
        self._liver_mesh_native = None
        self._gall_mesh_native = None
        self._valid_vertex_stride = 8
        self._valid_bone_cache = {}

    def _nominal_view_frame(self, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nominal image frame containing the model GB center and axis."""

        forward = _unit(self._center_base - origin, np.array([0., 1., 0.]))
        # The normal is orthogonal to both the contact-to-center ray and the
        # GB axis, so the nominal image plane is a genuine long-axis plane.
        normal = np.cross(forward, self._axis_base)
        if np.linalg.norm(normal) < 1.e-8:
            normal = np.cross(forward, np.array([1., 0., 0.]))
        normal = _unit(normal, np.array([0., 0., 1.]))
        right = _unit(np.cross(normal, forward), np.array([1., 0., 0.]))
        return forward, right, normal

    def view(self, chart, rib=None):
        """Return the existing model.View for ``(y, theta, roll, rock)``.

        Roll and rock are continuous angles in degrees.  ``rib`` is an
        optional ``(dy_mm, dz_mm)`` displacement routed through the inherited
        :class:`Bones` ray caster; no target anatomy is consulted here.
        """

        q = np.asarray(chart, dtype=float).reshape(-1)
        if q.size != 4 or not np.isfinite(q).all():
            raise ValueError("chart must be four finite values (y, theta, roll, rock)")
        # Keep the base material surface implementation, which is the SMPL-X
        # transported torso rather than a synthetic abdominal window.
        origin = np.asarray(self.surface(q), dtype=float)
        forward, right, normal = self._nominal_view_frame(origin)
        roll, rock = np.deg2rad(q[2:])
        right = np.cos(roll) * right + np.sin(roll) * np.cross(forward, right)
        forward = np.cos(rock) * forward + np.sin(rock) * np.cross(forward, right)
        forward = _unit(forward, np.array([0., 1., 0.]))
        normal = _unit(np.cross(forward, right), normal)
        address = np.array([(q[0] - self.atlas.y0) / (self.atlas.y1 - self.atlas.y0),
                            np.sin(np.deg2rad(q[1])), np.cos(np.deg2rad(q[1]))])
        result = _ModelView(q.copy(), address, origin, forward, right, normal,
                            np.full(len(READOUT), np.inf, dtype=float))
        return result if rib is None else self.with_ribs(result, np.asarray(rib, dtype=float).reshape(2))

    def anatomy(self, params) -> Anatomy:
        """Instantiate native anatomy from ``dx,dy,dz,yaw,pitch,logL,logW``."""

        p = np.asarray(params, dtype=float).reshape(-1)
        if p.size != 7 or not np.isfinite(p).all():
            raise ValueError("anatomy parameters must be seven finite values")
        translation = p[:3]
        # Clipping only protects numerical overflow.  Body.valid remains the
        # anatomical rejection gate used by the benchmark root.
        length = float(np.exp(np.clip(p[5], -4., 4.)))
        width = float(np.exp(np.clip(p[6], -4., 4.)))
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler("yx", [p[3], p[4]], degrees=True).as_matrix()

        neck0 = self._landmarks_base["neck"]
        relative = self._gall_base - neck0
        axial = relative @ self._axis_base
        transverse = relative - axial[:, None] * self._axis_base
        scaled = neck0 + length * axial[:, None] * self._axis_base + width * transverse
        gall = neck0 + (scaled - neck0) @ rotation.T + translation
        liver = self._liver_base + translation

        def transform_landmark(point):
            rel = point - neck0
            t = float(rel @ self._axis_base)
            q = neck0 + length * t * self._axis_base + width * (rel - t * self._axis_base)
            return neck0 + (q - neck0) @ rotation.T + translation

        landmarks = {name: transform_landmark(point) for name, point in self._landmarks_base.items()}
        axis = _unit(rotation @ self._axis_base, self._axis_base)
        # The model center is a material landmark too: apply the same axial
        # and transverse scale about the fossa neck before yaw/pitch and the
        # shared translation.  This keeps ``center`` consistent with the
        # transformed mesh when a planner uses non-unit GB scales.
        center = transform_landmark(self._center_base)
        return Anatomy(liver, gall, landmarks, axis, center)

    @staticmethod
    def _polydata(points, faces):
        """Build a PyVista mesh lazily for exact native collision checks."""

        try:
            import pyvista as pv
            cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
            return pv.PolyData(np.asarray(points, dtype=float), cells)
        except Exception:
            return None

    def _skin_ok(self, points: np.ndarray) -> bool:
        """Check that all sampled organ points remain inside the skin atlas."""

        sample = np.asarray(points)[::self._valid_vertex_stride]
        if len(sample) == 0:
            return False
        try:
            h, theta, rho = self.atlas.to_disk(sample)
        except Exception:
            return False
        # The material map is torso-wide, so allow only a small interpolation
        # tolerance at the atlas caps.  Organs must not be on the outside shell.
        if not (np.isfinite(h).all() and np.isfinite(theta).all() and np.isfinite(rho).all()
                and np.all((h >= -.02) & (h <= 1.02))
                and np.all(rho <= 1.005) and np.all(rho >= -.02)):
            return False
        # Evaluate the displaced SMPL-X shell at each material address.  The
        # first rho test above is a cheap atlas-domain guard; this radial test
        # uses the actual body rim/core correction carried by ``warp`` and
        # catches shape-dependent shell escape without a 3-D skin collision.
        hq = np.clip(h, 0., 1.)
        try:
            shell = self.atlas.to_world(hq, theta, np.ones_like(hq)) + self.rim.evaluate(hq, theta)
            core = self.atlas.core(hq) + self.core(hq)
            shell_radius = np.linalg.norm(shell - core, axis=-1)
            point_radius = np.linalg.norm(sample - core, axis=-1)
        except Exception:
            return False
        return bool(np.isfinite(shell_radius).all() and np.isfinite(point_radius).all()
                    and np.all(point_radius <= shell_radius + 1.5))

    def _bone_ok(self, points: np.ndarray, faces: np.ndarray, rib_shift) -> bool:
        """Native rib collision test with a fast clearance precheck."""

        shift = np.zeros(3)
        if rib_shift is not None:
            rs = np.asarray(rib_shift, dtype=float).reshape(-1)
            if rs.size != 2 or not np.isfinite(rs).all():
                return False
            shift[[1, 2]] = rs
        shifted = np.asarray(points, dtype=float) - shift
        # The locator's closest-point query is inexpensive and rejects obvious
        # penetrations before invoking VTK's exact triangle collision pass.
        sample = shifted[::self._valid_vertex_stride]
        if not len(sample):
            return False
        try:
            if self.bones.clearance(sample, np.zeros(len(sample))) < -1.e-5:
                return False
        except Exception:
            pass
        mesh = self._polydata(shifted, faces)
        if mesh is None:
            # Validity must never silently turn into a proxy test when native
            # triangle collision support is unavailable.
            return False
        try:
            _, count = mesh.collision(self.bones.mesh)
            return int(count) == 0
        except Exception:
            # Do not claim an anatomy is valid from a clearance heuristic when
            # the exact native collision filter cannot run.
            return False

    def valid(self, anatomy: Anatomy, rib_shift=None) -> bool:
        """Reject non-finite, out-of-skin, or native rib-colliding anatomy."""

        if not isinstance(anatomy, Anatomy):
            return False
        if not (np.isfinite(anatomy.liver_vertices).all() and np.isfinite(anatomy.gallbladder_vertices).all()):
            return False
        # Liver and GB share exactly the requested translation; use both meshes
        # in the skin/collision envelope rather than a proxy ellipsoid.
        if not self._skin_ok(anatomy.liver_vertices) or not self._skin_ok(anatomy.gallbladder_vertices):
            return False
        return (self._bone_ok(anatomy.liver_vertices, self.liver_faces, rib_shift)
                and self._bone_ok(anatomy.gallbladder_vertices, self.gallbladder_faces, rib_shift))

    @staticmethod
    def _fan_polygon():
        # A true radial fan: near-field boundary at 4 mm and a circular
        # far-field boundary at DEPTH_MM, rather than an artificial rectangle.
        angles = np.linspace(-float(BEAM_ANGLE), float(BEAM_ANGLE), 96)
        outer = np.column_stack((DEPTH_MM * np.sin(angles), DEPTH_MM * np.cos(angles)))
        left = np.array([4. * np.sin(-BEAM_ANGLE), 4. * np.cos(-BEAM_ANGLE)])
        right = np.array([4. * np.sin(BEAM_ANGLE), 4. * np.cos(BEAM_ANGLE)])
        return Polygon(np.vstack((left, outer, right)))

    @staticmethod
    def _shadow_polygon(view):
        """Build the exact piecewise-READOUT radial shadow region."""

        distances = np.asarray(view.bone_distance, dtype=float)
        if distances.size == 0 or not np.isfinite(distances).any():
            return None
        angles = np.linspace(-float(BEAM_ANGLE), float(BEAM_ANGLE), len(distances))
        edges = np.r_[angles[0], .5 * (angles[:-1] + angles[1:]), angles[-1]]
        pieces = []
        for i, radius in enumerate(distances):
            if not np.isfinite(radius):
                continue
            radius = float(np.clip(radius, 0., DEPTH_MM))
            a0, a1 = edges[i], edges[i + 1]
            # Four points describe the radial cell beyond the rib hit.  The
            # arc is approximated by the cell edges only; this matches
            # View.beam_blocked's nearest-READOUT convention exactly.
            inner0 = [radius * np.sin(a0), radius * np.cos(a0)]
            inner1 = [radius * np.sin(a1), radius * np.cos(a1)]
            outer1 = [DEPTH_MM * np.sin(a1), DEPTH_MM * np.cos(a1)]
            outer0 = [DEPTH_MM * np.sin(a0), DEPTH_MM * np.cos(a0)]
            pieces.append(Polygon([inner0, inner1, outer1, outer0]))
        if not pieces:
            return None
        try:
            return unary_union(pieces)
        except Exception:
            return pieces[0]

    @staticmethod
    def _section(points: np.ndarray, faces: np.ndarray, view):
        """Return a Shapely geometry for the exact triangle-plane section."""

        if trimesh is None or polygonize is None:
            return None
        mesh = trimesh.Trimesh(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64), process=False)
        section = mesh.section(plane_origin=view.origin, plane_normal=view.normal)
        if section is None:
            return None
        lines = []
        for path in section.discrete:
            rel = np.asarray(path, dtype=float) - view.origin
            xy = np.column_stack((rel @ view.right, rel @ view.forward))
            if len(xy) >= 3 and np.isfinite(xy).all():
                # mesh.section usually closes paths, but explicitly closing
                # handles older trimesh releases too.
                if np.linalg.norm(xy[0] - xy[-1]) > 1.e-8:
                    xy = np.vstack((xy, xy[0]))
                lines.append(LineString(xy))
        if not lines:
            return None
        try:
            polygons = list(polygonize(MultiLineString(lines)))
            if not polygons:
                return None
            return unary_union(polygons)
        except Exception:
            return None

    @staticmethod
    def _boundary_points(geometry):
        if geometry is None or geometry.is_empty:
            return np.empty((0, 2), dtype=float)
        boundary = geometry.boundary
        if hasattr(boundary, "geoms"):
            chunks = [np.asarray(g.coords, dtype=float) for g in boundary.geoms if len(g.coords)]
        else:
            chunks = [np.asarray(boundary.coords, dtype=float)]
        return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=float)

    @classmethod
    def _feature(cls, geometry, axis, view):
        """Compute area/centroid/long-span/orientation from visible polygons."""

        if geometry is None or geometry.is_empty or float(geometry.area) <= 1.e-9:
            return np.zeros(5, dtype=float)
        area = float(geometry.area)
        centroid = geometry.centroid
        points = cls._boundary_points(geometry)
        if len(points) < 2:
            return np.array([area, float(centroid.x), float(centroid.y), 0., 0.])
        centered = points - points.mean(axis=0)
        cov = centered.T @ centered / max(len(centered), 1)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        if direction[1] < 0.:
            direction = -direction
        projection = centered @ direction
        span = float(projection.max() - projection.min())
        # atan2(horizontal, depth) makes zero a depth-aligned long axis and is
        # naturally periodic under a 180-degree contour flip.
        angle = _wrap_pi(float(np.arctan2(direction[0], direction[1])))
        return np.array([area, float(centroid.x), float(centroid.y), span, angle], dtype=float)

    @staticmethod
    def _axis_range(geometry, axis2):
        points = Body._boundary_points(geometry)
        if len(points) == 0:
            return None
        direction = _unit(axis2, np.array([0., 1.]))
        values = points @ direction
        return float(values.min()), float(values.max())

    def measure(self, view, anatomy: Anatomy) -> Slice:
        """Measure true mesh sections clipped by the fan and rib shadow."""

        if not isinstance(view, _ModelView):
            raise TypeError("view must be the model.View returned by Body.view")
        if not isinstance(anatomy, Anatomy):
            raise TypeError("anatomy must be returned by Body.anatomy")
        fan = self._fan_polygon()
        shadow = self._shadow_polygon(view)

        raw_gall = self._section(anatomy.gallbladder_vertices, self.gallbladder_faces, view)
        raw_liver = self._section(anatomy.liver_vertices, self.liver_faces, view)

        def clip(raw):
            if raw is None or raw.is_empty:
                return None, None
            inside = raw.intersection(fan)
            if inside.is_empty:
                return inside, inside
            return inside, (inside.difference(shadow) if shadow is not None else inside)

        gall_inside, gall_visible = clip(raw_gall)
        liver_inside, liver_visible = clip(raw_liver)
        gall = self._feature(gall_visible, anatomy.axis, view)
        liver = self._feature(liver_visible, anatomy.axis, view)
        raw_gall_feature = self._feature(raw_gall, anatomy.axis, view)
        raw_liver_feature = self._feature(raw_liver, anatomy.axis, view)

        # Visible fractions are relative to the un-clipped mesh section.  None
        # explicitly records a tissue miss (as opposed to a visible zero area
        # after a rib shadow).
        raw_gall_area = 0. if raw_gall is None else float(raw_gall.area)
        raw_liver_area = 0. if raw_liver is None else float(raw_liver.area)
        inside_gall_area = 0. if gall_inside is None else float(gall_inside.area)
        inside_liver_area = 0. if liver_inside is None else float(liver_inside.area)
        visible_area = float(gall[0])
        visible_liver_area = float(liver[0])
        visible = None if inside_gall_area <= 1.e-8 else float(np.clip(visible_area / max(raw_gall_area, 1.e-8), 0., 1.))
        visible_liver = None if inside_liver_area <= 1.e-8 else float(np.clip(visible_liver_area / max(raw_liver_area, 1.e-8), 0., 1.))

        # Axis coverage is a geometric line-span fraction, measured on the
        # actual section and not inferred from centroid centering.  The
        # denominator is the subject anatomy's complete source-derived
        # neck-to-fundus range, so a short in-frame chord cannot qualify just
        # because it fills the local section polygon.
        axis2 = np.array([anatomy.axis @ view.right, anatomy.axis @ view.forward])
        raw_range = self._axis_range(raw_gall, axis2)
        lumen = gall_visible
        if lumen is not None and hasattr(lumen, "geoms"):
            lumen = max(lumen.geoms, key=lambda component: component.area, default=None)
        visible_range = self._axis_range(lumen, axis2)
        full_axis_mm = float(abs((anatomy.landmarks["neck"] - anatomy.landmarks["fundus"]) @ anatomy.axis))
        projected_axis = float(np.linalg.norm(axis2))
        if (raw_range is None or visible_range is None or raw_range[1] <= raw_range[0] + 1.e-8
                or full_axis_mm <= 1.e-8 or projected_axis <= .35):
            coverage = 0.
        else:
            visible_axis_mm = (visible_range[1] - visible_range[0]) / projected_axis
            coverage = float(np.clip(visible_axis_mm / full_axis_mm, 0., 1.))

        # Long-axis qualification requires in-frame tissue and all three
        # source-derived landmarks to have visible support.  A small normal
        # tolerance accounts for a triangulated section passing near (rather
        # than exactly through) a tip, while the projected point must still be
        # inside the ray-clipped polygon.
        support = False
        if lumen is not None and not lumen.is_empty and visible_area > 1.e-8:
            support_values = []
            for name in ("neck", "body", "fundus"):
                point = np.asarray(anatomy.landmarks[name])
                rel = point - view.origin
                x, depth, normal = float(rel @ view.right), float(rel @ view.forward), float(rel @ view.normal)
                in_fan = depth >= 4. and np.hypot(x, depth) <= DEPTH_MM and abs(x) <= depth * np.tan(BEAM_ANGLE)
                clear = not bool(view.beam_blocked(np.array([x]), np.array([depth]))[0])
                # The body landmark is a source-mesh median region rather than
                # a point on the principal-axis chord, so retain a modest
                # tissue-thickness tolerance for that triangulated support.
                near_plane = abs(normal) <= 12.
                support_values.append(in_fan and clear and near_plane
                                     and float(lumen.distance(Point(x, depth))) <= 8.)
            support = bool(np.all(support_values))
        # A long-axis section is established by the source-derived span and
        # landmark support, not an arbitrary in-image angle threshold.  Keep a
        # connected visible lumen requirement so a rib-split sliver cannot be
        # accepted as a qualified image.
        connected = lumen is not None and not lumen.is_empty and float(lumen.area) >= .70 * visible_area
        long_span = 0. if visible_range is None else visible_range[1] - visible_range[0]
        qualified = bool(visible is not None and visible_area >= 7.5 and coverage >= .50
                         and long_span >= 18. and support and connected and liver[0] > 0.)
        return Slice(gall=gall, liver=liver, coverage=coverage, visible=visible,
                     qualified=qualified, visible_liver=visible_liver,
                     raw_gall=raw_gall_feature, raw_liver=raw_liver_feature)

    def aim(self, xy, center, axis):
        return aim(xy, center, axis, body=self)


_DEFAULT_AIM_BODY = None


def aim(xy, center, axis, body=None):
    """Approximate roll/rock for a known model center and long axis.

    The three-argument public form lazily uses a nominal reference body.  A
    controller that already owns a subject body should pass it as the fourth
    argument or call ``body.aim(xy, center, axis)``; both use the same contact
    surface and frame calculation.  The returned chart is ``(y, theta, roll,
    rock)`` in degrees.
    """

    global _DEFAULT_AIM_BODY
    # Match the existing heart helper's convenient ``aim(body, xy, target,
    # axis)`` form as well as the requested public ``aim(xy, target, axis)``
    # form.  Both call styles share one implementation and angle convention.
    if isinstance(xy, _MaterialBody):
        xy, center, axis, body = center, axis, body, xy
    if body is None:
        if _DEFAULT_AIM_BODY is None:
            _DEFAULT_AIM_BODY = Body(Reference(), np.zeros(2))
        body = _DEFAULT_AIM_BODY
    q = np.asarray(xy, dtype=float).reshape(-1)
    if q.size != 2 or not np.isfinite(q).all():
        raise ValueError("xy must be two finite chart values (y, theta)")
    target = np.asarray(center, dtype=float).reshape(-1)
    direction = _unit(np.asarray(axis, dtype=float).reshape(-1), np.array([0., 1., 0.]))
    if target.size != 3 or not np.isfinite(target).all():
        raise ValueError("center must be three finite millimetres")
    base = body.view(np.r_[q, 0., 0.])
    # A long-axis plane contains the center-to-contact ray and target axis;
    # this is the closest valid plane to the nominal view normal.
    ray = _unit(target - base.origin, base.forward)
    desired_normal = np.cross(ray, direction)
    if np.linalg.norm(desired_normal) < 1.e-8:
        desired_normal = base.normal - direction * float(base.normal @ direction)
    desired_normal = _unit(desired_normal, base.normal)
    if desired_normal @ base.normal < 0.:
        desired_normal = -desired_normal
    roll = np.arctan2(-desired_normal @ base.right, desired_normal @ base.normal)
    after_roll = -np.sin(roll) * base.right + np.cos(roll) * base.normal
    rock = np.arctan2(-desired_normal @ base.forward, desired_normal @ after_roll)
    return np.r_[q, np.rad2deg([roll, rock])]


__all__ = ["Reference", "Body", "Anatomy", "Slice", "aim", "ATLAS_PATH"]
