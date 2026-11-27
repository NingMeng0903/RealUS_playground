"""Render recorded search trajectories and their observation-conditioned vessel maps."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import igl
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.special import ndtri
import trimesh

import rl_rollouts as rollout
from audit_smooth_skin import build_pose_mesh
from audit_trajectory import DynamicScan, load_rows
from smooth_switch import SmoothAtlas, learned_shapes
from validate_smooth_switch import GEOMETRY, HERE, OUT, SEED, ReferenceRecorder


SIZE = (1280, 720)
FPS = 24
PAD = 32
SCENE = (PAD, 116, 896, 435)
PROFILE = (PAD, 448, 896, 642)
PANEL = (915, 116, 1248, 642)
FONT_DIR = Path("/usr/share/fonts/truetype/noto")
BG = "#f1f5f6"
INK = "#152e3a"
MUTED = "#506a73"
LINE = "#d4e1e4"
VESSEL = "#994a22"
PRIOR = "#748c96"
VESSEL_BAND = (233, 157, 71, 97)
BRANCH = "#714a85"
GOLD = "#b87718"
GREEN = "#176b57"
Z90 = float(ndtri(.95))
INITIAL = "Initial scan"
RECHECK = "Recheck"


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def label(draw, xy, value, size, *, color=INK, bold=False, anchor=None):
    draw.text(xy, value, fill=color, font=font(size, bold), anchor=anchor)


def normalize(vector):
    return vector / np.linalg.norm(vector)


class Camera:
    def __init__(self, atlas):
        centers = atlas.targets[2, :, 3, :3, 3]
        self.origin = centers.mean(axis=0)
        axis = normalize(centers[-1] - centers[0])
        outward = -atlas.targets[2, :, 3, :3, 2].mean(axis=0)
        outward = normalize(outward - axis * (axis @ outward))
        toward = normalize(.88 * outward + .47 * np.cross(axis, outward) + .13 * axis)
        horizontal = normalize(axis - toward * (axis @ toward))
        vertical = np.cross(toward, horizontal)
        self.axes = np.column_stack((horizontal, vertical, toward))
        self.center = np.array([.5 * (SCENE[0] + SCENE[2]), 252.])
        self.scale = 1530.

    def local(self, points):
        return (np.asarray(points) - self.origin) @ self.axes

    def project(self, points):
        xyz = self.local(points)
        xy = xyz[..., :2] * [self.scale, -self.scale] + self.center
        return xy, xyz[..., 2]


def xy(point):
    return tuple(map(float, point))


def line(draw, points, color, width=2):
    if len(points) >= 2:
        draw.line([xy(p) for p in points], fill=color, width=width, joint="curve")


def dot(draw, point, radius, fill, outline=None, width=1):
    x, y = xy(point)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def dashed(draw, points, color, *, dash=8., gap=5., width=2):
    phase = 0.
    period = dash + gap
    for left, right in zip(points, points[1:]):
        delta = right - left
        length = np.linalg.norm(delta)
        if length <= 1.e-8:
            continue
        progress = 0.
        while progress < length - 1.e-8:
            part = (phase + progress) % period
            step = min(length - progress, (dash if part < dash else period) - part)
            if step <= 1.e-8:
                progress += 1.e-8
                continue
            if part < dash:
                line(draw, np.array([left + (progress / length) * delta,
                                     left + ((progress + step) / length) * delta]), color, width)
            progress += step
        phase = (phase + length) % period


class Anatomy:
    def __init__(self, geometry, scan, atlas, camera):
        self.camera = camera
        mesh_at, faces = build_pose_mesh(geometry, atlas)
        original = mesh_at(atlas.angles[0])
        coarse = trimesh.Trimesh(original, faces, process=False).simplify_quadric_decimation(face_count=2400)
        # Keep the simplified drawing mesh anchored to the baked skin.
        _, indices, closest = igl.point_mesh_squared_distance(coarse.vertices, original, faces)
        carriers = faces[indices]
        bary = trimesh.triangles.points_to_barycentric(original[carriers], closest)
        posed = np.array([
            np.einsum("ni,nij->nj", bary, mesh_at(angle)[carriers]) for angle in atlas.angles
        ])
        coarse_faces = np.asarray(coarse.faces)
        center_x = camera.local(posed[2])[coarse_faces, 0].mean(axis=1)
        self.faces = coarse_faces[abs(center_x) < .234]
        self.mesh = CubicSpline(atlas.angles, posed)
        transform = geometry["robot_from_atlas"]
        rotation, translation = transform[:3, :3], transform[:3, 3]
        lateral = scan.frames[..., 1] @ rotation.T
        trunk = scan.trunk @ rotation.T + translation
        branch = scan.dense_branch @ rotation.T + translation
        self.curves = (
            VesselCurve(scan, atlas.angles, trunk, lateral, branch=False),
            VesselCurve(scan, atlas.angles, branch, lateral, branch=True),
        )
        self.reference_angle = atlas.angles[scan.bent]
        self.station_mm = scan.s
        fade = np.zeros(SIZE[0], dtype=np.uint8)
        left, top, right, bottom = SCENE
        width = 55
        fade[left:left + width] = np.rint(np.linspace(255., 0., width)).astype(np.uint8)
        fade[right - width:right + 1] = np.rint(np.linspace(0., 255., width + 1)).astype(np.uint8)
        mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
        mask[top + 28:bottom - 22, left + 3:right - 2] = fade[left + 3:right - 2]
        self.fade = Image.fromarray(mask)

    @lru_cache(maxsize=84)
    def background(self, angle_tenth):
        angle = np.deg2rad(angle_tenth / 10.)
        image = Image.new("RGB", SIZE, BG)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(SCENE, radius=16, fill="#fdfefe", outline=LINE)
        v = self.mesh(angle)
        screen, depth = self.camera.project(v)
        triangles = v[self.faces]
        face_n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.maximum(np.linalg.norm(face_n, axis=1), 1.e-12)
        light = np.abs(face_n @ self.camera.axes[:, 2] / length)
        order = np.argsort(depth[self.faces].mean(axis=1))
        for i in order:
            shade = float(light[i])
            rgb = (int(207 + 22 * shade), int(223 + 17 * shade), int(227 + 17 * shade))
            draw.polygon([xy(p) for p in screen[self.faces[i]]], fill=rgb)
        image.paste("#fdfefe", (0, 0, *SIZE), self.fade)
        return image

    def draw_vessels(self, draw, angle, belief):
        paths = [curve.locations(angle, belief) for curve in self.curves]
        for _, _, lower, upper in paths:
            band = self.camera.project(np.vstack((lower, upper[::-1])))[0]
            draw.polygon([xy(point) for point in band], fill=VESSEL_BAND)
        for nominal, mean, _, _ in paths:
            dashed(draw, self.camera.project(nominal)[0], PRIOR, width=2)
            line(draw, self.camera.project(mean)[0], VESSEL, width=3)

    def draw_profile(self, draw, belief, site):
        left, top, right, bottom = PROFILE
        draw.rounded_rectangle(PROFILE, radius=16, fill="#fdfefe", outline=LINE)
        label(draw, (left + 22, top + 10), "Lateral offset at 15° (mm)", 15, bold=True)
        for x, text, color in ((right - 172, "Main", VESSEL), (right - 90, "Branch", BRANCH)):
            draw.line((x, top + 20, x + 14, top + 20), fill=color, width=3)
            label(draw, (x + 21, top + 20), text, 13, color=MUTED, anchor="lm")
        x0, x1, y0, y1 = left + 53, right - 26, top + 37, bottom - 36
        start, end = self.station_mm[[0, -1]]
        limit = 16.

        def x_of(s):
            return x0 + (x1 - x0) * (s - start) / (end - start)

        def y_of(offset):
            return y0 + (y1 - y0) * (limit - offset) / (2. * limit)

        for offset in (-16, -8, 0, 8, 16):
            y = y_of(offset)
            draw.line((x0, y, x1, y), fill="#cedbdf" if offset == 0 else "#e8eff1", width=1)
            label(draw, (x0 - 8, y), f"{offset}", 12, color=MUTED, anchor="rm")
        if site is not None:
            x = x_of(self.station_mm[site])
            draw.line((x, y0, x, y1), fill="#9fb2bc", width=1)
        for s in (0, 100, 200, 300, end):
            x = x_of(s)
            draw.line((x, y1, x, y1 + 3), fill="#b9cbd0")
            label(draw, (x, y1 + 12), f"{s:.0f}", 12, color=MUTED, anchor="mm")
        label(draw, (.5 * (x0 + x1), bottom - 9), "Material arc length (mm)", 12, color=MUTED, anchor="mm")

        styles = (("#efb982", VESSEL), ((171, 133, 194, 118), BRANCH))
        for curve, (fill, stroke) in zip(self.curves, styles):
            mean, half_width = curve.offsets(self.reference_angle, belief)
            if np.any(np.abs(mean) + half_width > limit + 1.e-9):
                raise ValueError("posterior interval exceeds the fixed profile axes")
            x = x_of(curve.s)
            lower = np.column_stack((x, y_of(mean - half_width)))
            upper = np.column_stack((x, y_of(mean + half_width)))
            draw.polygon([xy(point) for point in np.vstack((lower, upper[::-1]))], fill=fill)
            line(draw, np.column_stack((x, y_of(mean))), stroke, width=2)


class VesselCurve:
    """Transport the lateral latent field to each point on a vessel template."""

    def __init__(self, scan, angles, nominal, lateral, *, branch):
        s = scan.branch_s if branch else np.unique(np.r_[scan.s, np.linspace(scan.s[0], scan.s[-1], 145)])
        self.s = s
        basis = np.column_stack([np.interp(s, scan.s, w) for w in np.eye(scan.n)])
        self.rows = np.zeros((len(s), scan.dim))
        self.rows[:, :scan.n] = basis
        if branch:
            self.rows[:, -2] = (s - s[0]) / (s[-1] - s[0])
        else:
            nominal = np.einsum("ij,pjc->pic", basis, nominal)
        self.slip = basis @ scan.feature
        self.center_deg = scan.center
        self.range_deg = scan.range
        self.nominal = CubicSpline(angles, nominal)
        self.lateral = CubicSpline(angles, np.einsum("ij,pjc->pic", basis, lateral))

    def offsets(self, angle, belief):
        rows = self.rows.copy()
        rows[:, -1] = self.slip * (np.rad2deg(angle) - self.center_deg) / self.range_deg
        mean = rows @ belief.mean
        variance = np.sum((rows @ belief.covariance) * rows, axis=1)
        return mean, Z90 * np.sqrt(np.maximum(variance, 0.))

    def locations(self, angle, belief):
        nominal = self.nominal(angle)
        lateral = self.lateral(angle)
        lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
        offset, half_width = self.offsets(angle, belief)
        mean = nominal + .001 * offset[:, None] * lateral
        spread = .001 * half_width[:, None] * lateral
        return nominal, mean, mean - spread, mean + spread


@dataclass(frozen=True)
class Belief:
    mean: np.ndarray
    covariance: np.ndarray
    half_width_mm: float
    ready_targets: int


def belief_timeline(scan, chapter, model):
    """Reproduce the public belief, including misses and the shared prefix."""
    queries = np.stack([scan.row(scan.bent, site, label, True)
                        for site, label in scan.targets if label < 2])
    book = rollout.branches.DetectionBook(model, gate="predictive", predictive_z=2.)
    mean, covariance = rollout._initial_belief(scan)

    def snapshot(mean, covariance, ready):
        variances = np.sum((queries @ covariance) * queries, axis=1)
        width = Z90 * np.median(np.sqrt(np.maximum(variances, 0.)))
        return Belief(mean.copy(), covariance.copy(), float(width), int(ready))

    records = [snapshot(mean, covariance, 0)]
    for step in chapter.prefix:
        mean, covariance = book.assimilate(
            (mean, covariance), step["pose"], step["action"], step["observation_mm"],
        )
        ready = book.status((mean, covariance))[2]
        records.append(snapshot(mean, covariance, ready.sum()))
    _, variance, _, _ = book.status((mean, covariance))
    if records[-1].ready_targets != chapter.suffix["initial_certified"]:
        raise ValueError("belief replay disagrees with the logged initial evidence")
    initial_index = len(records) - 1
    for step in chapter.suffix["steps"]:
        mean, covariance = book.assimilate(
            (mean, covariance), step["pose"], step["action"], step["centroids_mm"],
        )
        _, variance, ready, _ = book.status((mean, covariance))
        if int(ready.sum()) != step["n_ready"]:
            raise ValueError("belief replay disagrees with the logged evidence")
        records.append(snapshot(mean, covariance, ready.sum()))
    logged_band = chapter.suffix["max_band_mm"]
    np.testing.assert_allclose(model.gamma * np.sqrt(max(variance)), logged_band, atol=1.e-9, rtol=0.)
    widths = {"prior": records[0].half_width_mm, "after_initial_scan": records[initial_index].half_width_mm,
              "final": records[-1].half_width_mm}
    start = 0 if chapter.show_prefix else initial_index - 1
    return records[start:], widths


@dataclass(frozen=True)
class Chapter:
    title: str
    subtitle: str
    seed: int
    color: str
    prefix: list
    suffix: dict
    show_prefix: bool
    highlight: int | None


def get_chapters():
    shown = load_rows([HERE / "control_24", HERE / "eval_q_24"], [])[SEED]
    prefix = shown["rl"]["prefix_steps"]
    b_seed = 4087610863
    b = load_rows([HERE / "replicate_0"], [])[b_seed]["immediate"]
    return (
        Chapter("Rule-based search", "Mapping the full bifurcation", SEED, "#236887", prefix, shown["rule"], True, 30),
        Chapter("RL rechecks", "From the shared first-scan exit", SEED, "#a14262", prefix, shown["rl"], False, 9),
        Chapter("Observation-guided B", "", b_seed, "#177467",
                b["prefix_steps"], b, True, 8),
    )


def playback(chapter, scan, atlas, shapes):
    steps = [*chapter.prefix, *chapter.suffix["steps"]] if chapter.show_prefix else [
        chapter.prefix[-1], *chapter.suffix["steps"],
    ]
    states = [(step["pose"], *step["action"]) for step in steps]
    recorder = ReferenceRecorder(atlas, shapes, scan.moving, 30.)
    paths = []
    for previous, goal, step in zip(states, states[1:], steps[1:]):
        recorder.append(previous, goal, step["path_skill"])
        segment = recorder.segments[-1]
        phase = np.linspace(0., segment.duration, 97)
        v, r, _ = segment.world_jet(phase, acceleration=False)
        p, rot, _ = atlas.pose_and_twist(v, r)
        angle = recorder.clock.sample(segment.time_s + phase)[0]
        np.testing.assert_allclose(p[[0, -1]], atlas.targets[..., :3, 3][[previous[0], goal[0]],
                                  [previous[1], goal[1]], [previous[2], goal[2]]], atol=1.e-9, rtol=0.)
        paths.append((p, rot.as_matrix(), angle))
    return steps, states, paths


def probe(draw, camera, p, r, color):
    outward = -r[:, 2]
    long, lateral = r[:, 0], r[:, 1]
    # The acoustic slice uses the modeled +/-25 mm lateral support.
    mouth = np.array([p - .025 * lateral, p + .025 * lateral])
    base = mouth - .041 * outward
    plane = camera.project(np.array([mouth[0], mouth[1], base[1], base[0]]))[0]
    draw.polygon([xy(a) for a in plane], fill=(62, 146, 172, 37), outline=(46, 119, 145, 115), width=1)
    center = p + .011 * outward
    bottom = np.array([center + a * .007 * long + b * .014 * lateral for a, b in ((-1,-1), (1,-1), (1,1), (-1,1))])
    corners = np.concatenate((bottom - .008 * outward, bottom + .008 * outward))
    screen, depth = camera.project(corners)
    sides = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    colors = ("#174654", "#286878", "#245767", "#1d5362", "#1a4858", "#3a7885")
    for i in sorted(range(len(sides)), key=lambda j: float(depth[list(sides[j])].mean())):
        draw.polygon([xy(q) for q in screen[list(sides[i])]], fill=colors[i], outline="#e3f2f2", width=1)
    dot(draw, camera.project(p)[0], 5., color, "white", 2)


def reading(draw, x, y, values, *, gains, prefix, prior, title):
    label(draw, (x, y), title, 17, bold=True)
    y += 41
    length = 260.
    lo, hi = x + 8., x + 8. + length
    draw.rounded_rectangle((lo, y - 8, hi, y + 8), radius=8, fill="#eaf2f4")
    for idx, text in enumerate(("-25", "0", "+25 mm")):
        u = lo + .5 * idx * length
        draw.line((u, y - 11, u, y + 10), fill="#90a9b2")
        label(draw, (u, y + 24), text, 13, color=MUTED, anchor="mm")
    for value in values:
        u = lo + ((value + 25.) / 50.) * length
        dot(draw, (u, y), 6, GREEN if gains else GOLD, "white", 1)
    confirmed = "Detection accepted" if prefix else "New evidence"
    status = (
        "Awaiting first frame" if prior else
        "No detection" if not values else
        confirmed if gains else "Detected · no new evidence"
    )
    label(draw, (x, y + 40), status, 15, color=GREEN if gains else MUTED)
    if values:
        label(draw, (x, y + 64), "  /  ".join(f"{value:+.2f}" for value in values) + " mm", 14, color=MUTED)


def frame(anatomy, atlas, chapter, path, active, landmarks, idx, shown_total, step, previous_step, p, r,
          angle, belief, prior_width, *, arrival, stage, result=False, prior=False):
    cam = anatomy.camera
    image = anatomy.background(int(round(np.rad2deg(angle) * 10))).copy()
    draw = ImageDraw.Draw(image, "RGBA")
    label(draw, (PAD, 29), chapter.title, 33, bold=True)
    if chapter.subtitle:
        label(draw, (PAD, 81), chapter.subtitle, 17, color=MUTED)
    draw.rounded_rectangle((1050, 31, 1248, 74), radius=21, fill="#dcecf0")
    label(draw, (1149, 52), "Simulation replay", 16, bold=True, anchor="mm")
    label(draw, (54, 130), "Vessel posterior", 18, color=MUTED)
    anatomy.draw_vessels(draw, angle, belief)
    projected = cam.project(path)[0]
    rgba = tuple(bytes.fromhex(chapter.color.lstrip("#")))
    line(draw, projected, rgba + (96,), 2)
    for q in cam.project(landmarks)[0]:
        dot(draw, q, 3, rgba + (100,), "#fdfefe")
    line(draw, cam.project(active)[0], chapter.color, 4)
    probe(draw, cam, p, r, chapter.color)
    if arrival and stage == RECHECK and idx == chapter.highlight:
        if chapter.suffix["arm"] == "rule":
            center = atlas.targets[step["pose"], step["action"][0], 3, :3, 3]
            q, point = cam.project(np.stack((center, p)))[0]
            dashed(draw, np.array([q, point]), GOLD, dash=3., gap=3., width=2)
            leader = q + [44., 44.]
            line(draw, np.array([q, q + [25., 44.], leader]), GOLD, 2)
            dot(draw, q, 4, "white", GOLD, 2)
            draw.rounded_rectangle((leader[0] - 3, leader[1] - 13, leader[0] + 97, leader[1] + 12),
                                   radius=5, fill="#fff9eb")
            label(draw, (leader[0] + 5, leader[1]), "Center window", 13, color=GOLD, bold=True, anchor="lm")

    anatomy.draw_profile(draw, belief, None if prior else step["action"][0])
    x = PANEL[0] + 23
    draw.rounded_rectangle(PANEL, radius=16, fill="#ffffff", outline=LINE)
    label(draw, (x, 142), "Prior" if prior else stage, 17, color=chapter.color, bold=True)
    label(draw, (x, 166), f"{0 if prior else idx:02d} / {shown_total:02d}", 34, bold=True)
    site, view = step["action"]
    offset = 1000. * atlas.offsets[view]
    if not prior:
        label(draw, (x, 218), f"Station {site}  ·  {offset:+.0f} mm", 17, bold=True)
    label(draw, (x, 245), f"Knee {np.rad2deg(angle):.1f}°", 15, color=MUTED)
    draw.line((x, 279, PANEL[2] - 23, 279), fill=LINE)
    readout = {} if prior else step if arrival else previous_step
    values = readout.get("centroids_mm", readout.get("observation_mm", []))
    gains = bool(readout.get("evidence_gains", readout.get("confirmed", False)))
    reading(draw, x, 292, values, gains=gains, prefix="evidence_gains" not in readout, prior=prior,
            title="Ultrasound observation" if arrival or prior else "Last observation")
    draw.line((x, 429, PANEL[2] - 23, 429), fill=LINE)
    label(draw, (x, 443), "Median 90% half-width", 16, bold=True)
    label(draw, (x, 469), f"±{belief.half_width_mm:.2f} mm", 28, color=VESSEL, bold=True)
    if not prior:
        label(draw, (PANEL[2] - 25, 487), f"Prior ±{prior_width:.2f} mm", 14, color=MUTED, anchor="rm")
    draw.rounded_rectangle((x, 517, PANEL[2] - 25, 524), radius=3, fill="#f2e8dd")
    fraction = min(belief.half_width_mm / prior_width, 1.)
    draw.rounded_rectangle((x, 517, x + fraction * (PANEL[2] - 25 - x), 524), radius=3, fill="#c77d40")
    label(draw, (x, 542), "Targets ready", 16, bold=True)
    label(draw, (PANEL[2] - 25, 553), f"{belief.ready_targets} / 32", 17, bold=True, anchor="rm")
    detail = ""
    if arrival and stage == INITIAL and idx == shown_total:
        detail = "Initial pass complete"
    if stage == RECHECK and step["changed_suffix"]:
        locked = step["tail_choice"]["locked"][0]
        detail = f"Retargeted · station {locked} to {site}"
    if stage == RECHECK and idx == chapter.highlight:
        if chapter.suffix["arm"] == "rule":
            detail = "Same-station side step"
        elif chapter.suffix["arm"] == "rl":
            detail = "Long return · offline QP failed"
    if result:
        detail = "Passed simulation checks" if chapter.suffix["complete"] else "Stopped · identity check failed"
    if detail:
        draw.rounded_rectangle((x - 5, 577, PANEL[2] - 18, 620), radius=9, fill="#eaf3f5")
        label(draw, (x + 4, 589), detail, 15, bold=True, color=chapter.color)

    legend_y = 659
    for start, text, color, dash in ((36, "Skin", "#d3e3e7", False),
                                      (128, "Prior centerline", PRIOR, True),
                                      (293, "Posterior mean", VESSEL, False),
                                      (467, "90% lateral band", "#e9b17f", False),
                                      (643, "Probe path", chapter.color, False)):
        coords = np.array([[start, legend_y], [start + 27, legend_y]])
        if dash:
            dashed(draw, coords, color, width=3)
        else:
            line(draw, coords, color, 4)
        label(draw, (start + 36, legend_y), text, 15, color=MUTED, anchor="lm")
    label(draw, (1248, 685), "Playback time compressed", 14, color=MUTED, anchor="ra")
    return image


def stages(chapter, n):
    prefix_count = len(chapter.prefix) if chapter.show_prefix else 1
    if n < prefix_count:
        return INITIAL, len(chapter.prefix) if not chapter.show_prefix else n + 1, len(chapter.prefix)
    return RECHECK, n - prefix_count + 1, len(chapter.suffix["steps"])


def render(out, chapters, scan, atlas, anatomy, *, preview=False):
    shapes = learned_shapes(scan.primitive)
    model = rollout.study.Recheck(scan, edge=None)
    manifest = {
        "fps": FPS, "size": list(SIZE), "chapters": [],
        "uncertainty": "Pointwise 90% lateral posterior bands; median half-width at 15 degrees.",
        "profile_offset_limits_mm": [-16, 16],
    }
    output = out / "SEARCH_REPLAY_UNCERTAINTY.mp4"
    encoder = None
    frame_count = 0
    try:
        if not preview:
            encoder = imageio_ffmpeg.write_frames(
                str(output), SIZE, fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                quality=None, ffmpeg_log_level="error",
                output_params=["-preset", "veryfast", "-crf", "19", "-movflags", "+faststart"],
            )
            encoder.send(None)
        for chapter in chapters:
            steps, states, paths = playback(chapter, scan, atlas, shapes)
            beliefs, widths = belief_timeline(scan, chapter, model)
            if len(beliefs) != len(steps) + 1:
                raise ValueError("belief and movement histories have different lengths")
            chapter_info = {"title": chapter.title, "seed": chapter.seed, "start_s": frame_count / FPS,
                            "shown_initial_scan": chapter.show_prefix,
                            "rechecks": len(chapter.suffix["steps"]), "strict_complete": chapter.suffix["complete"],
                            "initial_acquisitions": len(chapter.prefix),
                            "ready_after_initial_scan": chapter.suffix["initial_certified"],
                            "ready_final": chapter.suffix["certified"],
                            "median_lateral_half_width_mm": widths}
            first = atlas.targets[states[0]]
            history = [first[:3, 3][None, :]]
            contacts = [first[:3, 3]]
            if chapter.show_prefix:
                first_step = steps[0]
                p, r, angle = first[:3, 3], first[:3, :3], atlas.angles[states[0][0]]
                image = frame(anatomy, atlas, chapter, history[0], p[None, :], np.empty((0, 3)), 0,
                              len(chapter.prefix), first_step, first_step, p, r, angle, beliefs[0],
                              widths["prior"], arrival=False, stage=INITIAL, prior=True)
                if preview:
                    image.save(out / f"PRIOR_{len(manifest['chapters']) + 1}.png")
                else:
                    rgb = np.asarray(image)
                    for _ in range(27):
                        encoder.send(rgb)
                        frame_count += 1
            for n, step in enumerate(steps):
                stage, idx, total = stages(chapter, n)
                old = steps[max(n - 1, 0)]
                if n == 0:
                    p, r, angle = first[:3, 3], first[:3, :3], atlas.angles[states[0][0]]
                    active = p[None, :]
                else:
                    points, rotations, angles = paths[n - 1]
                    p, r, angle = points[-1], rotations[-1], angles[-1]
                    if encoder is not None:
                        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                        moving_frames = int(np.clip(np.rint(4. + 44. * length), 4, 17))
                        trail = np.concatenate(history, axis=0)
                        for j in np.linspace(0, len(points) - 2, moving_frames, dtype=int):
                            image = frame(anatomy, atlas, chapter, trail, points[:j + 1], np.asarray(contacts),
                                          idx, total, step, old, points[j], rotations[j], angles[j], beliefs[n],
                                          widths["prior"], arrival=False, stage=stage)
                            encoder.send(np.asarray(image))
                            frame_count += 1
                    history.append(points[1:])
                    contacts.append(p)
                    active = points
                focus = stage == RECHECK and idx == chapter.highlight
                if encoder is None and not focus:
                    continue
                image = frame(anatomy, atlas, chapter, np.concatenate(history, axis=0), active, np.asarray(contacts),
                              idx, total, step, old, p, r, angle, beliefs[n + 1], widths["prior"],
                              arrival=True, stage=stage)
                if focus:
                    chapter_info["highlight_s"] = frame_count / FPS
                    chapter_info["highlight_recheck"] = idx
                    if preview:
                        image.save(out / f"PREVIEW_{len(manifest['chapters']) + 1}.png")
                if encoder is None:
                    continue
                obs = step.get("centroids_mm", step.get("observation_mm", []))
                hold = 23 if n == 0 else 5 if obs else 10
                if stage == INITIAL and idx == total:
                    hold = max(hold, 40)
                    chapter_info["initial_pass_complete_s"] = frame_count / FPS
                if step.get("changed_suffix", False):
                    hold = 20
                if focus:
                    hold = 48
                rgb = np.asarray(image)
                for _ in range(hold):
                    encoder.send(rgb)
                    frame_count += 1
            end = frame(anatomy, atlas, chapter, np.concatenate(history, axis=0), active, np.asarray(contacts),
                        idx, total, step, old, p, r, angle, beliefs[-1], widths["prior"],
                        arrival=True, stage=stage, result=True)
            if preview:
                end.save(out / f"FINAL_{len(manifest['chapters']) + 1}.png")
            else:
                rgb = np.asarray(end)
                for _ in range(40):
                    encoder.send(rgb)
                    frame_count += 1
                chapter_info["end_s"] = frame_count / FPS
            manifest["chapters"].append(chapter_info)
            print(chapter.title, chapter_info, flush=True)
    finally:
        if encoder is not None:
            encoder.close()
    if not preview:
        manifest["duration_s"] = frame_count / FPS
        (out / "VIDEO_CHAPTERS_UNCERTAINTY.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "search_replay")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(GEOMETRY, allow_pickle=False) as geometry:
        scan, atlas = DynamicScan(geometry), SmoothAtlas(geometry)
        anatomy = Anatomy(geometry, scan, atlas, Camera(atlas))
    render(args.out, get_chapters(), scan, atlas, anatomy, preview=args.preview)


if __name__ == "__main__":
    main()
