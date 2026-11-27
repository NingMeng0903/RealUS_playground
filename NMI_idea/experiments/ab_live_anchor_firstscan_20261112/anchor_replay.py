"""Replay acquisition timing and frame changes on the baked knee geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ASSET = HERE.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
OUTPUT = HERE / "anchor"
DT = .005
ATLAS_INTERVAL = 200
ATLAS_DELAY = 40
LOCAL_INTERVAL = 8
LOCAL_DELAY = 8
STATIONS = np.arange(2, 28, 3)
TICKS = np.arange(0, 1601, 4)
RUNS = 32
SEED = 2026111211
METHODS = ("Delayed atlas", "Atlas with pose prediction", "Time-aligned local plane")


def unit(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def rotate(rotation, vector):
    return np.einsum("...ij,...j->...i", rotation, vector)


def frame(tangent, normal):
    normal = unit(normal)
    tangent = unit(tangent - np.sum(tangent * normal, axis=-1, keepdims=True) * normal)
    return np.stack((tangent, np.cross(normal, tangent), normal), axis=-1)


class KneeGeometry:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            moving = data["pose_knee_deg"] > 0.
            central = np.flatnonzero((data["view_offsets_mm"] == 0.) & (data["view_tilts_deg"] == 0.)).item()
            self.angles = data["pose_knee_deg"][moving]
            self.point = 1000. * data["posed_views_m"][moving][:, STATIONS, central]
            self.axes = data["posed_frames"][moving][:, STATIONS, central]
        order = self.angles.argsort()
        self.angles = self.angles[order]
        self.point = self.point[order]
        self.axes = self.axes[order]

    def sample(self, q):
        q = np.clip(q, self.angles[0], self.angles[-1])
        index = np.clip(np.searchsorted(self.angles, q, side="right"), 1, len(self.angles) - 1)
        fraction = (q - self.angles[index - 1]) / (self.angles[index] - self.angles[index - 1])
        point = self.point[index - 1] + fraction[:, None, None] * (self.point[index] - self.point[index - 1])
        axes = self.axes[index - 1] + fraction[:, None, None, None] * (self.axes[index] - self.axes[index - 1])
        return point, frame(axes[..., 0], -axes[..., 2])


def statistics(errors):
    return {"median": float(np.median(errors)), "p95": float(np.quantile(errors, .95))}


def signed_height(point, normal, estimate_point, estimate_normal):
    # Intersect the estimated plane with the material station's normal line.
    return (np.sum((estimate_point - point) * estimate_normal, axis=-1)
            / np.sum(estimate_normal * normal, axis=-1))


def errors(point, normal, estimate_point, estimate_normal):
    height = np.abs(signed_height(point, normal, estimate_point, estimate_normal))
    angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(normal, estimate_normal), axis=-1),
                                 np.sum(normal * estimate_normal, axis=-1)))
    return height, angle


def replay(geometry, rng, *, stationary=False, biased=True):
    # Generate every simulated sensor and the measured FK log on one physical clock.
    ticks = np.arange(-2 * ATLAS_INTERVAL, TICKS[-1] + 1)
    t = ticks * DT
    amplitude = 0. if stationary else rng.uniform(2.6, 3.85)
    omega = 2. * np.pi / rng.uniform(5.4, 9.)
    phase = omega * t + rng.uniform(0., 2. * np.pi)
    q = 15. + amplitude * np.sin(phase)
    dq = amplitude * omega * np.cos(phase)
    nominal, axes = geometry.sample(q)
    tangent, lateral, nominal_normal = np.moveaxis(axes, -1, 0)

    height_bias = rng.uniform(-6., 6., len(STATIONS)) if biased else np.zeros(len(STATIONS))
    tilt = np.deg2rad(rng.uniform(-8., 8., (len(STATIONS), 2))) if biased else np.zeros((len(STATIONS), 2))
    actual = nominal + height_bias[None, :, None] * nominal_normal
    actual_normal = unit(nominal_normal + tilt[None, :, :1] * tangent + tilt[None, :, 1:] * lateral)

    # The FK stream is generated independently of the hidden skin bias. There is no servo model.
    probe_point = nominal + 3. * nominal_normal + (6. * np.sin(1.7 * t))[:, None, None] * tangent
    rocking = Rotation.from_rotvec(np.stack((.2 * np.sin(1.9 * t), .16 * np.cos(1.4 * t),
                                             np.zeros_like(t)), axis=-1)).as_matrix()
    probe_rotation = axes @ rocking[:, None, :, :]

    now = TICKS - ticks[0]
    atlas = (TICKS - ATLAS_DELAY) // ATLAS_INTERVAL * ATLAS_INTERVAL - ticks[0]
    local = (TICKS - LOCAL_DELAY) // LOCAL_INTERVAL * LOCAL_INTERVAL - ticks[0]
    delayed_point, delayed_axes = nominal[atlas], axes[atlas]
    predict_q = q[atlas] + dq[atlas] * (now - atlas) * DT
    predicted_point, predicted_axes = geometry.sample(predict_q)

    capture = np.unique(local)
    packet = np.searchsorted(capture, local)
    capture_probe = probe_point[capture]
    capture_rotation = probe_rotation[capture]
    capture_normal = actual_normal[capture]
    distance = np.sum((actual[capture] - capture_probe) * capture_normal, axis=-1)
    closest = capture_probe + distance[..., None] * capture_normal
    world_to_probe = capture_rotation.swapaxes(-1, -2)
    local_point = rotate(world_to_probe, closest - capture_probe)
    local_normal = rotate(world_to_probe, capture_normal)
    local_point += rng.normal(0., .25, distance.shape)[..., None] * local_normal
    local_normal = unit(local_normal + rng.normal(0., np.deg2rad(.5), local_normal.shape))
    local_world_point = (capture_probe + rotate(capture_rotation, local_point))[packet]
    local_world_normal = rotate(capture_rotation, local_normal)[packet]

    capture_wrong_point = probe_point[now] + rotate(probe_rotation[now], local_point[packet])
    capture_wrong_normal = rotate(probe_rotation[now], local_normal[packet])
    capture_true_point = actual[local]
    capture_true_normal = actual_normal[local]
    estimates = (
        (delayed_point, delayed_axes[..., 2]),
        (predicted_point, predicted_axes[..., 2]),
        (local_world_point, local_world_normal),
    )
    current = [errors(actual[now], actual_normal[now], p, n) for p, n in estimates]
    capture_error = [
        errors(capture_true_point, capture_true_normal, local_world_point, local_world_normal),
        errors(capture_true_point, capture_true_normal, capture_wrong_point, capture_wrong_normal),
    ]

    # If scan progress is zero, this is the required displacement of a fixed material station.
    pause_start = np.array([1., 3., 5., 7.])
    pause_ticks = np.rint(pause_start / DT).astype(int) - ticks[0]
    pause_end = pause_ticks + round(.5 / DT)
    paused = np.linalg.norm(actual[pause_end] - actual[pause_ticks], axis=-1)
    return current, capture_error, paused, (actual[now], actual_normal[now], estimates)


def check_rebase(rng):
    n = 256
    rotation0 = Rotation.random(n, random_state=rng).as_matrix()
    rotation1 = Rotation.from_rotvec(rng.normal(0., .07, (n, 3))).as_matrix() @ rotation0
    origin0 = rng.normal(0., 300., (n, 3))
    origin1 = origin0 + rng.normal(0., 3., (n, 3))
    point = origin0 + rotate(rotation0, rng.normal(0., 15., (n, 3)))
    orientation = Rotation.random(n, random_state=rng).as_matrix()
    velocity = rng.normal(0., 20., (n, 3))
    angular_velocity = rng.normal(0., .35, (n, 3))
    anchor_velocity = rng.normal(0., 10., (n, 3))
    anchor_angular_velocity = rng.normal(0., .2, (n, 3))

    previous_local_point = rotate(rotation0.swapaxes(-1, -2), point - origin0)
    direct = origin1 + rotate(rotation1, previous_local_point)
    inverse = rotation1.swapaxes(-1, -2)
    relative = rotate(inverse, point - origin1)
    relative_rotation = inverse @ orientation
    transported_velocity = anchor_velocity + np.cross(anchor_angular_velocity, point - origin1)
    relative_velocity = rotate(inverse, velocity - transported_velocity)
    relative_angular = rotate(inverse, angular_velocity - anchor_angular_velocity)
    rebased = origin1 + rotate(rotation1, relative)

    point_error = np.linalg.norm(rebased - point, axis=-1)
    rotation_error = np.linalg.norm(rotation1 @ relative_rotation - orientation, axis=(-2, -1))
    velocity_error = np.linalg.norm(transported_velocity + rotate(rotation1, relative_velocity) - velocity, axis=-1)
    angular_error = np.linalg.norm(anchor_angular_velocity + rotate(rotation1, relative_angular) - angular_velocity, axis=-1)
    assert max(point_error.max(), rotation_error.max(), velocity_error.max(), angular_error.max()) < 1e-10
    return {
        "frame_updates": n,
        "unchanged_local_coordinates_position_jump_mm": statistics(np.linalg.norm(direct - point, axis=-1)),
        "rebased_max_position_jump_mm": float(point_error.max()),
        "rebased_max_rotation_matrix_error": float(rotation_error.max()),
        "rebased_max_linear_velocity_jump_mm_s": float(velocity_error.max()),
        "rebased_max_angular_velocity_jump_rad_s": float(angular_error.max()),
    }


def figure(example):
    surface, normal, estimates = example
    span = np.linalg.norm(surface[-1] - surface[0], axis=-1)
    station = int(span.argmax())
    point = surface[:, station]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("Normal offsets at one moving station", "Surface height error over time"))
    color = ("#bd633d", "#7f639c", "#217589")
    fig.add_trace(go.Scatter3d(x=point[:, 0], y=point[:, 1], z=point[:, 2], mode="lines",
                              name="Simulated surface", line=dict(width=7, color="#25323c")), row=1, col=1)
    for method, (p, n), rgb in zip(METHODS, estimates, color):
        true_normal = normal[:, station]
        height = signed_height(point, true_normal, p[:, station], n[:, station])
        # Render the same normal residual as the right panel. Tangential plane origins are arbitrary.
        xyz = point + height[:, None] * true_normal
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                  name=method, legendgroup=method, line=dict(width=4, color=rgb)), row=1, col=1)
        fig.add_trace(go.Scatter(x=TICKS * DT, y=np.abs(height), mode="lines", name=method,
                                 legendgroup=method, showlegend=False, line=dict(color=rgb)), row=1, col=2)
    fig.update_scenes(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data")
    fig.update_xaxes(title_text="Physical time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Absolute height error (mm)", row=1, col=2)
    fig.update_layout(title=f"Knee surface timing replay · material station {STATIONS[station]}",
                      template="plotly_white", height=610, legend=dict(orientation="h", y=-.14),
                      margin=dict(l=42, r=25, t=100, b=105))
    fig.write_html(OUTPUT / "ANCHOR_3D.html", include_plotlyjs=True)


def summarize(rows, names):
    values = np.asarray(rows)
    return {name: {"height_mm": statistics(values[:, i, 0]), "normal_deg": statistics(values[:, i, 1])}
            for i, name in enumerate(names)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    geometry = KneeGeometry(ASSET)
    results = [replay(geometry, rng) for _ in range(RUNS)]
    stationary = replay(geometry, rng, stationary=True, biased=False)
    report = {
        "scope": "Offline kinematic replay with synthetic local plane readings. No contact-controller execution.",
        "geometry_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "seed": SEED, "motions": RUNS, "stations": STATIONS.tolist(), "samples_per_station": len(TICKS),
        "atlas_interval_s": ATLAS_INTERVAL * DT, "atlas_delay_s": ATLAS_DELAY * DT,
        "local_interval_s": LOCAL_INTERVAL * DT, "local_delay_s": LOCAL_DELAY * DT,
        "surface_height_bias_range_mm": [-6., 6.], "surface_tilt_components_range_deg": [-8., 8.],
        "local_height_noise_sd_mm": .25, "local_normal_noise_component_sd_deg": .5,
        "now": summarize([row[0] for row in results], METHODS),
        "same_packet_at_capture": summarize([row[1] for row in results], ("Capture-time FK", "Current FK")),
        "zero_motion_zero_bias": summarize([stationary[0]], METHODS),
        "required_body_displacement_when_scan_pauses_0_5s_mm": statistics([row[2] for row in results]),
        "stationary_pause_max_mm": float(stationary[2].max()),
        "rebase": check_rebase(rng),
    }
    (OUTPUT / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    figure(results[0][3])


if __name__ == "__main__":
    main()
