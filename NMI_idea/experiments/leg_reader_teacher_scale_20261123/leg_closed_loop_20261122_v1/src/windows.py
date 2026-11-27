"""Train-only design of recoverable acoustic windows on a fixed layout."""

from dataclasses import dataclass, replace

import numpy as np

import motion
from scan_kernel import DEPTH_MAX, DEPTH_MIN, LATERAL, Trial


@dataclass(frozen=True)
class AcousticWindow:
    station: int
    sector: int
    alternate_sector: int
    competing_role: int
    validation_ticks: int
    reference_resolved: int
    alternate_resolved: int
    reference_mixed: int
    alternate_mixed: int


def _visible(points, gate):
    return ((np.abs(points[:, 0]) < LATERAL - 2.) & (points[:, 1] > DEPTH_MIN + 2.)
            & (points[:, 1] < DEPTH_MAX - 3.) & (gate < .64))


def _sensor_pair(trial, poses, role, station, angles, count):
    scores = []
    for angle, (points, gate) in zip(angles, poses, strict=True):
        resolved, merged = 0, 0
        coupling = 1. - .12 * abs(np.sin(angle * motion.ANGLE_STEP))
        for tick in range(1, count + 1):
            z, labels, _ = trial.observe(points, gate, tick, station=station, sector=angle, contact_quality=coupling)
            target = np.flatnonzero(labels == 0)
            rival = np.flatnonzero(labels == role)
            if len(target) and len(rival):
                resolved += int(np.linalg.norm(z[target[0]] - z[rival[0]]) >= 4.0)
            merged += int(-2 in labels)
        scores.append((resolved, merged))
    return scores


def configure(env, node, spec, case, trial, max_ticks, max_mm):
    """Use private design checks to create examples, never a policy input.

    The selected patch only changes the renderer's anisotropic multiplet
    resolution. Layout, junction positions and other measurement conditions
    remain constant during the scan and during each replay.
    """
    scanner, reader = env.scanner, node.reader
    candidates = []
    stations = np.unique(np.linspace(spec[1], min(spec[2] - 1, spec[1] + 28), 7).astype(int))
    for station in stations:
        prior = env.route.predictive(reader.model, int(station))[:, 0] @ reader.model.prior
        preferred = np.argsort(np.where(scanner.valid[station], prior, -1.))[-2:]
        for angle in preferred:
            angle = int(angle)
            points, gate = scanner._anatomy.training_truth(scanner.surface, trial, int(station), angle)
            if not _visible(points, gate)[0]:
                continue
            for role in np.flatnonzero(_visible(points, gate)[1:4]) + 1:
                dx, dz = points[role] - points[0]
                distance = np.hypot(dx, dz)
                blur_separation = (dx / 14.7) ** 2 + (dz / 11.7) ** 2
                if distance < 4.5 or blur_separation > 1.48:
                    continue
                for turn in (-2, 2):
                    alternate = (angle + turn) % motion.SECTORS
                    if not scanner.valid[station, alternate]:
                        continue
                    comparison = scanner._anatomy.training_truth(scanner.surface, trial, int(station), alternate)
                    if not _visible(*comparison)[[0, role]].all():
                        continue
                    merit = (.28 * min(1., distance / 10.) + .53 * prior[angle]
                             - .16 * blur_separation - .24 * max(comparison[1][[0, role]]))
                    candidates.append((float(merit), int(station), angle, int(role), int(turn),
                                       (points, gate), comparison))
    proof_count = 28
    # Selection uses a separate proof tape. It never inspects the student's
    # future noise tape or changes which ground-truth anatomy is used.
    audit_trial = Trial(case.source, case.tape_seed + 4_000_000_000_000_000_000,
                        case.profile, world_seed=case.world_seed, split=case.split, n=64)
    audit_trial.layout = trial.layout
    for _, station, sector, role, turn, base, other in sorted(candidates, key=lambda row: row[0], reverse=True)[:8]:
        anchor = replace(node, station=station, sector=sector)
        controls = np.array([turn / motion.ANGLE_RANGE, 0., 0.])
        plan = env.material.plan_probe(anchor, spec, 1, controls, max_ticks, max_mm, use_dird=False)
        if plan is None or abs(motion._sector_delta(sector, plan.goal[1])) < 1.55:
            continue
        alternate = (sector + turn) % motion.SECTORS
        window = AcousticWindow(station, sector, alternate, role, proof_count, 0, 0, 0, 0)
        audit_trial.window = window
        (reference, mix_a), (alternative, mix_b) = _sensor_pair(audit_trial, (base, other), role, station,
                                                                  (sector, alternate), proof_count)
        if alternative >= 5 and alternative >= reference + 3 and mix_a >= mix_b + 2:
            return replace(window, reference_resolved=reference, alternate_resolved=alternative,
                           reference_mixed=mix_a, alternate_mixed=mix_b)
    return None
