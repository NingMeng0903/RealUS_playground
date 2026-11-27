"""Simulation gates. Evaluation assets are exercised only in their own process."""

import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _isolated(code, *, allow=False):
    env = os.environ.copy()
    env.pop("AGPS_EVAL_HOLDOUT_ALLOWED", None)
    if allow:
        env["AGPS_EVAL_HOLDOUT_ALLOWED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), str(ROOT.parents[2]), env.get("PYTHONPATH"),
    )))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_holdout_denied_at_import_and_after_revocation():
    _isolated('''
try:
    import sim.holdout_eval_assets
except RuntimeError as error:
    assert str(error) == 'Holdout assets forbidden during training'
else:
    raise AssertionError('unauthorized import succeeded')
''')
    _isolated('''
import os
from sim import holdout_eval_assets as assets
os.environ.pop('AGPS_EVAL_HOLDOUT_ALLOWED')
for loader in (assets.load_blender_legacy_routes, assets.load_blender_nysora_thigh):
    try:
        loader()
    except RuntimeError as error:
        assert str(error) == 'Holdout assets forbidden during training'
    else:
        raise AssertionError('revoked-loader authorization was ignored')
''', allow=True)


def test_training_import_and_data_access_lack_holdout():
    _isolated('''
import os
import sys
import numpy as np

def audit(event, args):
    if event not in ('open', 'os.listdir', 'os.scandir') or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(raw).lower()
    if any(part in path for part in (
        'blender_holdout', 'blender_topology_holdout', 'smplx_vascular_retarget_audit',
        'arterial_paths.json', 'sim/holdout_eval_assets',
    )):
        raise AssertionError(f'training tried to access holdout: {path}')

sys.addaudithook(audit)
from sim.anatomy_assets import TrainingAnatomyBank
from sim.metric_graph_gp import MetricGraphAnatomyGenerator
from sim.smplx_dynamic_atlas import SMPLXLegKinematics
bank = TrainingAnatomyBank.load()
for source_id in bank.source_ids:
    assert bank.source(source_id).supported.any()
gen = MetricGraphAnatomyGenerator(bank)
graph = gen.graph('medseg_right')
assert gen.audit_centers(graph, graph.knot_xyz_mm).checked_points > 0
kin = SMPLXLegKinematics()
assert np.isfinite(kin.forward_pose(None, np.zeros(4), chart_points_mm=[[0., 0., 12.]])[0]).all()
assert 'sim.holdout_eval_assets' not in sys.modules
''')


def test_permuting_role_labels_and_input_order_is_bit_identical():
    from sim.physics_ultrasound_kernel import (
        PhysicsUltrasoundKernel, ProbePose, UltrasoundTruth, VesselTruth,
    )

    truth = (
        VesselTruth(np.array([-6., 0., 30.]), 8., 0.125, 0.025, role=0, noise_key=173),
        VesselTruth(np.array([5., 0., 31.]), 5., 0.02, 0.14, role=1),
        VesselTruth(np.array([14., 0., 33.]), 5., 0.11, 0.02, role=5, noise_key=736),
    )
    relabeled = tuple(replace(v, role=r) for r, v in zip((0, 5, 1), truth[::-1], strict=True))
    kernel = PhysicsUltrasoundKernel(seed=23)
    pose = ProbePose(np.zeros(3), np.eye(3), dwell_s=0.45)
    for mode in ("centroid_only", "centroid_caliber", "full_physical"):
        a = kernel.render(UltrasoundTruth(truth), pose, 4.5, 1, mode).echoes
        b = kernel.render(UltrasoundTruth(relabeled), pose, 4.5, 1, mode).echoes
        assert len(a) == 3 and np.all(np.diff(a[:, 0]) > 0)
        assert a.tobytes() == b.tobytes()
        assert np.isfinite(a[:, :2]).all()
        if mode == "centroid_only":
            assert np.isnan(a[:, 2:]).all()
        elif mode == "centroid_caliber":
            assert np.isfinite(a[:, 2]).all() and np.isnan(a[:, 3:]).all()
        else:
            assert np.isfinite(a).all()
    short_dwell = replace(pose, dwell_s=0.34)
    assert np.isnan(kernel.render(UltrasoundTruth(truth), short_dwell, 0., 1).echoes[:, 3:]).all()
    # Neither suppressed pulse nor calibration nuisance changes centroid noise.
    acoustic_change = replace(truth[1], pulse_amplitude=0.33, pulse_amp_std=0.08)
    a = kernel.render(UltrasoundTruth(truth), short_dwell, 2.5, 11, "centroid_only").echoes
    b = kernel.render(UltrasoundTruth((truth[0], acoustic_change, truth[2])),
                      short_dwell, 2.5, 11, "centroid_only").echoes
    assert a.tobytes() == b.tobytes()
    print("\nPublic echoes: byte-identical under order/role permutations in all modes")


def test_world_anchored_ghost_extinguishes_after_rewindow():
    from sim.physics_ultrasound_kernel import (
        PhysicsUltrasoundKernel, ProbePose, SpecularReflector, UltrasoundTruth,
    )

    reflector = SpecularReflector(
        np.array([0., 0., 35.]), np.array([0., 0., -1.]), np.array([0., 0., 20.]),
    )
    truth = UltrasoundTruth((), reflectors=(reflector,))
    kernel = PhysicsUltrasoundKernel(seed=42)
    head_on = ProbePose(np.zeros(3), np.eye(3))
    # Keep the reflection point in view: translate origin and rotate the probe
    # together so the re-window test cannot pass from depth FOV loss alone.
    rotation = Rotation.from_euler("y", 45, degrees=True).as_matrix()
    rewindow = ProbePose(reflector.point_mm - 35. * rotation[:, 2], rotation)
    before = kernel.render(truth, head_on, 2., 0, "centroid_only").echoes
    after = kernel.render(truth, rewindow, 2., 0, "centroid_only").echoes
    assert len(before) == 1
    assert len(after) == 0 or abs(after[0, 0] - before[0, 0]) > 5.
    print("\nSpecular ghost: displaced or extinguished under a 45-degree re-window")


def test_paired_noise_and_force_dwell_cues():
    from sim.physics_ultrasound_kernel import (
        PhysicsUltrasoundKernel, ProbePose, UltrasoundTruth, VesselTruth,
    )

    pose = ProbePose(np.zeros(3), np.eye(3), dwell_s=0.45)
    vein = VesselTruth(np.array([-6., 0., 35.]), 8., 0.24, 0.02, 0.10, 0.02, noise_key=871)
    artery = VesselTruth(np.array([6., 0., 35.]), 5., 0.04, 0.14, 0.05, 0.04, noise_key=123)
    kernel = PhysicsUltrasoundKernel(seed=17)
    truth = UltrasoundTruth((vein,))
    transported = UltrasoundTruth((replace(vein, center_mm=vein.center_mm + [2., 0., 0.]),))
    base = kernel.render(truth, pose, 4.5, 3).echoes
    moved = kernel.render(transported, pose, 4.5, 3).echoes
    np.testing.assert_allclose(moved - base, [[2., 0., 0., 0., 0.]], rtol=0, atol=1e-12)

    truth = UltrasoundTruth((vein, artery))
    nominal = kernel.render(truth, pose, 2.5, 8).echoes
    assert np.isnan(nominal[:, 3]).all() and np.isfinite(nominal[:, 4]).all()
    early = kernel.render(truth, replace(pose, dwell_s=0.349), 4.5, 8).echoes
    assert np.isfinite(early[:, 3]).all() and np.isnan(early[:, 4]).all()
    measurements = np.stack([kernel.render(truth, pose, 4.5, i).echoes for i in range(300)])
    np.testing.assert_allclose(measurements[:, :, 3:].mean(axis=0),
                               [[-0.48, 0.02], [-0.08, 0.14]], rtol=0, atol=0.025)


def test_bone_and_fascia_shadow_and_merge():
    from sim.physics_ultrasound_kernel import (
        PhysicsUltrasoundKernel, ProbePose, UltrasoundTruth, VesselTruth,
    )

    kernel = PhysicsUltrasoundKernel(seed=42)
    probe = ProbePose(np.zeros(3), np.eye(3))
    a = VesselTruth(np.array([0., 0., 40.]), 6.)
    b = VesselTruth(np.array([0.5, 0., 40.5]), 6.)
    scene = UltrasoundTruth((a,))
    blocker = np.array([[[-8., -8., 20.], [8., -8., 20.], [0., 8., 20.]]])
    assert len(kernel.render(scene, probe, 2., 0).echoes) == 1
    assert len(kernel.render(UltrasoundTruth((a, b)), probe, 2., 0).echoes) == 1
    assert len(kernel.render(replace(scene, bone_triangles_mm=blocker), probe, 2., 0).echoes) == 0
    assert len(kernel.render(replace(scene, fascia_triangles_mm=blocker[:, ::-1]), probe, 2., 0).echoes) == 0
    assert len(kernel.render(replace(scene, bone_triangles_mm=blocker * [1., 1., 3.]), probe, 2., 0).echoes) == 1


def test_harmonic_round_trip_and_moving_smplx_pose_smoothness():
    import pytest
    from sim.smplx_dynamic_atlas import SMPLXLegKinematics

    kinematics = SMPLXLegKinematics()
    chart_points = np.array([[-24., 0., 12.], [0., 12., 18.], [24., -12., 18.]])
    xi = kinematics.chart.xi(chart_points)
    assert kinematics.material_support(xi).all()
    neutral, _, _ = kinematics.forward_pose(None, np.zeros(4), material_coordinates=xi)
    np.testing.assert_allclose(neutral, kinematics.chart.world_from_xi(xi), rtol=0, atol=3e-4)
    np.testing.assert_allclose(neutral, kinematics.chart.world(chart_points), rtol=0, atol=3e-4)
    unreachable = xi.copy()
    unreachable[:, 1] = 10.
    assert not kinematics.material_support(unreachable).any()
    with pytest.raises(ValueError):
        kinematics.forward_pose(None, np.zeros(4), material_coordinates=unreachable)

    q = np.deg2rad([7., -4., 2., 25.])
    pos, frame, jac = kinematics.forward_pose(None, q, material_coordinates=xi)
    np.testing.assert_allclose(frame.transpose(0, 2, 1) @ frame,
                               np.broadcast_to(np.eye(3), frame.shape), rtol=0, atol=1e-8)
    assert (np.linalg.det(frame) > 0.).all()
    assert np.max(np.linalg.norm(pos - neutral, axis=1)) > 10.
    independent_directions = 1e-5 * np.eye(4)
    truth_jac = np.stack([
        (kinematics.forward_pose(None, q + d, material_coordinates=xi)[0]
         - kinematics.forward_pose(None, q - d, material_coordinates=xi)[0]) / 2e-5
        for d in independent_directions
    ], axis=-1)
    max_error = np.max(np.abs(jac - truth_jac))
    assert max_error < 2e-3
    print(f"\nSMPL-X material transport: max independent pose derivative error {max_error:.3e} mm/rad")


def test_tracker_lag_jitter_and_ou_are_separate():
    from sim.smplx_dynamic_atlas import DynamicMotionProfile

    t = 1.0
    lag = 0.06
    noiseless = DynamicMotionProfile(7, jitter_deg=0., latency_s=lag)
    truth, tracker = noiseless.sample(t)
    assert abs(tracker.measurement_time_s - (t - lag)) < 1e-12
    np.testing.assert_array_equal(tracker.q_hat, noiseless.truth_at(t - lag).q)
    assert np.linalg.norm(truth.q - tracker.q_hat) > 0.
    noisy = DynamicMotionProfile(7, jitter_deg=[0., 2., 4., 2.], latency_s=lag)
    np.testing.assert_array_equal(truth.q, noisy.truth_at(t).q)
    noisy_tracker = noisy.tracker_at(t)
    np.testing.assert_allclose(noisy_tracker.Sigma_q,
                               np.diag(np.deg2rad([0., 2., 4., 2.])**2))
    np.testing.assert_array_equal(noisy_tracker.q_hat, noisy.tracker_at(t).q_hat)
    assert 65. <= noisy.heart_rate_bpm <= 85.
    decay, covariance = noisy.ou_transition(1.5)
    np.testing.assert_allclose(decay, np.exp(-1.))
    np.testing.assert_allclose(covariance, (1. - np.exp(-2.)) * np.eye(3))
    initial = noisy.ou_initial((5,))
    np.testing.assert_array_equal(noisy.ou_step(initial, 0.), initial)


def test_training_assets_preserve_observed_support_and_extract_dfa():
    from sim.anatomy_assets import (
        FAMILY_HELD_OUT_ATLASES, LOAO_BODYPARTS, LOAO_OPEN3D,
        NYSORA_STATIONS, OPEN3D_REGISTERED_PATH, TrainingAnatomyBank, VesselRole,
    )

    bank = TrainingAnatomyBank.load()
    assert len(bank.source_ids) == 4
    assert len(NYSORA_STATIONS) == 5
    for split in (LOAO_BODYPARTS, LOAO_OPEN3D):
        assert "same atlas family" in split.relationship
        assert not set(split.train_source_ids).intersection(split.test_source_ids)
    train_families = {bank.source(s).provenance["source_family"]
                      for s in FAMILY_HELD_OUT_ATLASES.train_source_ids}
    test_families = {bank.source(s).provenance["source_family"]
                     for s in FAMILY_HELD_OUT_ATLASES.test_source_ids}
    assert not train_families.intersection(test_families)
    graph = bank.native_graph
    # The mounted native archive contains 117/118 (not 330/322 in the
    # planning summary); assert its available voxel-contact semantics.
    assert len(graph["nodes"]) == 117
    assert len(graph["sampled_edges"]) + len(graph["verified_junction_edges"]) == 118
    assert {tuple(row["labels"]) for row in graph["verified_junction_edges"]} == {
        (21, 35), (21, 36), (23, 27), (23, 29), (23, 39),
    }
    for key in bank.source_ids:
        source = bank.source(key)
        np.testing.assert_array_equal(source.h_mm, np.arange(-480., -1281., -5.))
        assert source.centers_mm.shape == (*source.h_mm.shape, 6, 3)
        np.testing.assert_array_equal(source.supported, np.isfinite(source.centers_mm).all(axis=-1))
        assert np.isnan(source.centers_mm[~source.supported]).all()
        h, _ = np.where(source.supported)
        np.testing.assert_allclose(source.centers_mm[source.supported, 1], source.h_mm[h], atol=1e-6)
        if "bodyparts" in key.lower():
            assert not source.supported[:, VesselRole.DFA].any()
    open3d = bank.source(next(key for key in bank.source_ids if "open3d" in key.lower()))
    dfa = open3d.supported[:, VesselRole.DFA]
    assert dfa.sum() >= 10
    assert np.all((-880 <= open3d.h_mm[dfa]) & (open3d.h_mm[dfa] <= -480))
    assert open3d.provenance["added_role_registration_file"] == str(OPEN3D_REGISTERED_PATH)
    print(f"\nOpen3D DFA: {dfa.sum()} source-supported stations")


def test_sparse_prior_has_shared_junctions_in_500_anatomies():
    from scipy.sparse import isspmatrix_csc
    from sim.anatomy_assets import TrainingAnatomyBank
    from sim.metric_graph_gp import MetricGraphAnatomyGenerator

    bank = TrainingAnatomyBank.load()
    generator = MetricGraphAnatomyGenerator(bank)
    rng = np.random.default_rng(953)
    total = junction_draws = 0
    for key in bank.source_ids:
        batch = generator.sample_batch(key, range(125), rng, geometry_policy="raw")
        graph = generator.graph(key)
        assert batch and isspmatrix_csc(batch[0].prior.precision)
        assert batch[0].prior.precision.nnz < 32 * batch[0].prior.layout.size
        endpoints = {}
        for edge in graph.edges:
            endpoints.setdefault(edge.vertex_u, []).append((edge.id, 0))
            endpoints.setdefault(edge.vertex_v, []).append((edge.id, -1))
        junctions = [members for members in endpoints.values() if len(members) >= 3]
        assert junctions, f"No measured or modeled branching vertices for {key}"
        for anatomy in batch:
            assert anatomy.attempts == 1 and anatomy.sampling_law == "raw_gaussian"
            for members in junctions:
                ends = np.stack([anatomy.edge_centers_mm(e)[i] for e, i in members])
                np.testing.assert_array_equal(ends, np.broadcast_to(ends[0], ends.shape))
                junction_draws += 1
            assert np.isnan(anatomy.centers_mm[~anatomy.supported]).all()
            total += 1
    assert total == 500
    print(f"\nKirchhoff trace: {total} draws, {junction_draws} common junction endpoints; max gap 0 mm")


def test_geometry_conditioning_and_independent_bone_proxy_check():
    from sim.anatomy_assets import BODY_PARTS3D_LEFT_MIRRORED, TrainingAnatomyBank
    from sim.metric_graph_gp import MetricGraphAnatomyGenerator

    bank = TrainingAnatomyBank.load()
    source_id = next(s for s in bank.source_ids if "medseg" in s.lower())
    generator = MetricGraphAnatomyGenerator(bank)
    batch = generator.sample_batch(source_id, range(64), np.random.default_rng(4),
                                   geometry_policy="reject", max_attempts=256)
    occ = bank.bone_occupancy
    get = occ.__getitem__ if hasattr(occ, "__getitem__") else lambda key: getattr(occ, key)
    axes = [np.asarray(get(f"{dim}_mm")) for dim in "xyz"]
    counts = np.asarray(get("count"))
    threshold = 0.55 * get("cohort_size")

    def raw_bone_counts(anatomy):
        # Query the raw raster independently, on exactly the exported
        # material centerline points used by the generator's audit.
        xyz = generator.audit_points(anatomy.graph, anatomy.knot_centers_mm,
                                     prior=anatomy.prior, latent=anatomy.latent)
        assert len(xyz) == anatomy.audit.checked_points
        known = np.ones(len(xyz), dtype=bool)
        nearby = []
        for axis, values in zip(axes, xyz.T, strict=True):
            known &= (values >= axis[0] - (axis[1]-axis[0])/2.) & (
                values < axis[-1] + (axis[-1]-axis[-2])/2.
            )
            j = np.searchsorted(axis, values).clip(1, len(axis)-1)
            nearby.append(np.where(values - axis[j-1] < axis[j] - values, j-1, j))
        hits = counts[tuple(idx[known] for idx in nearby)] >= threshold
        return len(xyz), int(known.sum()), int(hits.sum())

    points, known_count, nonbone_count, attempts, skin_known_count = 0, 0, 0, 0, 0
    for anatomy in batch:
        assert anatomy.sampling_law == "geometry_conditioned_gaussian"
        assert anatomy.attempts == anatomy.rejections.draws + 1
        assert anatomy.audit.passes_known_checks
        attempts += anatomy.attempts
        skin_known_count += anatomy.audit.checked_points - anatomy.audit.skin_unknown
        n, known, hits = raw_bone_counts(anatomy)
        assert n - known == anatomy.audit.bone_unknown == 0
        assert hits == anatomy.audit.in_excluded_bone == 0
        points += n
        known_count += known
        nonbone_count += known - hits
    assert known_count > 0 and skin_known_count == sum(a.audit.checked_points for a in batch)
    assert known_count / points > 0.99 and nonbone_count / known_count > 0.99
    print(f"\nMedSeg ordinary: {len(batch)} conditioned draws / {attempts} proposals; "
          f"bone raster coverage={known_count/points:.3%}, known points outside excluded bone="
          f"{nonbone_count/known_count:.3%}; {skin_known_count} known skin checks")

    # Also check popliteal material arcs where Cartesian knot chords can
    # cut through condyles. Count all hits on the exported raw proposals.
    stress = generator.sample_batch(BODY_PARTS3D_LEFT_MIRRORED, range(4),
                                    np.random.default_rng(58), "stress", geometry_policy="raw")
    for anatomy in stress:
        n, known, hits = raw_bone_counts(anatomy)
        assert n - known == anatomy.audit.bone_unknown
        assert hits == anatomy.audit.in_excluded_bone


def test_authorized_holdout_keeps_raw_gaps_and_separates_inferred_arteries():
    report = _isolated('''
import numpy as np
from sim.holdout_eval_assets import load_blender_legacy_routes, load_blender_nysora_thigh

legacy = load_blender_legacy_routes()
np.testing.assert_array_equal(legacy.physically_reachable_mask, ~legacy.inside_bone)
assert set(legacy.routes) == {
    'saphenofemoral', 'femoral', 'popliteal', 'calf', 'groin_to_distal_calf',
}
np.testing.assert_array_equal(legacy.h_mm, np.arange(-510., -1281., -5.))
route = legacy.routes['groin_to_distal_calf']
assert len(route.h_mm) == 147
assert np.sum(~route.physically_reachable_mask[:, 0]) == 10
assert np.sum(route.physically_reachable_mask[:, 0] & route.supported[:, 0]) == 137
thigh = load_blender_nysora_thigh()
np.testing.assert_array_equal(thigh.h_mm, np.arange(-510., -876., -5.))
gap = (thigh.h_mm >= -545) & (thigh.h_mm <= -520)
assert gap.sum() == 6 and not thigh.supported[gap, 1].any()
assert np.isnan(thigh.centers_mm[gap, 1]).all()
dfa = thigh.supported[:, 5]
assert dfa.any()
assert np.all(thigh.support_source[dfa, 5] == 2)
assert not thigh.directly_supported[:, 5].any()
assert thigh.station_role_support.shape == (len(thigh.h_mm), 5)
assert np.all(thigh.station_named_role_support <= thigh.station_role_support)
pop = np.flatnonzero(thigh.h_mm == -850.)[0]
assert thigh.station_role_support[pop, 0] and not thigh.station_named_role_support[pop, 0]
assert thigh.station_named_role_support[pop, 4]
assert tuple(thigh.named_roles[pop, :2]) == ("R_POPV", "R_PA")
assert not thigh.reachability_known[:, 5].any()
assert not thigh.physically_reachable_mask[:, 5].any()
assert np.isnan(thigh.bone_distance_mm[..., 5]).all()
print('Holdout: six proximal raw arterial rows missing, anatomical DFA stored separately')
''', allow=True)
    print(f"\n{report.strip()}")
