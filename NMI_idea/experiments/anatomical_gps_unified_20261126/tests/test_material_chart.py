"""Beta=0 geometry, analytic derivatives, and matched material GP regression."""
import numpy as np
import pytest
from scipy import sparse

from sim import material_chart
from sim.anatomy_assets import MEDSEG_RIGHT, NEUTRAL_SMPLX_SURFACE_PATH, OPEN3DMODEL_RIGHT, TrainingAnatomyBank
from sim.material_chart import RightLegMaterialChart
from sim.metric_graph_gp import _TrainingValidity, MetricGraphAnatomyGenerator, MetricGraphPrior
from sim.smplx_dynamic_atlas import NEUTRAL_BETA, SMPLXLegKinematics


@pytest.fixture(scope='module')
def chart():
    return RightLegMaterialChart()


@pytest.fixture(scope='module')
def bank():
    return TrainingAnatomyBank.load()


def _jacobian(f, x, step=1.e-5):
    return np.stack([(f(x + d) - f(x - d)) / (2. * step) for d in np.eye(x.shape[-1]) * step], axis=-1)


def test_material_chart_round_trip_and_analytic_jacobians(chart):
    grid = np.array([
        [-np.pi + .00003, -1265.4, .22], [np.pi - .00003, -1195.1, .81],
        [-1.17, -900.3, .49], [1.43, -875.6, .28], [.71, -773.8, .58],
        [.11, -550.12, .80], [1.62, -479.11, .32],
    ])
    xyz = chart.xi_to_world(grid)
    assert np.max(abs(chart.world_to_xi(xyz) - grid)) < 1.e-10
    assert np.max(abs(chart.xi_to_world(chart.world_to_xi(xyz)) - xyz)) < 1.e-10
    wrapped = grid + [4. * np.pi, 0., 0.]
    np.testing.assert_allclose(chart.xi_to_world(wrapped), xyz, rtol=0., atol=1.e-10)
    latent, _, audit = chart.encode_with_audit(xyz)
    assert audit['n_projected_knots'] == 0
    np.testing.assert_allclose(chart.tilde_xi_to_world(latent), xyz, rtol=0., atol=1.e-10)
    for values, f, j in ((grid, chart.xi_to_world, chart.dworld_dxi),
                         (latent, chart.tilde_xi_to_world, chart.dworld_dtilde_xi)):
        max_error = np.max(abs(_jacobian(f, values) - j(values)))
        assert max_error < 1.e-4
        print(f'\n{j.__name__}: maximum central-difference error = {max_error:.3e}')


def test_canonical_material_frame_and_pose_jacobian(chart):
    kin = SMPLXLegKinematics()
    addresses = np.array([[-1.18, -880., -.21], [.51, -850.77, .63], [2.51, -742.15, .48],
                          [-2.65, -603.8, .42], [.74, -506.18, -.72], [1.11, -475., .42]])
    q = np.deg2rad([19., -6., 4., 28.])
    points = chart.tilde_xi_to_world(addresses)
    np.testing.assert_allclose(kin.forward_pose_canonical(np.zeros(4), points)[0], points, rtol=0., atol=1.e-10)
    beta_shape = np.zeros_like(NEUTRAL_BETA)
    beta_shape[:3] = [.3, -.2, .1]
    assert np.max(abs(kin.forward_pose_canonical(np.zeros(4), points, beta=beta_shape)[0] - points)) > .1
    for beta in (NEUTRAL_BETA, beta_shape):
        phi, frame, jq, jxi = kin.pose_points_and_material_frame(beta, q, chart, addresses)
        np.testing.assert_allclose(phi, kin.forward_pose_canonical(q, points, beta=beta)[0], rtol=0., atol=1.e-10)
        np.testing.assert_allclose(frame.transpose(0, 2, 1) @ frame, np.broadcast_to(np.eye(3), frame.shape), atol=1.e-12)
        np.testing.assert_allclose(np.linalg.det(frame), np.ones(len(frame)), atol=1.e-12)
        q_numeric = _jacobian(lambda pose: kin.forward_pose_canonical(pose, points, beta=beta)[0], q)
        max_q_error = np.max(abs(jq - q_numeric))
        assert max_q_error < 1.e-5
        print(f'\nCanonical pose Jacobian [{points[:, 1].min():g}, {points[:, 1].max():g}] mm: {max_q_error:.3e} mm/rad')
        xi_numeric = _jacobian(lambda a: kin.forward_pose_canonical(q, chart.tilde_xi_to_world(a), beta=beta)[0], addresses[:-1])
        assert np.max(abs(jxi[:-1] - xi_numeric)) < 1.e-4


def test_material_dual_precision_and_world_jacobian(chart, bank):
    prior = MetricGraphAnatomyGenerator(bank).prior(MEDSEG_RIGHT)
    cartesian = MetricGraphPrior(prior.graph, coordinate_mode='cartesian_xyz')
    assert sparse.isspmatrix_csc(prior.prior_precision_csc())
    np.testing.assert_array_equal(prior.precision.indices, cartesian.precision.indices)
    np.testing.assert_array_equal(prior.precision.indptr, cartesian.precision.indptr)
    np.testing.assert_array_equal(prior.kirchhoff_laplacian().toarray(), cartesian.kirchhoff_laplacian().toarray())
    stiffness = prior.kirchhoff_laplacian()
    np.testing.assert_allclose(stiffness @ np.ones(stiffness.shape[0]), 0., atol=1.e-12)
    assert prior.knot_design_matrix().shape == (3 * len(prior.graph.knot_xyz_mm), prior.layout.size)
    rng = np.random.default_rng(142)
    state = prior.sample_latent(rng)
    direction = rng.standard_normal(prior.layout.size)
    jac = prior.dworld_dx(state, chart)
    numeric = (prior.decode_knots_to_world(state + 1.e-5 * direction, chart)
               - prior.decode_knots_to_world(state - 1.e-5 * direction, chart)).ravel() / 2.e-5
    assert sparse.isspmatrix_csr(jac)
    error = np.max(abs(jac @ direction - numeric))
    print(f'\nMaterial GP dworld/dx directional error = {error:.3e}')
    assert error < 1.e-4


def test_material_sweep_independent_validity_and_unmodified_source(chart, bank):
    generator = MetricGraphAnatomyGenerator(bank)
    source = bank.source(OPEN3DMODEL_RIGHT)
    before = source.centers_mm.tobytes()
    raw = source.centers_mm[source.supported]
    _, _, audit = chart.encode_with_audit(raw)
    assert audit['n_projected_knots'] > 0
    assert source.centers_mm.tobytes() == before and not source.centers_mm.flags.writeable
    swept = [generator.shift_sweep_material(source.source_id, offset) for offset in range(0, 41, 10)]
    assert all(d.sampling_law == 'deterministic_intervention' and d.audit.passes_known_checks
               and d.audit.all_locations_known for d in swept)
    nominal = swept[0].prior.material_knots(swept[0].latent)
    scale = chart.nominal_reference_radius(nominal[:, 1])
    for offset, draw in enumerate(swept):
        shift = draw.prior.material_knots(draw.latent) - nominal
        np.testing.assert_allclose(scale * shift[:, 0], offset * 10., atol=1.e-11)
        np.testing.assert_array_equal(shift[:, 1:], np.zeros_like(shift[:, 1:]))
        for edge in draw.graph.edges:
            line = draw.edge_polyline_mm(edge.id)
            np.testing.assert_array_equal(line[[0, -1]], draw.knot_centers_mm[edge.knot_ids[[0, -1]]])
            assert np.linalg.norm(np.diff(line, axis=0), axis=1).max() <= 1.5 + 1.e-10
    assert source.centers_mm.tobytes() == before
    print('\nOpen3D circumferential sweep 0/10/20/30/40 mm: all sampled centerlines valid')


def test_canonical_skin_coverage_and_full_leg_shape_channel(bank):
    generator = MetricGraphAnatomyGenerator(bank, domain='full_leg', shape_std=.4)
    assert generator._validity.lower_body_boundary_edges == 0
    points = np.column_stack((-100. * np.ones(1351), np.arange(-1350., 1.), np.zeros(1351)))
    assert generator.classify_points(points)[1].all()
    graph = generator.graph(OPEN3DMODEL_RIGHT)
    assert graph.h_mm.min() == -1280. and graph.h_mm.max() == -480.
    anatomy = generator.sample_anatomy(MEDSEG_RIGHT, None, np.random.default_rng(34), geometry_policy='raw')
    assert np.all(anatomy.shape_beta[:10] != 0.) and not anatomy.shape_beta[10:].any()
    assert anatomy.audit.all_locations_known


def test_thin_corridor_preserves_independent_raw_bone_boundary(chart, bank):
    xi = np.array([np.deg2rad(-173.), -880., .925])
    assert chart.corridor_supported(*xi[:2])
    xyz = chart.xi_to_world(xi)
    occ = bank.bone_occupancy
    cell = tuple(np.abs(axis - value).argmin() for axis, value in zip((occ.x_mm, occ.y_mm, occ.z_mm), xyz, strict=True))
    assert occ.count[cell] / occ.cohort_size < .55
    inside, skin_known, bone_hit, bone_known = MetricGraphAnatomyGenerator(bank).classify_points(xyz[None])
    assert inside[0] and skin_known[0] and bone_known[0] and not bone_hit[0]
    np.testing.assert_allclose(chart.dworld_dxi(xi), _jacobian(chart.xi_to_world, xi), rtol=0., atol=1.e-4)
    assert not chart.corridor_supported(np.deg2rad(38.), -480.)


def test_material_bone_guard_is_continuous_across_voxel_slabs(chart, bank):
    generator = MetricGraphAnatomyGenerator(bank)
    eps = 1.e-6
    theta, boundary, depth = np.deg2rad(-42.), -856.5, .925
    xi = np.array([[theta, boundary - eps, depth], [theta, boundary + eps, depth]])
    points = chart.xi_to_world(xi)
    jump = np.linalg.norm(points[1] - points[0])
    assert jump < 1.e-3
    inside, skin_known, bone_hit, bone_known = generator.classify_points(points)
    assert np.all(inside & skin_known & bone_known & ~bone_hit)

    prior = generator.prior(OPEN3DMODEL_RIGHT)
    knot = 286
    nominal = prior.nominal_knots[knot]
    xyz_rows = slice(3 * knot, 3 * knot + 3)
    start = prior.layout.relative.start + 3 * knot
    latent_cols = np.arange(start, start + 3)
    scale = prior.knot_design_matrix()[xyz_rows, :][:, latent_cols].diagonal()
    direction = np.zeros_like(prior.mean)
    direction[start + 1] = 1.
    state = prior.mean.copy()
    state[start] = (theta - nominal[0]) / scale[0]
    state[start + 2] = (chart.eta_from_d(depth) - nominal[2]) / scale[2]
    edges = bank.bone_occupancy.y_mm - 1.5
    edges = edges[(edges >= -910.) & (edges <= -825.)]
    assert boundary in edges
    error = 0.
    for h in edges:
        state[start + 1] = (h - eps - nominal[1]) / scale[1]
        before = prior.decode_knots_to_world(state)[knot]
        after = prior.decode_knots_to_world(state + 2. * eps * direction)[knot]
        jac = (prior.dworld_dx(state) @ direction)[xyz_rows]
        error = max(error, np.linalg.norm(after - before - 2. * eps * jac))
    print(f'\nSlab crossing: chart displacement {jump:.3e} mm; max latent linearization residual {error:.3e} mm')
    assert error < 1.e-6


def test_geometry_loaders_reject_neutral_skin_digest_mismatch(chart, tmp_path, monkeypatch):
    corrupt = tmp_path / NEUTRAL_SMPLX_SURFACE_PATH.name
    # The archive is still readable and carries a canonical schema, but its
    # bytes no longer match the seal. The chart cache is already populated.
    corrupt.write_bytes(NEUTRAL_SMPLX_SURFACE_PATH.read_bytes() + b'altered')
    monkeypatch.setattr(material_chart, 'NEUTRAL_SMPLX_SURFACE_PATH', corrupt)
    with pytest.raises(ValueError, match='neutral skin SHA-256'):
        RightLegMaterialChart()
    with pytest.raises(ValueError, match='neutral skin SHA-256'):
        _TrainingValidity(None, corrupt)
