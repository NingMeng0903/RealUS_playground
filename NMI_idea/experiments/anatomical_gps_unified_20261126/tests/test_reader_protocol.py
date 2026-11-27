"""Controls for Gate 2 station support and label/offset-blind Step 3 acquisition."""
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from reader.split_prior import DynamicScanSimulator, station_windows
from sim.anatomy_assets import (
    BODY_PARTS3D_LEFT_MIRRORED, BODY_PARTS3D_RIGHT, FAMILY_HELD_OUT_ATLASES, LOAO_BODYPARTS,
    LOAO_OPEN3D, MEDSEG_RIGHT, TrainingAnatomyBank,
)
from sim.audit_training_support import _coverage
from sim.metric_graph_gp import MetricGraphAnatomyGenerator
from sim.smplx_dynamic_atlas import SMPLXLegKinematics
from theory.station_identifiability import station_population_bounds


@pytest.fixture(scope='module')
def bank():
    return TrainingAnatomyBank.load()


@pytest.fixture(scope='module')
def kinematics():
    return SMPLXLegKinematics()


@pytest.fixture(scope='module')
def sim(bank, kinematics):
    return DynamicScanSimulator(bank, LOAO_OPEN3D, kinematics=kinematics)


def test_gate2_source_windows_and_declared_split_support(bank, sim, kinematics):
    path = Path(__file__).resolve().parents[1] / 'results' / 'step2_training_support.json'
    archived = json.loads(path.read_text(encoding='utf-8'))['axis_1_original_ct_coverage']
    generator = MetricGraphAnatomyGenerator(bank)
    assert json.loads(json.dumps(_coverage(bank, generator))) == archived

    raw_windows = {source_id: station_windows(generator.graph(source_id)) for source_id in bank.source_ids}
    for source_id, windows in raw_windows.items():
        assert {name: len(h) for name, h in windows.items()} == archived['sources'][source_id][
            'junction_window_role_copresence_rows']
        assert all(not h.flags.writeable for h in windows.values())
    for name, split in (('LOAO_OPEN3D', LOAO_OPEN3D), ('LOAO_BODYPARTS', LOAO_BODYPARTS),
                        ('FAMILY_HELD_OUT_ATLASES', FAMILY_HELD_OUT_ATLASES)):
        simulator = sim if split is LOAO_OPEN3D else DynamicScanSimulator(bank, split, kinematics=kinematics)
        for source_id, expected in archived['splits'][name]['per_test_source_supported_stations'].items():
            assert sorted(simulator.station_ids(source_id)) == expected
            for station in expected:
                assert_array_equal(simulator._windows(source_id)[station], raw_windows[source_id][station])


def _controls(episode):
    """All reader inputs other than the echoes, with array values compared bitwise."""
    return tuple((f.probe_pose.origin_mm.tobytes(), f.probe_pose.rotation.tobytes(),
                  f.probe_pose.speed_mm_s, f.probe_pose.dwell_s, f.f_contact,
                  f.q_hat.tobytes(), f.Sigma_q.tobytes(), f.dt) for f in episode.frames)


def test_offset_geometry_changes_without_reaiming_public_probe_or_tracker(sim):
    source = BODY_PARTS3D_RIGHT
    nominal = sim.episode(source, 42, benchmark='offset')
    expected_controls = _controls(nominal)
    for offset in (10., 20., 30., 40.):
        shifted = sim.episode(source, 42, benchmark='offset', offset_mm=offset)
        assert shifted.h_start_mm == nominal.h_start_mm
        assert _controls(shifted) == expected_controls
    assert any(not np.allclose(a.centers_mm, b.centers_mm, equal_nan=True)
               for a, b in zip(nominal.frames, shifted.frames, strict=True))
    assert any(a.obs.echoes.tobytes() != b.obs.echoes.tobytes()
               for a, b in zip(nominal.frames, shifted.frames, strict=True))
    other_source = sim.episode(BODY_PARTS3D_LEFT_MIRRORED, 42, benchmark='offset')
    assert other_source.h_start_mm == nominal.h_start_mm
    assert _controls(other_source) == expected_controls


def test_shared_height_different_requested_tags_give_identical_reader_inputs(sim):
    s2, s3 = 'S2_SFJ_MICKEY', 'S3_ART_BIF'
    windows = sim._windows(MEDSEG_RIGHT)
    shared = np.intersect1d(windows[s2], windows[s3])
    assert shared.size and windows[s2].size != windows[s3].size
    # Find a matched train-only seed using just the uniform station-site stream;
    # do not render or screen episodes for echo visibility when picking it.
    def opening_height(seed, station):
        site_seed = np.random.SeedSequence(seed, spawn_key=(4,))
        return np.random.default_rng(site_seed).choice(windows[station])

    seed = next(seed for seed in range(128) if opening_height(seed, s2) == opening_height(seed, s3))
    a = sim.episode(MEDSEG_RIGHT, seed, benchmark='station', station_id=s2)
    b = sim.episode(MEDSEG_RIGHT, seed, benchmark='station', station_id=s3)
    assert a.h_start_mm == b.h_start_mm and a.h_start_mm in shared
    assert (a.station_id, b.station_id) == (s2, s3)
    assert _controls(a) == _controls(b)
    for f, g in zip(a.frames, b.frames, strict=True):
        assert f.obs.echoes.tobytes() == g.obs.echoes.tobytes()
        assert f.echo_targets == g.echo_targets
        assert f.centers_mm.tobytes() == g.centers_mm.tobytes()


def test_identical_conditional_station_laws_separate_accuracy_and_f1_bound():
    # Two balanced labels, each with the same (.25, .75) law over two sites.
    population = station_population_bounds(np.array([[.125, .125], [.375, .375]]))
    assert population['ambiguous_cell_mass'] == pytest.approx(1.)
    assert population['oracle_accuracy'] == pytest.approx(.5)
    assert population['macro_f1_upper_bound'] == pytest.approx(2. / 3.)
    # Each one-vs-rest optimum predicts its own tag everywhere. These optima
    # conflict, so their average is not an achievable single-label macro-F1.
    for result in population['per_class']:
        assert result['f1_upper_bound'] == pytest.approx(2. / 3.)
        assert result['precision_at_f1_optimum'] == pytest.approx(.5)
        assert result['recall_at_f1_optimum'] == pytest.approx(1.)
