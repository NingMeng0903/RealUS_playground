import numpy as np
import pytest

from peirastic.contact_qp.geometry import point_normal_row
from peirastic.contact_qp.region_visual import (
    better_half_reference, preferred_relative_alpha, preferred_scan_alpha, region_visual_request,
)
from peirastic.contact_qp.types import ProbeGeometry


def request(quality,sign=1,valid=None):
    n=len(quality)
    return region_visual_request(quality,np.ones(n,dtype=bool) if valid is None else valid,
        np.linspace(0.,1.,n+1),ProbeGeometry.synthetic(image_x_sign=sign),c_min=.8,repair_speed_m_s=.002)


@pytest.mark.parametrize('n',[4,10,17])
@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('index_side',[0,-1])
def test_all_region_geometry_requests_rocking_toward_actual_bad_contact_point(n,sign,index_side):
    q=np.full(n,.9);q[index_side]=.3
    task=request(q,sign)
    index=index_side%n
    assert task.bad_region_indices==(index,)
    assert task.region_deficits[index]==pytest.approx(.5)
    tcp=np.array([0,0,0,0,task.request_rad_s,0.])
    indentation=point_normal_row(ProbeGeometry.synthetic(image_x_sign=sign),task.region_centers_m[index]) @ tcp
    assert indentation>0.
    assert task.alpha_preferred<1.


def test_central_local_defect_is_detected_with_zero_unobservable_direction():
    q=np.full(11,.9);q[5]=.2
    task=request(q)
    assert task.bad_region_indices==(5,)
    assert task.bad_region_positions_m==(0.,)
    assert task.request_rad_s==0.
    assert task.reason=='symmetric_deficit_no_rocking_direction'
    assert task.alpha_preferred==pytest.approx(.4375)


def test_ten_region_central_pair_and_disconnected_symmetric_regions_remain_visible():
    for bad in ((4,5),(0,3,6,9)):
        q=np.full(10,.9);q[list(bad)]=.3
        task=request(q)
        assert task.bad_region_indices==bad
        assert task.request_rad_s==0.
        assert task.deficit_area>0.
        assert task.alpha_preferred<1.


def test_same_left_right_means_different_bad_locations_produce_different_spatial_tasks():
    outer=np.full(10,.9);outer[0]=.3
    inner=np.full(10,.9);inner[4]=.3
    assert np.mean(outer[:5])==np.mean(inner[:5])
    assert np.mean(outer[5:])==np.mean(inner[5:])
    a,b=request(outer),request(inner)
    assert a.bad_region_indices==(0,) and b.bad_region_indices==(4,)
    assert abs(a.request_rad_s)>8*abs(b.request_rad_s)


def test_unknown_central_region_withdraws_visual_even_with_healthy_legacy_sides():
    q=np.full(10,.9);valid=np.ones(10,dtype=bool);valid[5]=False
    task=request(q,valid=valid)
    assert not task.valid and task.reason=='unknown_regions'
    assert task.request_rad_s==0. and task.alpha_preferred==.25


def test_scan_alpha_counts_from_cmin_and_floors_at_typical_hole():
    assert preferred_scan_alpha(0., .85, alpha_min_at_quality=.79) == pytest.approx(1.)
    assert preferred_scan_alpha(.85 - .79, .85, alpha_min_at_quality=.79) == pytest.approx(.25)
    q = np.full(10, .99)
    q[0] = .85
    task = region_visual_request(q, np.ones(10, dtype=bool), np.linspace(0., 1., 11),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=.85, repair_speed_m_s=.03,
        alpha_min_at_quality=.79)
    assert task.reason == 'all_regions_sufficient'
    assert task.request_rad_s == 0.
    assert task.alpha_preferred == pytest.approx(1.)
    q[0] = .79
    task = region_visual_request(q, np.ones(10, dtype=bool), np.linspace(0., 1., 11),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=.85, repair_speed_m_s=.03,
        alpha_min_at_quality=.79)
    assert task.reason == 'regional_deficit_moment'
    assert task.alpha_preferred == pytest.approx(.25)
    assert abs(task.request_rad_s) > 0.


def test_rocking_reference_is_proportional_above_cmin():
    q = np.ones(10)
    q[0] = 0.98
    gated = region_visual_request(
        q, np.ones(10, dtype=bool), np.linspace(0., 1., 11),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=0.97, repair_speed_m_s=0.09,
        max_request_rad_s=0.20,
    )
    amp = region_visual_request(
        q, np.ones(10, dtype=bool), np.linspace(0., 1., 11),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=0.97, repair_speed_m_s=0.09,
        max_request_rad_s=0.20, rocking_reference=1.0,
    )
    assert gated.request_rad_s == 0.0
    assert gated.alpha_preferred == pytest.approx(1.0)
    assert abs(amp.request_rad_s) > 0.0
    assert amp.alpha_preferred == pytest.approx(1.0)
    deeper = np.ones(10)
    deeper[0] = 0.70
    deep = region_visual_request(
        deeper, np.ones(10, dtype=bool), np.linspace(0., 1., 11),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=0.97, repair_speed_m_s=0.09,
        max_request_rad_s=0.20, rocking_reference=1.0,
    )
    assert abs(deep.request_rad_s) > abs(amp.request_rad_s)


def test_uniform_deficit_does_not_raise_force_or_invent_torque_direction():
    for n in (3,10,17):
        task=request(np.full(n,.2))
        assert len(task.bad_region_indices)==n
        assert task.request_rad_s==0.
        assert task.alpha_preferred<1.
        assert not any('force' in field for field in task.to_dict())


def _relative(quality, **kwargs):
    n = len(quality)
    return region_visual_request(
        quality, np.ones(n, dtype=bool), np.linspace(0., 1., n + 1),
        ProbeGeometry.synthetic(image_x_sign=-1), c_min=0.97, repair_speed_m_s=0.09,
        max_request_rad_s=0.16, deficit_mode='relative', balance_deadband=0.03,
        relative_alpha_span=0.15, **kwargs,
    )


def test_relative_deficit_adapts_to_tissue_and_phantom_and_still_emits_omega_and_alpha():
    human = np.linspace(0.867, 0.901, 10)
    tissue = _relative(human)
    assert tissue.deficit_mode == 'relative'
    assert tissue.quality_reference == pytest.approx(better_half_reference(human))
    assert tissue.request_rad_s == pytest.approx(0.0)
    assert tissue.alpha_preferred == pytest.approx(1.0)
    assert tissue.reason == 'all_regions_sufficient'

    uniform = _relative(np.full(10, 0.72))
    assert uniform.request_rad_s == pytest.approx(0.0)
    assert uniform.alpha_preferred == pytest.approx(1.0)

    phantom = np.full(10, 0.995)
    phantom[0] = 0.70
    hole = _relative(phantom)
    assert hole.reason == 'regional_deficit_moment'
    assert abs(hole.request_rad_s) > 0.0
    assert hole.alpha_preferred < 1.0
    assert hole.quality_reference == pytest.approx(better_half_reference(phantom))

    assert preferred_relative_alpha(0.0, 0.15) == pytest.approx(1.0)
    assert preferred_relative_alpha(0.15, 0.15) == pytest.approx(0.25)


def test_spread_floor_shrinks_tissue_imbalance_and_keeps_a_phantom_hole():
    tissue = np.full(20, 0.92)
    tissue[:10] = 0.81
    mild = _relative(tissue, relative_spread_floor=0.12)
    raw = _relative(tissue)
    assert abs(mild.request_rad_s) < abs(raw.request_rad_s)
    assert mild.alpha_preferred > raw.alpha_preferred
    phantom = np.full(20, 0.995)
    phantom[:4] = 0.60
    hole = _relative(phantom, relative_spread_floor=0.12)
    bare = _relative(phantom)
    assert hole.reason == 'regional_deficit_moment'
    assert abs(hole.request_rad_s) == pytest.approx(abs(bare.request_rad_s), rel=1e-6)
    assert hole.alpha_preferred == pytest.approx(bare.alpha_preferred)
