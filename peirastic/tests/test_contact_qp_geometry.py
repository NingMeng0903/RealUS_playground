import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.geometry import (
    accepted_alpha, aperture_rows, constraints_to_base, motion_basis, point_normal_row,
    twist_tcp_to_face, window_rows, wrench_tcp_to_face,
)
from peirastic.contact_qp.types import ProbeGeometry, TwistConstraints


def test_offset_frame_change_preserves_full_port_power():
    rng = np.random.default_rng(9)
    for _ in range(30):
        t = np.eye(4); t[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        t[:3, 3] = rng.uniform(-.1, .1, 3)
        geometry = ProbeGeometry(.025, t)
        v, w = rng.standard_normal((2, 6))
        vf = twist_tcp_to_face(geometry) @ v
        wf = wrench_tcp_to_face(w, geometry)
        assert wf @ vf == pytest.approx(w @ v, abs=2e-14)


def test_point_velocity_matches_finite_rigid_rotation():
    t = np.eye(4); t[:3, :3] = Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix()
    t[:3, 3] = [.01, -.02, .03]
    g = ProbeGeometry(.025, t)
    velocity = np.array([.002, -.003, .004, .1, .2, -.1])
    x = .012
    point = t[:3, 3]+x*t[:3, 0]
    dt = 1e-7
    moved = Rotation.from_rotvec(dt*velocity[3:]).apply(point)+dt*velocity[:3]
    derivative = t[:3, 2] @ ((moved-point)/dt)
    assert point_normal_row(g, x) @ velocity == pytest.approx(derivative, abs=2e-9)


def test_aligned_geometry_and_image_flip_have_explicit_signs():
    g = ProbeGeometry.synthetic()
    v = np.array([0, 0, .001, 0, .1, 0])
    rows = window_rows(g)
    assert (rows @ v)[0] > .001 and (rows @ v)[2] < .001
    np.testing.assert_allclose(window_rows(ProbeGeometry.synthetic(image_x_sign=-1)), rows[::-1])
    endpoint = aperture_rows(g) @ v
    assert max(abs(endpoint)) == pytest.approx(.001+g.half_length_m*.1)


def test_tcp_pivot_is_not_silently_moved_to_face_center():
    t = np.eye(4); t[:3, 3] = [.01, 0, .03]
    geometry = ProbeGeometry(.025, t)
    h = motion_basis([.01, 0, 0, 0, 0, 0])
    v = h @ [0., .2, 0.]
    np.testing.assert_array_equal(v[:3], [0, 0, 0])
    face = twist_tcp_to_face(geometry) @ v
    np.testing.assert_allclose(face[:3], [.006, 0, -.002])


def test_world_constraint_transform_and_final_progress_compatibility():
    r = Rotation.from_euler("xyz", [.5, -.2, .4]).as_matrix()
    c = TwistConstraints(np.eye(6), -np.ones(6), np.ones(6))
    b = constraints_to_base(c, r)
    v = np.array([2., .1, .2, .4, 0, 0])
    world = np.r_[r@v[:3], r@v[3:]]
    assert b.violation(world) == pytest.approx(c.violation(v))
    h = motion_basis([.02, 0, 0, 0, 0, 0])
    tol = np.array([1e-5]*3+[1e-4]*3)
    assert accepted_alpha(h, .8, h@[.001, .1, .7], h@[.001, .1, .6], tol) == pytest.approx(.6)
    assert accepted_alpha(h, 0., h@[.001, .1, 0], h@[.001, .1, 0], tol) == 0
    with pytest.raises(ValueError):
        accepted_alpha(h, .8, [0, .01, 0, 0, 0, 0], np.zeros(6), tol)
    with pytest.raises(ValueError):
        accepted_alpha(h, .8, h@[0, 0, 1.], h@[0, 0, .8], tol)


def test_invalid_geometry_and_inequalities_rejected():
    with pytest.raises(ValueError):
        ProbeGeometry(-.1)
    with pytest.raises(ValueError):
        ProbeGeometry(.02, np.zeros((4, 4)))
    with pytest.raises(ValueError):
        TwistConstraints(np.ones((2, 6)), [1, 0], [0, 1])


def test_zero_path_still_checks_final_sent_and_predicted_motion():
    h = motion_basis(np.zeros(6))
    tol = np.array([1e-5]*3+[1e-4]*3)
    valid = h @ [.001, .1, 0.]
    assert accepted_alpha(h, .8, valid, valid, tol) == 0.
    for invalid in ([.01, 0, 0, 0, 0, 0], [np.nan]*6):
        with pytest.raises(ValueError):
            accepted_alpha(h, .8, invalid, valid, tol)
        with pytest.raises(ValueError):
            accepted_alpha(h, .8, valid, invalid, tol)
