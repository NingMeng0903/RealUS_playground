from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    _assert_proper_rotation,
    _validate_leg_hinge_solve_entry_v1,
    solve_leg_hinge_v1,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def _valid_entry(**overrides):
    entry = {
        "femur_bone": 1,
        "knee_bone": 2,
        "ankle_bone": 3,
        "smplx_hip": 1,
        "smplx_knee": 4,
        "smplx_ankle": 7,
        "hinge_axis_femur_local": [1.0, 0.0, 0.0],
        "femoral_head_femur_local": [0.0, 0.0, 0.0],
        "femoral_head_vertex_indices": [0, 1, 2, 3],
        "hinge_axis_sign": 1,
        "blend_lo_deg": 5.0,
        "blend_hi_deg": 15.0,
    }
    entry.update(overrides)
    return entry


def test_validate_leg_hinge_solve_malformed_metadata():
    with pytest.raises(ValueError, match="missing fields"):
        _validate_leg_hinge_solve_entry_v1(
            {"femur_bone": 1},
            side="left",
            bone_count=10,
            joint_count=55,
        )
    with pytest.raises(ValueError, match="invalid bone"):
        _validate_leg_hinge_solve_entry_v1(
            _valid_entry(knee_bone=1),
            side="left",
            bone_count=10,
            joint_count=55,
        )


def test_synthetic_two_segment_ankle_and_hinge_alignment():
    H0 = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    K0 = np.asarray((0.0, -0.40, 0.0), dtype=np.float64)
    A0 = np.asarray((0.0, -0.80, 0.0), dtype=np.float64)
    R_bind = np.eye(3, dtype=np.float64)
    hinge_local = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    thigh = float(np.linalg.norm(K0 - H0))
    shank = float(np.linalg.norm(A0 - K0))

    for flex_deg in (20.0, 45.0, 90.0, 120.0):
        theta = np.radians(flex_deg)
        H = H0.copy()
        K = H + np.asarray((0.0, -thigh, 0.0), dtype=np.float64)
        A = K + Rotation.from_rotvec(hinge_local * theta).apply(
            np.asarray((0.0, -shank, 0.0), dtype=np.float64)
        )
        R_driver = R_bind.copy()
        R_femur, solved_theta, raw_theta, hinge_world = solve_leg_hinge_v1(
            hip=H,
            knee=K,
            ankle=A,
            bind_hip=H0,
            bind_knee=K0,
            bind_ankle=A0,
            bind_femur_rotation=R_bind,
            hinge_axis_femur_local=hinge_local,
            driver_femur_rotation=R_driver,
            blend_lo_deg=5.0,
            blend_hi_deg=15.0,
        )
        _assert_proper_rotation(R_femur, "synthetic femur")
        assert float(np.degrees(solved_theta)) == pytest.approx(flex_deg, abs=1.0e-5)
        assert float(raw_theta) == pytest.approx(float(solved_theta), abs=1.0e-12)

        K_posed = H + R_femur @ (K0 - H0)
        assert np.linalg.norm(K_posed - K) == pytest.approx(0.0, abs=1.0e-9)

        # Authored hinge stays on the femur; shank lands on the drive ankle.
        h_posed = _unit(R_femur @ hinge_local)
        assert np.allclose(h_posed, _unit(hinge_world), atol=1.0e-9)
        A_posed = K_posed + Rotation.from_rotvec(h_posed * solved_theta).apply(
            R_femur @ (A0 - K0)
        )
        assert np.linalg.norm(A_posed - A) == pytest.approx(0.0, abs=1.0e-8)


def test_hyperextension_is_clamped_to_zero():
    H0 = np.zeros(3, dtype=np.float64)
    K0 = np.asarray((0.0, -0.40, 0.0), dtype=np.float64)
    A0 = np.asarray((0.0, -0.80, 0.0), dtype=np.float64)
    R_bind = np.eye(3, dtype=np.float64)
    hinge_local = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    theta = np.radians(-20.0)
    H = H0.copy()
    K = K0.copy()
    A = K + Rotation.from_rotvec(hinge_local * theta).apply(A0 - K0)
    _R_femur, applied, raw, _hinge = solve_leg_hinge_v1(
        hip=H,
        knee=K,
        ankle=A,
        bind_hip=H0,
        bind_knee=K0,
        bind_ankle=A0,
        bind_femur_rotation=R_bind,
        hinge_axis_femur_local=hinge_local,
        driver_femur_rotation=R_bind,
        blend_lo_deg=5.0,
        blend_hi_deg=15.0,
    )
    assert float(np.degrees(raw)) == pytest.approx(-20.0, abs=1.0e-5)
    assert float(applied) == pytest.approx(0.0, abs=1.0e-12)


def test_straight_leg_fallback_uses_driver_rotation():
    H0 = np.zeros(3, dtype=np.float64)
    K0 = np.asarray((0.0, -0.40, 0.0), dtype=np.float64)
    A0 = np.asarray((0.0, -0.80, 0.0), dtype=np.float64)
    R_bind = np.eye(3, dtype=np.float64)
    hinge_local = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    R_driver = Rotation.from_rotvec(
        np.asarray((0.0, -1.0, 0.0)) * np.radians(17.0)
    ).as_matrix()
    H = H0.copy()
    K = K0.copy()
    A = A0.copy()
    R_femur, theta, _raw, _hinge = solve_leg_hinge_v1(
        hip=H,
        knee=K,
        ankle=A,
        bind_hip=H0,
        bind_knee=K0,
        bind_ankle=A0,
        bind_femur_rotation=R_bind,
        hinge_axis_femur_local=hinge_local,
        driver_femur_rotation=R_driver,
        blend_lo_deg=5.0,
        blend_hi_deg=15.0,
    )
    # Near-straight: phi fades out so the femur keeps the direction-aligned driver.
    d0 = _unit(K0 - H0)
    assert np.allclose(R_femur @ d0, d0, atol=1.0e-12)
    assert abs(float(np.degrees(theta))) < 1.0e-6
    # Driver long-axis-aligned attitude is recovered when flex ~ 0.
    d_driver = _unit(R_driver @ d0)
    assert np.allclose(d_driver, d0, atol=1.0e-12)
    twist = float(Rotation.from_matrix(R_driver.T @ R_femur).magnitude())
    assert twist < 1.0e-9


def test_bent_bind_leg_does_not_twist_the_femur_near_its_bind_pose():
    """A bind leg that is not straight must still fade the twist in from zero.

    The real anatomy's bind femur axis and shank sit 16.4 deg apart on the left
    and 10.3 deg on the right. Measuring flexion from a straight leg puts that
    bind pose above the fade band, so the twist arrives at full strength on the
    first degree of drive: a 0.5 deg knee input twisted the femur -34.9 deg about
    its own shaft and dragged the trochlea 63.6 mm off the patella. Flexion is
    therefore measured as an excursion away from the bind angle.
    """
    H0 = np.zeros(3, dtype=np.float64)
    K0 = np.asarray((0.0, -0.40, 0.0), dtype=np.float64)
    # Bind shank carries a 16 degree bend, as the authored anatomy does.
    bend = np.radians(16.0)
    A0 = K0 + 0.40 * np.asarray(
        (0.0, -np.cos(bend), np.sin(bend)), dtype=np.float64
    )
    R_bind = np.eye(3, dtype=np.float64)
    hinge_local = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    d0 = _unit(K0 - H0)

    # Drive axis tilted well off the authored hinge, which is what makes the
    # closed-form twist large and is the situation the SMPL-X drive presents.
    drive_axis = _unit(
        Rotation.from_rotvec(d0 * np.radians(50.0)).as_matrix() @ hinge_local
    )
    for drive_deg in (0.25, 0.5, 1.0, 2.0):
        rotation = Rotation.from_rotvec(
            drive_axis * np.radians(drive_deg)
        ).as_matrix()
        R_femur, _theta, _raw, _hinge = solve_leg_hinge_v1(
            hip=H0,
            knee=K0,
            ankle=K0 + rotation @ (A0 - K0),
            bind_hip=H0,
            bind_knee=K0,
            bind_ankle=A0,
            bind_femur_rotation=R_bind,
            hinge_axis_femur_local=hinge_local,
            driver_femur_rotation=R_bind,
            blend_lo_deg=5.0,
            blend_hi_deg=15.0,
        )
        axial_twist_deg = abs(
            float(
                np.degrees(
                    np.dot(Rotation.from_matrix(R_femur).as_rotvec(), d0)
                )
            )
        )
        assert axial_twist_deg < 1.0e-6, (
            f"{drive_deg} deg of drive injected {axial_twist_deg:.3f} deg of "
            "femoral twist near the bind pose"
        )


def test_blend_region_is_c1_continuous():
    H0 = np.zeros(3, dtype=np.float64)
    K0 = np.asarray((0.0, -0.40, 0.0), dtype=np.float64)
    A0 = np.asarray((0.0, -0.80, 0.0), dtype=np.float64)
    R_bind = np.eye(3, dtype=np.float64)
    hinge_local = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    R_driver = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
    thigh = 0.40
    shank = 0.40

    def evaluate(flex_deg: float) -> np.ndarray:
        theta = np.radians(flex_deg)
        H = H0.copy()
        K = H + np.asarray((0.0, -thigh, 0.0), dtype=np.float64)
        A = K + Rotation.from_rotvec(hinge_local * theta).apply(
            np.asarray((0.0, -shank, 0.0), dtype=np.float64)
        )
        R, _theta, _raw, _hinge = solve_leg_hinge_v1(
            hip=H,
            knee=K,
            ankle=A,
            bind_hip=H0,
            bind_knee=K0,
            bind_ankle=A0,
            bind_femur_rotation=R_bind,
            hinge_axis_femur_local=hinge_local,
            driver_femur_rotation=R_driver,
            blend_lo_deg=5.0,
            blend_hi_deg=15.0,
        )
        return Rotation.from_matrix(R).as_quat()

    degrees = np.linspace(4.0, 16.0, 25)
    quats = [evaluate(float(deg)) for deg in degrees]
    for q in quats:
        q /= max(float(np.linalg.norm(q)), 1.0e-12)
    deltas = []
    for previous, current in zip(quats, quats[1:]):
        if float(np.dot(previous, current)) < 0.0:
            current = -current
        deltas.append(float(np.linalg.norm(current - previous)))
    median = float(np.median(deltas))
    assert max(deltas) < max(8.0 * median, 1.0e-3)


def test_source_bone_posed_global_rejects_malformed_leg_metadata():
    with pytest.raises(ValueError, match="missing fields"):
        _validate_leg_hinge_solve_entry_v1(
            {"femur_bone": 0, "knee_bone": 1},
            side="left",
            bone_count=4,
            joint_count=55,
        )
    with pytest.raises(ValueError, match="blend thresholds"):
        _validate_leg_hinge_solve_entry_v1(
            _valid_entry(blend_lo_deg=15.0, blend_hi_deg=5.0),
            side="right",
            bone_count=10,
            joint_count=55,
        )


def test_femur_pose_about_head_keeps_centre_invariant():
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        _femur_pose_about_head,
    )

    parent = np.eye(4, dtype=np.float64)
    bind_local = np.eye(4, dtype=np.float64)
    head_local = np.asarray((0.02, -0.01, 0.03), dtype=np.float64)
    head_target = np.asarray((0.1, -0.4, 0.0), dtype=np.float64)
    R = Rotation.from_euler("xyz", [20.0, -35.0, 12.0], degrees=True).as_matrix()
    _local, posed = _femur_pose_about_head(
        parent_global=parent,
        bind_local=bind_local,
        femur_rotation_world=R,
        head_femur_local=head_local,
        head_target_world=head_target,
    )
    reconstructed = posed[:3, 3] + posed[:3, :3] @ head_local
    assert np.linalg.norm(reconstructed - head_target) == pytest.approx(0.0, abs=1.0e-12)
