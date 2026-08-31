"""C++ wbc_rt SRS / PostureRetarget must match the Python originals."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import PostureRetarget
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary
from rm75_control.control.joint_admittance_8dof.wbc_rt.config_dump import dump_wbc_config
from rm75_control.kinematics.srs_ik import (
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    shoulder_y_from_q_rail,
    srs_ik,
)

Q_ARM_SAFE = [
    np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40]),
    np.array([-0.50, 1.10, 0.30, -1.20, -0.25, 0.85, 0.10]),
    np.array([0.80, 0.55, 0.40, 0.75, 1.00, 0.65, 1.20]),
    np.array([-0.20, -0.85, -0.35, -1.05, 0.55, -0.75, -0.90]),
    np.array([1.20, 0.40, 0.60, 0.50, -0.80, 0.55, 0.30]),
]

_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
_BIN = find_wbc_rt_binary()


def _env() -> dict[str, str]:
    env = os.environ.copy()
    cmeel = env.get(
        "CMEEL_PREFIX",
        "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix",
    )
    lib = str(Path(cmeel) / "lib")
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib + (":" + prev if prev else "")
    return env


def _run(args: list[str]) -> str:
    assert _BIN is not None
    out = subprocess.check_output([str(_BIN), *args], env=_env(), text=True)
    return out.strip()


def _parse_floats(line: str) -> np.ndarray:
    return np.fromstring(line, sep=" ", dtype=float)


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    return RobotKinematics()


@pytest.fixture(scope="module")
def dumped_cfg(kin: RobotKinematics, tmp_path_factory) -> Path:
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    path = tmp_path_factory.mktemp("wbc_rt_srs") / "wbc.cfg"
    dump_wbc_config(cfg, path, urdf_path=kin.urdf_path, kin=kin)
    return path


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
@pytest.mark.parametrize("q_arm", Q_ARM_SAFE)
@pytest.mark.parametrize("q_rail", [0.10, 0.375])
def test_cxx_srs_ik_matches_python(kin, q_arm, q_rail) -> None:
    q = full_q_from_arm(q_arm, rail_m=q_rail)
    pose = np.asarray(kin.fk_pose(q), dtype=float).reshape(6)
    psi = float(psi_from_q(q))
    branch = int(branch_from_q(q))
    y_s = shoulder_y_from_q_rail(q_rail)
    R, t = flange_tcp_from_kin(kin)
    py = srs_ik(pose, psi, branch, y_rail=y_s, R_flange_tcp=R, t_flange_tcp=t)
    assert py is not None
    args = [
        "--srs-ik",
        "--pose",
        *[f"{x:.17g}" for x in pose],
        "--psi",
        f"{psi:.17g}",
        "--branch",
        str(branch),
        "--y-shoulder",
        f"{y_s:.17g}",
        "--R",
        *[f"{x:.17g}" for x in np.asarray(R, dtype=float).reshape(-1)],
        "--t",
        *[f"{x:.17g}" for x in np.asarray(t, dtype=float).reshape(-1)],
    ]
    line = _run(args)
    assert line != "none"
    cxx = _parse_floats(line)
    assert cxx.shape == (7,)
    assert np.allclose(cxx, py, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_psi_and_branch_match_python() -> None:
    line = _run(["--psi-from-q", "--q", *[f"{x:.17g}" for x in _SEED_Q]])
    psi_c, branch_c = line.split()
    assert float(psi_c) == pytest.approx(float(psi_from_q(_SEED_Q)), abs=1e-12)
    assert int(branch_c) == int(branch_from_q(_SEED_Q))


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_fk_pose_matches_python(kin, dumped_cfg) -> None:
    py = np.asarray(kin.fk_pose(_SEED_Q), dtype=float).reshape(6)
    cxx = _parse_floats(
        _run(["--fk-pose", "--config", str(dumped_cfg), "--q", *[f"{x:.17g}" for x in _SEED_Q]])
    )
    assert cxx.shape == (6,)
    assert np.allclose(cxx[:3], py[:3], atol=1e-10)
    assert np.allclose(cxx[3:], py[3:], atol=1e-9)


@pytest.fixture(scope="module")
def kin_synced() -> RobotKinematics:
    kin = RobotKinematics()
    offset = kin.tcp_offset_pose.copy()
    offset[5] += np.deg2rad(1.5)
    kin.apply_link7_to_tcp_offset(offset)
    return kin


@pytest.fixture(scope="module")
def dumped_cfg_synced(kin_synced: RobotKinematics, tmp_path_factory) -> Path:
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    path = tmp_path_factory.mktemp("wbc_rt_srs_synced") / "wbc.cfg"
    dump_wbc_config(cfg, path, urdf_path=kin_synced.urdf_path, kin=kin_synced)
    return path


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_fk_pose_matches_synced_tool(kin_synced, dumped_cfg_synced) -> None:
    py = np.asarray(kin_synced.fk_pose(_SEED_Q), dtype=float).reshape(6)
    urdf_kin = RobotKinematics()
    urdf_pose = np.asarray(urdf_kin.fk_pose(_SEED_Q), dtype=float).reshape(6)
    assert not np.allclose(py[3:], urdf_pose[3:], atol=1e-4)
    cxx = _parse_floats(
        _run(
            [
                "--fk-pose",
                "--config",
                str(dumped_cfg_synced),
                "--q",
                *[f"{x:.17g}" for x in _SEED_Q],
            ]
        )
    )
    assert cxx.shape == (6,)
    assert np.allclose(cxx[:3], py[:3], atol=1e-10)
    assert np.allclose(cxx[3:], py[3:], atol=1e-9)


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_posture_matches_synced_tool(kin_synced, dumped_cfg_synced) -> None:
    dt = 0.005
    rail_lo, rail_hi = 0.005, 0.78
    ticks = 40
    py = _py_posture_rows(kin_synced, _SEED_Q, dt, rail_lo, rail_hi, False, ticks)
    cxx = _cxx_posture_rows(dumped_cfg_synced, _SEED_Q, dt, rail_lo, rail_hi, False, ticks)
    assert py.shape == cxx.shape
    assert np.allclose(py[:, 0], cxx[:, 0], atol=1e-9)
    assert np.allclose(py[:, 1], cxx[:, 1], atol=1e-9)
    assert np.allclose(py[:, 2], cxx[:, 2], atol=1e-9)
    assert np.allclose(py[:, 3], cxx[:, 3], atol=1e-9)
    assert np.allclose(py[:, 4:], cxx[:, 4:], atol=1e-7)


def _py_posture_rows(kin, q, dt, rail_lo, rail_hi, hold, ticks):
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    pr = PostureRetarget(kin, cfg.psi_retarget)
    pr.reset(q)
    rows = [
        np.concatenate(
            (
                [pr.homotopy_s, pr.d_star_m, pr._psi_cmd, pr.psi_star_rad],
                np.asarray(pr.q_star_rad, dtype=float).reshape(-1),
            )
        )
    ]
    for _ in range(ticks):
        pr.step(q, dt, rail_lo=rail_lo, rail_hi=rail_hi, hold_setpoint=hold)
        rows.append(
            np.concatenate(
                (
                    [pr.homotopy_s, pr.d_star_m, pr._psi_cmd, pr.psi_star_rad],
                    np.asarray(pr.q_star_rad, dtype=float).reshape(-1),
                )
            )
        )
    return np.asarray(rows, dtype=float)


def _cxx_posture_rows(dumped_cfg, q, dt, rail_lo, rail_hi, hold, ticks):
    text = _run(
        [
            "--posture-tick",
            "--config",
            str(dumped_cfg),
            "--q",
            *[f"{x:.17g}" for x in q],
            "--dt",
            f"{dt:.17g}",
            "--rail-lo",
            f"{rail_lo:.17g}",
            "--rail-hi",
            f"{rail_hi:.17g}",
            "--hold",
            "1" if hold else "0",
            "--ticks",
            str(ticks),
        ]
    )
    return np.asarray([_parse_floats(line) for line in text.splitlines()], dtype=float)


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_posture_homotopy_matches_python(kin, dumped_cfg) -> None:
    dt = 0.005
    rail_lo, rail_hi = 0.005, 0.78
    ticks = 200
    py = _py_posture_rows(kin, _SEED_Q, dt, rail_lo, rail_hi, False, ticks)
    cxx = _cxx_posture_rows(dumped_cfg, _SEED_Q, dt, rail_lo, rail_hi, False, ticks)
    assert py.shape == cxx.shape
    # s, d*, ψ_cmd, ψ*, q*
    assert np.allclose(py[:, 0], cxx[:, 0], atol=1e-9)
    assert np.allclose(py[:, 1], cxx[:, 1], atol=1e-9)
    assert np.allclose(py[:, 2], cxx[:, 2], atol=1e-9)
    assert np.allclose(py[:, 3], cxx[:, 3], atol=1e-9)
    assert np.allclose(py[:, 4:], cxx[:, 4:], atol=1e-7)


@pytest.mark.skipif(_BIN is None, reason="wbc_rt binary not built")
def test_cxx_posture_hold_matches_live_python(kin, dumped_cfg) -> None:
    dt = 0.005
    rail_lo, rail_hi = 0.005, 0.78
    ticks = 40
    py = _py_posture_rows(kin, _SEED_Q, dt, rail_lo, rail_hi, True, ticks)
    cxx = _cxx_posture_rows(dumped_cfg, _SEED_Q, dt, rail_lo, rail_hi, True, ticks)
    assert py[-1, 0] > 0.0
    assert cxx[-1, 0] > 0.0
    assert np.allclose(py[:, 0], cxx[:, 0], atol=1e-9)
    assert np.allclose(py[:, 1], cxx[:, 1], atol=1e-9)
    assert np.allclose(py[:, 2], cxx[:, 2], atol=1e-9)
