"""Schema v2 JSON: payload (m,h) vs session bias vs tool binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.force.compensation.regressor import PHI_NAMES
from rm75_control.force.compensation.v2.frames import WRENCH_SEMANTICS

SCHEMA_VERSION = 2


class ToolBindingError(ValueError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def urdf_sha256(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def empty_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "payload": {
            "tool_id": "",
            "mass_kg": None,
            "first_moment_kg_m": [None, None, None],
            "inertia_kg_m2": None,
            "covariance": [],
        },
        "calibration_session": {
            "bias0": [0.0] * 6,
            "bias_drift_per_s": [0.0] * 6,
            "drift_enabled": False,
        },
        "tool_binding": {
            "active_tool_name": "",
            "T_link7_sensor": np.eye(4).tolist(),
            "T_link7_tcp": np.eye(4).tolist(),
            "urdf_sha256": "",
            "force_sign": [],
            "wrench_semantics": WRENCH_SEMANTICS,
        },
        "provenance": {
            "git_commit": "",
            "fit_config_sha256": "",
            "source_log_sha256": [],
        },
        "validation": {
            "force_dynamic_valid": False,
            "moment_dynamic_valid": False,
            "unmodeled_inertia_torque_bound_nm": None,
            "dynamic_kinematics_identified": False,
            "dynamic_kinematics_free_air_validated": False,
            "dynamic_kinematics_contact_validated": False,
            "recommended_online_enable": False,
            "inertia_ident_failed": None,
        },
        "delay": {
            "delay_sensor_vs_joint_s": None,
            "delay_online_effective_s": None,
            "delay_ci95_s": None,
            "delay_per_axis_s": [],
            "delay_hit_search_boundary": False,
        },
        "validity_envelope": {
            "rail_stationary_only": True,
            "max_linear_accel_m_s2": 0.30,
            "max_angular_accel_rad_s2": 0.8,
            "max_sensor_age_s": 0.015,
            "max_qdd_covariance": 0.0,
        },
        "phi_mhb": {},
        "phi_recommended": {},
    }


def _phi_mhb_dict(mass_kg: float, h_L: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    names = ["m", "mc_x", "mc_y", "mc_z", "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0"]
    vals = [mass_kg, *list(np.asarray(h_L, dtype=float).reshape(3)), *list(np.asarray(bias, dtype=float).reshape(6))]
    return {n: float(v) for n, v in zip(names, vals, strict=True)}


def phi16(mass_kg: float, h_L: np.ndarray, bias: np.ndarray, I_voigt: np.ndarray | None = None) -> np.ndarray:
    phi = np.zeros(16, dtype=float)
    phi[0] = float(mass_kg)
    phi[1:4] = np.asarray(h_L, dtype=float).reshape(3)
    if I_voigt is not None:
        phi[4:10] = np.asarray(I_voigt, dtype=float).reshape(6)
    phi[10:16] = np.asarray(bias, dtype=float).reshape(6)
    return phi


def phi_dict16(phi: np.ndarray) -> dict[str, float]:
    return {PHI_NAMES[i]: float(phi[i]) for i in range(16)}


def promote_mhb_to_live(
    live_path: Path,
    rec: dict[str, Any],
    *,
    com: dict[str, Any] | None = None,
    rms_all: float | None = None,
    per_pose: dict[str, Any] | None = None,
) -> None:
    """Write identified m, h, b into live ``force_id_phi.json``. I stays 0."""

    live_path = Path(live_path)
    doc: dict[str, Any] = {}
    if live_path.is_file():
        doc = json.loads(live_path.read_text())
    mhb_keys = ("m", "mc_x", "mc_y", "mc_z", "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0")
    block = {k: float(rec[k]) for k in mhb_keys}
    for k in ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"):
        block[k] = 0.0
    for key in ("phi_recommended", "phi_10"):
        prev = dict(doc.get(key) or {})
        prev.update(block)
        doc[key] = prev
    m = float(block["m"])
    if com is not None:
        doc["com_recommended"] = com
    elif m > 1e-9:
        com_mm = {
            "Cx": 1e3 * float(block["mc_x"]) / m,
            "Cy": 1e3 * float(block["mc_y"]) / m,
            "Cz": 1e3 * float(block["mc_z"]) / m,
        }
        doc["com_recommended"] = {"sensor_mm": dict(com_mm), "link7_mm": dict(com_mm)}
    if rms_all is not None:
        doc["rms_10"] = float(rms_all)
    if per_pose is not None:
        doc["per_pose_residual"] = per_pose
    doc["recommended"] = "phi_recommended (payload_id_v2 m,h,b; I=0)"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(json.dumps(doc, indent=2) + "\n")


def write_phi_v2(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def load_phi_v2(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_tool_binding(doc: dict[str, Any], live: dict[str, Any]) -> None:
    b = doc.get("tool_binding") or {}
    for key in ("active_tool_name", "urdf_sha256", "wrench_semantics"):
        if live.get(key) is not None and str(b.get(key)) != str(live.get(key)):
            raise ToolBindingError(f"tool binding mismatch on {key}: {b.get(key)!r} != {live.get(key)!r}")
    if live.get("force_sign") is not None:
        a = [int(x) for x in (b.get("force_sign") or [])]
        c = [int(x) for x in live["force_sign"]]
        if a and a != c:
            raise ToolBindingError(f"force_sign mismatch: {a} != {c}")
    for Tkey in ("T_link7_sensor", "T_link7_tcp"):
        if live.get(Tkey) is None or b.get(Tkey) is None:
            continue
        A = np.asarray(b[Tkey], dtype=float)
        B = np.asarray(live[Tkey], dtype=float)
        if A.shape != B.shape or not np.allclose(A, B, atol=1e-6):
            raise ToolBindingError(f"{Tkey} mismatch")


def phi16_from_document(doc: dict[str, Any]) -> np.ndarray:
    if int(doc.get("schema_version", 0)) == SCHEMA_VERSION and "payload" in doc:
        p = doc["payload"]
        s = doc.get("calibration_session") or {}
        I = p.get("inertia_kg_m2")
        return phi16(float(p["mass_kg"]), p["first_moment_kg_m"], s.get("bias0") or [0.0] * 6, I)
    if "phi_recommended" in doc and isinstance(doc["phi_recommended"], dict):
        rec = doc["phi_recommended"]
        if rec and all(k in rec for k in PHI_NAMES):
            return np.array([rec[k] for k in PHI_NAMES], dtype=float)
    raise ValueError(f"unrecognized phi document {list(doc)[:8]}")


def runtime_bias_refresh(wrench_L: np.ndarray, mass_kg: float, h_L: np.ndarray, g_L: np.ndarray) -> np.ndarray:
    from rm75_control.force.compensation.v2.regressor_v2 import payload_wrench_mhb

    pred = payload_wrench_mhb(
        mass_kg=mass_kg,
        h_L=h_L,
        a_L=np.zeros(3),
        g_L=g_L,
        omega_L=np.zeros(3),
        alpha_L=np.zeros(3),
    )
    return np.asarray(wrench_L, dtype=float).reshape(6) - pred
