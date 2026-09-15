"""Comparison-study run folders, frozen plans, and optional US recording."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

COMPARISON_MODES = ("ultrapoc", "admittance_1d", "ac2d", "tafac")
COMPARISON_REUSE_MODES = ("admittance_1d", "ac2d", "tafac")
COMPARISON_SPEED_M_S = 0.010
DEFAULT_DATA_ROOT = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
DEFAULT_CONTACT_QP = (
    Path(__file__).resolve().parents[2] / "config" / "contact_qp" / "active_probe50_delay_kf_cop.yaml"
)
ICRA_SCRIPT_DIR = Path(os.environ.get("ICRA_SCRIPT_DIR", "/media/camp/EXT_DRIVE/ICRA_YM/script"))
FEATURE_ENDPOINT = "tcp://127.0.0.1:17361"
US_ENDPOINT = "tcp://127.0.0.1:17359"
DEMO_FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def allocate_run(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    occupied = {int(path.name) for path in root.iterdir() if path.name.isdigit()}
    number = 1
    while True:
        if number not in occupied:
            path = root / f"{number:03d}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                pass
        number += 1


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_reused_plan(directory: Path):
    summary = json.loads(Path(directory, "plan.json").read_text(encoding="utf-8"))
    dims = summary.get("centerline_dimensions_m") or [0.0, 0.0]
    return SimpleNamespace(
        poses=np.asarray(summary["poses"], dtype=float),
        standoff=np.asarray(summary["standoff"], dtype=float),
        lift=np.asarray(summary["preview_lift"], dtype=float),
        normal=np.asarray(summary["normal_outward"], dtype=float),
        normals=np.asarray(summary["waypoint_normals_outward"], dtype=float),
        n_rows=int(summary.get("rows") or 0),
        stride_m=float(summary.get("row_spacing_m") or 0.0),
        u_span_m=float(dims[0]),
        v_span_m=float(dims[1]),
        length_m=float(summary["length_m"]),
        pattern=str(summary.get("pattern") or "lissajous"),
        orientation_diagnostics=summary.get("orientation_diagnostics") or {},
        right=np.asarray(summary["long_axis"], dtype=float),
        outline_dimensions_m=np.asarray(summary["outline_dimensions_m"], dtype=float),
        lissajous_yaw_deg=float(summary.get("lissajous_yaw_deg") or 0.0),
        normal_offset_deg=float(summary.get("normal_offset_deg") or 0.0),
    )


def write_meta(directory: Path, payload: dict) -> Path:
    path = Path(directory) / "meta.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_hfpc_polyline(run_dir: Path, poses) -> Path:
    """Vessel-format polyline the live controller already loads from disk."""
    dest = Path(run_dir) / "hfpc_polyline.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(poses, dtype=float).reshape(-1, 6)
    dest.write_text(
        json.dumps({"ok": True, "scan_poses": arr.tolist()}, separators=(",", ":")),
        encoding="utf-8",
    )
    return dest


def write_contact_qp_run_config(run_dir: Path, config_path: Path | None) -> Path:
    import yaml
    from peirastic.contact_qp.runtime_config import load_study_config

    config = load_study_config(config_path or DEFAULT_CONTACT_QP)
    dest = Path(run_dir) / "contact_qp_run.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    config["log_path"] = str(Path(run_dir) / "contact_qp.jsonl")
    dest.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return dest


def comparison_force_axes(mode: str) -> list[float]:
    """UltraPoC and 2D AC own tool-z and ωy; 1D AC/TAFAC follow path attitude."""
    if mode not in COMPARISON_MODES:
        raise ValueError(f"mode must be one of {COMPARISON_MODES}, got {mode!r}")
    if mode in ("ultrapoc", "ac2d"):
        from peirastic.scan_path import SCAN_FORCE_AXES

        return list(SCAN_FORCE_AXES)
    return list(DEMO_FORCE_AXES)


def comparison_hfpc_options(mode: str, *, force: float, run_dir: Path, config_path: Path | None):
    del force
    if mode not in COMPARISON_MODES:
        raise ValueError(f"mode must be one of {COMPARISON_MODES}, got {mode!r}")
    if mode == "ultrapoc":
        from peirastic.scan_path import force_profile

        return dict(
            law="contact_qp",
            contact_qp=str(write_contact_qp_run_config(run_dir, config_path)),
            extra=dict(force_profile("icra")),
        )
    extra = {
        "admittance_mass": 1.0,
        "admittance_damping": 40.0,
        "max_vz_tool_m_s": 0.012,
        "feature_endpoint": FEATURE_ENDPOINT,
        "comparison_log_path": str(Path(run_dir) / "comparison_law.jsonl"),
    }
    if mode == "ac2d":
        from peirastic.realman8dof.force.comparison_laws import (
            DEFAULT_COULOMB_YY,
            DEFAULT_DAMPING_YY,
            DEFAULT_INERTIA_YY,
            DEFAULT_VMAX_OMEGA_Y,
        )

        extra.update(
            admittance_inertia_yy=DEFAULT_INERTIA_YY,
            admittance_damping_yy=DEFAULT_DAMPING_YY,
            max_omega_y_rad_s=DEFAULT_VMAX_OMEGA_Y,
            admittance_coulomb_yy=DEFAULT_COULOMB_YY,
        )
    return dict(law=mode, contact_qp=None, extra=extra)


def start_us_recorder(directory: Path, *, python: str | None = None):
    script_dir = Path(ICRA_SCRIPT_DIR)
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from scan_io import RecorderProcess

    args = SimpleNamespace(
        recorder_python=Path(python or os.environ.get("ICRA_RECORD_PYTHON", sys.executable)),
        repo=Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[3])),
        state_shm="rm75_state",
        shm_prefix="",
        us_endpoint=os.environ.get("REALUS_US_ENDPOINT", US_ENDPOINT),
        force_shm=None,
    )
    recorder = RecorderProcess(directory, args)
    recorder.wait_ready(lambda: None)
    return recorder
