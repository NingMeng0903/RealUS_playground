"""Per-segment nullspace summary CSV."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

def _load_analyze():
    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_nullspace.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_nullspace", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(t: float, *, lx: float, err_mm: float, psi_deg: float, d_star: float, q4: float) -> dict:
    row = {
        "t_wall_s": f"{t:.6f}",
        "track_err_mm": f"{err_mm:.4f}",
        "psi_deg": f"{psi_deg:.4f}",
        "d_star_m": f"{d_star:.6f}",
        "slack_norm": "0.01",
        "secondary_suppressed": "0",
        "pad_lx": f"{lx:.3f}",
        "pad_ly": "0.0",
        "pad_lt": "-1.0",
        "pad_rx": "0.0",
        "pad_ry": "0.0",
        "pad_rt": "-1.0",
        "pad_lb": "0",
        "pad_rb": "0",
        "qpik_nullspace_norm": "0.2",
        "qpik_sec_target_norm": "0.25",
    }
    for i in range(8):
        row[f"q_cmd_{i}"] = f"{(0.4 if i == 0 else (q4 if i == 4 else 0.1)):.6f}"
    return row


def test_analyze_nullspace_writes_segment_csv(tmp_path) -> None:
    mod = _load_analyze()
    dt = 0.005
    rows = []
    t = 0.0
    for i in range(40):
        rows.append(_row(t, lx=-1.0, err_mm=2.0, psi_deg=50.0 + 0.1 * i, d_star=-0.10, q4=1.6))
        t += dt
    for _ in range(40):
        rows.append(_row(t, lx=0.0, err_mm=12.0, psi_deg=54.0, d_star=-0.12, q4=1.7))
        t += dt
    for _ in range(80):
        rows.append(_row(t, lx=0.0, err_mm=0.2, psi_deg=54.0, d_star=-0.12, q4=1.7))
        t += dt
    summaries = mod.analyze_rows(rows)
    kinds = [s["kind"] for s in summaries]
    assert "active" in kinds
    assert "hold_pullback" in kinds
    assert "quiet_hold" in kinds
    quiet = [s for s in summaries if s["kind"] == "quiet_hold"]
    assert quiet and quiet[0]["duration_s"] >= 0.25
    src = tmp_path / "run_fake.csv"
    with src.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out = mod.write_analysis(src)
    assert out.name == "ns_analysis_run_fake.csv"
    with out.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0].keys()) == mod.SUMMARY_FIELDS
    assert {r["kind"] for r in written} >= {"active", "hold_pullback", "quiet_hold"}
