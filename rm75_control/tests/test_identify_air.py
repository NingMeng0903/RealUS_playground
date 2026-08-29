"""Air-campaign analyzer: required plots from a synthetic vel_ff_vz plant."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from peirastic.apps.identify_air import (
    AIR_LOG_FIELDS,
    AIR_STEPS_MM_S,
    REQUIRED_PLOTS,
    analyze_air_paths,
    analyze_tdpa_contact,
    assert_plots_complete,
    detect_step_edges,
    fit_contact_press,
    load_tool_z_arrays,
    shadow_tdpa_from_rows,
    load_tool_z_rows,
    synthesize_air_csv,
    write_identification_report,
)
from peirastic.apps.identify_plant import STEPS_MM_S


def test_air_campaign_writes_required_plots(tmp_path: Path) -> None:
    csv_path = tmp_path / "air.csv"
    out_dir = tmp_path / "id_air"
    synthesize_air_csv(csv_path, chirp_s=8.0)
    result = analyze_air_paths([csv_path], out_dir=out_dir)
    assert_plots_complete(out_dir)
    for name in REQUIRED_PLOTS:
        assert (out_dir / name).is_file()
        assert (out_dir / name.replace(".png", ".svg")).is_file()
    assert len(result.edges) >= 2 * len(AIR_STEPS_MM_S)
    speeds = {round(e.cmd_mm_s) for e in result.edges}
    for mm in AIR_STEPS_MM_S:
        assert mm in speeds
    assert len(result.chirps) == 3
    assert result.jitter is not None
    assert result.jitter.feedback_age["p95"] < 0.005
    assert (out_dir / "edges.csv").is_file()
    assert (out_dir / "chirp_fits.csv").is_file()
    assert (out_dir / "jitter.json").is_file()
    assert (out_dir / "tdpa_shadow.json").is_file()
    delays = [e.delay_s for e in result.edges if e.cmd_mm_s >= 8.0]
    assert any(math.isfinite(d) and d > 0.0 for d in delays)


def test_air_log_schema_has_force_and_tdpa_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "air.csv"
    synthesize_air_csv(csv_path, chirp_s=8.0)
    rows = load_tool_z_rows(csv_path)
    assert rows
    for key in AIR_LOG_FIELDS:
        assert key in rows[0]
    assert float(rows[10]["fz"]) > 0.0
    assert rows[10]["v_cmd_z"] == rows[10]["vel_ff_vz"]


def test_step_xcorr_uses_pre_window(tmp_path: Path) -> None:
    csv_path = tmp_path / "air.csv"
    synthesize_air_csv(csv_path, chirp_s=8.0)
    t, u, y, *_ = load_tool_z_arrays(csv_path)
    edges = detect_step_edges(t, u, y)
    hold_edges = [e for e in edges if e.cmd_mm_s >= 8.0]
    assert hold_edges
    assert all(math.isfinite(e.delay_s) for e in hold_edges[:8])


def test_analyze_tdpa_contact_requires_rising_e_obs(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "hybrid.csv"
    fields = [
        "t_wall_s",
        "fz",
        "v_force_cmd_z",
        "contact_present",
        "tdpa_e_obs_j",
        "tdpa_alpha",
        "tdpa_clamped",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        e = 0.0
        for i in range(250):
            e += 0.0002
            writer.writerow(
                {
                    "t_wall_s": f"{i * 0.005:.3f}",
                    "fz": "1.20",
                    "v_force_cmd_z": "0.010",
                    "contact_present": "1",
                    "tdpa_e_obs_j": f"{e:.6f}",
                    "tdpa_alpha": "6.0",
                    "tdpa_clamped": "0",
                }
            )
    verdict = analyze_tdpa_contact(path)
    assert verdict["ok"] is True
    assert verdict["e_obs_press_delta_j"] > 0.0
    assert verdict["window_s"] >= 1.0
    assert verdict["passivity_claimed"] is True


def test_analyze_tdpa_contact_rejects_concatenated_bounces(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "hybrid.csv"
    fields = [
        "t_wall_s",
        "fz",
        "v_force_cmd_z",
        "contact_present",
        "tdpa_e_obs_j",
        "tdpa_alpha",
        "tdpa_clamped",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        e = 0.002
        t = 0.0
        for burst in range(8):
            for i in range(20):
                e += 0.00015
                writer.writerow(
                    {
                        "t_wall_s": f"{t:.3f}",
                        "fz": "1.40",
                        "v_force_cmd_z": "0.010",
                        "contact_present": "1",
                        "tdpa_e_obs_j": f"{e:.6f}",
                        "tdpa_alpha": "400.0",
                        "tdpa_clamped": "1",
                    }
                )
                t += 0.005
            e -= 0.004
            for i in range(40):
                writer.writerow(
                    {
                        "t_wall_s": f"{t:.3f}",
                        "fz": "0.80",
                        "v_force_cmd_z": "-0.020",
                        "contact_present": "1",
                        "tdpa_e_obs_j": f"{e:.6f}",
                        "tdpa_alpha": "400.0",
                        "tdpa_clamped": "1",
                    }
                )
                t += 0.005
    verdict = analyze_tdpa_contact(path)
    assert verdict["ok"] is False
    assert verdict["window_s"] < 1.0
    assert "wrong excitation" in verdict["reason"]
    assert "port pairing" not in verdict["reason"]


def test_analyze_tdpa_contact_uses_open_loop_phase(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "tdpa_press.csv"
    fields = [
        "t_wall_s",
        "phase",
        "fz",
        "v_cmd_z",
        "tdpa_e_obs_j",
        "tdpa_alpha",
        "tdpa_clamped",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        e = 0.0
        for i in range(80):
            writer.writerow(
                {
                    "t_wall_s": f"{i * 0.005:.3f}",
                    "phase": "tdpa_seek",
                    "fz": "0.10",
                    "v_cmd_z": "0.010",
                    "tdpa_e_obs_j": f"{e:.6f}",
                    "tdpa_alpha": "0.0",
                    "tdpa_clamped": "0",
                }
            )
        for i in range(250):
            e += 0.00025
            writer.writerow(
                {
                    "t_wall_s": f"{(80 + i) * 0.005:.3f}",
                    "phase": "tdpa_press",
                    "fz": "1.10",
                    "v_cmd_z": "0.008",
                    "tdpa_e_obs_j": f"{e:.6f}",
                    "tdpa_alpha": "0.0",
                    "tdpa_clamped": "0",
                }
            )
    verdict = analyze_tdpa_contact(path)
    assert verdict["ok"] is True
    assert verdict["window_s"] >= 1.0
    assert verdict["n_press"] == 250


def test_tdpa_shadow_reads_fz_times_v_cmd(tmp_path: Path) -> None:
    csv_path = tmp_path / "air.csv"
    synthesize_air_csv(csv_path, chirp_s=8.0)
    shadow = shadow_tdpa_from_rows(load_tool_z_rows(csv_path))
    assert shadow["air_only"] is True
    assert shadow["rigid_press_sign_check"] is False
    assert shadow["n"] > 100
    assert math.isfinite(shadow["e_obs_final_j"])


def test_air_campaign_dry_run_prints_dense_steps(capsys, monkeypatch) -> None:
    import sys

    from peirastic.apps.identify_plant import main

    monkeypatch.setattr(
        sys, "argv", ["identify_plant", "--air-campaign", "--dry-run"]
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "4.0" in out
    assert "air-campaign" in out
    assert "air_campaign.csv" in out
    assert "Window A log" in out
    assert "stop-reverse" not in out


def test_tdpa_press_dry_run_prints_open_loop_protocol(capsys, monkeypatch) -> None:
    import sys

    from peirastic.apps.identify_plant import main

    monkeypatch.setattr(
        sys, "argv", ["identify_plant", "--tdpa-press", "--dry-run"]
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "tdpa-press" in out
    assert "force-loop OFF" in out
    assert "hybrid" in out.lower()
    assert "3" in out


def test_analyze_tdpa_contact_accepts_short_open_loop_ramp(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "tdpa_press.csv"
    fields = [
        "t_wall_s",
        "phase",
        "fz",
        "v_cmd_z",
        "tdpa_e_obs_j",
        "tdpa_alpha",
        "tdpa_clamped",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        e = 0.0008
        for i in range(38):
            e += 0.00005
            writer.writerow(
                {
                    "t_wall_s": f"{i * 0.005:.3f}",
                    "phase": "tdpa_press",
                    "fz": f"{0.43 + i * 0.055:.3f}",
                    "v_cmd_z": "0.008",
                    "tdpa_e_obs_j": f"{e:.6f}",
                    "tdpa_alpha": "0.0",
                    "tdpa_clamped": "0",
                }
            )
    verdict = analyze_tdpa_contact(path)
    assert verdict["ok"] is True
    assert verdict["sign_ok"] is True
    assert verdict["window_s"] < 1.0
    assert "sign ok" in verdict["reason"]
    fit = fit_contact_press(path)
    assert fit["ke_n_m"] > 100.0
    assert fit["sign_ok"] is True


def test_write_identification_report_reads_air_artifacts(tmp_path: Path) -> None:
    import csv
    import json

    air = tmp_path / "id_air"
    air.mkdir()
    with (air / "chirp_fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["amp_mm_s", "t0_s", "tp_s", "gain", "group_delay_3hz_s"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "amp_mm_s": "8",
                "t0_s": "0.035",
                "tp_s": "0.018",
                "gain": "0.85",
                "group_delay_3hz_s": "0.065",
            }
        )
        writer.writerow(
            {
                "amp_mm_s": "25",
                "t0_s": "0.035",
                "tp_s": "0.012",
                "gain": "0.98",
                "group_delay_3hz_s": "0.020",
            }
        )
    (air / "jitter.json").write_text(
        json.dumps(
            {
                "linear_speed_mm_s": 40.0,
                "t0_spread_s": 0.0,
                "td_is_band": True,
                "jitter": {"feedback_age": {"p95": 0.0063}},
            }
        )
    )
    press = tmp_path / "press.csv"
    with press.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t_wall_s",
                "phase",
                "fz",
                "v_cmd_z",
                "dt_actual_s",
                "tdpa_e_obs_j",
                "tdpa_alpha",
                "tdpa_clamped",
            ],
        )
        writer.writeheader()
        e = 0.0
        for i in range(40):
            e += 0.0001
            writer.writerow(
                {
                    "t_wall_s": f"{i * 0.005:.3f}",
                    "phase": "tdpa_press",
                    "fz": f"{0.5 + i * 0.04:.3f}",
                    "v_cmd_z": "0.008",
                    "dt_actual_s": "0.005",
                    "tdpa_e_obs_j": f"{e:.6f}",
                    "tdpa_alpha": "0.0",
                    "tdpa_clamped": "0",
                }
            )
    out = tmp_path / "id_fit"
    report = write_identification_report(
        air_dir=air, press_paths=[press], out_dir=out
    )
    assert report["plant"]["t0_s"] == pytest.approx(0.035)
    assert report["plant"]["write_single_fopdt"] is True
    assert report["tdpa_sign"]["ok"] is True
    assert (out / "identification.json").is_file()
    assert (out / "contact_ke.csv").is_file()
    assert (out / "08_contact_press.png").is_file()
    assert (out / "controller_ref.csv").is_file()
    assert (out / "id_reference.log").is_file()


def test_air_campaign_does_not_reuse_stop_speeds() -> None:
    assert AIR_STEPS_MM_S != STEPS_MM_S
    assert 4.0 in AIR_STEPS_MM_S
    assert 12.0 in AIR_STEPS_MM_S
    assert 15.0 in AIR_STEPS_MM_S
