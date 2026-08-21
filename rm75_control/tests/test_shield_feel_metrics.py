"""Offline shield feel report."""

from __future__ import annotations

import csv
from pathlib import Path

from peirastic.apps.shield_feel_metrics import compute_metrics, report


def test_feel_metrics_counts_activation_and_pierce(tmp_path: Path) -> None:
    rows = [
        {
            "contact_present": "1",
            "lambda_obs": "1.00",
            "u_nom_capped": "0.020",
            "u_sent": "0.020",
            "fz": "2.0",
            "f_ub_n": "3.0",
            "physical_contact_loss_event": "0",
            "shield_applied": "0",
        },
        {
            "contact_present": "1",
            "lambda_obs": "0.40",
            "u_nom_capped": "0.040",
            "u_sent": "0.010",
            "fz": "4.2",
            "f_ub_n": "3.5",
            "physical_contact_loss_event": "1",
            "shield_applied": "1",
        },
        {
            "contact_present": "1",
            "lambda_obs": "0.50",
            "u_nom_capped": "0.030",
            "u_sent": "0.012",
            "fz": "3.0",
            "f_ub_n": "3.4",
            "physical_contact_loss_event": "0",
            "shield_applied": "1",
        },
        {
            "contact_present": "0",
            "lambda_obs": "1.00",
            "u_nom_capped": "0.000",
            "u_sent": "0.000",
            "fz": "0.1",
            "f_ub_n": "3.0",
            "physical_contact_loss_event": "0",
            "shield_applied": "0",
        },
    ]
    metrics = compute_metrics(rows, eps_lambda=0.02)
    assert metrics["contact_n"] == 3
    assert metrics["shield_active_frac"] == 2.0 / 3.0
    assert metrics["lambda_p05"] < 0.45
    assert metrics["rms_u_sent_minus_u_nom"] > 0.0
    assert metrics["longest_active_ticks"] == 2
    assert metrics["f_ub_pierce"] == 1
    assert metrics["losses"] == 1
    assert metrics["failsafe_frac"] == 0.0
    assert metrics["effective_intervention_frac"] == 2.0 / 3.0

    path = tmp_path / "run.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assert report(path) == 0
