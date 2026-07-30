from __future__ import annotations

import sys

from projects.genesis_ue_sync.anatomy_retarget.cli.run_upgrade_anatomy_v810 import (
    parse_args,
)
from projects.genesis_ue_sync.anatomy_retarget.version_v8 import (
    SOURCE_OPERATOR_ALGORITHM_VERSION,
    SOURCE_OPERATOR_CORRECTION_VERSION,
    SOURCE_OPERATOR_ORACLE_VERSION,
    SUBJECT_SOLVER_VERSION,
)


def test_v811_cache_versions_and_upgrade_defaults(
    monkeypatch,
) -> None:
    assert SOURCE_OPERATOR_ALGORITHM_VERSION == "contact-first-joint-chain-v8.11"
    assert SOURCE_OPERATOR_ORACLE_VERSION == "smplx-joint-contact-chain-v8.11"
    assert (
        SOURCE_OPERATOR_CORRECTION_VERSION
        == "selective-fk-volume-corrective-v8.11"
    )
    assert SUBJECT_SOLVER_VERSION == "selective-fk-foot-tube-v8.11"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_upgrade_anatomy_v810.py",
            "--operator",
            "operator",
            "--output",
            "output",
            "--foot-product",
            "foot-product",
        ],
    )
    args = parse_args()

    assert args.algorithm_version == SOURCE_OPERATOR_ALGORITHM_VERSION
    assert args.oracle_version == SOURCE_OPERATOR_ORACLE_VERSION
    assert args.correction_version == SOURCE_OPERATOR_CORRECTION_VERSION
