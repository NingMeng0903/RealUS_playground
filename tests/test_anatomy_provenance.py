from __future__ import annotations

import json
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.provenance import (
    CANONICAL_COORDINATE_SYSTEM,
    VIEWER_COORDINATE_SYSTEM,
    atomic_write_json,
    sha256_file,
    stable_json_hash,
)


def test_stable_json_hash_ignores_mapping_order() -> None:
    assert stable_json_hash({"a": 1, "b": [2, 3]}) == stable_json_hash(
        {"b": [2, 3], "a": 1}
    )


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "latest.json"
    atomic_write_json(target, {"run": "first"})
    atomic_write_json(target, {"run": "second", "passed": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "passed": True,
        "run": "second",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_file_hash_and_coordinate_contract_are_explicit(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"anatomy")
    assert sha256_file(payload) == sha256_file(payload)
    assert CANONICAL_COORDINATE_SYSTEM == "smplx_y_up_m"
    assert VIEWER_COORDINATE_SYSTEM == "genesis_z_up_m"
