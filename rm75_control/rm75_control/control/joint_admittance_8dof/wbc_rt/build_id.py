"""Content hashes for the native inner and its Python protocol twin."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve()
_NATIVE = _ROOT.parents[4] / "native" / "wbc_rt"
_PROTOCOL_PY = Path(__file__).resolve().parent / "protocol.py"

_HASH_SOURCES = (
    _NATIVE / "include" / "wbc_rt" / "protocol.hpp",
    _NATIVE / "include" / "wbc_rt" / "inner.hpp",
    _NATIVE / "include" / "wbc_rt" / "types.hpp",
    _NATIVE / "src" / "inner.cpp",
    _NATIVE / "src" / "main.cpp",
    _PROTOCOL_PY,
)


def file_sha256(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def tree_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in _HASH_SOURCES:
        rel = str(path)
        try:
            rel = str(path.relative_to(_NATIVE.parents[1]))
        except ValueError:
            rel = path.name
        out[rel] = file_sha256(path) if path.is_file() else ""
    return out


def combined_hash(manifest: dict[str, str] | None = None) -> str:
    items = manifest or tree_manifest()
    blob = "".join(f"{k}={v}\n" for k, v in sorted(items.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_native_matches_tree(embedded: str | None) -> None:
    """Refuse to talk to a binary built from a different source snapshot."""

    expect = combined_hash()
    if not embedded:
        raise RuntimeError(
            "wbc_rt binary has no embedded source hash; rebuild native/wbc_rt"
        )
    if str(embedded).strip() != expect:
        raise RuntimeError(
            "wbc_rt source/binary hash mismatch: "
            f"tree={expect[:12]} binary={str(embedded).strip()[:12]}. "
            "Rebuild native/wbc_rt before hardware."
        )
