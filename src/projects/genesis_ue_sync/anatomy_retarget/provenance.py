"""Content-addressed provenance and coordinate contracts for anatomy runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable


CANONICAL_COORDINATE_SYSTEM = "smplx_y_up_m"
VIEWER_COORDINATE_SYSTEM = "genesis_z_up_m"
RUN_MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
        "diff_hash": hashlib.sha256(
            subprocess.run(
                ("git", "diff", "--binary", "HEAD"),
                cwd=repo_root,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest(),
    }


def _hash_existing(paths: Iterable[Path | str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256_file(path)
    return result


def build_run_manifest(
    *,
    repo_root: Path | str,
    blend_file: Path | str,
    motion_npz: Path | str | None,
    canonical_files: Iterable[Path | str],
    config_files: Iterable[Path | str],
    code_files: Iterable[Path | str],
    solver_versions: dict[str, str],
    random_seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest before any cache lookup or publish."""
    root = Path(repo_root).resolve()
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "git": _git_state(root),
        "inputs": {
            "blend": {
                "path": str(Path(blend_file).resolve()),
                "sha256": sha256_file(blend_file),
            },
            "motion_npz": (
                None
                if motion_npz is None
                else {
                    "path": str(Path(motion_npz).resolve()),
                    "sha256": sha256_file(motion_npz),
                }
            ),
            "canonical_files": _hash_existing(canonical_files),
            "config_files": _hash_existing(config_files),
        },
        "code_files": _hash_existing(code_files),
        "solver_versions": dict(sorted((str(k), str(v)) for k, v in solver_versions.items())),
        "coordinate_contract": {
            "asset": CANONICAL_COORDINATE_SYSTEM,
            "viewer": VIEWER_COORDINATE_SYSTEM,
            "viewer_transform_count": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "random_seed": int(random_seed),
        },
        "extra": dict(extra or {}),
    }
    manifest["content_hash"] = stable_json_hash(manifest)
    return manifest


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Durably replace a small manifest/pointer in one filesystem."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
