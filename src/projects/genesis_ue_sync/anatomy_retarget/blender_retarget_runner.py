"""Headless Blender runners for anatomy retargeting tasks."""

from __future__ import annotations

import os
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import discover_blender_executable


BLENDER_SCRIPT_DIR = Path(__file__).resolve().parent / "blender_scripts"


@dataclass(frozen=True)
class BlenderRunResult:
    ok: bool
    command: list[str]
    log_path: Path
    elapsed_s: float
    returncode: int


def resolve_blender_binary() -> str:
    env_bin = os.environ.get("AMONGUS_BLENDER_BIN", "").strip()
    if env_bin:
        candidate = Path(env_bin).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return discover_blender_executable(project_paths(__file__))


def _run_blender(
    cmd: list[str],
    *,
    log_path: Path,
    timeout_s: float,
) -> BlenderRunResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
            check=False,
        )
    elapsed = float(time.perf_counter() - t0)
    return BlenderRunResult(
        ok=(int(proc.returncode) == 0),
        command=list(cmd),
        log_path=log_path,
        elapsed_s=elapsed,
        returncode=int(proc.returncode),
    )


def _mapping_for_blender(mapping_path: Path, *, work_dir: Path) -> Path:
    src = Path(mapping_path).expanduser().resolve()
    if src.suffix.lower() == ".json":
        return src
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    except Exception:
        payload = json.loads(src.read_text(encoding="utf-8"))
    out = work_dir / "anatomy_retarget_mapping.runtime.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload or {}, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def run_rig_inspect(
    *,
    blend_path: Path,
    output_json: Path,
    log_path: Path | None = None,
    timeout_s: float = 120.0,
    max_vertex_groups: int = 256,
) -> BlenderRunResult:
    blender = resolve_blender_binary()
    script = BLENDER_SCRIPT_DIR / "blender_rig_inspect.py"
    log = log_path or (output_json.parent / "blender_rig_inspect.log")
    cmd = [
        blender,
        "-b",
        str(Path(blend_path).expanduser().resolve()),
        "--python",
        str(script),
        "--",
        f"--output={Path(output_json).expanduser().resolve()}",
        f"--max-vertex-groups={int(max_vertex_groups)}",
    ]
    return _run_blender(cmd, log_path=log, timeout_s=timeout_s)


def run_retarget(
    *,
    blend_path: Path,
    canonical_dir: Path,
    mapping_path: Path,
    output_npz: Path,
    output_glb: Path,
    report_json: Path,
    log_path: Path | None = None,
    timeout_s: float = 900.0,
) -> BlenderRunResult:
    blender = resolve_blender_binary()
    script = BLENDER_SCRIPT_DIR / "blender_retarget_script.py"
    log = log_path or (report_json.parent / "blender_retarget.log")
    mapping_for_blender = _mapping_for_blender(Path(mapping_path), work_dir=Path(report_json).parent)
    cmd = [
        blender,
        "-b",
        str(Path(blend_path).expanduser().resolve()),
        "--python",
        str(script),
        "--",
        f"--canonical-dir={Path(canonical_dir).expanduser().resolve()}",
        f"--mapping={mapping_for_blender}",
        f"--output-npz={Path(output_npz).expanduser().resolve()}",
        f"--output-glb={Path(output_glb).expanduser().resolve()}",
        f"--report-json={Path(report_json).expanduser().resolve()}",
    ]
    return _run_blender(cmd, log_path=log, timeout_s=timeout_s)

