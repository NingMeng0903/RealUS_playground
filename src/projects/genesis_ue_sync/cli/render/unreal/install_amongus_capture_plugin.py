"""Install AmongUsRealtimeCapture C++ plugin into the BE_IBL UE project (idempotent).

Steps performed:
1. Copy the plugin source tree into ``<project>/Plugins/AmongUsRealtimeCapture``.
2. Optionally compile it offline via ``RunUAT.sh BuildPlugin`` so the editor finds the
   Linux binary at startup (UE 5.3 launched with ``-stdout`` does not show the
   "Missing modules - rebuild?" prompt and would otherwise abort).
3. Patch ``<project>.uproject`` to mark the plugin enabled.

Run once per UE engine + project after pulling fresh source. Safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_PROJECT_PATH = Path(
    "assets/humans/bedlam2/unreal/projects/BE_IBL/BE_IBL.uproject"
)
PLUGIN_SOURCE_REL = Path(
    "src/projects/genesis_ue_sync/integrations/ue/cpp_plugin/AmongUsRealtimeCapture"
)
PLUGIN_NAME = "AmongUsRealtimeCapture"


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Cannot locate repository root (expected /src and /configs).")


def _copy_plugin_tree(src: Path, dst: Path) -> dict:
    if not src.is_dir():
        raise FileNotFoundError(f"Plugin source not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = sum(1 for _ in dst.rglob("*") if _.is_file())
    return {"plugin_dir": str(dst), "copied_file_count": int(files)}


def _patch_uproject(uproject_path: Path, plugin_name: str) -> dict:
    if not uproject_path.is_file():
        raise FileNotFoundError(f".uproject not found: {uproject_path}")
    data = json.loads(uproject_path.read_text(encoding="utf-8"))
    plugins = list(data.get("Plugins") or [])
    existing_names = {str((entry or {}).get("Name", "")) for entry in plugins}
    added = False
    if plugin_name not in existing_names:
        plugins.append({"Name": plugin_name, "Enabled": True})
        added = True
    data["Plugins"] = plugins
    uproject_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"uproject": str(uproject_path), "added_entry": bool(added)}


def _unpatch_uproject(uproject_path: Path, plugin_name: str) -> dict:
    if not uproject_path.is_file():
        return {"uproject": str(uproject_path), "removed_entry": False}
    data = json.loads(uproject_path.read_text(encoding="utf-8"))
    plugins = list(data.get("Plugins") or [])
    new_plugins = [entry for entry in plugins if str((entry or {}).get("Name", "")) != plugin_name]
    removed = len(new_plugins) != len(plugins)
    data["Plugins"] = new_plugins
    uproject_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"uproject": str(uproject_path), "removed_entry": bool(removed)}


def _resolve_run_uat(unreal_root: Path | None) -> Path:
    candidates: list[Path] = []
    if unreal_root is not None:
        candidates.append(unreal_root / "Engine" / "Build" / "BatchFiles" / "RunUAT.sh")
    env = os.environ.get("UNREAL_ENGINE_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser() / "Engine" / "Build" / "BatchFiles" / "RunUAT.sh")
    candidates.extend(
        [
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine-5.3.2/Engine/Build/BatchFiles/RunUAT.sh"),
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine/Engine/Build/BatchFiles/RunUAT.sh"),
        ]
    )
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    raise FileNotFoundError(
        "Cannot locate RunUAT.sh; pass --unreal-root /path/to/UnrealEngine or set UNREAL_ENGINE_DIR."
    )


def _build_plugin_in_place(uplugin_path: Path, run_uat: Path, target_platforms: list[str]) -> dict:
    """Run RunUAT.sh BuildPlugin into a temp dir, then copy Binaries/ + Intermediate/Build/ into the project plugin dir."""
    import tempfile

    plugin_root = uplugin_path.parent
    with tempfile.TemporaryDirectory(prefix="among_us_plugin_pkg_") as tmp:
        out_dir = Path(tmp) / "packaged"
        cmd = [
            str(run_uat),
            "BuildPlugin",
            f"-Plugin={uplugin_path}",
            f"-Package={out_dir}",
            f"-TargetPlatforms={'+'.join(target_platforms)}",
            "-Rocket",
        ]
        logging.info("Compiling plugin: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_tail = "\n".join((proc.stdout or "").splitlines()[-30:] + (proc.stderr or "").splitlines()[-30:])
        if proc.returncode != 0:
            logging.error("BuildPlugin failed (exit=%s):\n%s", proc.returncode, log_tail)
            raise RuntimeError("RunUAT.sh BuildPlugin failed; see log above.")
        binaries_src = out_dir / "Binaries"
        if not binaries_src.is_dir():
            raise RuntimeError(f"BuildPlugin succeeded but {binaries_src} missing.")
        binaries_dst = plugin_root / "Binaries"
        if binaries_dst.exists():
            shutil.rmtree(binaries_dst)
        shutil.copytree(binaries_src, binaries_dst)
        intermediate_src = out_dir / "Intermediate"
        intermediate_dst = plugin_root / "Intermediate"
        if intermediate_src.is_dir():
            if intermediate_dst.exists():
                shutil.rmtree(intermediate_dst)
            shutil.copytree(intermediate_src, intermediate_dst)
        artifact_files = sum(1 for _ in binaries_dst.rglob("*") if _.is_file())
    return {
        "binaries_dir": str(binaries_dst),
        "artifact_file_count": int(artifact_files),
        "target_platforms": list(target_platforms),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install AmongUsRealtimeCapture into BE_IBL UE project.")
    p.add_argument("--uproject", type=Path, default=None, help="Path to BE_IBL.uproject (default: repo BE_IBL).")
    p.add_argument("--plugin-source", type=Path, default=None, help="Override plugin source dir.")
    p.add_argument("--unreal-root", type=Path, default=None, help="Override UnrealEngine root dir.")
    p.add_argument("--target-platforms", type=str, default="Linux", help="+ separated UE target platforms (default Linux).")
    p.add_argument("--skip-build", action="store_true", help="Copy + patch only; skip RunUAT BuildPlugin.")
    p.add_argument("--keep-source-only", action="store_true", help="Erase Binaries/Intermediate before patching (rebuild required).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    repo = _resolve_repo_root()
    uproject = (args.uproject or (repo / DEFAULT_PROJECT_PATH)).resolve()
    plugin_src = (args.plugin_source or (repo / PLUGIN_SOURCE_REL)).resolve()

    plugin_dst = uproject.parent / "Plugins" / PLUGIN_NAME
    plugin_dst.parent.mkdir(parents=True, exist_ok=True)

    # First detach the .uproject entry so a previous failed launch state cannot block this run.
    _unpatch_uproject(uproject, PLUGIN_NAME)

    copy_summary = _copy_plugin_tree(plugin_src, plugin_dst)
    if args.keep_source_only:
        for sub in ("Binaries", "Intermediate"):
            target = plugin_dst / sub
            if target.exists():
                shutil.rmtree(target)

    build_summary: dict | None = None
    if not args.skip_build:
        run_uat = _resolve_run_uat(args.unreal_root)
        targets = [item.strip() for item in str(args.target_platforms).split("+") if item.strip()]
        uplugin_path = plugin_dst / f"{PLUGIN_NAME}.uplugin"
        build_summary = _build_plugin_in_place(uplugin_path, run_uat, targets)

    patch_summary = _patch_uproject(uproject, PLUGIN_NAME)

    out = {"plugin": copy_summary, "build": build_summary, "uproject": patch_summary}
    logging.info("AmongUsRealtimeCapture installed: %s", json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
