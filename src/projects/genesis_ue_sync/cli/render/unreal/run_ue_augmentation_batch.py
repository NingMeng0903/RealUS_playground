#!/usr/bin/env python3
"""Run a batch of UE augmentation renders for the same SyncSceneSpec through one reusable editor session."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import discover_unreal_editor_executable
from projects.genesis_ue_sync.cli.render.unreal.official_retarget_fbx_host import ensure_official_retarget_fbx_cached
from projects.genesis_ue_sync.integrations.ue import (
    EditorCommand,
    EditorSessionPaths,
    amongus_tool_env_for_ue_editor,
    ensure_editor_session,
    wait_for_command_result,
)
from projects.genesis_ue_sync.sim_platform.scenes import (
    SceneAugmentationSpec,
    load_scene_augmentation_spec,
    scene_augmentation_to_dict,
)

PROJECT_PATHS = project_paths(__file__)
UE_WATCHER_SCRIPT = (
    REPO_ROOT
    / "src"
    / "projects"
    / "genesis_ue_sync"
    / "cli"
    / "render"
    / "unreal"
    / "ue_editor_session_watch.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path, default=PROJECT_PATHS.default_scene_spec_path)
    parser.add_argument("--augmentation-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_PATHS.outputs_root / "ue_augmentation_batch")
    parser.add_argument("--session-dir", type=Path, default=PROJECT_PATHS.outputs_root / "ue_sessions" / "be_ibl")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--force-rebuild-motion", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--unreal-editor", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def _variant_spec(
    base: SceneAugmentationSpec,
    *,
    sample_index: int,
    seed: int,
) -> SceneAugmentationSpec:
    task = {**dict(base.task), "sample_index": int(sample_index), "batch_seed": int(seed)}
    metadata = {**dict(base.metadata), "sample_index": int(sample_index), "batch_seed": int(seed)}
    return dataclasses.replace(
        base,
        name=f"{base.name}_sample_{sample_index:03d}",
        seed=int(seed),
        task=task,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    base_spec = load_scene_augmentation_spec(args.augmentation_spec.expanduser().resolve())
    seed_start = int(args.seed_start) if args.seed_start is not None else int(base_spec.seed or 0)
    output_root = args.output_root.expanduser().resolve()
    generated_root = output_root / "_generated_augmentations"
    generated_root.mkdir(parents=True, exist_ok=True)

    unreal_cmd = args.unreal_editor or discover_unreal_editor_executable(PROJECT_PATHS)
    ensure_official_retarget_fbx_cached(
        scene_spec_path=args.scene_spec.expanduser().resolve(),
        augmentation_spec_path=args.augmentation_spec.expanduser().resolve(),
        output_root=output_root,
        force_rebuild=bool(args.force_rebuild_motion),
        unreal_cmd=unreal_cmd,
    )
    session_paths = EditorSessionPaths(args.session_dir.expanduser().resolve())
    ensure_editor_session(
        session_paths,
        unreal_cmd=unreal_cmd,
        project_path=PROJECT_PATHS.bedlam_unreal_project_file,
        watcher_script=UE_WATCHER_SCRIPT,
        log_path=session_paths.root / "editor_session.log",
        timeout_s=float(args.timeout_s),
    )

    summary: list[dict[str, object]] = []
    for sample_index in range(max(int(args.sample_count), 0)):
        seed = int(seed_start + sample_index * int(args.seed_step))
        variant = _variant_spec(base_spec, sample_index=sample_index, seed=seed)
        variant_path = generated_root / f"{variant.name}.json"
        variant_path.write_text(json.dumps(scene_augmentation_to_dict(variant), indent=2), encoding="utf-8")
        sample_output_root = output_root / f"sample_{sample_index:03d}"
        command = EditorCommand(
            command_type="prepare_render_pipeline",
            payload={
                "output_root": str(sample_output_root),
                "scene_spec_path": str(args.scene_spec.expanduser().resolve()),
                "augmentation_spec_path": str(variant_path),
                "render_now": not bool(args.prepare_only),
                "force_rebuild_motion": bool(args.force_rebuild_motion),
                "quit_editor_on_finish": False,
                "tool_env": amongus_tool_env_for_ue_editor(),
            },
        )
        command.write(session_paths)
        result = wait_for_command_result(session_paths, command.request_id, timeout_s=max(float(args.timeout_s), 3600.0))
        if not result.success:
            raise RuntimeError(f"UE augmentation sample {sample_index} failed: {result.detail}")
        summary.append(
            {
                "sample_index": sample_index,
                "seed": seed,
                "augmentation_spec": str(variant_path),
                "output_root": str(sample_output_root),
                "request_id": command.request_id,
                "sequence_names": result.payload.get("sequence_names", []),
            }
        )

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"UE augmentation batch complete: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
