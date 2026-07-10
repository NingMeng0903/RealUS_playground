from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import unreal


SCRIPT_PATH = Path(__file__).resolve()
for _candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
    _common = _candidate / "common" / "project.py"
    if _common.is_file():
        _src_root = str(_candidate)
        if _src_root not in sys.path:
            sys.path.insert(0, _src_root)
        break
else:
    raise RuntimeError(f"Cannot locate src root (common/project.py) for {SCRIPT_PATH}")

from common.project import discover_project_root

REPO_ROOT = discover_project_root(SCRIPT_PATH)
RETARGET_MODULE_PATH = REPO_ROOT / "ref_code_library" / "bedlam2_retargeting" / "retargeting" / "Content" / "Python" / "retarget.py"
DEFAULT_IK_RETARGETER_PATH = "/Game/BodyModels/Smplx/smplx_IKRetargeter"
DEFAULT_SOURCE_IK_RIG_PATH = "/Game/BodyModels/Smplx/smplx_IKRig"
DEFAULT_TARGET_IK_RIG_PATH = "/Game/BodyModels/Smplx/smplx_IKRig"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preferred_fbx_anim_length_mode():
    lt = getattr(unreal, "FBXAnimationLengthImportType", None)
    if lt is None:
        return None
    # Prefer keyframe extent: Blender FBX time span headers are often wrong (~2s) while keys cover full clip.
    for name in ("FBXALIT_ANIMATED_KEY", "FBXALIT_EXPORTED_TIME"):
        mode = getattr(lt, name, None)
        if mode is not None:
            return mode
    return None


def _apply_anim_sequence_import_settings(options: unreal.FbxImportUI) -> None:
    if not bool(getattr(options, "import_animations", False)):
        return
    asd = getattr(options, "anim_sequence_import_data", None)
    if asd is None:
        unreal.log_warning("ANIMATION_ASSET_SESSION: anim_sequence_import_data is None")
        return
    mode = _preferred_fbx_anim_length_mode()
    if mode is None:
        unreal.log_warning("ANIMATION_ASSET_SESSION: FBXAnimationLengthImportType unavailable")
        return
    try:
        asd.set_editor_property("animation_length", mode)
    except Exception:
        try:
            asd.animation_length = mode
        except Exception as exc:
            unreal.log_warning(f"ANIMATION_ASSET_SESSION: could not set animation_length: {exc}")
    unreal.log(f"ANIMATION_ASSET_SESSION: anim import animation_length={mode}")


def _build_fbx_options(animation: bool, skeleton_path: str | None) -> unreal.FbxImportUI:
    options = unreal.FbxImportUI()
    options.import_mesh = True
    options.import_textures = False
    options.import_materials = False
    options.import_as_skeletal = True
    options.import_animations = bool(animation)
    options.create_physics_asset = False
    if animation:
        options.mesh_type_to_import = unreal.FBXImportType.FBXIT_SKELETAL_MESH
        _apply_anim_sequence_import_settings(options)
    if skeleton_path:
        skeleton = unreal.load_asset(skeleton_path)
        if skeleton is None:
            raise RuntimeError(f"Cannot load skeleton: {skeleton_path}")
        options.skeleton = skeleton
    return options


def _purge_assets_with_prefix(package_dir: str, name_prefix: str) -> None:
    if not unreal.EditorAssetLibrary.does_directory_exist(package_dir):
        return
    for ap in unreal.EditorAssetLibrary.list_assets(package_dir, recursive=True, include_folder=False):
        short = ap.rsplit("/", maxsplit=1)[-1]
        if short.startswith(name_prefix):
            unreal.log(f"ANIMATION_ASSET_SESSION: delete stale asset {ap}")
            unreal.EditorAssetLibrary.delete_asset(ap)


def import_fbx_asset(
    fbx_path: Path,
    destination_path: str,
    destination_name: str,
    *,
    animation: bool = True,
    skeleton_path: str | None = None,
) -> list[str]:
    if not fbx_path.is_file():
        raise RuntimeError(f"FBX does not exist: {fbx_path}")
    unreal.EditorAssetLibrary.make_directory(destination_path)
    task = unreal.AssetImportTask()
    task.filename = str(fbx_path)
    task.destination_path = destination_path
    task.destination_name = destination_name
    task.automated = True
    task.save = True
    task.replace_existing = True
    task.options = _build_fbx_options(animation=animation, skeleton_path=skeleton_path)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported_paths = [str(item) for item in task.get_editor_property("imported_object_paths")]
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
        save_map_packages=True, save_content_packages=True
    )
    return imported_paths


def export_anim_sequence(asset_path: str, output_fbx: Path, *, export_preview_mesh: bool = False) -> None:
    asset = unreal.load_asset(asset_path)
    if asset is None:
        raise RuntimeError(f"Cannot load asset: {asset_path}")
    if not isinstance(asset, unreal.AnimSequence):
        raise RuntimeError(f"Asset is not an AnimSequence: {asset_path}")
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    export_task = unreal.AssetExportTask()
    export_task.automated = True
    export_task.object = asset
    export_task.prompt = False
    export_task.filename = str(output_fbx)
    export_task.options = unreal.FbxExportOption()
    export_task.options.set_editor_property("bExportPreviewMesh", bool(export_preview_mesh))
    export_task.exporter = unreal.AnimSequenceExporterFBX()
    export_task.replace_identical = True
    export_task.exporter.run_asset_export_task(export_task)


def _retarget_asset_name(target_asset_dir: str, source_asset_name: str) -> str:
    target_name = target_asset_dir.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return f"{target_name}+{source_asset_name}_Anim"


def _discover_retargeted_anim_package_path(
    output_asset_dir: str,
    import_destination_path: str,
    source_asset_name: str,
    target_short: str,
) -> str | None:
    expected_stem = f"{target_short}+{source_asset_name}_Anim"
    for base in (output_asset_dir, import_destination_path):
        if not unreal.EditorAssetLibrary.does_directory_exist(base):
            continue
        for ap in unreal.EditorAssetLibrary.list_assets(base, recursive=True, include_folder=False):
            leaf = ap.rsplit("/", maxsplit=1)[-1]
            stem = leaf.split(".", 1)[0]
            if stem != expected_stem and not stem.startswith(expected_stem):
                continue
            pkg_path = ap.split(".", 1)[0] if "." in ap else ap
            asset = unreal.load_asset(pkg_path)
            if asset is not None and isinstance(asset, unreal.AnimSequence):
                unreal.log(f"ANIMATION_ASSET_SESSION: discovered retargeted AnimSequence at {pkg_path}")
                return str(pkg_path)
    return None


def run_animation_asset_session(cfg: dict) -> None:
    retarget = _load_module("bedlam_retarget", RETARGET_MODULE_PATH)

    input_fbx_path = Path(cfg["input_fbx_path"]).resolve()
    import_destination_path = str(cfg["import_destination_path"]).strip()
    import_destination_name = str(cfg["import_destination_name"]).strip()
    source_asset_root = str(cfg["source_asset_root"]).strip()
    source_asset_name = str(cfg["source_asset_name"]).strip()
    target_asset_dir = str(cfg["target_asset_dir"]).strip()
    output_asset_dir = str(cfg["output_asset_dir"]).strip()
    output_fbx_path = Path(cfg["output_fbx_path"]).resolve()
    export_preview_mesh = bool(cfg.get("export_preview_mesh", False))
    ik_retargeter_path = str(cfg.get("ik_retargeter_path", DEFAULT_IK_RETARGETER_PATH)).strip()
    source_ik_rig_path = str(cfg.get("source_ik_rig_path", DEFAULT_SOURCE_IK_RIG_PATH)).strip()
    target_ik_rig_path = str(cfg.get("target_ik_rig_path", DEFAULT_TARGET_IK_RIG_PATH)).strip()

    unreal.log("ANIMATION_ASSET_SESSION: import FBX")
    _purge_assets_with_prefix(import_destination_path, import_destination_name)
    target_short = target_asset_dir.rstrip("/").rsplit("/", maxsplit=1)[-1]
    retarget_prefix = f"{target_short}+{source_asset_name}"
    _purge_assets_with_prefix(output_asset_dir, retarget_prefix)

    import_fbx_asset(
        fbx_path=input_fbx_path,
        destination_path=import_destination_path,
        destination_name=import_destination_name,
        animation=True,
        skeleton_path=None,
    )

    source_skel_path = f"{source_asset_root}/animations/{source_asset_name}/{source_asset_name}"
    source_anim_path = f"{source_asset_root}/animations/{source_asset_name}/{source_asset_name}_Anim"
    unreal.log(
        "ANIMATION_ASSET_SESSION: "
        f"source_skel={source_skel_path} source_anim={source_anim_path} "
        f"target_dir={target_asset_dir} out_dir={output_asset_dir}"
    )
    retarget.execute(
        target_dir=target_asset_dir,
        source_skel_path=source_skel_path,
        source_anim_path=source_anim_path,
        ik_retargeter_path=ik_retargeter_path,
        source_ik_rig_path=source_ik_rig_path,
        target_ik_rig_path=target_ik_rig_path,
        out_dir=output_asset_dir,
    )
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
        save_map_packages=True, save_content_packages=True
    )

    def _log_anim_len(label: str, asset_path: str) -> None:
        asset = unreal.load_asset(asset_path)
        if asset is None:
            unreal.log_warning(f"ANIMATION_ASSET_SESSION: {label} missing {asset_path}")
            return
        if not isinstance(asset, unreal.AnimSequence):
            unreal.log_warning(f"ANIMATION_ASSET_SESSION: {label} not AnimSequence: {asset_path}")
            return
        try:
            tlen = float(asset.get_editor_property("sequence_length"))
        except Exception:
            tlen = -1.0
        unreal.log(f"ANIMATION_ASSET_SESSION: {label} sequence_length_s={tlen} path={asset_path}")

    _log_anim_len("after_import_source", source_anim_path)

    expected_name = _retarget_asset_name(target_asset_dir, source_asset_name)
    output_asset_path = f"{output_asset_dir.rstrip('/')}/{expected_name}"
    discovered = _discover_retargeted_anim_package_path(
        output_asset_dir, import_destination_path, source_asset_name, target_short
    )
    if discovered is not None:
        output_asset_path = discovered
    elif not unreal.EditorAssetLibrary.does_asset_exist(output_asset_path):
        raise RuntimeError(
            "ANIMATION_ASSET_SESSION: retargeted AnimSequence not found at "
            f"{output_asset_path}; check IK retarget logs and asset directories."
        )
    _log_anim_len("after_retarget", output_asset_path)
    unreal.log(
        f"ANIMATION_ASSET_SESSION: export asset={output_asset_path} -> {output_fbx_path} "
        f"mesh={int(export_preview_mesh)}"
    )
    export_anim_sequence(
        asset_path=output_asset_path,
        output_fbx=output_fbx_path,
        export_preview_mesh=export_preview_mesh,
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: ue_animation_asset_session.py <session_config.json>\n"
            "JSON keys: input_fbx_path, import_destination_path, import_destination_name, "
            "source_asset_root, source_asset_name, target_asset_dir, output_asset_dir, output_fbx_path; "
            "optional export_preview_mesh, ik_retargeter_path, source_ik_rig_path, target_ik_rig_path."
        )
    config_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_animation_asset_session(cfg)


if __name__ == "__main__":
    main()
