"""Build a small, offline HTML index for saved V16 Genesis renders.

The page only links to RGB files that already exist under a V16 render root.
It never loads a mesh, reruns Genesis, changes a geometry cell, or draws a
replacement model.  The optional skin slider is a CSS wipe between two saved
Genesis images from the same cell, variant, camera, and pose.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


SKIN_LAYERS = frozenset({"skin", "skin_only", "smplx_skin"})
INTERNAL_LAYER_PREFERENCE = ("context", "viscera", "torso", "pelvis")


def _resolve_artifact(value: Any, *, manifest_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_root / path
    return path.resolve()


def _rel(path: Path | None, *, page_root: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    return os.path.relpath(path, page_root)


def _subject_from_cell(cell: str) -> str:
    token = str(cell).split("_", 1)[0]
    return token or str(cell)


def _read_manifest(render_root: Path) -> tuple[Path, dict[str, Any]]:
    candidate = render_root.resolve()
    manifest_path = candidate if candidate.is_file() else candidate / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V16 manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cells"), dict):
        raise ValueError(f"invalid V16 manifest (missing cells): {manifest_path}")
    return manifest_path.parent.resolve(), data


def _comparison_candidates(
    *, cell_root: Path, layer: str, view: str
) -> list[Path]:
    return [
        cell_root / "comparison" / layer / f"{view}.png",
        cell_root / "comparison" / f"{view}.png",
    ]


def _sibling_skin(
    *, variant_root: Path, view: str
) -> Path | None:
    for layer in ("skin", "skin_only", "smplx_skin"):
        path = variant_root / layer / "rgb" / f"{view}.png"
        if path.is_file():
            return path.resolve()
    return None


def _sibling_internal(
    *, variant_root: Path, view: str, preferred: str | None = None
) -> Path | None:
    names: list[str] = []
    if preferred:
        names.append(preferred)
    names.extend(INTERNAL_LAYER_PREFERENCE)
    seen: set[str] = set()
    for layer in names:
        if layer in seen or layer in SKIN_LAYERS:
            continue
        seen.add(layer)
        path = variant_root / layer / "rgb" / f"{view}.png"
        if path.is_file():
            return path.resolve()
    return None


def _make_entries(
    *,
    render_root: Path,
    manifest: dict[str, Any],
    page_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str, str, str, str]] = set()
    comparisons: dict[tuple[str, str, str], Path] = {}
    flat_comparisons: dict[tuple[str, str], Path] = {}

    # First collect explicit comparison files.  A nested V16 path is tied to
    # its layer; a historical flat path is attached to the first matching
    # layer below and remains visible in the extras gallery either way.
    for cell_name in manifest.get("cells", {}):
        cell_root = render_root / str(cell_name)
        comparison_root = cell_root / "comparison"
        if not comparison_root.is_dir():
            continue
        for path in sorted(comparison_root.rglob("*.png")):
            relative = path.relative_to(comparison_root)
            if len(relative.parts) >= 2:
                layer, filename = relative.parts[-2], relative.parts[-1]
                comparisons[(str(cell_name), layer, path.stem)] = path.resolve()
            else:
                flat_comparisons[(str(cell_name), path.stem)] = path.resolve()

    for raw_cell, raw_cell_data in manifest["cells"].items():
        cell_name = str(raw_cell)
        cell_data = raw_cell_data if isinstance(raw_cell_data, dict) else {}
        cell_root = render_root / cell_name
        subject = _subject_from_cell(cell_name)
        input_path = cell_data.get("input")
        input_sha = cell_data.get("input_sha256")
        variants = cell_data.get("variants", {})
        if not isinstance(variants, dict):
            variants = {}
        for raw_variant, raw_variant_data in variants.items():
            variant = str(raw_variant)
            variant_data = raw_variant_data if isinstance(raw_variant_data, dict) else {}
            layers = variant_data.get("layers", {})
            if not isinstance(layers, dict):
                layers = {}
            variant_root = cell_root / variant
            for raw_layer, raw_layer_data in layers.items():
                layer = str(raw_layer)
                layer_data = raw_layer_data if isinstance(raw_layer_data, dict) else {}
                renders = layer_data.get("renders", [])
                if not isinstance(renders, list):
                    renders = []
                for raw_render in renders:
                    if not isinstance(raw_render, dict):
                        continue
                    view = str(raw_render.get("camera", ""))
                    image = _resolve_artifact(raw_render.get("rgb"), manifest_root=render_root)
                    if image is None or not image.is_file() or not view:
                        # The report can be stale after temporary OBJ cleanup;
                        # discover the same file from the normal V16 layout.
                        image = variant_root / layer / "rgb" / f"{view}.png"
                    image = image.resolve()
                    if not image.is_file() or not view:
                        continue
                    key = (subject, cell_name, variant, layer, view)
                    if key in seen_entries:
                        continue
                    seen_entries.add(key)
                    skin = _sibling_skin(variant_root=variant_root, view=view)
                    internal = _sibling_internal(
                        variant_root=variant_root,
                        view=view,
                        preferred=None if layer in SKIN_LAYERS else layer,
                    )
                    if layer in SKIN_LAYERS:
                        display = skin or image
                        internal = internal or image
                    else:
                        display = image
                        internal = internal or image
                    comparison = comparisons.get((cell_name, layer, view))
                    if comparison is None:
                        comparison = flat_comparisons.get((cell_name, view))
                    entries.append(
                        {
                            "subject": subject,
                            "pose": cell_name,
                            "variant": variant,
                            "layer": layer,
                            "view": view,
                            "path": _rel(display, page_root=page_root),
                            "internal_path": _rel(internal, page_root=page_root),
                            "skin_path": _rel(skin, page_root=page_root),
                            "comparison_path": _rel(comparison, page_root=page_root),
                            "input": str(input_path) if input_path else None,
                            "input_sha256": str(input_sha) if input_sha else None,
                        }
                    )

    # Discovery supplements an incomplete/stale manifest.  This covers
    # candidate-only renders and comparison images produced by a rerun while
    # keeping the manifest as the authoritative source for pose metadata.
    for path in sorted(render_root.rglob("*.png")):
        parts = path.relative_to(render_root).parts
        if "comparison" in parts:
            kind = "comparison"
        elif len(parts) >= 4 and parts[-2] == "rgb":
            kind = "candidate_or_variant"
        else:
            continue
        extras_key = (str(path.resolve()), kind)
        if any(item.get("_key") == extras_key for item in extras):
            continue
        extras.append(
            {
                "_key": extras_key,
                "kind": kind,
                "label": " / ".join(parts),
                "path": _rel(path.resolve(), page_root=page_root),
            }
        )

    # A comparison file already appears in the per-entry selector when it can
    # be paired; keep the full scan in the gallery too so no saved review is
    # silently hidden.
    for item in extras:
        item.pop("_key", None)
    entries.sort(key=lambda item: (item["subject"], item["pose"], item["variant"], item["layer"], item["view"]))
    extras.sort(key=lambda item: (item["kind"], item["label"]))
    return entries, extras


def _filter_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    fields = {
        "subject": args.subject,
        "pose": args.pose,
        "layer": args.layer,
        "view": args.view,
    }
    result = []
    for entry in entries:
        if all(value is None or str(entry[field]).casefold() == str(value).casefold() for field, value in fields.items()):
            result.append(entry)
    if not result:
        raise ValueError("filters selected no saved Genesis RGB entries")
    return result


def _json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _html(*, entries: list[dict[str, Any]], extras: list[dict[str, Any]], render_root: Path) -> str:
    entry_json = _json_script(entries)
    extras_json = _json_script(extras)
    root_label = html.escape(str(render_root))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V16 内脏与骨血管 Genesis 图审</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0d141b;color:#e7edf2;font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:22px}}h1{{font-size:24px;margin:0 0 8px}}
a{{color:#8dccff}}.status{{border-left:4px solid #e2a95d;background:#27231e;padding:12px 16px;margin:14px 0}}
.controls{{display:flex;flex-wrap:wrap;gap:12px 18px;margin:16px 0}}label{{min-width:180px}}select,input{{display:block;margin-top:5px;box-sizing:border-box;padding:8px;background:#1d2a35;border:1px solid #587184;color:#fff;border-radius:4px;width:100%}}
.stage{{background:#071018;border:1px solid #2b3b47;padding:10px}}.wipe{{position:relative;overflow:hidden;background:#071018;min-height:240px}}
.wipe>img{{display:block;width:100%;height:auto;max-height:76vh;object-fit:contain;object-position:center}}
#skinClip{{position:absolute;inset:0;width:100%;height:100%;overflow:hidden;pointer-events:none;clip-path:inset(0 50% 0 0);border-right:2px solid #fff;box-sizing:border-box}}
#skinClip img{{display:block;width:100%;height:100%;max-height:76vh;object-fit:contain;object-position:center}}
.sliderrow{{display:flex;align-items:center;gap:10px;margin:10px 0 0}}.sliderrow input{{margin:0;flex:1}}
.caption{{overflow-wrap:anywhere;color:#b9c7d1}}.secondary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.secondary img{{width:100%;background:#071018;display:block}}details{{margin-top:24px}}summary{{cursor:pointer;font-weight:600}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.tile{{background:#131e27;padding:8px;overflow-wrap:anywhere}}.tile img{{width:100%;display:block;background:#071018;margin-bottom:5px}}
small{{color:#9eacb7}}
</style></head><body><main>
<h1>V16：内脏、静脉、神经与骨骼的真实 Genesis 图审</h1>
<p class="status"><b>解剖验收：未通过（publishable=false）。</b>本页只索引已经保存的 Genesis RGB 图，不重新 retarget、不重新 bake、不从 mesh 重新绘图。图像用于核对同一姿态下内脏—骨架—SMPL-X 皮肤边界以及血管/神经联动。</p>
<p>颜色约定：Artery 红、Vein 蓝、神经 黄；器官按真实 source mesh 名称分别着色。<code>UNCUT_Digestive_Tract</code> 与 <code>UNCUT_Cerebrum_*</code> 只在独立 <code>uncut_reference</code> 层显示，不和分割器官叠加。clean internal 层不放皮肤，以避免 Genesis 透明深度排序遮住肺、骨和细血管；不透明 skin 层单独作为外边界参照。</p>
<p><small>render root：{root_label}</small></p>
<div class="controls">
<label>体型 / subject<select id="subject"></select></label>
<label>姿态 / pose<select id="pose"></select></label>
<label>层 / layer<select id="layer"></select></label>
<label>视角 / view<select id="view"></select></label>
<label>版本 / variant<select id="variant"></select></label>
</div>
<div class="stage"><div class="wipe"><img id="base" alt="已保存的 Genesis 内部或 skin 图"><div id="skinClip"><img id="skinImage" alt="同相机不透明 SMPL-X skin"></div></div>
<div class="sliderrow"><span>内部</span><input id="slider" type="range" min="0" max="100" value="50"><span>skin</span></div></div>
<p id="caption" class="caption"></p>
<div id="comparisonBlock" class="secondary" hidden><div><b>已保存 comparison 图（若有）</b><img id="comparison" alt="已保存 comparison Genesis 图"></div></div>
<details open><summary>扫描到的 candidate / comparison 图片</summary><div id="gallery" class="gallery"></div></details>
<p id="count"></p>
</main><script>
const entries={entry_json};const extras={extras_json};
const subject=document.querySelector('#subject'),pose=document.querySelector('#pose'),layer=document.querySelector('#layer'),view=document.querySelector('#view'),variant=document.querySelector('#variant');
const base=document.querySelector('#base'),skinClip=document.querySelector('#skinClip'),skinImage=document.querySelector('#skinImage'),slider=document.querySelector('#slider'),caption=document.querySelector('#caption'),comparisonBlock=document.querySelector('#comparisonBlock'),comparison=document.querySelector('#comparison');
function unique(values){{return [...new Set(values)]}}
function refill(select,values){{const old=select.value;select.replaceChildren(...unique(values).map(v=>new Option(v,v)));if(unique(values).includes(old))select.value=old;}}
function by(field){{return entries.map(e=>e[field])}}
function updateSubject(){{refill(subject,by('subject'));if(!subject.value&&[...subject.options].some(o=>o.value==='213328'))subject.value='213328';updatePose();}}
function updatePose(){{let xs=entries.filter(e=>e.subject===subject.value);refill(pose,xs.map(e=>e.pose));if(!pose.value||!xs.some(e=>e.pose===pose.value)){{const t=xs.find(e=>e.pose.toLowerCase().includes('tpose'));pose.value=t?t.pose:(xs[0]&&xs[0].pose)||'';}}updateLayer();}}
function updateLayer(){{let xs=entries.filter(e=>e.subject===subject.value&&e.pose===pose.value);refill(layer,xs.map(e=>e.layer));if(!layer.value||!xs.some(e=>e.layer===layer.value))layer.value=xs.some(e=>e.layer==='context')?'context':(xs[0]&&xs[0].layer)||'';updateView();}}
function updateView(){{let xs=entries.filter(e=>e.subject===subject.value&&e.pose===pose.value&&e.layer===layer.value);refill(view,xs.map(e=>e.view));if(!view.value||!xs.some(e=>e.view===view.value))view.value=xs.some(e=>e.view==='torso_anterior')?'torso_anterior':(xs[0]&&xs[0].view)||'';updateVariant();}}
function updateVariant(){{let xs=entries.filter(e=>e.subject===subject.value&&e.pose===pose.value&&e.layer===layer.value&&e.view===view.value);refill(variant,xs.map(e=>e.variant));if(!variant.value||!xs.some(e=>e.variant===variant.value))variant.value=xs.some(e=>e.variant==='candidate')?'candidate':(xs[0]&&xs[0].variant)||'';show();}}
function selected(){{return entries.find(e=>e.subject===subject.value&&e.pose===pose.value&&e.layer===layer.value&&e.view===view.value&&e.variant===variant.value)}}
function show(){{const e=selected();if(!e)return;const skin=e.skin_path;const display=(e.layer==='skin'||e.layer==='skin_only'||e.layer==='smplx_skin')?(skin||e.path): (e.path||e.internal_path||'');base.src=display;skinImage.src=skin||'';skinClip.style.clipPath=skin?'inset(0 '+(100-Number(slider.value))+'% 0 0)':'inset(0 100% 0 0)';slider.disabled=!skin;comparisonBlock.hidden=!e.comparison_path;if(e.comparison_path)comparison.src=e.comparison_path;const bits=[e.subject,e.pose,e.variant,e.layer,e.view];const a=document.createElement('a');a.href=display;a.textContent='打开原始 RGB：'+bits.join(' / ');caption.replaceChildren(a);if(e.input_sha256)caption.append(' · input SHA256 '+e.input_sha256);}}
function renderGallery(){{const box=document.querySelector('#gallery');for(const item of extras){{const tile=document.createElement('div');tile.className='tile';const img=document.createElement('img');img.loading='lazy';img.src=item.path;const a=document.createElement('a');a.href=item.path;a.textContent=item.kind+'：'+item.label;tile.append(img,a);box.append(tile);}}}}
subject.onchange=updatePose;pose.onchange=updateLayer;layer.onchange=updateView;view.onchange=updateVariant;variant.onchange=show;slider.oninput=show;updateSubject();renderGallery();document.querySelector('#count').textContent=entries.length+' 个可选 Genesis 图，'+extras.length+' 个 candidate/comparison 扫描文件。';
</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", "--manifest", dest="render_root", type=Path, required=True,
                        help="V16 render directory or its manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="directory receiving index.html and gallery.json")
    parser.add_argument("--subject")
    parser.add_argument("--pose")
    parser.add_argument("--layer")
    parser.add_argument("--view")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_root, manifest = _read_manifest(args.render_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries, extras = _make_entries(render_root=render_root, manifest=manifest, page_root=output)
    selected = _filter_entries(entries, args)
    gallery = {
        "schema_version": 16,
        "artifact_kind": "VisceraReviewPageV16",
        "render_root": str(render_root),
        "manifest_source": str((render_root / "manifest.json").resolve()),
        "anatomical_passed": False,
        "publishable": False,
        "image_source": "saved Genesis RGB files only",
        "images": selected,
        "all_manifest_images": entries,
        "candidate_and_comparison_scan": extras,
    }
    (output / "gallery.json").write_text(
        json.dumps(gallery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(
        _html(entries=selected, extras=extras, render_root=render_root), encoding="utf-8"
    )
    print(json.dumps({"output": str(output / "index.html"), "images": len(selected), "scanned": len(extras)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
