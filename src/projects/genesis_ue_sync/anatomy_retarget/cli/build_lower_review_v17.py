"""Build an offline selector for saved V17 lower-chain Genesis renders.

The page references RGB files already produced by the renderer.  It never
loads a mesh, reruns a fit, or claims a whole-body anatomical pass.  Render
roots ending in ``_001`` are labelled as the failed/reference run and roots
ending in ``_002`` as the candidate run.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any


def _manifest_root(value: Path) -> tuple[Path, Path]:
    path = value.resolve()
    manifest = path if path.is_file() else path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"V17 render manifest not found: {manifest}")
    return manifest.parent.resolve(), manifest


def _artifact(value: Any, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    return path if path.is_file() else None


def _relative(path: Path | None, *, page_root: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    return os.path.relpath(path, page_root)


def _beta_label(root: Path) -> str:
    stem = root.name
    if "beta_axis0_plus" in stem:
        return "β=[1.5,0,…]"
    if "beta0" in stem:
        return "β=0"
    if "worst_native" in stem:
        return "213328 · worst native frames"
    for subject in ("213328", "213712"):
        if subject in stem:
            return subject
    return stem


def _release_label(root: Path) -> str:
    match = re.search(r"_(00[12])$", root.name)
    if match and match.group(1) == "001":
        return "001 · failed/reference"
    if match and match.group(1) == "002":
        return "002 · candidate"
    return "unclassified"


def _add_entry(
    entries: list[dict[str, Any]],
    seen: set[tuple[str, ...]],
    *,
    render_root: Path,
    page_root: Path,
    beta: str,
    release: str,
    pose: str,
    variant: str,
    layer: str,
    view: str,
    path: Path | None,
    input_path: Any,
    input_sha256: Any,
) -> None:
    if path is None:
        return
    key = (beta, release, pose, variant, layer, view, str(path))
    if key in seen:
        return
    seen.add(key)
    entries.append(
        {
            "beta": beta,
            "release": release,
            "pose": pose,
            "variant": variant,
            "layer": layer,
            "view": view,
            "path": _relative(path, page_root=page_root),
            "root": str(render_root),
            "input": str(input_path) if input_path else None,
            "input_sha256": str(input_sha256) if input_sha256 else None,
        }
    )


def _manifest_renders(layer_data: Any) -> list[dict[str, Any]]:
    if not isinstance(layer_data, dict) or not isinstance(layer_data.get("renders"), list):
        return []
    return [item for item in layer_data["renders"] if isinstance(item, dict)]


def _discover(render_root: Path, *, page_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root, manifest_path = _manifest_root(render_root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cells"), dict):
        raise ValueError(f"invalid V17 manifest: {manifest_path}")
    beta = _beta_label(root)
    release = _release_label(root)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_pose, raw_cell in data["cells"].items():
        pose = str(raw_pose)
        cell = raw_cell if isinstance(raw_cell, dict) else {}
        variants = cell.get("variants", {})
        if not isinstance(variants, dict):
            variants = {}
        for raw_variant, raw_variant_data in variants.items():
            variant = str(raw_variant)
            variant_data = raw_variant_data if isinstance(raw_variant_data, dict) else {}
            variant_root = root / pose / variant
            for render in _manifest_renders(variant_data):
                view = str(render.get("camera", ""))
                if not view:
                    continue
                path = _artifact(render.get("rgb"), root=root)
                if path is None:
                    path = variant_root / "rgb" / f"{view}.png"
                _add_entry(
                    entries, seen, render_root=root, page_root=page_root,
                    beta=beta, release=release, pose=pose, variant=variant,
                    layer="transparent_internal", view=view, path=path,
                    input_path=cell.get("input"), input_sha256=cell.get("input_sha256"),
                )
            overlay = variant_data.get("opaque_skin_overlay")
            for render in _manifest_renders(overlay):
                view = str(render.get("camera", ""))
                if not view:
                    continue
                path = _artifact(render.get("rgb"), root=root)
                if path is None:
                    path = variant_root / "opaque_skin_overlay" / "rgb" / f"{view}.png"
                _add_entry(
                    entries, seen, render_root=root, page_root=page_root,
                    beta=beta, release=release, pose=pose, variant=variant,
                    layer="opaque_skin_overlay", view=view, path=path,
                    input_path=cell.get("input"), input_sha256=cell.get("input_sha256"),
                )
            # Saved comparison sheets are useful when reviewing before/after
            # without selecting a single side of the split image.
            for layer, directory in (
                ("comparison", root / pose / "comparison"),
                ("comparison_opaque_skin_overlay", root / pose / "comparison_opaque_skin_overlay"),
            ):
                for path in sorted(directory.glob("*.png")) if directory.is_dir() else ():
                    _add_entry(
                        entries, seen, render_root=root, page_root=page_root,
                        beta=beta, release=release, pose=pose, variant="comparison",
                        layer=layer, view=path.stem, path=path,
                        input_path=cell.get("input"), input_sha256=cell.get("input_sha256"),
                    )
        for render in _manifest_renders(cell.get("skin_only")):
            view = str(render.get("camera", ""))
            if not view:
                continue
            path = _artifact(render.get("rgb"), root=root)
            if path is None:
                path = root / pose / "smplx_skin" / "rgb" / f"{view}.png"
            _add_entry(
                entries, seen, render_root=root, page_root=page_root,
                beta=beta, release=release, pose=pose, variant="skin",
                layer="smplx_skin", view=view, path=path,
                input_path=cell.get("input"), input_sha256=cell.get("input_sha256"),
            )
    entries.sort(key=lambda item: tuple(str(item[key]) for key in (
        "beta", "release", "pose", "variant", "layer", "view"
    )))
    return entries, {
        "root": str(root),
        "manifest": str(manifest_path),
        "beta": beta,
        "release": release,
        "whole_body_pass": False,
        "static_visual_review_only": True,
    }


def _html(entries: list[dict[str, Any]], roots: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    roots_payload = json.dumps(roots, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V17 lower-chain Genesis visual review</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}
header{padding:16px 20px;background:#1e1e1e;position:sticky;top:0;z-index:2}
h1{font-size:20px;margin:0 0 8px}p{margin:5px 0;color:#bbb}
.warning{color:#ffd36a}.controls{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
label{display:flex;gap:5px;align-items:center;color:#ccc}select{background:#292929;color:#fff;border:1px solid #555;padding:5px;max-width:240px}
main{padding:20px}.frame{max-width:1100px;margin:auto;background:#191919;padding:12px;border-radius:6px}
img{display:block;width:100%;height:auto;background:#06111d;image-rendering:auto}
#caption{margin-top:10px;line-height:1.5;color:#ccc;word-break:break-word}
a{color:#8ecbff}.empty{padding:40px;color:#ffb0a8}
</style></head><body>
<header><h1>V17 lower-chain Genesis review</h1>
<p class="warning">001 is marked failed/reference; 002 is marked candidate. This gallery is static RGB visual QA only. whole_body_pass=false; it is not a collision certificate.</p>
<div class="controls">
<label>β <select id="beta"></select></label>
<label>release <select id="release"></select></label>
<label>pose <select id="pose"></select></label>
<label>variant <select id="variant"></select></label>
<label>layer <select id="layer"></select></label>
<label>view <select id="view"></select></label>
</div></header>
<main><div class="frame"><img id="image" alt="saved Genesis render"><div id="caption"></div></div></main>
<script>
const entries=__ENTRIES__;
const roots=__ROOTS__;
const fields=["beta","release","pose","variant","layer","view"];
const labels={transparent_internal:"transparent internal",opaque_skin_overlay:"opaque skin overlay",smplx_skin:"SMPL-X skin",comparison:"comparison",comparison_opaque_skin_overlay:"comparison opaque skin"};
function values(field){return [...new Set(entries.map(e=>e[field]))].sort();}
function fill(field){const s=document.getElementById(field);const old=s.value;s.replaceChildren();for(const v of values(field)){const o=document.createElement("option");o.value=v;o.textContent=labels[v]||v;s.append(o);}if([...s.options].some(o=>o.value===old))s.value=old;}
function selected(){return fields.reduce((a,f)=>(a[f]=document.getElementById(f).value,a),{});}
function refresh(){const q=selected();let e=entries.find(x=>fields.every(f=>x[f]===q[f]));if(!e)e=entries[0];const img=document.getElementById("image"),caption=document.getElementById("caption");if(!e){img.removeAttribute("src");caption.className="empty";caption.textContent="No saved render matches these filters.";return;}img.src=e.path;caption.className="";caption.replaceChildren();caption.append(`${e.beta} · ${e.release} · ${e.pose} · ${e.variant} · ${labels[e.layer]||e.layer} · ${e.view}`);if(e.path){caption.append(" · ");const a=document.createElement("a");a.href=e.path;a.textContent="open RGB";caption.append(a);}if(e.input_sha256)caption.append(` · input SHA256 ${e.input_sha256}`);}
for(const f of fields)document.getElementById(f).addEventListener("change",refresh);
for(const f of fields)fill(f);
const preferred={beta:entries.find(e=>e.beta==="213328"),release:entries.find(e=>e.release==="002 · candidate"),layer:entries.find(e=>e.layer==="opaque_skin_overlay"),variant:entries.find(e=>e.variant==="candidate"),pose:entries.find(e=>e.pose.includes("knees_90")),view:entries.find(e=>e.view==="right_knee_lateral")};
for(const f of fields)if(preferred[f]&&[...document.getElementById(f).options].some(o=>o.value===preferred[f][f]))document.getElementById(f).value=preferred[f][f];
refresh();
</script></body></html>
""".replace("__ENTRIES__", payload).replace("__ROOTS__", roots_payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, nargs="+", required=True,
                        help="V17 Genesis render root(s), each containing manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory in which index.html is written")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review directory: {output}")
    output.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for render_root in args.render_root:
        found, metadata = _discover(render_root, page_root=output)
        entries.extend(found)
        roots.append(metadata)
    if not entries:
        raise RuntimeError("no saved RGB renders found in supplied roots")
    page = output / "index.html"
    page.write_text(_html(entries, roots), encoding="utf-8")
    print(f"V17 lower review -> {page} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
