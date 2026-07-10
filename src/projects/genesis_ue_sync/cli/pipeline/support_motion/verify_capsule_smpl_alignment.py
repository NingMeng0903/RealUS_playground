#!/usr/bin/env python3
"""Offline checks for SMPL capsule URDF + DOF packing (no Genesis required).

Optional Genesis smoke: if ``import genesis`` works, loads cached runtime URDF and prints entity DOF counts.

Examples::

  # Use real path (no ellipsis). ``<shape_key>`` is 16 hex chars from your cache filename.
  PYTHONPATH=src python scripts/pipeline/support_motion/verify_capsule_smpl_alignment.py \\
    --runtime-urdf outputs/genesis_capsule_urdf_cache/smpl_proxy_urdf/crisp_smpl_capsule_g3_099d0723a57e7725_runtime_visual.urdf

  # Or auto-pick the newest geometry-tagged runtime URDF under the default cache (no path typing).
  PYTHONPATH=src python scripts/pipeline/support_motion/verify_capsule_smpl_alignment.py --discover-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_THIS_FILE = Path(__file__).resolve()
SRC = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from projects.genesis_ue_sync.sim_platform.embodiments.crisp_smpl_euler_retarget import (  # noqa: E402
    pack_floating_capsule_dof,
    smpl_mujoco_permutation_from_crisp,
)

_DEFAULT_CACHE = REPO / "outputs" / "genesis_capsule_urdf_cache"


def _discover_latest_runtime_urdf(cache_root: Path) -> Path | None:
    """Legacy URDF capsule discovery (removed); returns None."""
    del cache_root
    return None


def _discover_latest_phc_mjcf(cache_root: Path) -> Path | None:
    d = cache_root / "phc_bundled_mjcf"
    if not d.is_dir():
        return None
    matches = sorted(d.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _urdf_capsule_rpy_stats(urdf_path: Path) -> dict[str, object]:
    tree = ET.parse(urdf_path)
    rpys: list[str] = []
    for link in tree.getroot().findall("link"):
        col = link.find("collision")
        if col is None:
            continue
        origin = col.find("origin")
        if origin is None or "rpy" not in origin.attrib:
            continue
        rpys.append(str(origin.attrib["rpy"]))
    legacy = sum(1 for r in rpys if "-1.5708" in r or "-1.57080" in r)
    return {
        "urdf": str(urdf_path),
        "collision_origin_count": len(rpys),
        "legacy_minus_90_x_count": int(legacy),
        "geometry_tag": "phc_bundled_mjcf",
    }


def _urdf_continuous_joint_count(urdf_path: Path) -> int:
    tree = ET.parse(urdf_path)
    return sum(
        1
        for joint in tree.getroot().findall("joint")
        if joint.attrib.get("type") in ("continuous", "revolute")
    )


def _genesis_dof_introspection(robot: object) -> dict[str, object]:
    out: dict[str, object] = {}
    for attr in ("get_dofs_name", "get_dof_names"):
        fn = getattr(robot, attr, None)
        if callable(fn):
            try:
                val = fn()
                out[attr] = list(val) if isinstance(val, (list, tuple)) else str(val)
            except Exception as exc:
                out[attr] = f"<error: {exc}>"
    out["members_with_dof"] = sorted(x for x in dir(robot) if "dof" in x.lower())
    return out


def _dof_pack_stats(*, n_revolute: int) -> dict[str, object]:
    pose = np.zeros(72, dtype=np.float32)
    pose[:3] = (0.1, -0.2, 0.05)
    pose[12:15] = (0.3, 0.0, 0.0)
    trans = np.array([0.5, -0.25, 1.1], dtype=np.float32)
    q = pack_floating_capsule_dof(pose, trans, body_euler_count=n_revolute)
    perm = smpl_mujoco_permutation_from_crisp()
    return {
        "q_len": int(q.size),
        "q0_trans": q[:3].tolist(),
        "q3_6_root_euler": q[3:6].tolist(),
        "perm_len": int(perm.size),
        "expected_len": int(6 + n_revolute),
    }


def _discover_latest_mjcf(cache_root: Path) -> Path | None:
    return _discover_latest_phc_mjcf(cache_root)


def _maybe_genesis_mjcf_smoke(mjcf_path: Path | None) -> dict[str, object] | None:
    if mjcf_path is None or not mjcf_path.is_file():
        return None
    try:
        import genesis as gs  # type: ignore
    except Exception as exc:
        return {"genesis_import": False, "mjcf": str(mjcf_path), "error": str(exc)}
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(gs.morphs.MJCF(file=str(mjcf_path), pos=(0.0, 0.0, 0.0)))
    scene.build()
    n_dofs = int(robot.n_dofs)
    q = np.asarray(robot.get_dofs_position()).reshape(-1)
    layout_meta: dict[str, object] = {}
    layout_path = mjcf_path.parent / f"{mjcf_path.stem}_dof_layout.json"
    if layout_path.is_file():
        try:
            lay = json.loads(layout_path.read_text(encoding="utf-8"))
            layout_meta["dof_layout_path"] = str(layout_path)
            layout_meta["layout_total_dofs"] = lay.get("total_dofs")
            layout_meta["layout_tag"] = lay.get("mjcf_layout_tag")
            layout_meta["packed_len_matches_layout_total_dofs"] = int(lay.get("total_dofs", -1)) == n_dofs
        except Exception as exc:
            layout_meta["dof_layout_read_error"] = str(exc)
    out = {
        "genesis_import": True,
        "mjcf": str(mjcf_path),
        "n_dofs": n_dofs,
        "get_dofs_position_len": int(q.size),
        "dof_introspection": _genesis_dof_introspection(robot),
    }
    out.update(layout_meta)
    return out


def _maybe_genesis_smoke(urdf_path: Path | None) -> dict[str, object] | None:
    if urdf_path is None or not urdf_path.is_file():
        return None
    try:
        import genesis as gs  # type: ignore
    except Exception as exc:
        return {"genesis_import": False, "error": str(exc)}
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(gs.morphs.URDF(file=str(urdf_path), fixed=False, merge_fixed_links=False))
    scene.build()
    n_dofs = int(robot.n_dofs)
    n_qs = int(getattr(robot, "n_qs", -1))
    q = np.asarray(robot.get_dofs_position()).reshape(-1)
    urdf_cont = _urdf_continuous_joint_count(urdf_path)
    expected_packed = int(6 + urdf_cont)
    return {
        "genesis_import": True,
        "n_dofs": n_dofs,
        "n_qs": n_qs,
        "get_dofs_position_len": int(q.size),
        "urdf_continuous_joint_count": urdf_cont,
        "expected_packed_dof_count_xyz_and_root_euler_plus_body": expected_packed,
        "packed_len_would_match_if_using_6_dof_free_joint": bool(n_dofs == expected_packed),
        "genesis_implied_free_dof_count": int(n_dofs - urdf_cont),
        "note_if_mismatch": "If implied_free_dof_count is 7, Genesis likely uses a quaternion free joint; "
        "pack_floating_capsule_dof assumes 6 (xyz + intrinsic euler).",
        "dof_introspection": _genesis_dof_introspection(robot),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runtime-urdf",
        type=Path,
        default=None,
        help="Path to *_runtime_visual.urdf (relative paths are under repo root). Do not use literal ... or <shape>.",
    )
    p.add_argument(
        "--discover-cache",
        action="store_true",
        help="Deprecated: handwritten URDF cache removed; use --discover-mjcf for PHC MJCF.",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Capsule cache root used by --discover-cache (default: outputs/genesis_capsule_urdf_cache).",
    )
    p.add_argument(
        "--discover-mjcf",
        action="store_true",
        help="Use newest PHC bundled MJCF *.xml under --cache-root/phc_bundled_mjcf and run Genesis MJCF smoke.",
    )
    p.add_argument(
        "--n-revolute",
        type=int,
        default=69,
        help="Tail scalars after 6-DOF float base for pack_floating_capsule_dof sanity: URDF≈69, MJCF v5≈61 (67 total).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out: dict[str, object] = {
        "dof_pack": _dof_pack_stats(n_revolute=int(args.n_revolute)),
    }
    urdf: Path | None = None
    if args.runtime_urdf is not None:
        candidate = args.runtime_urdf if args.runtime_urdf.is_absolute() else (REPO / args.runtime_urdf)
        s = str(args.runtime_urdf)
        placeholder = "..." in s or "<shape" in s.lower() or "<shape_key" in s.lower()
        if candidate.is_file():
            urdf = candidate
        elif placeholder:
            out["urdf_resolve_note"] = "Placeholder path ignored; use a real filename or --discover-cache."
        else:
            out["urdf_resolve_note"] = f"File not found: {candidate}; try --discover-cache."
    if urdf is None and args.discover_cache:
        urdf = _discover_latest_runtime_urdf(Path(args.cache_root))
        if urdf is not None:
            out["urdf_discovered"] = str(urdf)
    if urdf is not None and urdf.is_file():
        out["urdf_stats"] = _urdf_capsule_rpy_stats(urdf)
        out["genesis"] = _maybe_genesis_smoke(urdf)
    elif urdf is None and args.discover_cache:
        out["urdf_stats"] = {
            "note": "Hand-written capsule URDF removed; use --discover-mjcf with PHC phc_bundled_mjcf cache.",
        }
    elif args.runtime_urdf is not None and "urdf_stats" not in out:
        cand = args.runtime_urdf if args.runtime_urdf.is_absolute() else (REPO / args.runtime_urdf)
        out["urdf_stats"] = {
            "error": f"not found: {cand}",
            "hint": "Use the real 16-hex shape_key in the filename, or run with --discover-cache.",
        }
    if args.discover_mjcf:
        mj = _discover_latest_mjcf(Path(args.cache_root))
        if mj is not None:
            out["mjcf_discovered"] = str(mj)
            out["genesis_mjcf"] = _maybe_genesis_mjcf_smoke(mj)
        else:
            out["genesis_mjcf"] = {
                "error": "discover_mjcf found no matching MJCF",
                "glob_dir": str(Path(args.cache_root) / "phc_bundled_mjcf"),
                "pattern": "*.xml",
            }
    pack = out.get("dof_pack")
    gen = out.get("genesis")
    if isinstance(pack, dict) and isinstance(gen, dict) and gen.get("genesis_import") is True:
        exp_cli = int(pack.get("expected_len", -1))
        nd = int(gen.get("n_dofs", -1))
        gen_exp = int(gen.get("expected_packed_dof_count_xyz_and_root_euler_plus_body", -1))
        out["cross_check"] = {
            "dof_pack_expected_len_from_cli_n_revolute": exp_cli,
            "urdf_based_expected_packed_len_6_plus_continuous": gen_exp,
            "genesis_n_dofs": nd,
            "cli_pack_matches_genesis": exp_cli == nd,
            "urdf_based_pack_matches_genesis": gen_exp == nd,
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
