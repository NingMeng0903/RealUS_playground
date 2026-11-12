"""Build the checked inverse-acquisition motion bank.

The inherited ``BANK_MAIN.json`` is copied byte-for-byte at the level of its
nodes and edges.  New nodes use the two inherited centre-contact dIRD
realizations and rotate the centre TCP target about its local probe axis by
the bounded angles ``+-3, +-6, +-9`` degrees.  Each target is solved from the
inherited centre ``q`` witness and rechecked with the pinned dIRD/URDF
evaluator.  Missing graph edges are sampled with the original full edge
checker, including FK, configured self-collision, joint limits and velocity
checks; endpoint-only records are never emitted.

This is intentionally a mechanics-only builder.  It never loads a case,
acoustic tape, label, or benchmark outcome.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


POSE_INDEX = 2
POSE_DEG = 15.0
SITES = (3, 4, 5, 6, 7)
REALIZATIONS = ("dird_top", "dird_diverse")
BASE_VIEW_ANGLES = {3: 0.0, 7: -12.0, 8: 12.0}
# New IDs are deliberately >= 9 so the inherited 3/7/8 IDs and all existing
# node IDs remain untouched.  The acoustic/world runner can use tilt_deg to
# distinguish these IDs from the legacy sign-only view encoding.
NEW_VIEW_ANGLES = {9: -9.0, 10: -6.0, 11: -3.0, 12: 3.0, 13: 6.0, 14: 9.0}
ALL_VIEW_ANGLES = {**BASE_VIEW_ANGLES, **NEW_VIEW_ANGLES}
ANGLE_ORDER = tuple(sorted(ALL_VIEW_ANGLES, key=lambda view: ALL_VIEW_ANGLES[view]))
EDGE_DURATION_S = 1.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    """Recursively convert numpy values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    # bool is an int subclass; keep logical fields logical in newly generated
    # records and in copied preflight metadata.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"non-finite numeric value in output: {value!r}")
        return value
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def load_base_builder(base: Path, repo: Path):
    """Load the exact sibling ``motion_bank.py`` used for the inherited bank."""
    candidates = (
        Path(repo)
        / "NMI_idea/experiments/ab_full_b_rollout_20261113/motion_bank.py",
        base.parent.parent / "motion_bank.py",
        Path(__file__).resolve().parents[2]
        / "RealUS_playground/NMI_idea/experiments/ab_full_b_rollout_20261113/motion_bank.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "could not locate inherited motion_bank.py next to --base or in RealUS_playground"
        )
    # Put the explicitly requested checkout first before loading the inherited
    # module.  This prevents its NMI package import from silently resolving to
    # the mounted archive when a caller passes a checkout-local builder.
    repo = Path(repo).resolve()
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    # The inherited builder computes its own fallback mount path from
    # ``__file__``.  When loaded from a checkout-local source that path is not
    # the checkout root, so the builder would otherwise prepend the mounted
    # archive and import a different capability module.  Preload the
    # capability package from the explicit repo before executing the builder;
    # Python then reuses this exact module even if the builder adds its mount
    # fallback to sys.path.
    capability_name = (
        "NMI_idea.experiments.ab_module_ladder_20261110.capability.capability"
    )
    capability_path = (
        repo
        / "NMI_idea/experiments/ab_module_ladder_20261110/capability/capability.py"
    ).resolve()
    if capability_path.exists():
        existing = sys.modules.get(capability_name)
        if existing is not None:
            try:
                existing_path = Path(inspect.getsourcefile(existing)).resolve()
            except (TypeError, OSError):
                existing_path = None
            if existing_path != capability_path:
                for name in list(sys.modules):
                    if name == "NMI_idea" or name.startswith("NMI_idea."):
                        del sys.modules[name]
        importlib.invalidate_caches()
        imported = importlib.import_module(capability_name)
        imported_path = Path(inspect.getsourcefile(imported)).resolve()
        if imported_path != capability_path:
            raise ImportError(
                f"explicit repo capability import resolved to {imported_path}, "
                f"expected {capability_path}"
            )
    spec = importlib.util.spec_from_file_location("inverse_b_inherited_motion_bank", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import inherited builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source, module


def geometry_path(repo: Path) -> Path:
    path = Path(repo) / "NMI_idea/experiments/ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing pinned geometry: {path}")
    return path


def rotate_target(center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the centre target in its local probe-x frame."""
    local = np.eye(4, dtype=float)
    local[:3, :3] = Rotation.from_rotvec([np.deg2rad(angle_deg), 0.0, 0.0]).as_matrix()
    return np.asarray(center, dtype=float) @ local


def view_target(geometry, site: int, angle_deg: float):
    """Return the synthetic target and atlas frame provenance for one view."""
    target_center = np.asarray(geometry["target_robot"][POSE_INDEX, site, 3], dtype=float)
    target = rotate_target(target_center, angle_deg)
    frame_center = np.asarray(geometry["posed_frames"][POSE_INDEX, site, 3], dtype=float)
    local = np.eye(3, dtype=float)
    local[:] = Rotation.from_rotvec([np.deg2rad(angle_deg), 0.0, 0.0]).as_matrix()
    frame = frame_center @ local
    return target, frame


def scalar(value) -> float:
    values = np.asarray(value).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"expected scalar score, got shape {np.asarray(value).shape}")
    return float(values[0])


def node_index(nodes):
    return {(int(node["site"]), int(node["view"]), str(node["realization"])): node for node in nodes}


def target_node(
    cap,
    geometry,
    center,
    site: int,
    view: int,
    realization: str,
    angle_deg: float,
    node_id: int,
    rejections: list,
):
    """Build one candidate node from a centre q/rail witness."""
    target, frame = view_target(geometry, site, angle_deg)
    seed = np.asarray(center["q"], dtype=float)
    rail = float(center["rail_m"])
    try:
        score = cap.score(target[None], [rail])
        raw_score = scalar(score["raw_score"])
        margin = scalar(score["raw_calibrated_margin"])
        proxy_pass = bool(np.asarray(score["proxy_pass"]).reshape(-1)[0])
    except Exception as exc:  # pragma: no cover - defensive audit record
        rejections.append(
            {
                "kind": "node",
                "site": site,
                "view": view,
                "tilt_deg": angle_deg,
                "realization": realization,
                "source_center_node_id": int(center["id"]),
                "reason": f"dIRD score exception: {type(exc).__name__}: {exc}",
            }
        )
        return None

    if not proxy_pass:
        rejections.append(
            {
                "kind": "node",
                "site": site,
                "view": view,
                "tilt_deg": angle_deg,
                "realization": realization,
                "source_center_node_id": int(center["id"]),
                "raw_score": raw_score,
                "calibrated_margin": margin,
                "reason": "live dIRD proxy below calibrated threshold",
            }
        )
        return None

    try:
        static = cap._static_pose(target, rail, seed)
    except Exception as exc:  # pragma: no cover - defensive audit record
        rejections.append(
            {
                "kind": "node",
                "site": site,
                "view": view,
                "tilt_deg": angle_deg,
                "realization": realization,
                "source_center_node_id": int(center["id"]),
                "raw_score": raw_score,
                "calibrated_margin": margin,
                "reason": f"seeded static IK exception: {type(exc).__name__}: {exc}",
            }
        )
        return None
    if not static.get("found", False):
        rejections.append(
            {
                "kind": "node",
                "site": site,
                "view": view,
                "tilt_deg": angle_deg,
                "realization": realization,
                "source_center_node_id": int(center["id"]),
                "raw_score": raw_score,
                "calibrated_margin": margin,
                "reason": "seeded static IK search found no valid witness",
                "dird_search": static.get("search"),
            }
        )
        return None

    q = np.asarray(static["q_solution"], dtype=float)
    check = cap.check_q(q, target)
    if not check.get("valid", False):
        rejections.append(
            {
                "kind": "node",
                "site": site,
                "view": view,
                "tilt_deg": angle_deg,
                "realization": realization,
                "source_center_node_id": int(center["id"]),
                "raw_score": raw_score,
                "calibrated_margin": margin,
                "reason": "seeded endpoint check failed",
                "endpoint_check": check,
            }
        )
        return None

    fk = cap.fk(q)
    source_dird = center.get("dird_score", {})
    node = {
        "id": int(node_id),
        "site": int(site),
        "view": int(view),
        "tilt_deg": float(angle_deg),
        "realization": realization,
        "q": q.tolist(),
        "tip_xyz_mm": (fk[:3, 3] * 1000.0).tolist(),
        "rail_m": rail,
        "endpoint_check": check,
        "min_self_collision_distance_m": check["min_self_collision_distance_m"],
        "min_joint_limit_margin": min(check["joint_limit_margin_by_coordinate"]),
        "dird_score": {
            "raw_score": raw_score,
            "calibrated_margin": margin,
            "proxy_pass": proxy_pass,
            "source_center_node_id": int(center["id"]),
            "source_center_archive_calibrated_margin": source_dird.get("archive_calibrated_margin"),
            "source_center_archive_candidate_index": source_dird.get("archive_candidate_index"),
        },
        "dird_search": {
            "branch": static.get("branch"),
            "psi_rad": static.get("psi_rad"),
            "robot_configurations_checked": static.get("search", {}).get(
                "robot_configurations_checked"
            ),
            "seed_source": "inherited centre q witness",
        },
        "target_robot": target.tolist(),
        "atlas_frame": frame.tolist(),
        "view_provenance": {
            "nominal_pose_index": POSE_INDEX,
            "nominal_knee_deg": POSE_DEG,
            "source_view_id": 3,
            "source_target": f"target_robot[{POSE_INDEX},{site},3]",
            "source_atlas_frame": f"posed_frames[{POSE_INDEX},{site},3]",
            "construction": "right_multiply_local_probe_x_rotation",
            "rotation_axis": "source target local x / atlas frame column 0",
            "tilt_deg": float(angle_deg),
            "outcome_independent": True,
        },
    }
    return node


def edge_record(base_builder, cap, nodes_by_id, start: int, goal: int, rejections, audit):
    qa = np.asarray(nodes_by_id[start]["q"], dtype=float)
    qb = np.asarray(nodes_by_id[goal]["q"], dtype=float)
    try:
        legal, metrics, trace = base_builder._edge_trace(cap, qa, qb, EDGE_DURATION_S)
    except Exception as exc:  # pragma: no cover - defensive audit record
        rejections.append(
            {
                "kind": "edge",
                "start": int(start),
                "goal": int(goal),
                "reason": f"sampled edge exception: {type(exc).__name__}: {exc}",
            }
        )
        audit.append({"start": int(start), "goal": int(goal), "legal": False,
                      "reason": rejections[-1]["reason"]})
        return None
    record = {"start": int(start), "goal": int(goal), **metrics}
    audit.append(record)
    if not legal:
        rejections.append(
            {
                "kind": "edge",
                "start": int(start),
                "goal": int(goal),
                "reason": metrics.get("reason", "sampled edge rejected"),
                "metrics": metrics,
            }
        )
        return None
    return {
        "start": int(start),
        "goal": int(goal),
        "duration_s": float(metrics["duration_s"]),
        "travel_mm": float(metrics["travel_mm"]),
        "turn_deg": float(metrics["turn_deg"]),
        "rail_mm": float(metrics["rail_mm"]),
        "max_joint_rate_fraction": float(metrics["max_joint_rate_fraction"]),
        "min_self_collision_distance_m": float(metrics["min_self_collision_distance_m"]),
        "min_joint_limit_margin": float(metrics["min_joint_limit_margin"]),
        "samples": trace,
    }


def build(repo: Path, base_path: Path, out_dir: Path, trace_mode: str = "joint"):
    started = time.perf_counter()
    repo = Path(repo).resolve()
    base_path = Path(base_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_builder_path, base_builder = load_base_builder(base_path, repo)
    if trace_mode not in {"joint", "cartesian"}:
        raise ValueError("trace_mode must be 'joint' or 'cartesian'")
    base_builder.TRACE_MODE = trace_mode

    base = json.loads(base_path.read_text())
    geometry_file = geometry_path(repo)
    with np.load(geometry_file, allow_pickle=False) as geometry:
        geometry_arrays = {name: geometry[name] for name in ("target_robot", "posed_frames")}

    nodes = copy.deepcopy(base["nodes"])
    edges = copy.deepcopy(base["edges"])
    rejections = copy.deepcopy(base.get("rejections", []))
    edge_audit = []
    nodes_by_id = {int(node["id"]): node for node in nodes}
    by_key = node_index(nodes)
    next_id = max(nodes_by_id) + 1

    cap = base_builder.CapabilityEvaluator(device="cpu")
    added_nodes = []
    for site in SITES:
        for view, angle_deg in sorted(NEW_VIEW_ANGLES.items(), key=lambda item: item[1]):
            for realization in REALIZATIONS:
                center = by_key[(site, 3, realization)]
                node = target_node(
                    cap, geometry_arrays, center, site, view, realization, angle_deg,
                    next_id, rejections,
                )
                if node is None:
                    continue
                nodes.append(node)
                nodes_by_id[next_id] = node
                by_key[(site, view, realization)] = node
                added_nodes.append(node)
                next_id += 1
    print(
        f"inverse-b nodes: preserved={len(base['nodes'])} added={len(added_nodes)} "
        f"retained={len(nodes)} rejected={sum(r.get('kind') == 'node' for r in rejections)}",
        flush=True,
    )

    # Preserve every inherited edge exactly.  New desired pairs include old
    # nodes as endpoints, so missing old-old diagonal tilts are checked too.
    existing_directed = {(int(edge["start"]), int(edge["goal"])) for edge in edges}
    desired_undirected = set()

    def add_pair(left, right):
        if left == right:
            return
        desired_undirected.add(tuple(sorted((int(left), int(right)))))

    angle_views = tuple(view for view in ANGLE_ORDER)
    for realization in REALIZATIONS:
        for left_site, right_site in ((5, 4), (4, 3), (5, 6), (6, 7)):
            for left_view in angle_views:
                for right_view in angle_views:
                    left = by_key.get((left_site, left_view, realization))
                    right = by_key.get((right_site, right_view, realization))
                    if left is not None and right is not None:
                        add_pair(left["id"], right["id"])
        for site in SITES:
            present = [by_key[(site, view, realization)] for view in angle_views
                       if (site, view, realization) in by_key]
            for i, left in enumerate(present):
                for right in present[i + 1:]:
                    add_pair(left["id"], right["id"])

    # Same pose / different dIRD realization is a checked regrasp.  This also
    # preserves the inherited swaps while adding swaps for each new tilt.
    for site in SITES:
        for view in angle_views:
            top = by_key.get((site, view, "dird_top"))
            diverse = by_key.get((site, view, "dird_diverse"))
            if top is not None and diverse is not None:
                add_pair(top["id"], diverse["id"])

    new_edges = []
    edge_attempts = 0
    for left, right in sorted(desired_undirected):
        for start, goal in ((left, right), (right, left)):
            if (start, goal) in existing_directed:
                continue
            edge_attempts += 1
            edge = edge_record(base_builder, cap, nodes_by_id, start, goal, rejections, edge_audit)
            if edge is not None:
                new_edges.append(edge)
                existing_directed.add((start, goal))
            if edge_attempts % 100 == 0:
                print(
                    f"inverse-b edges: attempted={edge_attempts} retained={len(new_edges)} "
                    f"rejected={sum(r.get('kind') == 'edge' for r in rejections)}",
                    flush=True,
                )

    # Add sampled hold edges for new nodes only; inherited holds are untouched.
    for node in added_nodes:
        node_id = int(node["id"])
        if (node_id, node_id) in existing_directed:
            continue
        edge_attempts += 1
        edge = edge_record(base_builder, cap, nodes_by_id, node_id, node_id, rejections, edge_audit)
        if edge is not None:
            edge["kind"] = "hold"
            new_edges.append(edge)
            existing_directed.add((node_id, node_id))
        if edge_attempts % 100 == 0:
            print(
                f"inverse-b edges: attempted={edge_attempts} retained={len(new_edges)} "
                f"rejected={sum(r.get('kind') == 'edge' for r in rejections)}",
                flush=True,
            )
    edges.extend(new_edges)

    # Advertise every retained view while preserving the inherited start and
    # route IDs.  The runner only needs task_starts, but these lists make the
    # bank self-describing for inverse-acquisition planners.
    def ids(site, view, realization="dird_top"):
        node = by_key.get((site, view, realization))
        return None if node is None else int(node["id"])

    task_targets = copy.deepcopy(base.get("task_targets", {}))
    for task, target_site in (("daughter", 3), ("trunk", 7)):
        if task not in task_targets:
            task_targets[task] = {}
        task_targets[task]["adjacent"] = [ids(4 if task == "daughter" else 6, view)
                                           for view in angle_views]
        task_targets[task]["final"] = [ids(target_site, view) for view in angle_views]
        task_targets[task]["adjacent"] = [x for x in task_targets[task]["adjacent"] if x is not None]
        task_targets[task]["final"] = [x for x in task_targets[task]["final"] if x is not None]

    base_hash = sha256_file(base_path)
    source_hash = sha256_file(base_builder_path)
    geometry_hash = sha256_file(geometry_file)
    failed_added = len([r for r in rejections if r.get("kind") in {"node", "edge"}])
    summary = dict(base.get("summary", {}))
    summary.update(
        {
            "base_node_count": len(base["nodes"]),
            "added_node_count": len(added_nodes),
            "node_count": len(nodes),
            "base_edge_count": len(base["edges"]),
            "added_edge_count": len(new_edges),
            "edge_count": len(edges),
            "hold_edge_count": sum(edge.get("kind") == "hold" for edge in edges),
            "requested_undirected_edge_count": len(desired_undirected),
            "rejected_count": len(rejections),
            "edge_duration_s": EDGE_DURATION_S,
            "edge_sample_dt_s": float(base_builder.SAMPLE_DT_S),
            "trace_mode": trace_mode,
            "all_retained_edges_uniform_duration": bool(
                edges and all(abs(float(edge["duration_s"]) - EDGE_DURATION_S) < 1e-12 for edge in edges)
            ),
            "all_retained_edges_legal": bool(
                edges and all(float(edge["max_joint_rate_fraction"]) <= 1.0 + 1e-10 for edge in edges)
            ),
            "angles_deg": [float(ALL_VIEW_ANGLES[view]) for view in angle_views],
            "new_view_ids": sorted(NEW_VIEW_ANGLES),
            "new_node_failures": failed_added,
        }
    )

    provenance = dict(base.get("provenance", {}))
    provenance.update(
        {
            "base_bank": str(base_path),
            "base_bank_sha256": base_hash,
            "builder": str(base_builder_path),
            "builder_sha256": source_hash,
            "geometry_path": str(geometry_file),
            "geometry_sha256": geometry_hash,
            "view_angles_deg": {str(k): float(v) for k, v in ALL_VIEW_ANGLES.items()},
            "view_construction": "target_robot[2,site,3] @ Rx(angle); posed_frames[2,site,3] @ Rx(angle)",
            "rotation_axis": "local probe x / atlas frame first column",
            "edge_duration_s": EDGE_DURATION_S,
            "trace_mode": trace_mode,
            "outcome_independent": True,
            "scope": list(base.get("provenance", {}).get("scope", []))
            + [
                "bounded continuous tilt extension at nominal 15 degree atlas pose",
                "seeded dIRD/SRS endpoint checks for every retained added node",
                "all added edges sampled by inherited FK/limits/collision/rate checker",
            ],
        }
    )

    bank = {
        "schema": "inverse_b_motion_bank_v1",
        "status": "screened_nominal_atlas_replay_bank_bounded_continuous_tilt",
        "selection_policy": {
            "angles_deg": [float(ALL_VIEW_ANGLES[v]) for v in angle_views],
            "new_angles_deg": [-9.0, -6.0, -3.0, 3.0, 6.0, 9.0],
            "source_realizations": list(REALIZATIONS),
            "target_rotation": "center target right-multiplied by local probe-x Rx",
            "q_seed": "corresponding inherited centre node q",
            "dird_and_endpoint_required": True,
            "outcome_independent": True,
        },
        "provenance": provenance,
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
        "task_starts": copy.deepcopy(base.get("task_starts", {})),
        "task_start_options": copy.deepcopy(base.get("task_start_options", {})),
        "task_targets": task_targets,
        "paths": copy.deepcopy(base.get("paths", {})),
        "view_catalog": {
            str(view): {
                "tilt_deg": float(angle),
                "source_view_id": 3 if view in NEW_VIEW_ANGLES else view,
                "construction": "existing bank view" if view in BASE_VIEW_ANGLES else "center_target_right_multiply_local_x_Rx",
            }
            for view, angle in sorted(ALL_VIEW_ANGLES.items())
        },
        "rejections": rejections,
    }
    bank = clean(bank)

    preflight = clean(
        {
            "schema": "inverse_b_motion_preflight_v1",
            "status": "pass" if not any(r.get("kind") in {"node", "edge"} for r in rejections) else "review",
            "base_bank_sha256": base_hash,
            "summary": summary,
            "added_node_checks": [
                {
                    "id": int(node["id"]),
                    "site": int(node["site"]),
                    "view": int(node["view"]),
                    "tilt_deg": float(node["tilt_deg"]),
                    "realization": node["realization"],
                    "dird_score": node["dird_score"],
                    "endpoint_check": node["endpoint_check"],
                }
                for node in added_nodes
            ],
            "added_edge_audit": edge_audit,
            "rejections": rejections,
            "preserved_base_nodes": len(base["nodes"]),
            "preserved_base_edges": len(base["edges"]),
            "elapsed_build_s": time.perf_counter() - started,
        }
    )
    output_bank = out_dir / "BANK.json"
    output_preflight = out_dir / "PREFLIGHT.json"
    output_rejections = out_dir / "REJECTIONS.json"
    output_bank.write_text(json.dumps(bank, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_preflight.write_text(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_rejections.write_text(json.dumps(clean(rejections), indent=2, sort_keys=True, allow_nan=False) + "\n")
    report = {
        "bank": str(output_bank),
        "preflight": str(output_preflight),
        "rejections": str(output_rejections),
        "summary": summary,
        "elapsed_build_s": time.perf_counter() - started,
    }
    (out_dir / "BUILD_REPORT.json").write_text(
        json.dumps(clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return bank, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True,
                        help="RealUS_playground checkout containing frozen geometry")
    parser.add_argument("--base", type=Path, required=True,
                        help="inherited BANK_MAIN.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; only bank/preflight/rejection reports are written")
    parser.add_argument("--trace-mode", choices=("joint", "cartesian"), default="joint",
                        help="inherited sampled edge trace mode (default: joint)")
    args = parser.parse_args()
    bank, _ = build(args.repo, args.base, args.out, args.trace_mode)
    print(json.dumps({"summary": bank["summary"], "out": str(args.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
