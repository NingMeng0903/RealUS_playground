"""Resolve frozen SMPL betas for realtime tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets import load_amass_sequence
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec

logger = logging.getLogger(__name__)


def _normalize_betas(raw: Any) -> np.ndarray:
    betas = np.asarray(raw, dtype=np.float32).reshape(-1)[:10]
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - int(betas.size)))
    return betas.astype(np.float32)


def _load_betas_from_amass_npz(npz_path: Path) -> np.ndarray:
    seq = load_amass_sequence(npz_path)
    return _normalize_betas(seq.betas)


def _load_betas_from_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"SMPL betas file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _normalize_betas(np.load(path))
    if suffix == ".npz":
        with np.load(path) as payload:
            if "betas" not in payload:
                raise KeyError(f"SMPL betas npz must contain a 'betas' array: {path}")
            return _normalize_betas(payload["betas"])
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("betas")
        if data is None:
            raise KeyError(f"SMPL betas json must contain a 'betas' field: {path}")
        return _normalize_betas(data)
    raise ValueError(f"Unsupported SMPL betas file extension: {path.suffix}")


def resolve_smpl_betas(
    smpl_fit_cfg: dict[str, Any] | None,
    *,
    scene_spec_path: Path | None = None,
    ik_cfg: dict[str, Any] | None = None,
    betas_amass_npz_override: Path | None = None,
    require_betas_path: bool = False,
) -> tuple[np.ndarray, str]:
    """Return (betas10, source_label). Priority: betas_path > AMASS debug fallback > yaml list > zeros."""
    cfg = dict(smpl_fit_cfg or {})
    ik = dict(ik_cfg or {})

    betas_path_raw = str(cfg.get("betas_path") or ik.get("betas_path") or "").strip()
    if betas_path_raw:
        path = project_paths(__file__).resolve_from_root(betas_path_raw)
        return _load_betas_from_file(path), f"betas_path:{path.name}"

    allow_debug_fallback = bool(cfg.get("allow_debug_betas_fallback", False) or ik.get("allow_debug_betas_fallback", False))
    if require_betas_path and not allow_debug_fallback:
        raise ValueError("smpl_fit.betas_path is required for realtime SMPL tracking.")

    if betas_amass_npz_override is not None:
        path = Path(betas_amass_npz_override).expanduser().resolve()
        betas = _load_betas_from_amass_npz(path)
        return betas, f"amass_npz:{path.name}"

    npz_raw = str(cfg.get("betas_from_amass_npz") or ik.get("betas_from_amass_npz") or "").strip()
    if npz_raw:
        path = project_paths(__file__).resolve_from_root(npz_raw)
        betas = _load_betas_from_amass_npz(path)
        return betas, f"amass_npz:{path.name}"

    source = str(cfg.get("betas_source") or ik.get("betas_source") or "").strip().lower()
    if source in {"scene_spec", "scene", "gt", "amass_scene"} and scene_spec_path is not None:
        spec = load_sync_scene_spec(scene_spec_path)
        amass_path = spec.motion.resolved_source_path
        if amass_path is not None and amass_path.is_file():
            betas = _load_betas_from_amass_npz(amass_path)
            return betas, f"scene_spec:{amass_path.name}"

    explicit = cfg.get("betas")
    if explicit is None:
        explicit = ik.get("betas")
    if explicit is not None:
        betas = _normalize_betas(explicit)
        if float(np.max(np.abs(betas))) > 0.0:
            return betas, "yaml_list"

    return np.zeros(10, dtype=np.float32), "zeros"
