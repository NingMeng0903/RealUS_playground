#!/usr/bin/env python3
"""Extract BABEL sequences for bed / lying activities.

Default: HF zero-shot NLI (facebook/bart-large-mnli) for semantic scoring; use `--classifier-mode lie_keyword` for lie/lay/sleep/bed on labels only (surface-agnostic; excludes sleepwalk-style strings); use `--classifier-mode rule` for bed-centric rules.

Why `MD/babel_amass_lie_related.md` lists more sequences than this extractor accepts:
- MD §2 indexes **lie / lay / sleep / bed / recline** in `raw_label`/`proc_label` only (56 seqs over all splits); it does **not** require “on a mattress”.
- This pipeline scores **on_bed vs other_surface**: “lie on floor / ground” often yields **high other_surface**, so `implicit_bed_recline_ok` used to fail at `--bert-other-surface-threshold` unless relaxed.
- MD includes **crawl + lay** clips; **locomotion** NLI scores can stay high, so `--bert-loco-threshold` still vetoes those unless you tune it.
- With `--rule-prefilter`, **strict rules** reject many floor+lie labels (`floor_or_ground_without_bed`); MD does not.

When MD §2-style lie labels are present, default **`--bert-lie-phrase-other-surface`** raises the cap **only** for the implicit-recline branch (still requires recline NLI scores and negative gates). This is not “regex accept”; it compensates for the bed-vs-floor axis mismatch.

Strict rules (--loose-rule off): bed/mattress/sit-on-bed OR recline without floor/gym cues — used in rule / rule_then_bert and optional --rule-prefilter.
Semantic mode uses two axes: support surface (`on_bed` vs `other_surface`) and pose (`lying/prone/sitting/standing/locomotion`).

Whole-clip on-bed proxy is **opt-in** (`--whole-clip-on-bed`): `frame_ann` vetoes obvious locomotion / non-bed surfaces; optional AMASS root XY/Z caps. Default is **BERT / rules only** so recall stays usable.
BABEL does not label global support; whole-clip mode is a conservative proxy, not ground-truth "on mattress".

**MD §2 parity + BERT scores:** `--bert-accept-md-lie-labels` keeps the same sequence-level label gate as `MD/babel_amass_lie_related.md` (`md_lie_index_hits`), runs the NLI classifier for `bert_scores`, and **accepts every sequence that passes the label gate** (BERT does not veto). **By default npz is not required** for emitting rows; `resolved_npz` is filled when the file exists (`--require-npz` to enforce local files).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.genesis_ue_sync.sim_platform.datasets.babel_bed_classifier import (
    _maybe_stub_torchvision_for_text_transformers,
    build_classifier,
)
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.datasets.layout import HumanDatasetLayout

_BABEL_EXTRA_TOP_PREFIXES: frozenset[str] = frozenset(
    {
        "BMLrub",
        "DFaust67",
        "EyesJapanDataset",
        "MPIHDM05",
        "MPILimits",
        "MPImosh",
        "SSMsynced",
        "TCDhandMocap",
        "Transitionsmocap",
    }
)


def normalize_feat_p(value: str) -> str:
    return Path(value.strip()).as_posix()


def feat_p_alias_variants(feat_p: str) -> list[str]:
    feat_p = normalize_feat_p(feat_p)
    if not feat_p:
        return []
    parts = feat_p.split("/")
    out: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    add(feat_p)
    if len(parts) >= 2 and parts[0] == parts[1]:
        add("/".join(parts[1:]))
    elif len(parts) >= 1:
        add("/".join([parts[0], parts[0], *parts[1:]]))
    if len(parts) >= 2 and parts[0] in _BABEL_EXTRA_TOP_PREFIXES:
        add("/".join(parts[1:]))
    return out


def resolve_babel_release_dir(path: Path) -> Path:
    path = Path(path)
    if (path / "train.json").exists():
        return path
    nested = path / "babel_v1.0_release"
    if (nested / "train.json").exists():
        return nested
    raise FileNotFoundError(
        f"No BABEL JSON splits under {path} (expected train.json or babel_v1.0_release/train.json)."
    )


def _labels_from_seq_ann(seq_ann: dict[str, Any] | None) -> list[str]:
    if seq_ann is None:
        return []
    labels = seq_ann.get("labels") or []
    texts: list[str] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        for key in ("raw_label", "proc_label"):
            v = item.get(key)
            if v:
                texts.append(str(v).strip())
        for ac in item.get("act_cat") or []:
            if ac:
                texts.append(str(ac).strip())
    return texts


def _labels_from_frame_ann(frame_ann: dict[str, Any] | None) -> list[str]:
    if frame_ann is None:
        return []
    texts: list[str] = []
    for item in frame_ann.get("labels") or []:
        if not isinstance(item, dict):
            continue
        for key in ("raw_label", "proc_label"):
            v = item.get(key)
            if v:
                texts.append(str(v).strip())
        for ac in item.get("act_cat") or []:
            if ac:
                texts.append(str(ac).strip())
    return texts


def collect_label_texts(ann: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.extend(_labels_from_seq_ann(ann.get("seq_ann")))
    if ann.get("seq_anns"):
        sa = ann["seq_anns"]
        first = sa[0] if isinstance(sa, (list, tuple)) else sa
        if isinstance(first, dict):
            parts.extend(_labels_from_seq_ann(first))
    parts.extend(_labels_from_frame_ann(ann.get("frame_ann")))
    if ann.get("frame_anns"):
        fa = ann["frame_anns"]
        first = fa[0] if isinstance(fa, (list, tuple)) else fa
        if isinstance(first, dict):
            parts.extend(_labels_from_frame_ann(first))
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        low = p.lower()
        if low and low not in seen:
            seen.add(low)
            uniq.append(p)
    return ". ".join(uniq)


def primary_frame_ann(ann: dict[str, Any]) -> dict[str, Any] | None:
    fa = ann.get("frame_ann")
    if isinstance(fa, dict):
        return fa
    fas = ann.get("frame_anns")
    if fas:
        first = fas[0] if isinstance(fas, (list, tuple)) else fas
        if isinstance(first, dict):
            return first
    return None


def collect_non_transition_label_texts(ann: dict[str, Any]) -> str:
    """Join frame_ann segments that are not pure transition (better NLI premise for whole-clip pose)."""
    fa = primary_frame_ann(ann)
    if not isinstance(fa, dict):
        return collect_label_texts(ann)
    parts: list[str] = []
    for item in fa.get("labels") or []:
        if not isinstance(item, dict):
            continue
        cats = [str(c).strip().lower() for c in (item.get("act_cat") or []) if c]
        if cats == ["transition"]:
            continue
        for key in ("raw_label", "proc_label"):
            v = item.get(key)
            if v and str(v).strip().lower() != "transition":
                parts.append(str(v).strip())
        for c in item.get("act_cat") or []:
            if c and str(c).strip().lower() != "transition":
                parts.append(str(c).strip())
    if not parts:
        return collect_label_texts(ann)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        low = p.lower()
        if low and low not in seen:
            seen.add(low)
            uniq.append(p)
    return ". ".join(uniq)


_SEGMENT_LOCOMOTION = re.compile(
    r"\b(walk|walking|walks|run|running|jog|jogging|stride|strolls?|step\b|steps\b|stepping|"
    r"stand\b|standing|stand up|get up|rise\b|crawl|crawling|climb|climbing|"
    r"jump|jumping|hop|hopping|skip|skipping|kick|kicking|dance|dancing|gallop|backpedal|stumble|"
    r"punch|throw|catch|play sport|sports move|martial|fight|evade|backwards movement|forward movement)\b",
    re.IGNORECASE,
)
_SEGMENT_NON_BED_SURFACE = re.compile(
    r"\b(chair|stool|bench|couch|sofa|desk|vehicle|car seat|on the floor|on floor|on the ground|"
    r"on ground|touch ground|outdoor|grass|street|stairs?)\b",
    re.IGNORECASE,
)
_SEGMENT_LOW_POSE = re.compile(
    r"\b(lie|lying|lay|laid|sleep|sleeping|asleep|supine|prone|recline|reclining|nap|napping|"
    r"sit|sitting|sit down|lay down|lie down|rest|resting|relax|relaxing)\b",
    re.IGNORECASE,
)
_SEGMENT_GYM = re.compile(
    r"\b(push\s*up|pushup|sit\s*up|situp|crunch|crunches|plank|burpee)\b",
    re.IGNORECASE,
)


def _segment_blob(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for ac in item.get("act_cat") or []:
        parts.append(str(ac))
    for key in ("proc_label", "raw_label"):
        v = item.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).lower()


def whole_clip_frame_ann_ok(frame_ann: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Reject clips with any segment that clearly leaves the bed support or exercises on floor.

    BABEL segments often omit pose words; do not require every segment to match lie/sit regex.
    Motion stationarity + NLI provide additional whole-clip signal.
    """
    if not frame_ann:
        return True, None
    labels = frame_ann.get("labels") or []
    for i, item in enumerate(labels):
        if not isinstance(item, dict):
            continue
        blob = _segment_blob(item)
        cats = [str(c).strip().lower() for c in (item.get("act_cat") or []) if c]
        transition_only = cats == ["transition"] or (
            not cats and "transition" in blob and not _SEGMENT_LOCOMOTION.search(blob)
        )
        if transition_only and not _SEGMENT_NON_BED_SURFACE.search(blob):
            continue
        if _SEGMENT_GYM.search(blob):
            return False, f"segment[{i}]_gym"
        if _SEGMENT_LOCOMOTION.search(blob):
            return False, f"segment[{i}]_locomotion"
        if _SEGMENT_NON_BED_SURFACE.search(blob):
            return False, f"segment[{i}]_non_bed_surface"
    return True, None


_FEAT_P_BED_HINT = re.compile(
    r"(bed|mattress|lie|lying|lay_down|lie_down|sleep|asleep|recline|rest_in_bed|in_bed)",
    re.IGNORECASE,
)


def feat_p_has_bed_motion_hint(feat_p: str) -> bool:
    return bool(_FEAT_P_BED_HINT.search(feat_p or ""))


_MD_LIE_INDEX_RE = re.compile(
    r"(\bcrouch\s+to\s+lie\b|\blie\s+to\s+crouch\b|"
    r"\blie\s+down\b|\blying\s+down\b|\blay\s+down\b|\blay\s+on\b|\blie\s+on\b|\blie\s+face\b|\blying\s+face\b|"
    r"\blie\s+on\s+stomach\b|\blay\s+on\s+back\b|\blay\s+on\s+stomach\b|\blying\s+flat\b|\blie\s+flat\b|"
    r"\blay\b|\blie\b|\blying\b|sleep(?!walk)|\brecline\b|\bbed\b|\bsupine\b|\bprone\b)",
    re.IGNORECASE,
)


def md_lie_index_hits(ann: dict[str, Any]) -> list[str]:
    """Labels matching `MD/babel_amass_lie_related.md` §2 (raw_label / proc_label only)."""
    hits: list[str] = []
    seen: set[str] = set()

    def scan_block(seq_ann: dict[str, Any] | None) -> None:
        if not isinstance(seq_ann, dict):
            return
        for item in seq_ann.get("labels") or []:
            if not isinstance(item, dict):
                continue
            for key in ("raw_label", "proc_label"):
                v = item.get(key)
                if not v:
                    continue
                s = str(v).strip()
                if not s or s in seen:
                    continue
                if _MD_LIE_INDEX_RE.search(s):
                    seen.add(s)
                    hits.append(s)

    scan_block(ann.get("seq_ann"))
    if ann.get("seq_anns"):
        sa = ann["seq_anns"]
        first = sa[0] if isinstance(sa, (list, tuple)) else sa
        scan_block(first if isinstance(first, dict) else None)
    scan_block(primary_frame_ann(ann))
    return hits


_SLEEP_WALK_VETO_RE = re.compile(
    r"sleep[-\s]*walk(?:ing)?|sleepwalk(?:ing)?",
    re.IGNORECASE,
)
_LIE_LAY_SLEEP_BED_RE = re.compile(
    r"\b("
    r"lie|lies|lying|lay|laid|laying|"
    r"sleep|sleeping|asleep|"
    r"bed|beds|"
    r"in\s+bed|on\s+(?:the|a)\s+bed"
    r")\b",
    re.IGNORECASE,
)


def lie_lay_sleep_bed_hits(ann: dict[str, Any]) -> list[str]:
    """raw_label / proc_label only: lie / lay / sleep / bed family; drop sleep-walk strings; surface-agnostic."""
    hits: list[str] = []
    seen: set[str] = set()

    def scan_block(seq_ann: dict[str, Any] | None) -> None:
        if not isinstance(seq_ann, dict):
            return
        for item in seq_ann.get("labels") or []:
            if not isinstance(item, dict):
                continue
            for key in ("raw_label", "proc_label"):
                v = item.get(key)
                if not v:
                    continue
                s = str(v).strip()
                if not s or s in seen:
                    continue
                if _SLEEP_WALK_VETO_RE.search(s):
                    continue
                if _LIE_LAY_SLEEP_BED_RE.search(s):
                    seen.add(s)
                    hits.append(s)

    scan_block(ann.get("seq_ann"))
    if ann.get("seq_anns"):
        sa = ann["seq_anns"]
        first = sa[0] if isinstance(sa, (list, tuple)) else sa
        scan_block(first if isinstance(first, dict) else None)
    scan_block(primary_frame_ann(ann))
    return hits


def load_sequence_for_motion(npz_path: Path) -> HumanMotionSequence:
    with np.load(npz_path, allow_pickle=True) as payload:
        keys = set(payload.files)
    if {"source_dataset", "poses", "trans"} <= keys:
        return HumanMotionSequence.load(npz_path)
    if {"poses", "trans", "betas"} <= keys:
        return load_amass_sequence(npz_path)
    raise ValueError(f"Unsupported motion npz: {npz_path}")


def motion_whole_clip_stationary_ok(
    npz_path: Path,
    *,
    max_xy_path_m: float,
    max_xy_step_m: float,
    max_z_range_m: float,
) -> tuple[bool, str | None, dict[str, float]]:
    try:
        seq = load_sequence_for_motion(npz_path)
    except Exception as e:
        return False, f"motion_load_error:{e}", {}
    trans = np.asarray(seq.trans, dtype=np.float64)
    if trans.shape[0] < 2:
        return True, None, {"frames": float(trans.shape[0]), "xy_path_m": 0.0, "xy_max_step_m": 0.0, "z_range_m": 0.0}
    d = np.diff(trans[:, :2], axis=0)
    steps = np.linalg.norm(d, axis=1)
    xy_path = float(steps.sum())
    xy_max_step = float(steps.max())
    z_rng = float(trans[:, 2].max() - trans[:, 2].min())
    stats = {
        "frames": float(trans.shape[0]),
        "xy_path_m": xy_path,
        "xy_max_step_m": xy_max_step,
        "z_range_m": z_rng,
        "fps": float(seq.fps),
    }
    if max_xy_path_m > 0 and xy_path > max_xy_path_m:
        return False, "motion_xy_path_too_large", stats
    if max_xy_step_m > 0 and xy_max_step > max_xy_step_m:
        return False, "motion_xy_step_too_large", stats
    if max_z_range_m > 0 and z_rng > max_z_range_m:
        return False, "motion_z_range_too_large", stats
    return True, None, stats


def default_positive_patterns() -> list[re.Pattern[str]]:
    """Loose mode only: broad keywords (many false positives e.g. rest, lay on ground)."""
    words = (
        r"\b(lie|lying|lay|laid|sleep|sleeping|asleep|rest|resting|recline|reclining|"
        r"supine|prone|nap|napping|bed|mattress|under covers|in bed)\b"
    )
    return [re.compile(words, re.IGNORECASE)]


def default_negative_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(r"\b(walk|walking|run|running|jog|jogging|stride|stroll)\b", re.IGNORECASE),
        re.compile(r"\b(drive|driving|vehicle|steering)\b", re.IGNORECASE),
    ]


def default_sit_bed_pattern() -> re.Pattern[str]:
    return re.compile(r"\b(sit|sitting)\b.{0,120}\b(bed|mattress)\b|\b(bed|mattress)\b.{0,120}\b(sit|sitting)\b", re.IGNORECASE)


_BED_ANCHOR = re.compile(
    r"\b(bed|mattress|duvet)\b|in bed|on the bed|on a bed|under (?:the )?covers|on (?:the )?covers",
    re.IGNORECASE,
)
_FLOOR_GROUND = re.compile(
    r"\b(?:on the ground|on the floor|on floor|to the floor|from the floor|laying on the ground|lie on the ground|"
    r"on earth|outdoors on)\b",
    re.IGNORECASE,
)
_BED_ACTIVITY = re.compile(
    r"\b(lie|lying|lay|laid|sleep|sleeping|asleep|supine|prone|nap|napping|recline|reclining|"
    r"sit|sitting|rest)\b",
    re.IGNORECASE,
)
_RECLINE_ONLY = re.compile(
    r"\b(lie|lying|lay|laid|sleep|sleeping|asleep|supine|prone|nap|napping|recline|reclining)\b",
    re.IGNORECASE,
)
_GYM_FLOOR = re.compile(
    r"\b(push\s*up|pushup|sit\s*up|situp|crunch|crunches|plank|burpee|jumping\s*jack)\b",
    re.IGNORECASE,
)


def rule_match_loose(
    text: str,
    *,
    positive_patterns: list[re.Pattern[str]],
    negative_patterns: list[re.Pattern[str]],
    allow_sit_on_bed: bool,
) -> tuple[bool, list[str], str | None]:
    t = text.lower()
    matched: list[str] = []
    for pat in negative_patterns:
        if pat.search(t):
            return False, matched, f"negative:{pat.pattern[:40]}"
    pos_hit = any(pat.search(t) for pat in positive_patterns)
    if not pos_hit and allow_sit_on_bed:
        pos_hit = bool(default_sit_bed_pattern().search(t))
        if pos_hit:
            matched.append("sit_on_bed_heuristic")
    if not pos_hit:
        return False, matched, "no_positive_keyword"
    for pat in positive_patterns:
        m = pat.search(t)
        if m:
            matched.append(m.group(0))
    return True, matched, None


def rule_match_strict(
    text: str,
    *,
    negative_patterns: list[re.Pattern[str]],
    allow_sit_on_bed: bool,
) -> tuple[bool, list[str], str | None]:
    """Prefer real bed cues; drop floor/gym false positives; allow recline when labels omit 'bed'."""
    t = (text or "").lower()
    matched: list[str] = []
    for pat in negative_patterns:
        if pat.search(t):
            return False, matched, f"negative:{pat.pattern[:40]}"

    has_bed_anchor = bool(_BED_ANCHOR.search(t))
    sit_on_bed = bool(allow_sit_on_bed and default_sit_bed_pattern().search(t))
    has_floor = bool(_FLOOR_GROUND.search(t))

    if has_floor and not has_bed_anchor and not sit_on_bed:
        return False, matched, "floor_or_ground_without_bed"

    if _GYM_FLOOR.search(t) and not has_bed_anchor:
        return False, matched, "gym_exercise_without_bed"

    if has_bed_anchor or sit_on_bed:
        if not _BED_ACTIVITY.search(t):
            return False, matched, "no_bed_related_pose"
        if has_bed_anchor:
            matched.append("bed_anchor")
        if sit_on_bed:
            matched.append("sit_on_bed")
        m_act = _BED_ACTIVITY.search(t)
        if m_act:
            matched.append(m_act.group(0))
        return True, matched, None

    if has_floor:
        return False, matched, "floor_cue_without_bed"

    m_rec = _RECLINE_ONLY.search(t)
    if m_rec:
        matched.append(m_rec.group(0))
        return True, matched, None

    if _BED_ACTIVITY.search(t) and re.search(r"\b(sit|sitting|rest)\b", t):
        return False, matched, "sit_or_rest_without_bed_cue"

    return False, matched, "no_bed_or_recline_cue"


def resolve_npz_path(feat_p: str, amass_roots: tuple[Path, ...]) -> Path | None:
    for variant in feat_p_alias_variants(feat_p):
        for root in amass_roots:
            cand = root / variant
            if cand.is_file():
                return cand.resolve()
            cand2 = root / "raw" / variant
            if cand2.is_file():
                return cand2.resolve()
    return None


class ProgressPrinter:
    """Lightweight terminal progress bar with rate and ETA."""

    def __init__(self, total: int, *, enabled: bool = True, width: int = 28, min_interval_s: float = 0.2) -> None:
        self.total = max(int(total), 0)
        self.enabled = enabled
        self.width = width
        self.min_interval_s = min_interval_s
        self.start_t = time.monotonic()
        self.last_t = 0.0

    def _eta_text(self, scanned: int, now: float) -> str:
        if scanned <= 0 or self.total <= 0:
            return "--:--"
        rate = scanned / max(now - self.start_t, 1e-6)
        remain = max(self.total - scanned, 0) / max(rate, 1e-6)
        mins, secs = divmod(int(remain), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def update(self, *, scanned: int, accepted: int, stage: str = "scan", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and scanned < self.total and (now - self.last_t) < self.min_interval_s:
            return
        self.last_t = now
        frac = (scanned / self.total) if self.total > 0 else 1.0
        frac = max(0.0, min(1.0, frac))
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(now - self.start_t, 1e-6)
        rate = scanned / elapsed
        eta = self._eta_text(scanned, now)
        line = (
            f"\r[{stage}] [{bar}] {scanned}/{self.total} "
            f"accepted={accepted} rate={rate:.2f}it/s eta={eta}"
        )
        print(line, end="", file=sys.stderr, flush=True)

    def close(self, *, scanned: int, accepted: int, stage: str = "done") -> None:
        if not self.enabled:
            return
        self.update(scanned=scanned, accepted=accepted, stage=stage, force=True)
        print(file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    layout = HumanDatasetLayout.default()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--babel-root", type=Path, default=layout.babel_release_root)
    p.add_argument("--split", type=str, default="train", help="train, val, test, extra_train, extra_val")
    p.add_argument(
        "--amass-root",
        type=Path,
        action="append",
        default=[],
        help="Repeatable. Defaults: dataset/raw/humans/amass, amass_hf",
    )
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "babel_bed_subset" / "manifest.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Max accepted rows (0=all)")
    p.add_argument("--scan-limit", type=int, default=0, help="Max BABEL entries scanned (0=all)")
    p.add_argument(
        "--classifier-mode",
        type=str,
        choices=("rule", "bert", "zero_shot", "finetuned", "rule_then_bert", "lie_keyword"),
        default="bert",
    )
    p.add_argument("--hf-model", type=str, default="facebook/bart-large-mnli")
    p.add_argument("--finetuned-dir", type=Path, default=None)
    p.add_argument("--bert-bed-threshold", type=float, default=0.38, help="Minimum on_bed score.")
    p.add_argument(
        "--bert-pose-threshold",
        type=float,
        default=0.35,
        help="Minimum max score over lying_or_sleeping / prone / sitting.",
    )
    p.add_argument(
        "--bert-lying-threshold",
        type=float,
        default=0.0,
        help="Optional extra floor for lying_or_sleeping itself (0 disables this extra gate).",
    )
    p.add_argument(
        "--bert-recline-threshold",
        type=float,
        default=0.35,
        help="When bed support is implicit rather than explicit, require max(lying_or_sleeping, prone) >= this.",
    )
    p.add_argument("--bert-loco-threshold", type=float, default=0.35)
    p.add_argument(
        "--bert-other-surface-threshold",
        type=float,
        default=0.32,
        help="Reject when non-bed support surface score exceeds this (implicit recline branch).",
    )
    p.add_argument(
        "--bert-lie-phrase-other-surface",
        type=float,
        default=0.52,
        help="When raw/proc labels match MD §2 lie regex (see md_lie_index_hits), allow at least this cap for other_surface on implicit recline only.",
    )
    p.add_argument(
        "--no-bert-lie-phrase-surface-relax",
        action="store_true",
        help="Do not raise other_surface cap when MD §2 lie labels are present.",
    )
    p.add_argument(
        "--bert-stand-threshold",
        type=float,
        default=0.35,
        help="Reject when standing score exceeds this.",
    )
    p.add_argument("--bert-floor-threshold", type=float, dest="bert_other_surface_threshold", help=argparse.SUPPRESS)
    p.add_argument("--bert-other-threshold", type=float, dest="bert_other_surface_threshold", help=argparse.SUPPRESS)
    p.add_argument(
        "--rule-prefilter",
        action="store_true",
        help="With bert/zero_shot/finetuned: skip entries that fail strict (or loose) keyword rules before calling the model.",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda for HF models")
    p.add_argument(
        "--no-sit-on-bed",
        action="store_true",
        help="Do not treat sit+sitting+near bed/mattress as positive (default: sitting on bed allowed).",
    )
    p.add_argument(
        "--loose-rule",
        action="store_true",
        help="Use old broad rules (lay/rest without bed keyword; more false positives).",
    )
    p.add_argument(
        "--require-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, drop rows when feat_p cannot be resolved to a local npz. Default off: emit rows anyway; resolved_npz set when found.",
    )
    p.add_argument(
        "--bert-accept-md-lie-labels",
        action="store_true",
        help="bert/zero_shot/finetuned only: require md_lie_index_hits (MD §2); run BERT for scores but accept on labels (ignore NLI gates).",
    )
    p.add_argument(
        "--whole-clip-on-bed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strict add-on: vetoes locomotion/non-bed cues in frame_ann + AMASS root motion caps (default off; use for fewer false positives, much lower recall).",
    )
    p.add_argument(
        "--motion-max-xy-path-m",
        type=float,
        default=0.55,
        help="Max cumulative root XY displacement over the clip (0 disables).",
    )
    p.add_argument(
        "--motion-max-xy-step-m",
        type=float,
        default=0.07,
        help="Max single-frame root XY step (0 disables).",
    )
    p.add_argument(
        "--motion-max-z-range-m",
        type=float,
        default=0.45,
        help="Max root Z max-min over the clip (0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.classifier_mode in ("bert", "zero_shot", "finetuned", "rule_then_bert"):
        _maybe_stub_torchvision_for_text_transformers()
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
        except (ImportError, RuntimeError) as e:
            extra = ""
            if "mpmath" in str(e).lower():
                extra = (
                    "SymPy needs mpmath inside the SAME env Python sees. With PYTHONNOUSERSITE=1, "
                    "~/.local is hidden — if mpmath is only there, import fails.\n"
                    "Fix: install into the active env (check path is under envs/.../site-packages):\n"
                    "  python3 -m pip install mpmath\n"
                    "  PYTHONNOUSERSITE=1 python3 -c \"import mpmath; print(mpmath.__file__)\"\n"
                    "Or: conda install -c conda-forge mpmath\n\n"
                )
            raise SystemExit(
                "Could not import Hugging Face text stack (transformers + torch).\n\n"
                + extra
                + "If packages are split between ~/.local and conda, use:\n"
                "  PYTHONNOUSERSITE=1 python3 scripts/pipeline/support_motion/extract_babel_bed_subset.py ...\n\n"
                "Broken torchvision alone should not block this script (no pipeline/vision imports).\n"
                "If torch fails: pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124\n\n"
                "No HF: --classifier-mode rule or lie_keyword\n\n"
                f"Detail: {e!r}"
            ) from e
    babel_dir = resolve_babel_release_dir(Path(args.babel_root))
    split_path = babel_dir / f"{args.split}.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    amass_roots = list(args.amass_root)
    if not amass_roots:
        layout = HumanDatasetLayout.default()
        amass_roots = [layout.amass_raw_root, layout.amass_hf_root]
    roots_t = tuple(Path(r).resolve() for r in amass_roots)

    pos_patterns = default_positive_patterns()
    neg_patterns = default_negative_patterns()

    clf = None
    if args.classifier_mode in ("bert", "zero_shot", "finetuned", "rule_then_bert"):
        print(
            f"[extract_babel_bed_subset] preparing classifier mode={args.classifier_mode} model={args.hf_model}",
            file=sys.stderr,
            flush=True,
        )
        mode = "finetuned" if args.classifier_mode == "finetuned" else "zero_shot"
        clf = build_classifier(
            mode,
            hf_model=args.hf_model,
            finetuned_dir=args.finetuned_dir,
            device=args.device,
        )
        print("[extract_babel_bed_subset] classifier ready", file=sys.stderr, flush=True)

    data: dict[str, Any] = json.loads(split_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_entries = min(len(data), args.scan_limit) if args.scan_limit else len(data)

    accepted = 0
    scanned = 0
    progress = ProgressPrinter(total_entries, enabled=total_entries > 0)
    with args.output.open("w", encoding="utf-8") as out_f:
        for sid, ann in data.items():
            if args.scan_limit and scanned >= args.scan_limit:
                break
            scanned += 1
            progress.update(scanned=scanned, accepted=accepted)
            if not isinstance(ann, dict):
                continue
            feat_p = str(ann.get("feat_p") or "").strip()
            if not feat_p:
                continue
            label_text = collect_label_texts(ann)
            semantic_text = collect_non_transition_label_texts(ann) if args.whole_clip_on_bed else label_text
            feat_hint = feat_p_has_bed_motion_hint(feat_p)
            clip_check: dict[str, Any] = {
                "whole_clip_on_bed": args.whole_clip_on_bed,
                "feat_p_bed_hint": feat_hint,
            }
            if args.whole_clip_on_bed:
                fo, fr = whole_clip_frame_ann_ok(primary_frame_ann(ann))
                clip_check["frame_ann_ok"] = fo
                clip_check["frame_reject"] = fr
                if not fo:
                    continue
            ok_rule = True
            matched_kw: list[str] = []
            reject: str | None = None
            need_rule = args.classifier_mode in ("rule", "rule_then_bert") or (
                args.rule_prefilter and args.classifier_mode in ("bert", "zero_shot", "finetuned")
            )
            bert_scores: dict[str, Any] | None = None
            if args.classifier_mode == "lie_keyword":
                kw_hits = lie_lay_sleep_bed_hits(ann)
                if not kw_hits:
                    continue
                matched_kw = list(kw_hits)
                bert_scores = {"lie_lay_sleep_bed_hits": kw_hits}
            if need_rule:
                if args.loose_rule:
                    ok_rule, matched_kw, reject = rule_match_loose(
                        label_text,
                        positive_patterns=pos_patterns,
                        negative_patterns=neg_patterns,
                        allow_sit_on_bed=not args.no_sit_on_bed,
                    )
                else:
                    ok_rule, matched_kw, reject = rule_match_strict(
                        label_text,
                        negative_patterns=neg_patterns,
                        allow_sit_on_bed=not args.no_sit_on_bed,
                    )
            if args.classifier_mode == "rule" and not ok_rule:
                continue
            if args.classifier_mode in ("bert", "zero_shot", "finetuned") and not args.rule_prefilter:
                ok_rule = True
                matched_kw = []
                reject = None
            if args.rule_prefilter and args.classifier_mode in ("bert", "zero_shot", "finetuned") and not ok_rule:
                continue
            if clf is not None and (args.classifier_mode != "rule"):
                md_gate_hits = md_lie_index_hits(ann)
                if args.bert_accept_md_lie_labels and not md_gate_hits:
                    continue
                scores = clf.score(semantic_text or label_text or feat_p)
                on_bed_sc = float(scores.on_bed_likelihood)
                other_surface_sc = float(scores.other_surface_likelihood)
                lying_sc = float(scores.lying_or_sleeping_likelihood)
                prone_sc = float(scores.prone_likelihood)
                sitting_sc = float(scores.sitting_likelihood)
                standing_sc = float(scores.standing_likelihood)
                positive_pose_sc = float(scores.positive_pose_likelihood)
                recline_pose_sc = max(lying_sc, prone_sc)
                _recline_src = semantic_text if args.whole_clip_on_bed else label_text
                implicit_recline_text_ok = bool(_RECLINE_ONLY.search(_recline_src))
                bert_scores = {
                    "on_bed_likelihood": on_bed_sc,
                    "lying_or_sleeping_likelihood": lying_sc,
                    "prone_likelihood": prone_sc,
                    "sitting_likelihood": sitting_sc,
                    "standing_likelihood": standing_sc,
                    "positive_pose_likelihood": positive_pose_sc,
                    "recline_pose_likelihood": recline_pose_sc,
                    "locomotion_likelihood": scores.locomotion_likelihood,
                    "other_surface_likelihood": other_surface_sc,
                    "bed_likelihood": on_bed_sc,
                    "floor_or_ground_likelihood": other_surface_sc,
                    "other_non_bed_likelihood": other_surface_sc,
                    "implicit_recline_text_ok": implicit_recline_text_ok,
                }

                explicit_bed_pose_ok = (
                    (on_bed_sc >= args.bert_bed_threshold or feat_hint)
                    and positive_pose_sc >= args.bert_pose_threshold
                )
                md_hits = md_gate_hits
                lie_phrase_surface_relax = bool(md_hits) and not args.no_bert_lie_phrase_surface_relax
                implicit_other_cap = (
                    max(args.bert_other_surface_threshold, args.bert_lie_phrase_other_surface)
                    if lie_phrase_surface_relax
                    else args.bert_other_surface_threshold
                )
                implicit_bed_recline_ok = (
                    implicit_recline_text_ok
                    and on_bed_sc < args.bert_bed_threshold
                    and not feat_hint
                    and other_surface_sc <= implicit_other_cap
                    and recline_pose_sc >= args.bert_recline_threshold
                )
                bert_scores["explicit_bed_pose_ok"] = explicit_bed_pose_ok
                bert_scores["implicit_bed_recline_ok"] = implicit_bed_recline_ok
                bert_scores["md_lie_hits"] = md_hits
                bert_scores["implicit_other_surface_cap"] = implicit_other_cap
                bert_scores["lie_phrase_surface_relax"] = lie_phrase_surface_relax
                lying_gate_ok = lying_sc >= args.bert_lying_threshold
                negative_gate_ok = (
                    scores.locomotion_likelihood < args.bert_loco_threshold and standing_sc <= args.bert_stand_threshold
                )
                ok_bert_core = negative_gate_ok and lying_gate_ok and (explicit_bed_pose_ok or implicit_bed_recline_ok)
                ok_bert = True if args.bert_accept_md_lie_labels else ok_bert_core
                bert_scores["ok_bert_core"] = ok_bert_core
                bert_scores["bert_accept_md_lie_labels"] = bool(args.bert_accept_md_lie_labels)
                if args.bert_accept_md_lie_labels and md_gate_hits:
                    matched_kw = list(dict.fromkeys([*matched_kw, *md_gate_hits]))
                if args.classifier_mode == "rule_then_bert":
                    if not ok_bert:
                        continue
                    if not ok_rule:
                        matched_kw = list(matched_kw)
                        matched_kw.append("bert_override")
                else:
                    if not ok_bert:
                        continue

            resolved = resolve_npz_path(feat_p, roots_t)
            if args.require_npz and resolved is None:
                row = {
                    "babel_sid": sid,
                    "feat_p": feat_p,
                    "resolved_npz": None,
                    "split": args.split,
                    "label_text": label_text,
                    "semantic_label_text": semantic_text,
                    "matched_keywords": matched_kw,
                    "reject_reason": "npz_not_found",
                    "classifier_mode": args.classifier_mode,
                    "bert_scores": bert_scores,
                    "whole_clip": clip_check,
                }
                out_f.write(json.dumps(row, ensure_ascii=True) + "\n")
                continue

            if resolved is not None and args.whole_clip_on_bed:
                mok, mrej, mstats = motion_whole_clip_stationary_ok(
                    resolved,
                    max_xy_path_m=args.motion_max_xy_path_m,
                    max_xy_step_m=args.motion_max_xy_step_m,
                    max_z_range_m=args.motion_max_z_range_m,
                )
                clip_check["motion_ok"] = mok
                clip_check["motion_reject"] = mrej
                clip_check["motion_stats"] = mstats
                if not mok:
                    continue
            else:
                clip_check["motion_ok"] = None
                clip_check["motion_stats"] = {}

            row = {
                "babel_sid": sid,
                "feat_p": feat_p,
                "resolved_npz": str(resolved) if resolved else None,
                "split": args.split,
                "label_text": label_text,
                "semantic_label_text": semantic_text,
                "matched_keywords": matched_kw,
                "reject_reason": None,
                "classifier_mode": args.classifier_mode,
                "bert_scores": bert_scores,
                "whole_clip": clip_check,
            }
            out_f.write(json.dumps(row, ensure_ascii=True) + "\n")
            accepted += 1
            progress.update(scanned=scanned, accepted=accepted)
            if args.limit and accepted >= args.limit:
                break

    progress.close(scanned=scanned, accepted=accepted)
    print(f"[extract_babel_bed_subset] wrote {args.output} accepted={accepted} scanned={scanned}", flush=True)


if __name__ == "__main__":
    main()
