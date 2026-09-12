"""Versioned random-walk confidence, computed outside the control thread.

Karamalis, Ultrasound Confidence Maps and Applications in Medical Image
Processing, thesis §4.3.4, eqs 4.10–4.14:
https://mediatum.ub.tum.de/doc/1129526/827108.pdf
Eight-neighbour weighted Laplacian, top=1/bottom=0 Dirichlet boundary.
This is a signal-propagation confidence policy, not a mechanical contact label.

Default v2 follows the public CAMP collaborator MATLAB B-mode implementation:
https://github.com/TJKlein/Nakagami_Confidence_Maps
Its depth weight is 1-exp(-alpha*d), unlike the thesis expression used by v1.
Both algorithms remain explicit for replay; their quality thresholds are not
interchangeable or mechanically calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import zoom

from .types import ContactObservation


def revision(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class FeatureConfig:
    algorithm_version: str = "randomwalk_camp_bmode_v2"
    width: int = 96
    height: int = 112
    attenuation: float = 2.0
    contrast: float = 90.0
    lateral_penalty: float = 0.03
    min_weight: float = 1e-10
    camp_weight_epsilon: float = 1e-5
    near_depth: tuple[float, float] = (0.04, 0.22)
    lateral_windows: tuple = ((0.04, 0.34), (0.34, 0.66), (0.66, 0.96))
    effective_delay_s: float = 0.15196365053143765
    calibration_version: str = "unverified"
    image_x_sign: int = 1
    low_confidence_threshold: float = 0.8
    unknown_scanline_range: float = 1.0

    def __post_init__(self):
        if self.algorithm_version not in ("randomwalk_thesis_v1", "randomwalk_camp_bmode_v2", "randomwalk_welleweerd2020_v3"):
            raise ValueError("unsupported confidence algorithm_version")
        if not 0 < self.low_confidence_threshold < 1 or not 0 <= self.unknown_scanline_range <= 255:
            raise ValueError("invalid experimental confidence/unknown thresholds")
        if self.width < 8 or self.height < 8 or self.width*self.height > 250000:
            raise ValueError("invalid processing resolution")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.attenuation, self.contrast, self.lateral_penalty, self.effective_delay_s)):
            raise ValueError("invalid confidence parameters")
        if not 0 < self.min_weight < 1:
            raise ValueError("min_weight must be positive")
        if not 0 < self.camp_weight_epsilon < 1:
            raise ValueError("camp_weight_epsilon must be positive")
        if len(self.lateral_windows) != 3 or self.image_x_sign not in (-1, 1):
            raise ValueError("three windows and an explicit image axis sign required")
        for lo, hi in (self.near_depth, *self.lateral_windows):
            if not 0 <= lo < hi <= 1:
                raise ValueError("window fractions must satisfy 0 <= lo < hi <= 1")

    @property
    def window_version(self):
        values = asdict(self)
        if self.algorithm_version != "randomwalk_welleweerd2020_v3":
            # These policies do not participate in historical v1/v2 observations.
            values.pop("low_confidence_threshold")
            values.pop("unknown_scanline_range")
        if self.algorithm_version == "randomwalk_thesis_v1":
            # Preserve historical v1 evidence hashes; this parameter is v2-only.
            values.pop("camp_weight_epsilon")
        return revision(values)


def random_walk_confidence(image, config=None):
    """Solve L_UU c_U = -L_UM c_M, without thresholded dark occupancy."""
    cfg = config or FeatureConfig()
    im = np.asarray(image)
    if im.ndim != 2 or min(im.shape) < 2 or not np.isfinite(im).all():
        raise ValueError("finite grayscale B-mode image required")
    if np.min(im) < 0 or np.max(im) > 255:
        raise ValueError("B-mode intensity must be on the [0,255] display scale")
    camp = cfg.algorithm_version in ("randomwalk_camp_bmode_v2", "randomwalk_welleweerd2020_v3")
    im = zoom(im.astype(np.float64)/(1. if camp else 255.),
              (cfg.height/im.shape[0], cfg.width/im.shape[1]),
              order=1, prefilter=False)
    h, w = im.shape
    depth_weight = np.exp(-cfg.attenuation*np.linspace(0, 1, h)[:, None])
    if camp:
        # MATLAB confMap normalizes its processing-resolution B-mode input.
        im = (im-im.min())/(np.ptp(im)+np.finfo(float).eps)
        depth_weight = 1.-depth_weight
    signal = im * depth_weight
    indices = np.arange(h*w).reshape(h, w)
    sources, targets, differences, penalties = [], [], [], []
    for dy, dx, penalty in ((1, 0, 0.), (0, 1, cfg.lateral_penalty),
                             (1, 1, math.sqrt(2)*cfg.lateral_penalty),
                             (1, -1, math.sqrt(2)*cfg.lateral_penalty)):
        y0, y1 = 0, h-dy
        x0, x1 = max(0, -dx), min(w, w-dx)
        a = indices[y0:y1, x0:x1].ravel()
        b = indices[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
        delta = np.abs(signal.ravel()[a]-signal.ravel()[b])
        if camp and dx:
            penalty = cfg.lateral_penalty
        sources.extend((a, b)); targets.extend((b, a))
        differences.extend((delta, delta))
        penalties.extend((np.full(delta.shape, penalty), np.full(delta.shape, penalty)))
    delta = np.concatenate(differences)
    penalty = np.concatenate(penalties)
    if camp:
        # MATLAB includes zero diagonal entries in both normalizations, so the
        # minimum is zero even when all actual graph edges have positive cost.
        delta = delta/(delta.max()+np.finfo(float).eps)+penalty
        delta = delta/(delta.max()+np.finfo(float).eps)
        weight = np.exp(-cfg.contrast*delta)+cfg.camp_weight_epsilon
    else:
        weight = np.maximum(np.exp(-cfg.contrast*(delta+penalty)), cfg.min_weight)
    adjacency = sparse.coo_matrix((weight,
                                  (np.concatenate(sources), np.concatenate(targets))),
                                 shape=(h*w, h*w)).tocsr()
    lap = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    unknown = np.arange(w, (h-1)*w)
    top = np.arange(w)
    rhs = -np.asarray(lap[unknown][:, top].sum(axis=1)).ravel()
    interior = np.asarray(spsolve(lap[unknown][:, unknown].tocsc(), rhs)).ravel()
    if not np.isfinite(interior).all() or interior.min() < -1e-6 or interior.max() > 1+1e-6:
        raise RuntimeError("random-walk linear solve violated probability bounds")
    result = np.zeros((h, w), dtype=float)
    result[0] = 1.
    result[1:-1] = np.clip(interior, 0, 1).reshape(h-2, w)
    return result


def window_quality(confidence, config=None):
    cfg = config or FeatureConfig()
    c = np.asarray(confidence, dtype=float)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("finite confidence map required")
    h, w = c.shape
    y0, y1 = (int(h*f) for f in cfg.near_depth)
    y1 = max(y0+1, y1)
    return np.array([c[y0:y1, int(lo*w):max(int(lo*w)+1, int(hi*w))].mean()
                     for lo, hi in cfg.lateral_windows])



def welleweerd_config(**overrides):
    """Declared reconstruction, not undisclosed paper parameters.

    Paper gives 100 by 145 without axis order: assume height=100, width=145.
    ROI depth 22%, graph parameters and threshold .8 are experimental choices.
    """
    values = dict(algorithm_version="randomwalk_welleweerd2020_v3", height=100,
                  width=145, near_depth=(0., .22))
    values.update(overrides)
    return FeatureConfig(**values)


def load_feature_config(path):
    """Read a bare mapping or the exact controller feature.config mapping."""
    from pathlib import Path
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        import yaml
        raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and "feature" in raw:
        raw = raw["feature"]["config"]
    if not isinstance(raw, dict):
        raise ValueError("feature config must be an explicit mapping")
    return FeatureConfig(**raw)


def unknown_scanlines(image, config):
    """Separate validity overlay: resized full-depth scanline range < 1 DN.

    Also mark every column unknown for globally flat (<1 DN range) frames.
    This is an experimental no-variation rule, not a physical contact detector;
    it does not modify the random-walk map or its reported statistics.
    """
    im = np.asarray(image, dtype=float)
    resized = zoom(im, (config.height/im.shape[0], config.width/im.shape[1]),
                   order=1, prefilter=False)
    return ((np.ptp(resized, axis=0) < config.unknown_scanline_range) |
            (np.ptp(im) < 1.0))


def confidence_features(image, confidence, config):
    """Paper eqs (2),(3) and explicit experimental diagnostic regions.

    Barycentre uses 1-based (row,column) pixel coordinates in processing image.
    Raw statistics include unknown columns; validity is reported separately.
    """
    c = np.asarray(confidence, dtype=float)
    if c.shape != (config.height, config.width) or not np.isfinite(c).all():
        raise ValueError("confidence shape must match feature configuration")
    h, w = c.shape
    y0, y1 = (int(h*f) for f in config.near_depth)
    y1 = max(y0+1, y1)
    roi = c[y0:y1]
    mass = float(roi.sum())
    unknown = unknown_scanlines(image, config)
    columns = roi.mean(axis=0)
    low = (columns < config.low_confidence_threshold) & ~unknown
    regions = []
    for label, mask in (("low_confidence", low), ("unknown", unknown)):
        edges = np.diff(np.r_[False, mask, False].astype(int))
        regions.extend(dict(label=label, x_start=int(a), x_stop=int(b))
                       for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return dict(top_roi_rows=[y0, y1], roi_mean=float(roi.mean()),
                paper_eq3_fullarea_mean=mass/(h*w), confidence_mass=mass,
                barycenter_row_col_1based=([float((roi*np.arange(y0+1, y1+1)[:, None]).sum()/mass),
                                            float((roi*np.arange(1, w+1)[None, :]).sum()/mass)]
                                           if mass > 0 else None),
                column_confidence=columns.tolist(), unknown_columns=unknown.tolist(),
                low_confidence_columns=low.tolist(), regions=regions,
                quality_lcr=window_quality(c, config).tolist(),
                frame_status="unknown" if unknown.all() else "partly_unknown" if unknown.any() else "valid",
                threshold=config.low_confidence_threshold,
                policy_notes=["Threshold and ROI depth are experimental, not specified by Welleweerd2020.",
                              "100 by 145 interpreted as height by width; paper axis order unspecified.",
                              "Unknown: full-depth resized scanline range below configured DN threshold, or globally flat frame.",
                              "Raw confidence and statistics are unmasked; regions are not calibrated physical contact labels.",
                              "Eq3 divides top-ROI sum by full image area; ROI mean divides by ROI area.",
                              "Barycentre is in pixels only; no physical angle without calibration."])


def effective_image_time(recorded_time_s, delay_s, *, already_aligned=False):
    """Aligned H5 rows already use image-effective time; never subtract again."""
    if not math.isfinite(recorded_time_s) or not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError("invalid image registration times")
    return float(recorded_time_s) if already_aligned else float(recorded_time_s-delay_s)


def registration_revision(config, *, source_id, crop_box=None, hflip=False,
                          already_aligned=False, clock_domain="host_monotonic"):
    return revision({"source": source_id, "crop": crop_box, "hflip": bool(hflip),
                     "calibration": config.calibration_version, "image_x_sign": config.image_x_sign,
                     "delay_s": config.effective_delay_s, "already_aligned": bool(already_aligned),
                     "clock_domain": clock_domain})


class FeatureExtractor:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.last_features = None

    def extract(self, image, *, frame_seq, source_id, capture_time_s, received_time_s,
                crop_box=None, hflip=False, already_aligned=False, clock_domain="host_monotonic",
                registration_source_id=None):
        cfg = self.config
        if clock_domain not in ("host_monotonic", "offline_aligned"):
            raise ValueError("convert image time to controller clock before extraction")
        # A publisher restart resets frame history, not the calibrated image geometry.
        # Keep its instance ID in observation.source_id, outside the registration hash.
        registration = registration_revision(cfg, source_id=source_id if registration_source_id is None else registration_source_id,
                                             crop_box=crop_box, hflip=hflip,
                                             already_aligned=already_aligned, clock_domain=clock_domain)
        c = random_walk_confidence(image, cfg)
        q = window_quality(c, cfg)
        # Blank/no-image frames are invalid observations, not reliable low quality.
        valid = np.full(3, float(np.ptp(image)) >= 1.0, dtype=bool)
        if cfg.algorithm_version == "randomwalk_welleweerd2020_v3":
            self.last_features = confidence_features(image, c, cfg)
            known = ~unknown_scanlines(image, cfg)
            valid &= np.array([known[int(lo*cfg.width):max(int(lo*cfg.width)+1, int(hi*cfg.width))].all()
                               for lo, hi in cfg.lateral_windows])
        obs = ContactObservation(int(frame_seq), str(source_id),
                                 effective_image_time(capture_time_s, cfg.effective_delay_s,
                                                      already_aligned=already_aligned),
                                 float(received_time_s), q, valid, registration, cfg.window_version,
                                 calibration_version=cfg.calibration_version)
        return obs, c


class LatestObservation:
    """Reject duplicates/late frames; version changes invalidate previous evidence."""
    def __init__(self):
        self.observation = None
        self.generation = 0
        self.last_reason = "empty"
        self._retired_sources = set()

    def accept(self, obs: ContactObservation):
        old = self.observation
        if old is not None:
            if obs.source_id in self._retired_sources or obs.received_time_s < old.received_time_s:
                self.last_reason = "retired_source_or_late_receipt"
                return False
            if obs.source_id == old.source_id and obs.frame_seq <= old.frame_seq:
                self.last_reason = "late_or_duplicate_sequence"
                return False
            if obs.version != old.version or obs.source_id != old.source_id:
                if obs.source_id != old.source_id:
                    self._retired_sources.add(old.source_id)
                self.generation += 1
                self.last_reason = "version_changed"
            else:
                if obs.effective_time_s <= old.effective_time_s:
                    self.last_reason = "late_or_duplicate_time"
                    return False
                self.last_reason = "updated"
        else:
            self.last_reason = "first"
        self.observation = obs
        return True
