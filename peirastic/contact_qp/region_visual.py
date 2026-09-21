"""Spatial deficit task over all configured equal-width confidence regions.

Each region remains identifiable. A single rocking axis can only respond to
the signed first spatial moment; symmetric defects are detected and slow path
progress, but cannot justify inventing a rocking direction or increasing force.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .types import ProbeGeometry, positive, vector


def preferred_scan_alpha(max_deficit, c_min, *, progress_loss_gain=0.75, alpha_min=0.25,
                         alpha_min_at_quality=None):
    """Map the worst-region deficit onto path α.

    Default is the paper law α=clip(1−0.75 max d_i/c_min, α_min, 1), which only
    reaches α_min at c=0. When alpha_min_at_quality is set, the same floor is
    reached at that confidence so a typical hole (≈0.79) already commands α_min.
    """
    if not math.isfinite(c_min) or not 0 < c_min < 1:
        raise ValueError('region confidence threshold must be in (0,1)')
    if not math.isfinite(alpha_min) or not 0 < alpha_min < 1:
        raise ValueError('alpha_min must be in (0,1)')
    if not math.isfinite(progress_loss_gain) or progress_loss_gain <= 0:
        raise ValueError('progress_loss_gain must be finite and positive')
    deficit = max(0., float(max_deficit))
    if alpha_min_at_quality is None:
        return float(np.clip(1. - progress_loss_gain * deficit / c_min, alpha_min, 1.))
    floor_q = float(alpha_min_at_quality)
    if not math.isfinite(floor_q) or not 0 <= floor_q < c_min:
        raise ValueError('alpha_min_at_quality must be in [0, c_min)')
    return float(np.clip(1. - (1. - alpha_min) * deficit / (c_min - floor_q), alpha_min, 1.))


def preferred_relative_alpha(max_deficit, span, *, alpha_min=0.25):
    """Map a relative (good-side) deficit onto path α.

    ``span`` is the imbalance that already commands α_min. A uniform map has
    deficit 0 so α stays 1 on both phantom gel and human tissue.
    """
    if not math.isfinite(span) or span <= 0:
        raise ValueError('relative_alpha_span must be finite and positive')
    if not math.isfinite(alpha_min) or not 0 < alpha_min < 1:
        raise ValueError('alpha_min must be in (0,1)')
    deficit = max(0., float(max_deficit))
    return float(np.clip(1. - (1. - alpha_min) * deficit / span, alpha_min, 1.))


def better_half_reference(quality):
    """Per-frame good-side median. Adapts to phantom (~0.99) and tissue (~0.8)."""

    q = np.asarray(quality, dtype=float).reshape(-1)
    if q.size == 0 or not np.isfinite(q).all():
        raise ValueError('quality reference requires a finite region map')
    half = q.size // 2
    if half == 0:
        return float(q[0])
    return float(max(np.median(q[:half]), np.median(q[half:])))


def better_half_spread(quality):
    """Median absolute deviation of the better half. A flat gel side is ~0."""

    q = np.asarray(quality, dtype=float).reshape(-1)
    half = q.size // 2
    if half == 0:
        return 0.
    left, right = q[:half], q[half:]
    side = left if float(np.median(left)) >= float(np.median(right)) else right
    center = float(np.median(side))
    return float(np.median(np.abs(side - center)))


@dataclass(frozen=True)
class RegionVisualRequest:
    request_rad_s: float
    alpha_preferred: float
    valid: bool
    reason: str
    region_edges: np.ndarray
    region_centers_m: np.ndarray
    region_weights: np.ndarray
    region_quality: np.ndarray | None
    region_valid: np.ndarray
    region_deficits: np.ndarray | None
    bad_region_mask: np.ndarray
    bad_region_indices: tuple[int, ...]
    bad_region_positions_m: tuple[float, ...]
    deficit_area: float | None
    deficit_first_moment_m: float | None
    spatial_second_moment_m2: float
    quality_reference: float | None
    deficit_mode: str

    def to_dict(self):
        return asdict(self)


def region_visual_request(quality, region_valid, region_edges, geometry: ProbeGeometry,
                          *, c_min, repair_speed_m_s, evidence_valid=True,
                          progress_loss_gain=0.75, alpha_min=0.25, alpha_min_at_quality=None,
                          max_request_rad_s=None, rocking_reference=None,
                          deficit_mode='absolute', balance_deadband=0.0,
                          relative_alpha_span=0.15, relative_spread_floor=None):
    """Least-squares rocking fit to per-region indentation deficits.

    v_z(x)=U-x*Omega and rocking d_i=max(ref-c_i,0). The scan default
    ``deficit_mode='absolute'`` uses ref=c_min, so Omega is gated.
    ``rocking_reference=1`` keeps a proportional first-moment amplitude while
    alpha still uses max(c_min-c_i,0). ``deficit_mode='relative'`` sets ref to
    this frame's better-half median, so a uniformly dark human map does not
    invent ω; a one-sided hole still produces the same first-moment Ω and a
    separate α from the relative span. Uniform or symmetric deficits have
    zero net first moment. Unknown/stale evidence withdraws the row.
    """
    edges=np.asarray(region_edges,dtype=float)
    if (edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all()
            or edges[0]!=0. or edges[-1]!=1. or np.any(np.diff(edges)<=0)):
        raise ValueError('region edges must partition the complete normalized image width')
    n=len(edges)-1
    if not math.isfinite(c_min) or not 0<c_min<1:
        raise ValueError('region confidence threshold must be in (0,1)')
    repair_speed_m_s=positive(repair_speed_m_s,'repair_speed_m_s',zero=True)
    preferred_scan_alpha(0.,c_min,progress_loss_gain=progress_loss_gain,
        alpha_min=alpha_min,alpha_min_at_quality=alpha_min_at_quality)
    if str(deficit_mode or 'absolute').lower()=='relative':
        preferred_relative_alpha(0.,relative_alpha_span,alpha_min=alpha_min)
    if max_request_rad_s is not None:
        max_request_rad_s=positive(max_request_rad_s,'max_request_rad_s')
    valid=np.zeros(n,dtype=bool) if region_valid is None else np.asarray(region_valid,dtype=bool)
    if valid.shape!=(n,):raise ValueError('one validity flag per configured region required')
    edges=vector(edges,(n+1,),name='region_edges')
    weights=vector(np.diff(edges),(n,),name='region_weights')
    centers=vector(geometry.image_x_sign*geometry.half_length_m*(edges[:-1]+edges[1:]-1.),(n,),name='region_centers_m')
    denominator=float(math.fsum(float(a*x*x) for a,x in zip(weights,centers)))
    q=None if quality is None else vector(np.clip(quality,0.,1.),(n,),name='region_quality')
    mode=str(deficit_mode or 'absolute').lower()
    if mode not in ('absolute','relative'):
        raise ValueError("deficit_mode must be 'absolute' or 'relative'")
    deadband=float(balance_deadband)
    if not math.isfinite(deadband) or deadband<0.:
        raise ValueError('balance_deadband must be finite and non-negative')
    rocking_ref=None
    if q is not None:
        if mode=='relative':
            if rocking_reference is not None:
                raise ValueError('relative deficit_mode does not take rocking_reference')
            rocking_ref=better_half_reference(q)
        elif rocking_reference is None:
            rocking_ref=c_min
        else:
            rocking_ref=float(rocking_reference)
            if not math.isfinite(rocking_ref) or not 0.<rocking_ref<=1.:
                raise ValueError('rocking_reference must be in (0, 1]')
    if q is None:
        deficits=None
    elif mode=='relative':
        excess=np.maximum(rocking_ref-q-deadband,0.)
        # Scale by this frame's good-side spread. Differences inside that
        # spread stay small; a hole deeper than the floor keeps full excess.
        # Cubic in the ratio is Welleweerd's small-error weight, setpoint
        # stays the better-half median (zero imbalance), not phantom c_min.
        if relative_spread_floor is not None:
            floor=float(relative_spread_floor)
            if not math.isfinite(floor) or floor<=0.:
                raise ValueError('relative_spread_floor must be finite and positive')
            scale=max(better_half_spread(q), floor)
            ratio=np.clip(excess/scale, 0., 1.)
            excess=excess*ratio*ratio
        deficits=vector(excess,(n,),name='region_deficits')
    else:
        deficits=vector(np.maximum(rocking_ref-q,0.),(n,),name='region_deficits')
    if q is None:
        alpha_deficits=None
    elif mode=='relative':
        alpha_deficits=deficits
    else:
        alpha_deficits=vector(np.maximum(c_min-q,0.),(n,),name='alpha_deficits')
    bad=np.zeros(n,dtype=bool) if q is None else valid & (deficits>0.)
    indices=tuple(int(i) for i in np.flatnonzero(bad))
    moment=area=None
    if deficits is not None:
        terms=weights*centers*deficits
        moment=float(math.fsum(map(float,terms)))
        # Remove only cancellation at floating-point precision, never a
        # quality/position deadband that could hide a detected weak region.
        if abs(moment)<=16*np.finfo(float).eps*float(np.sum(np.abs(terms))):moment=0.
        area=float(weights @ deficits)
    enabled=bool(evidence_valid and q is not None and valid.all())
    if not enabled:
        request=0.;alpha=float(alpha_min)
        reason='missing_regions' if q is None else 'unknown_regions' if not valid.all() else 'stale_or_invalid_prediction'
    else:
        if mode=='relative':
            alpha=preferred_relative_alpha(float(np.max(alpha_deficits)),relative_alpha_span,
                alpha_min=alpha_min)
        else:
            alpha=preferred_scan_alpha(float(np.max(alpha_deficits)),c_min,progress_loss_gain=progress_loss_gain,
                alpha_min=alpha_min,alpha_min_at_quality=alpha_min_at_quality)
        request=-repair_speed_m_s*moment/denominator if denominator>0. else 0.
        if max_request_rad_s is not None and abs(request)>max_request_rad_s:
            request=math.copysign(max_request_rad_s,request)
        reason=('all_regions_sufficient' if not bad.any() else
                'symmetric_deficit_no_rocking_direction' if request==0. else 'regional_deficit_moment')
    valid=valid.copy();valid.setflags(write=False)
    bad=bad.copy();bad.setflags(write=False)
    return RegionVisualRequest(float(request),alpha,enabled,reason,edges,centers,weights,q,valid,
        deficits,bad,indices,tuple(float(centers[i]) for i in indices),area,moment,denominator,
        rocking_ref,mode)
