"""Spatial deficit task over all configured equal-width confidence regions.

Each region remains identifiable. A single rocking axis can only respond to
the signed first spatial moment; symmetric defects are detected and slow path
progress, but cannot justify inventing a rocking direction or increasing force.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .types import ProbeGeometry, positive, vector


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

    def to_dict(self):
        return asdict(self)


def region_visual_request(quality, region_valid, region_edges, geometry: ProbeGeometry,
                          *, c_min, repair_speed_m_s, evidence_valid=True):
    """Least-squares rocking fit to per-region indentation deficits.

    v_z(x)=U-x*Omega and d_i=max(c_min-c_i,0), so a positive-side
    deficit requests negative Omega. Uniform or symmetric deficits have zero
    net first moment but still reduce alpha using the worst individual region.
    Unknown/stale evidence withdraws the entire visual row and sets alpha=.25.
    """
    edges=np.asarray(region_edges,dtype=float)
    if (edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all()
            or edges[0]!=0. or edges[-1]!=1. or np.any(np.diff(edges)<=0)):
        raise ValueError('region edges must partition the complete normalized image width')
    n=len(edges)-1
    if not math.isfinite(c_min) or not 0<c_min<1:
        raise ValueError('region confidence threshold must be in (0,1)')
    repair_speed_m_s=positive(repair_speed_m_s,'repair_speed_m_s',zero=True)
    valid=np.zeros(n,dtype=bool) if region_valid is None else np.asarray(region_valid,dtype=bool)
    if valid.shape!=(n,):raise ValueError('one validity flag per configured region required')
    edges=vector(edges,(n+1,),name='region_edges')
    weights=vector(np.diff(edges),(n,),name='region_weights')
    centers=vector(geometry.image_x_sign*geometry.half_length_m*(edges[:-1]+edges[1:]-1.),(n,),name='region_centers_m')
    denominator=float(math.fsum(float(a*x*x) for a,x in zip(weights,centers)))
    q=None if quality is None else vector(np.clip(quality,0.,1.),(n,),name='region_quality')
    deficits=None if q is None else vector(np.maximum(c_min-q,0.),(n,),name='region_deficits')
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
        request=0.;alpha=.25
        reason='missing_regions' if q is None else 'unknown_regions' if not valid.all() else 'stale_or_invalid_prediction'
    else:
        alpha=float(np.clip(1.-.75*float(np.max(deficits))/c_min,.25,1.))
        request=-repair_speed_m_s*moment/denominator if denominator>0. else 0.
        reason=('all_regions_sufficient' if not bad.any() else
                'symmetric_deficit_no_rocking_direction' if request==0. else 'regional_deficit_moment')
    valid=valid.copy();valid.setflags(write=False)
    bad=bad.copy();bad.setflags(write=False)
    return RegionVisualRequest(float(request),alpha,enabled,reason,edges,centers,weights,q,valid,
        deficits,bad,indices,tuple(float(centers[i]) for i in indices),area,moment,denominator)
