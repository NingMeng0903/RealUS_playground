"""Conservative global injectivity certificate for a periodic radial spline tube.

The tested extension is F(h,u,v)=C(h)+rho*(S(h,theta)-C(h)), where
rho=hypot(u,v), theta labels the circle, and 0 <= rho <= 1.  The extension
is Lipschitz at its axis.  Away from the axis its Jacobian is affine in
rho.  One fixed left preconditioner and one fixed coordinate scaling are
used at BOTH radial endpoints.  A strictly positive lower bound for the
symmetric part implies strong monotonicity on the convex solid cylinder,
by integration along line segments.  The hollow material shell therefore
cannot fold or overlap globally.  The axis needs no differentiability:
a non-axial line meets it at at most one point, while an axial line has
the ordinary derivative C'(h).

Polynomial bounds use the Bernstein convex-hull property and every basic
interval operation is rounded outwards with nextafter.  Interval trig
uses mpmath's directed-rounding interval context.  Numerical eigenvalues
only select a candidate lower bound; interval Sylvester minors verify it.
The proof assumes the continuously glued periodic spline specified by the
caller.  Floating-point power-coefficient trace residuals are checked and
reported separately; this is not a formal exact-arithmetic spline solver.
"""

from __future__ import annotations

from math import comb
from time import perf_counter

import numpy as np
from mpmath.ctx_iv import MPIntervalContext


NEG_INF = -np.inf
POS_INF = np.inf


def _down(x):
    return np.nextafter(x, NEG_INF)


def _up(x):
    return np.nextafter(x, POS_INF)


def _point(x):
    a = np.asarray(x, dtype=float)
    return a, a


def _add(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul(a, b):
    products = np.stack(np.broadcast_arrays(a[0] * b[0], a[0] * b[1],
                                            a[1] * b[0], a[1] * b[1]))
    return _down(products.min(axis=0)), _up(products.max(axis=0))


def _square(a):
    lo, hi = a
    low = np.where((lo <= 0.) & (hi >= 0.), 0., np.minimum(lo * lo, hi * hi))
    return np.maximum(0., _down(low)), _up(np.maximum(lo * lo, hi * hi))


def _reciprocal_positive(a):
    if np.any(a[0] <= 0.):
        raise ValueError("nonpositive interval denominator")
    return _down(1. / a[1]), _up(1. / a[0])


def _derivative(poly, axis, inverse_width):
    low, high = (np.zeros_like(poly[0]), np.zeros_like(poly[1]))
    for i in range(1, poly[0].shape[axis]):
        source = [slice(None)] * low.ndim
        target = source.copy()
        source[axis], target[axis] = i, i - 1
        value = _mul((poly[0][tuple(source)], poly[1][tuple(source)]), _point(float(i)))
        low[tuple(target)], high[tuple(target)] = value
    return _mul((low, high), inverse_width)


def _linear(poly, matrix):
    """Apply an exact stored floating-point matrix to interval vector coefficients."""
    rows = []
    for row in matrix:
        value = _point(np.zeros(poly[0].shape[:-1]))
        for k, scalar in enumerate(row):
            value = _add(value, _mul((poly[0][..., k], poly[1][..., k]), _point(scalar)))
        rows.append(value)
    return tuple(np.stack([row[j] for row in rows], axis=-1) for j in (0, 1))


def _bernstein_axis(poly, axis):
    n = poly[0].shape[axis] - 1
    low, high = np.empty_like(poly[0]), np.empty_like(poly[1])
    for i in range(n + 1):
        target = [slice(None)] * low.ndim
        target[axis] = i
        value = _point(np.zeros_like(low[tuple(target)]))
        for k in range(i + 1):
            source = target.copy()
            source[axis] = k
            numerator, denominator = comb(i, k), comb(n, k)
            coefficient = numerator / denominator
            factor = (_down(coefficient), _up(coefficient))
            value = _add(value, _mul((poly[0][tuple(source)], poly[1][tuple(source)]), factor))
        low[tuple(target)], high[tuple(target)] = value
    return low, high


def _bernstein(poly):
    return _bernstein_axis(_bernstein_axis(poly, 2), 3)


def _split(poly, axis):
    """Outward-rounded de Casteljau subdivision at one half."""
    level = tuple(np.moveaxis(v, axis, 0) for v in poly)
    left, right = [tuple(v[0] for v in level)], [tuple(v[-1] for v in level)]
    for _ in range(level[0].shape[0] - 1):
        level = _mul(_add(tuple(v[:-1] for v in level), tuple(v[1:] for v in level)), _point(.5))
        left.append(tuple(v[0] for v in level))
        right.append(tuple(v[-1] for v in level))
    def collect(items):
        return tuple(np.moveaxis(np.stack([v[j] for v in items]), 0, axis) for j in (0, 1))
    return collect(left), collect(right[::-1])


def _sylvester(symmetric, margin):
    low, high = symmetric[0].copy(), symmetric[1].copy()
    for k in range(3):
        low[..., k, k], high[..., k, k] = _sub(
            (low[..., k, k], high[..., k, k]), _point(margin))
    def entry(i, j):
        return low[..., i, j], high[..., i, j]
    a, b, c = entry(0, 0), entry(0, 1), entry(0, 2)
    d, e, f = entry(1, 1), entry(1, 2), entry(2, 2)
    minor2 = _sub(_mul(a, d), _square(b))
    minor3 = _add(_sub(_mul(a, _sub(_mul(d, f), _square(e))),
                      _mul(b, _sub(_mul(b, f), _mul(c, e)))),
                  _mul(c, _sub(_mul(b, e), _mul(c, d))))
    return (a[0] > 0.) & (minor2[0] > 0.) & (minor3[0] > 0.)


def _matrix_bounds(polys, cosine, sine, minimum_margin):
    # polys: box, kind=(h_at_r0,h_at_r1,D,D_phi), h_degree, theta_degree, xyz.
    vector = (polys[0].min(axis=(2, 3)), polys[1].max(axis=(2, 3)))
    def field(i):
        return vector[0][:, i], vector[1][:, i]
    cosine = tuple(v[:, None] for v in cosine)
    sine = tuple(v[:, None] for v in sine)
    fu = _sub(_mul(cosine, field(2)), _mul(sine, field(3)))
    fv = _add(_mul(sine, field(2)), _mul(cosine, field(3)))
    endpoint_bounds = []
    for endpoint in (0, 1):
        matrix = tuple(np.stack((field(endpoint)[j], fu[j], fv[j]), axis=-1) for j in (0, 1))
        symmetric = _mul(_add(matrix, tuple(np.swapaxes(v, -1, -2) for v in matrix)), _point(.5))
        midpoint = .5 * (symmetric[0] + symmetric[1])
        radius = .5 * (symmetric[1] - symmetric[0])
        approximate = np.linalg.eigvalsh(midpoint)[:, 0] - radius.sum(axis=-1).max(axis=-1)
        proposed = np.maximum(minimum_margin, .5 * approximate)
        accepted = np.zeros(len(proposed), dtype=bool)
        result = np.zeros_like(proposed)
        for _ in range(10):
            passed = _sylvester(symmetric, proposed) & ~accepted
            result[passed] = proposed[passed]
            accepted |= passed
            proposed = np.maximum(minimum_margin, .5 * proposed)
            if accepted.all():
                break
        endpoint_bounds.append(result)
    return np.minimum(*endpoint_bounds)


def _gluing(surface, core, h, theta):
    """Check the mathematical C1 construction at floating-point precision."""
    dh, dt = np.diff(h), np.diff(theta)
    power = np.arange(4.)
    errors = {}
    errors["surface_h_value_m"] = float(np.max(np.abs(surface[:-1].sum(axis=2) - surface[1:, :, 0])))
    end_h = (surface * power[None, None, :, None, None]).sum(axis=2) / dh[:, None, None, None]
    start_h = surface[:, :, 1] / dh[:, None, None, None]
    errors["surface_h_derivative"] = float(np.max(np.abs(end_h[:-1] - start_h[1:])))
    end_t = surface.sum(axis=3)
    start_t = np.roll(surface[:, :, :, 0], -1, axis=1)
    errors["surface_theta_value_m"] = float(np.max(np.abs(end_t - start_t)))
    end_t = (surface * power[None, None, None, :, None]).sum(axis=3) / dt[None, :, None, None]
    start_t = np.roll(surface[:, :, :, 1] / dt[None, :, None, None], -1, axis=1)
    errors["surface_theta_derivative"] = float(np.max(np.abs(end_t - start_t)))
    errors["core_value_m"] = float(np.max(np.abs(core[:-1].sum(axis=1) - core[1:, 0])))
    end_c = (core * power[None, :, None]).sum(axis=1) / dh[:, None]
    start_c = core[:, 1] / dh[:, None]
    errors["core_derivative"] = float(np.max(np.abs(end_c[:-1] - start_c[1:])))
    passed = all(value <= (1.e-10 if "value" in key else 1.e-8) for key, value in errors.items())
    return {"pass": bool(passed), "value_tolerance_m": 1.e-10,
            "derivative_tolerance": 1.e-8, **errors}


def _sample_preconditioner(surface, core, h, theta):
    weights = .5 ** np.arange(4)
    derivative = np.array([0., 1., 1., .75])
    s = np.einsum("i,j,abijc->abc", weights, weights, surface)
    sh = np.einsum("i,j,abijc->abc", derivative, weights, surface) / np.diff(h)[:, None, None]
    st = np.einsum("i,j,abijc->abc", weights, derivative, surface) / np.diff(theta)[None, :, None]
    c = np.einsum("i,aic->ac", weights, core)[:, None]
    ch = np.einsum("i,aic->ac", derivative, core)[:, None] / np.diff(h)[:, None, None]
    phase = (np.mean(np.stack((theta[:-1], theta[1:])), axis=0) - theta[0]) * (2. * np.pi / (theta[-1] - theta[0]))
    st *= (theta[-1] - theta[0]) / (2. * np.pi)
    cosine, sine = np.cos(phase)[None, :, None], np.sin(phase)[None, :, None]
    fu, fv = cosine * (s - c) - sine * st, sine * (s - c) + cosine * st
    j0 = np.stack((np.broadcast_to(ch, sh.shape), fu, fv), axis=-1)
    j1 = np.stack((sh, fu, fv), axis=-1)
    jacobians = np.concatenate((j0.reshape(-1, 3, 3), j1.reshape(-1, 3, 3)))
    preconditioner = np.linalg.inv(jacobians.mean(axis=0))
    normalized = np.einsum("ij,njk->nik", preconditioner, jacobians)
    candidates = []
    for scale in (1., 2., 3., 4., 5., 7., 9., 12., 16., 24.):
        z = np.array((scale, 1., 1.))
        transformed = z[None, :, None] * normalized / z[None, None, :]
        lower = float(np.linalg.eigvalsh(.5 * (transformed + transformed.swapaxes(-1, -2)))[:, 0].min())
        candidates.append((lower, scale))
    lower, scale = max(candidates)
    return preconditioner, scale, lower, float(np.linalg.det(jacobians).min())


def certify_tube(surface_coeff, core_coeff, h, theta, *, max_depth=8,
                 minimum_margin=1.e-8, max_boxes=250000):
    """Certify a spline tube, or return a fail-closed JSON-serializable report.

    Coefficients are ascending normalized POWER coefficients.  ``surface``
    has shape (H-1,T-1,4,4,3), ``core`` shape (H-1,4,3); h/theta contain
    cell edges, including the repeated periodic theta endpoint.  Every
    polynomial cell must belong to one continuously glued periodic C1
    spline.  A false result means *not certified*, rather than necessarily
    folded.  The certificate covers the whole solid radial extension and
    therefore any shell with radial fractions in (0,1].
    """
    started = perf_counter()
    surface, core = np.asarray(surface_coeff, float), np.asarray(core_coeff, float)
    h, theta = np.asarray(h, float), np.asarray(theta, float)
    if (h.ndim != 1 or theta.ndim != 1 or len(h) < 3 or len(theta) < 3
            or surface.shape != (len(h)-1, len(theta)-1, 4, 4, 3)
            or core.shape != (len(h)-1, 4, 3)
            or not all(np.isfinite(v).all() for v in (surface, core, h, theta))
            or np.any(np.diff(h) <= 0.) or np.any(np.diff(theta) <= 0.)
            or max_depth < 0 or minimum_margin <= 0. or max_boxes < 1):
        raise ValueError("invalid periodic cubic tube inputs or certificate limits")
    report = {
        "method": "outward_bernstein_interval_sylvester_strong_monotonicity",
        "certificate_pass": False, "global_injective": False,
        "scope": "continuous periodic spline on convex solid cylinder; shell inherits injectivity",
        "roundoff_scope": "directed interval polynomial/trig bounds; C1 spline gluing checked to recorded tolerance",
        "coefficient_gluing": _gluing(surface, core, h, theta),
        "radial_endpoints": [0., 1.], "same_preconditioner_at_both_endpoints": True,
        "minimum_required_margin": float(minimum_margin), "max_subdivision_depth": int(max_depth),
    }
    if not report["coefficient_gluing"]["pass"]:
        report.update(reason="spline coefficient gluing failed", elapsed_s=perf_counter()-started)
        return report
    try:
        p, scale, sampled_min, sampled_det = _sample_preconditioner(surface, core, h, theta)
    except np.linalg.LinAlgError:
        report.update(reason="singular sampled global preconditioner", elapsed_s=perf_counter()-started)
        return report
    report.update(preconditioner=p.tolist(), coordinate_scale=[scale, 1., 1.],
                  sampled_min_symmetric_eigenvalue=sampled_min,
                  sampled_min_cartesian_jacobian_det=sampled_det)
    if sampled_min <= minimum_margin:
        report.update(reason="fixed global monotonicity preconditioner has no sampled margin",
                      elapsed_s=perf_counter()-started)
        return report

    ctx = MPIntervalContext()
    ctx.dps = 40
    period = ctx.mpf(float(theta[-1])) - ctx.mpf(float(theta[0]))
    angular_factor = period / (2 * ctx.pi)
    factor = (_down(float(angular_factor.a)), _up(float(angular_factor.b)))
    dh = _sub(_point(h[1:]), _point(h[:-1]))
    dt = _sub(_point(theta[1:]), _point(theta[:-1]))
    ih = tuple(v[:, None, None, None, None] for v in _reciprocal_positive(dh))
    it = tuple(v[None, :, None, None, None] for v in _reciprocal_positive(dt))
    core_grid = np.zeros_like(surface)
    core_grid[:, :, :, 0] = core[:, None]
    sp, cp = _point(surface), _point(core_grid)
    delta = _sub(sp, cp)
    sh, ch = _derivative(sp, 2, ih), _derivative(cp, 2, ih)
    st = _mul(_derivative(sp, 3, it), factor)
    left = np.array((scale, 1., 1.))[:, None] * p
    # Treat the stored left matrix as exact.  Division of its first-column
    # fields by scale corresponds to the same fixed coordinate transform.
    h0 = _mul(_linear(ch, left), _reciprocal_positive(_point(scale)))
    h1 = _mul(_linear(sh, left), _reciprocal_positive(_point(scale)))
    fields = [_bernstein(v) for v in (h0, h1, _linear(delta, left), _linear(st, left))]
    polys = tuple(np.stack([v[j] for v in fields], axis=2).reshape(-1, 4, 4, 4, 3) for j in (0, 1))
    nt = len(theta) - 1
    # Track exact dyadic subdivisions within each original polynomial cell.
    # Floating midpoint labels could otherwise undershoot a child's true
    # angular range by an ulp when paired with exact de Casteljau halves.
    # Fields: original h/theta cell, h numerator/depth, theta numerator/depth,
    # total recursive depth.
    boxes = [(i, j, 0, 0, 0, 0, 0) for i in range(len(h)-1) for j in range(nt)]
    trig_cache = {}
    def trig(boxes_now):
        cosine, sine = [], []
        for _, j, _, _, numerator, level, _ in boxes_now:
            key = j, numerator, level
            if key not in trig_cache:
                fraction = ctx.mpf([numerator / 2**level, (numerator+1) / 2**level])
                angle = ctx.mpf(float(theta[j])) + (ctx.mpf(float(theta[j+1]))
                                                   - ctx.mpf(float(theta[j]))) * fraction
                phase = (angle - ctx.mpf(float(theta[0]))) * (2 * ctx.pi) / period
                cv, sv = ctx.cos(phase), ctx.sin(phase)
                trig_cache[key] = ((_down(float(cv.a)), _up(float(cv.b))),
                                   (_down(float(sv.a)), _up(float(sv.b))))
            cv, sv = trig_cache[key]
            cosine.append(cv)
            sine.append(sv)
        return tuple(np.array(cosine).T), tuple(np.array(sine).T)

    evaluated, certified, deepest = 0, 0, 0
    lower_bound = np.inf
    unresolved = []
    while boxes:
        evaluated += len(boxes)
        cosine, sine = trig(boxes)
        bounds = _matrix_bounds(polys, cosine, sine, minimum_margin)
        passed = bounds >= minimum_margin
        certified += int(passed.sum())
        if passed.any():
            lower_bound = min(lower_bound, float(bounds[passed].min()))
        next_boxes, next_polys = [], []
        for index in np.flatnonzero(~passed):
            i, j, hn, hd, tn, td, depth = boxes[index]
            if depth >= max_depth or evaluated + len(next_boxes) + 2 > max_boxes:
                unresolved.append({"h_cell": i, "theta_cell": j,
                                   "h_dyadic_interval": [hn, hn+1, 2**hd],
                                   "theta_dyadic_interval": [tn, tn+1, 2**td], "depth": depth})
                continue
            # Alternation controls both longitudinal and angular variation.
            split_theta = depth % 2 == 0
            children = _split(tuple(v[index] for v in polys), 2 if split_theta else 1)
            if split_theta:
                child_boxes = [(i,j,hn,hd,2*tn,td+1,depth+1),
                               (i,j,hn,hd,2*tn+1,td+1,depth+1)]
            else:
                child_boxes = [(i,j,2*hn,hd+1,tn,td,depth+1),
                               (i,j,2*hn+1,hd+1,tn,td,depth+1)]
            next_boxes.extend(child_boxes)
            next_polys.extend(children)
            deepest = max(deepest, depth+1)
        boxes = next_boxes
        if boxes:
            polys = tuple(np.stack([v[j] for v in next_polys]) for j in (0, 1))
    passed = not unresolved
    report.update(certificate_pass=passed, global_injective=passed,
                  certified_min_symmetric_eigenvalue=float(lower_bound) if np.isfinite(lower_bound) else None,
                  initial_polynomial_cells=(len(h)-1)*nt, interval_boxes_evaluated=evaluated,
                  certified_leaf_boxes=certified, deepest_subdivision=deepest,
                  unresolved_boxes=len(unresolved), unresolved_examples=unresolved[:8],
                  elapsed_s=perf_counter()-started,
                  reason="all continuous cells certified" if passed else "unresolved interval boxes; geometry not admitted")
    return report


__all__ = ["certify_tube"]
