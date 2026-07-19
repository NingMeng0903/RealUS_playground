"""Optional region network — deferred.

You do **not** need to distill from query-side Region A.

Preferred supervision for a future region head:
  densely sample (μ, extents) offline, label each with discrete-map or point-field
  Monte Carlo mean/softmin/coverage over that anisotropic patch, then train
  ``g_φ(ΔT, extents) → RegionScore``.

Region A remains the transparent query aggregator over the generic point field.
A region head is only an optional accelerator with the **same** semantics.
"""

REGION_B_STATUS = "deferred_dense_region_gt_not_distill_from_a"
