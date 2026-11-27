# Endpoint contact-surface coordinates

This bundle covers the complete retained ipsilateral leg skin, including the
proximal/distal boundary regions omitted by shaft ROI v2. It is a **surface
atlas at d=0**, with addresses `(chart_id, u_m, v_m)`. It does not extend the
shaft's medial-depth volume or supply a new global `(theta,h,d)` cube.

```python
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas

surface = SurfaceMaterialAtlas.load('contact_multichart_right.npz')
address = surface.query(canonical_surface_points)[0]  # strict by default
point, face_id, bary = surface.decode(address['chart_id'], address['coordinate_uv_m'])
```

`query(..., project=True)` explicitly projects a noisy/template point and returns
its projection distance. It never silently treats a general 3D point as already
on the surface. `query(..., chart_id=id)` restricts lookup to an existing chart.
Canonical physical coordinates and decoded points retain float64.

Charts overlap on complete unchanged source triangles. For a supported overlap:

```python
new_uv, J = surface.transition(old_chart, old_uv, new_chart)
```

The same physical point and its source barycentric weights are preserved. At a
piecewise-linear edge/vertex, `transition_options(...)` exposes the separate
face Jacobian modes; these derivatives are not one globally smooth derivative.
A transition requires a shared source triangle. Do not interpolate numeric
chart IDs or treat these local meter coordinates as the old global theta/h.

`posed_vertices(joint_transforms)` applies the bundle's fixed nonnegative SMPL-X
weights. Passing the resulting vertices to `decode(..., vertices=posed)` or
`query(..., vertices=posed)` uses the same material chart, with exact shared
triangle transport across charts. This is a fixed template deformation model,
not patient registration or tissue mechanics.

`{side}_template_prior_chart_addresses.npz` records actual reprojections of the
old template skin-projected path. It preserves raw original vessel points,
source skin-projected points, old labels/xi, new chart IDs/coordinates, new
physical points and projection residuals. These template assets are legitimate
prior/coverage inputs under the current task authorization. They are not
classified as hidden patient truth merely because their filenames are old.
No template path or vessel labels are passed to the chart builder.

The old shaft ROI v2 is unchanged and remains the only certified deep volume
in this work. The normal-collar and global radial-volume experiments were
rejected. See REPORT.txt and manifest.json for exact scope, hashes and tests.
No DMP/QP/controller defaults were changed by this follow-up.
