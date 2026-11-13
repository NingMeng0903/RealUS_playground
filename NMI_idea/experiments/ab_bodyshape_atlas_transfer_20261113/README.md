# Leg material addresses across SMPL-X shapes

This study checks material address reuse and physical orientation under SMPL-X shape changes. It also replays the existing knee estimator to check the proposed split of the total dynamic displacement into a nominal input response and an online residual.

## Coordinate definitions

The earlier production chart solves surface h with a cotangent Dirichlet problem. It seeds surface theta from a geometric leg axis and extends h and theta with a tetrahedral FEM solve. The historical depth implementation computes `dist_skin / (dist_skin + dist_core)`. The runtime frame uses the bone-axis tangent and direction to the medial curve. The slide's h-gradient construction defines a separate frame. [Legacy chart](../../../src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/harmonic.py#L891-L1008) · [Runtime frame](../../../src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/pose_bundle.py#L120-L157)

The repaired `shaft_roi_v2` bundle covers h in [0.14, 0.94]. It builds its template cells by cutting the skin along a smoothed centerline with parallel-transport frames. The longitudinal station h and periodic ring label theta are geometric labels. Depth d solves a positive-edge-weight graph Dirichlet problem between the outer and inner shells. Earlier audits cover a fixed body at seven knee angles. The old production chart and the repaired chart use distinct coordinate fields and domains. [Fixed-body audit](../ab_validation_20261110/repair_20261110/audit/FINAL_HARMONIC_AUDIT.txt)

For a new shape we evaluate the original male SMPL-X shape bases, shape-dependent rest joints, and native knee pose correctives. The outer chart nodes retain their source-face indices and barycentric weights after applying the original production stencil to the new surface. Interior nodes use the fixed radial fractions and a core extension from ring-average rest displacement and fixed skinning weights. The tetrahedra and material labels stay attached to those nodes. This defines a model-based volume deformation and a coarse anatomical search region. Validation of actual vascular correspondence requires independent internal targets. [Transport code](transport.py) · [SMPL-X model](https://download.is.tue.mpg.de/smplx/SMPL-X.pdf)

The geometry check covers all 322,056 tetrahedra per leg. It requires positive determinant ratios above 0.001, zero positive-volume overlap in the BVH/SAT check, and zero intersections in the CGAL checks of the volume boundary and retained production skin. The query audit on accepted geometry uses 256 points per configuration, including internal cells, periodic seam cells, and both shells. It tests the existing coordinate API without supplying reference points to the inverse. The rotation comparison holds the knee angle fixed and uses the same cells between each pair of shapes.

| Shape grid | Configurations passing all checks | Largest h/d frame turn | Largest centerline frame turn | Largest change in remeasured azimuth |
| --- | ---: | ---: | ---: | ---: |
| Nine offsets in {-2, 0, 2}² around the recovered beta | 45/54 | 12.17° | 15.70° | 22.60° |
| Eight nonzero offsets in {-1, 0, 1}², exploratory follow-up | 46/48 | 5.49° | 7.12° | 10.16° |

Each grid uses both legs and knee angles 0°, 45°, and 90°. Shape offsets refer to the first two actual shape coefficients. Among the admitted configurations, the largest roundtrip error is 3.25e-8 m. Eight stress cases and two follow-up cases are rejected for intersections in the retained production skin. The right-leg stress case (2, 2) at 90° has 37 nonpositive tetrahedra and 658 positive-volume overlap pairs. Rejections are recorded even when other geometry checks pass. These are finite sampled configurations. The follow-up grid was chosen after seeing stress failures. [Stress grid](verification/GEOMETRY.json) · [Follow-up grid](verification/GEOMETRY_MODERATE.json)

The h/d frame uses the slide's convention with the depth direction orthogonalized against the h gradient. It is evaluated on the transported v2 fields. Separately, the geometric diagnostic reconstructs parallel-transport frames along a spline through the moved ring centers. The same outer material point can have a different geometric azimuth when remeasured in that frame. The label is kept by its fixed attachment. With local deformation gradient A, the direct coordinate gradients agree with inverse-transpose transport to floating-point precision. The exact transport of a material tangent is A times the tangent. A material-plane normal is transported by the inverse transpose and normalized. Probe contact normals can be computed from the current skin. [Rotation and azimuth checks](verification/FRAME_ANGLES.json)

Keeping the stored d preserves the material address. In the admitted straight-leg stress cases, recomputing d with the same graph rule on the changed geometry shifts d by up to 0.01571 at a fixed point. Along a fixed material ray the point labeled d = 0.5 shifts by up to 1.122 mm. The maxima come from different configurations. Rigid and uniform-scale controls change d by less than 3.0e-8. The original material field is the repeatable reference. [Depth comparison](verification/GEOMETRY.json)

## Dynamic response and online residual

Let the old total dynamic displacement obey `x_next = a*x + b*u + noise`. Propagating a deterministic `r_next = a*r + b*u` gives `z_next = a*z + noise` when `z = x-r`. The same likelihood uses `r+z`. The residual grid and its bin boundaries move with r. The case state and all uncertainty are retained under this reparameterization. A mismatch in the actual time constant or input gain can leave the residual correlated with the measured input.

The 256 archived traces reproduce the saved measurements and fixed-parameter predictions. A separate residual update on the shifted grid reproduces the old posterior within 2.8e-16 and the physical prediction mean within 5.7e-13 mm. A five-point online posterior over a case-specific input gain changes the 0.5 s and 1.0 s RMSE by +0.00192 and +0.00204 mm, with paired intervals crossing zero. The nominal 90% per-XYZ-component mean ± 1.645 standard-deviation interval covers about one point more. This replay supports using the split as a model interpretation and keeping the tested gain adaptation as an ablation. The scans and their motion choices are held fixed. [Paired replay and source hashes](verification/RESPONSE.json)

## Reproduce

Use Python 3.10 with NumPy, SciPy, Trimesh, rtree, and libigl with CGAL. Geometry needs the local Among_US model and stencil assets, the frozen repair bundle, and the existing `harmonic_independent/tet_overlap` checker. The replay uses the existing knee archive with its 256 saved synthetic rule traces. All new results include input hashes. The repository contains the analysis code and result summaries.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python run_geometry.py --among-us "$among" --repair "$repair" --grid stress --out verification/GEOMETRY.json
python run_geometry.py --among-us "$among" --repair "$repair" --grid moderate --out verification/GEOMETRY_MODERATE.json
python frame_angles.py --among-us "$among" --repair "$repair" --geometry verification/GEOMETRY.json verification/GEOMETRY_MODERATE.json --out verification/FRAME_ANGLES.json
python response_replay.py --archive "$knee" --out verification/RESPONSE.json
```
