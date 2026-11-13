# Cross-shape material atlas in streaming MPC and ProDMP

The scan entry now accepts ten native male SMPL-X shape coefficients and a fixed knee angle. It binds the right-leg template once, checks the instantiated body geometry, and uses that geometry in motion generation and the observation loop. The benchmark uses one registered artery and one registered vein, represented by 65 centreline samples per vessel. [Entry point](run.py) · [Atlas](atlas_geometry.py) · [Scan](atlas_scan.py)

## Material addresses and attached probe orientation

The reference template defines the periodic angular label `theta`, longitudinal label `h`, and the graph-harmonic depth `d` with Dirichlet values zero on the skin and one on the inner shell. The smooth reference centreline and its parallel frame establish the reference ring labels. Material identity is part of the chart definition. For the skin it uses the retained SMPL-X vertex stencil, source triangle indices and barycentric weights. For the volume it uses a fixed tetrahedral mesh. A material point has the same cell and weights at any admitted shape.

For material address $a=(K,b)$ with cell $K$ and weights $b$, let $X_i(\beta,q)$ be the new physical positions of the cell vertices. The map and fixed-label condition are

$$
\Phi_{\beta,q}(a)=\sum_{i\in K} b_i X_i(\beta,q),
\qquad
\xi_{\beta,q}(\Phi_{\beta,q}(a))=\xi_{\mathrm{ref}}(a).
$$

The equation holds up to the precision of the discrete coordinate queries. The depth field is harmonic for the reference graph and is carried with its fixed labels. The reshaped graph generally has a different Euclidean Laplacian. The native skin and fixed interior extension define an anatomical prior. Physical correspondence of real vessel anatomy across people needs independent measurement. [Underlying geometry study](../ab_bodyshape_atlas_transfer_20261113/README.md)

The earlier pose bundle stores the orientation of the probe relative to a local frame at a material address. The same rule is used here. If $G_0(a)$ is the reference frame and $R_0(a)$ the registered probe orientation, the attached rotation is $\rho(a)=G_0(a)^\top R_0(a)$. The scan uses

$$
R_{\beta,q}(a)=G_{\beta,q}(a)\rho(a).
$$

The material surface tangent along $h$ and the inward normal determine the current orthogonal frame. The tangent uses differences at corresponding material grid vertices. The normal interpolates area-weighted vertex normals. The registered reference incidence is expressed in that same frame convention. This preserves the locally expressed probe rotation, including any reference tilt. The historical bone-axis/core frame uses a different convention. Changing frame from $F$ to $G$ for a given physical orientation changes its components to $G^\top F\rho_F$. [Earlier pose bundle](../../../src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/pose_bundle.py#L170-L212) · [Current reconstruction](atlas_geometry.py)

The scan imports its legacy registered origins at 15°. Their one-time projection onto the reference outer chart moves them at most 1.234 mm. The imported vessel samples bind to reference tetrahedra with a reconstruction residual below $3\times10^{-16}$ m. Every tested geometry reuses the same material attachment hash. The bidirectional coordinate API uses the transported volume and the baked coordinate fields, including the periodic angle. [Geometry metadata](verification/ATLAS/reference/SUMMARY.json) · [Replay checks](verification/ATLAS/VERIFIED.json)

## Closed-loop protocol

The experiment fixes longitudinal progression through the middle 80% of the registered segment. A local two-step MPC evaluates nominal one-second trajectories with the shared, servo-trained lateral ProDMP. It uses the predicted probability of a centred arterial return at three exposures per segment and a penalty on physical TCP travel. It re-evaluates at each segment boundary. If none of the nominal lateral targets is reachable under the sampled limits, it also tries carrying the current lateral offset forward or holding the current lateral coordinate. This fallback was introduced after a development case reached the next segment with an offset outside the nominal candidate window. The archived topology MPC is a different planner.

ProDMP references are mapped to the instantiated skin geometry before sampled fixed-rail IK. The simulator generates an anonymous centroid packet from the current FK plane every 100 ms. Its model intersects the two centreline polylines with that plane. The perturbation consists of a case-level offset, a small difference between measured and physical offsets, persistent vessel dropouts and frame noise. The planner and visual feedback use public packets and the association posterior. The evaluator uses physical identities and positions. Image feedback can change the current segment's suffix while preserving its scalar lateral position, velocity and deadline. A suffix rejected by the shared sampled joint, FK or robot self-collision checks leaves the existing valid suffix in use. A recorded revision changes the next exposure's physical probe position by at least 1 micrometre.

The three 15° configurations use identical draws of offset coefficients, bias coefficients, dropout states and noise. The two directions in which the offset coefficients are applied come from each instantiated body frame. Success requires five coverage bins with new accepted arterial centroids within 3 mm of the physical image centre, correct identity and completion of the scheduled scan. One accepted wrong identity fails the case. Acquisitions continue through temporary loss of evidence. The verification reruns FK, the sensor and association for each stored exposure, and derives task outcomes and executed costs from those records. [Scan and outcome calculation](atlas_scan.py) · [Replayer](verify_atlas_scan.py)

## Results

| Shape offset from reference | Knee | Correct completion | Wrong identity | Mean TCP travel | Mean turn | Mean effective within-segment revisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 15° | 12/12 | 0 | 22.99 mm | 5.27° | 22.08 |
| Δβ₀ = −1, Δβ₁ = 1 | 15° | 12/12 | 0 | 24.14 mm | 5.28° | 21.67 |
| Δβ₀ = 1, Δβ₁ = −1 | 15° | 11/12 | 0 | 22.03 mm | 5.50° | 21.83 |
| Δβ₂ = 0.5 through the ten-coefficient input | 15° | 2/2 | 0 | 23.30 mm | 5.36° | 26.00 |
| Reference | 25° | 2/2 | 0 | 22.60 mm | 5.83° | 25.50 |

All 40 executions reached 4 s. The remaining failure missed one coverage bin. Holding the material address fixed at the nine sampled surface stations, the two main shape changes moved the probe origin by up to 31.07 mm and rotated it by up to 2.24°. Across five configurations, the address-to-position roundtrip error was at most 25 nm. The largest elementwise difference between the relative rotation matrices before and after transport was below `3.4e-15` at the checked addresses. The 25° run adds a static pose check. These results establish execution through the transported chart for these configurations. The sensor uses synthetic centroids and the plant uses sampled ideal kinematics. Validation of real image quality, force, body contact, real-time compute and arbitrary body shapes remains open. [Per-case records and trace replay](verification/ATLAS)

## Reproduce

The benchmark uses the local Among_US male SMPL-X model and precision-coordinate source, plus the frozen repair assets and overlap checker from the original geometry audit. The archive stores derived observations and kinematic trajectories. Its input hashes identify the assets needed for geometry regeneration. Run from the repository root with the RM75 capability runtime, SciPy, Trimesh with rtree, and libigl with CGAL.

```sh
exp=NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python "$exp/run.py" atlas --among-us "$among" --repair "$repair" --cases 12 --out /tmp/atlas/reference
python "$exp/run.py" atlas --among-us "$among" --repair "$repair" --delta-beta -1 1 --cases 12 --out /tmp/atlas/minus-plus
python "$exp/run.py" atlas --among-us "$among" --repair "$repair" --delta-beta 1 -1 --cases 12 --out /tmp/atlas/plus-minus
python "$exp/verify_atlas_scan.py" --among-us "$among" --repair "$repair" --runs /tmp/atlas/reference /tmp/atlas/minus-plus /tmp/atlas/plus-minus --out /tmp/atlas/VERIFIED.json
```

For a full shape or a different static pose, pass `--beta` with all ten coefficients or `--knee-deg`. Each run's JSON contains the full inputs. `run.py --out …` reproduces the older frozen-section study. Its regression replay preserves all discrete decisions and the 174, 213 and 214 completion counts over 256 cases per arm. At the extreme shape offset `(2, 2)` and 90° knee, the scan rejects the geometry before constructing paths. The checker finds 37 nonpositive tetrahedra and 658 positive-volume overlaps. [Rejection record](verification/ATLAS/REJECTION.json)
