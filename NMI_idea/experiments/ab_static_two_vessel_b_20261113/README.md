# Static artery–vein reinspection

On one registered static artery–vein pair, anticipatory B improved modeled correct completion over the two-diagnostic forecast of live MPC by 6.13 percentage points. A fixed central first acquisition followed by the same live replanner achieved essentially the same completion as B. A simple option to reacquire the near frame after a miss explains the practical benefit in this model.

[Frozen protocol](PROTOCOL.md) · [Theory and estimand](THEORY.md) · [Exact and paired results](verification/SUMMARY.json) · [Independent implementation audit](verification/INDEPENDENT_AUDIT.json) · [Independent theory verdict](verification/THEORY_AUDIT.md)

## Geometry and task

The Among_US retarget pipeline exports the `Artery` and `Vein` meshes in the same canonical SMPL-X coordinates. Applying the same archived 15° joint transforms places them in the existing probe geometry. The three selected stations lie on the right popliteal-to-proximal lower-leg scan. Each of nine pose planes and 25 static registration shifts yields one local closed contour from each class. Nominal central-plane artery–vein centroid separations are 14.15, 17.85 and 20.87 mm at stations 3, 4 and 5. The archived checked tip positions agree with the registered plane origins within 0.00122 mm. At a given station the three view orientations use the same plane and rotate the image axes and aperture. [Registered cross-sections](sections.png) · [Geometry and checked edge costs](GEOMETRY.json) · [Retarget provenance](ASSET_PROVENANCE.json)

The common MPC prescan actually acquires two anonymous centroid packets at stations 4 and 3 and updates the joint static registration belief. All suffix arms start from that same state. The positive endpoint requires new images at stations 4 and 5, whose selected candidates have joint modeled arterial identity probability of at least 0.87. Both selected candidates must belong to the artery in the evaluator. A third new image is available as an extra observation or near retry. The same 63 checked directed/hold motion edges and four physical resource caps apply to all arms. The observation model uses only aperture and role-independent synthetic detection/noise. This is a finite simulation with a single anatomical template. Its identifiability does not require pulsatility, flow or raw image segmentation. It does not test actual U-Net output or tissue contact.

## Results

The exact column integrates the 25 registration hypotheses and all material prescan outcomes. Prescan outcomes with individual mass below `1e-12` have aggregate probability `2.61e-11`; this bounds their omitted effect on completion. The 512 paired replays sample shifts and noise from this same geometry. They are execution checks, not independent anatomies.

| Policy | Correct completion, exact | Wrong declaration, exact | Correct, paired replay | Mean TCP, exact | Mean rotation, exact | Mean time, exact |
|---|---:|---:|---:|---:|---:|---:|
| Live two-diagnostic MPC | 75.7160% | 0.00058% | 377/512 | 61.07 mm | 51.82° | 6.777 s |
| Contingent B | 81.8484% | 0.00051% | 413/512 | 57.49 mm | 45.37° | 6.202 s |
| Fixed central first near, then live MPC | 81.8459% | 0.00117% | 413/512 | 54.57 mm | 47.00° | 6.202 s |
| Two new images, no third acquisition | 70.5594% | 0.00095% | 354/512 | 55.13 mm | 44.08° | 5.763 s |

B wins 57 paired replays and loses 21 against the live MPC, with zero incorrect declarations in these two arms and the fixed-center arm. The B-versus-MPC resource differences favor B on the modeled path and time metrics, with equal rail motion. They are outside the strict two-sided equivalence bands. Against fixed center, B gains only 0.00253 percentage points and uses 2.92 mm more TCP travel on average. B and fixed center tie on completion and wrong declaration for every paired replay, with different paths on 325 of 512. The fixed control was selected as the best of the three declared fixed near views under the finite prior.

Let `p=0.84` denote intrinsic detection probability. Even with free true identity and perfect aperture, at least one of two near observations and the single far observation must actually contain the artery. Under independent intrinsic misses, the upper bound is

```text
p * [1 - (1 - p)^2] = 0.818496.
```

The best fixed first view comes within 0.00371 percentage points of this optimistic bound. The bound assumes at most two near attempts and exactly one far attempt. A fixed first near view can collect the qualifying image immediately, retry only as needed, and use the common far choice and posterior declaration. The value of B's extra root calculation is therefore limited here.

The replay exposes the action change. MPC spent its first fresh acquisition back at station 3 on 243 of 512 tapes. B and fixed center both first moved to station 4 on all 512 tapes, used an optional station-4 retry on 93, and reached station 5 on 488. Their capture counts agree on every paired tape, although their chosen probe views can differ.

MPC's fixed-continuation forecast is 70.5596%. Its **executed** value is 75.7160% after real updating and replanning. The reported 6.13-point contrast uses the latter. The control's forecast distinguishes a plan that proceeds from the current near packet from one that acquires a replacement near packet for all first outcomes. A broader forecast that explicitly values the conditional retry, or a different task that permits earlier near frames to count after intervening acquisitions, changes this comparison. This screen isolates one restricted lookahead difference. It provides a positive mechanism test for spending an extra acquisition conditionally, with negligible independent benefit for adaptive root selection over the fixed-first control.

## Reproduce

The small public geometry file contains derived plane centroids, radii and costs. The local source mesh and canonical asset are used only by the optional extraction path. The replay uses Python 3.10, NumPy 2.2.6 and SciPy 1.15.3. The optional extraction used Trimesh 4.11.5 and Blender 4.5.8.

```bash
output_dir="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python benchmark.py --geometry GEOMETRY.json --out "$output_dir/run" --replay 512
python audit_results.py --results "$output_dir/run"
```

For the local extraction, run `retarget.py --among-us /path/to/Among_US --blender /path/to/blender --out /path/to/local-output`. Run `extract_geometry.py --asset /path/to/local-output/vessels_registered.npz --repo /path/to/RealUS_playground --out /path/to/recreated-geometry.json`. Geometry values and motion costs are deterministic on the recorded runtime. NPZ source hashes can differ if the retarget export's embedded configuration path changes.

The release compresses the exhaustive prescan records and paired traces as JSONL XZ files. `audit_results.py` accepts the compressed files in a copy of the released `verification` directory.
