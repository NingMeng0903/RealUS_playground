# V14 continuous driver-axes sitting review — 2026-09-08

This is a read-only review of the frozen 213328 continuous sitting2 package:

- [Genesis motion video](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis_motion.mp4)
- [motion report](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/report.json)
- [Genesis frame directory](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis/)
- [video-frame directory](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/video_frames/)

The report records 36 evaluated frames at 12 fps, with no unsupported or
error frames. It also records `used_for_fit=false`,
`runtime_optimization=false`, `runtime_blender=false`,
`anatomical_passed=false`, and `publishable=false`. The display removes the
same root orientation and translation from skin and anatomy for an upright
review. No fitting, tuning, re-rendering, or file duplication was done here.

## Frames opened

Each Genesis contact sheet contains the whole-body, left-elbow-lateral, and
left-wrist-AP views. I opened both the contact sheet and the corresponding
wide video frame for frame 00000, 00018, and 00035.

| frame | source frame | time | Genesis contact sheet | wide video frame |
|---:|---:|---:|---|---|
| 00000 | 120 | 0.000 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis/00000/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/video_frames/00000.png) |
| 00018 | 300 | 1.500 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis/00018/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/video_frames/00018.png) |
| 00035 | 470 | 2.917 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis/00035/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/video_frames/00035.png) |

## Numeric sequence result

The motion report's selected frame metrics are:

| frame | all bones max outside | left-arm bones max outside | vessels max outside | left-arm vertices >1 mm |
|---:|---:|---:|---:|---:|
| 00000 | 31.374 mm | 7.549 mm | 25.758 mm | 3,200 |
| 00018 | 42.981 mm | 7.184 mm | 38.349 mm | 2,376 |
| 00035 | 36.174 mm | 10.541 mm | 30.612 mm | 3,122 |

Across all 36 frames, all-bone maximum outside distance ranges from 31.374
to 56.406 mm (maximum at frame 00012), left-arm bone maximum ranges from
4.332 to 13.290 mm (maximum at frame 00015), and vessel maximum ranges from
25.758 to 50.202 mm (maximum at frame 00012). The 1 mm left-arm count is
nonzero in every frame. These are skin-boundary failures despite the fact
that the rendered whole-body silhouette remains visually continuous.

The motion report contains no complete bone triangle-pair audit. It therefore
cannot establish bone–bone or tube–bone non-intersection for any frame.

## Visual observations

At frame 00000 the whole-body view is upright with the arms near the sides.
The left-elbow view shows the distal humerus and proximal radius/ulna in a
close, readable arrangement, with red and yellow structures running around
the joint. The wrist view shows the forearm bones and carpal chain, while a
large red vessel lies directly alongside the bones. The narrow depth ordering
and the crop prevent a visual clearance decision.

At frame 00018 the whole-body view has progressed into the sitting posture,
with the lower limbs flexed and separated. The elbow view shows the upper arm
slanting across the rib region and the forearm dropping from the joint; the
red vessel bundle follows the bend but crowds the bone surfaces. In the wrist
view, the carpal/forearm structures remain visible, with a pale fragment at
the edge of the crop whose identity cannot be established from the image
alone. This needs a geometry query rather than a visual pass.

At frame 00035 the whole-body view has returned toward a more upright
transition while one lower limb remains flexed. The elbow view keeps the
upper-arm/forearm chain visually connected and shows the vessels alongside
the joint. The wrist view has a thick red branch crossing the distal forearm
and hand region; the bones remain visible but the vessel-to-bone corridor is
crowded.

The three samples show continuous motion at the whole-body scale, yet the
local elbow and wrist views preserve close vessel/bone contact and do not
resolve the large numerical skin excursions. No image is treated as proof of
anatomical clearance or a retarget acceptance.

## Identity and conclusion

The report identity is compiled manifest SHA-256
`5358ab93e59586d893467e4e15d6b5b29a2cb3e230a3be719f97e106179d53a8`, motion
SHA-256 `9ea10483f9fdc94d6d4943b4bb660fdec916e6cc7382bdfd82bf179c1ee00988`,
and video SHA-256
`2f40715ec95c302186421fdfc0ac806e0e8f9c53631af6170d5471d039b3857b`.

This continuous Genesis review confirms that all frames can be evaluated and
played back, but it does not turn the frozen driver-axes package into an
anatomical pass. The multi-frame skin errors and the unresolved local
bone/vessel spacing require further geometry/runtime work outside this
bounded review.

## 213712 sitting2 supplement

The same bounded visual check was completed for the second frozen sitting2
package:

- [Genesis motion video](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis_motion.mp4)
- [motion report](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/report.json)
- [Genesis frame directory](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis/)
- [video-frame directory](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/video_frames/)

The report contains 36 evaluated frames at 12 fps and zero unsupported or
error frames. It records `used_for_fit=false`, `runtime_optimization=false`,
`runtime_blender=false`, `anatomical_passed=false`, and
`publishable=false`. I opened both the Genesis contact sheet and the wide
video frame at 00000, 00018, and 00035.

| frame | source frame | time | Genesis contact sheet | wide video frame |
|---:|---:|---:|---|---|
| 00000 | 120 | 0.000 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis/00000/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/video_frames/00000.png) |
| 00018 | 300 | 1.500 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis/00018/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/video_frames/00018.png) |
| 00035 | 470 | 2.917 s | [contact sheet](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis/00035/contact_sheet.png) | [frame](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/video_frames/00035.png) |

The selected numeric rows are:

| frame | all bones max outside | left-arm bones max outside | vessels max outside | left-arm vertices >1 mm |
|---:|---:|---:|---:|---:|
| 00000 | 30.430 mm | 6.215 mm | 23.233 mm | 2,036 |
| 00018 | 42.481 mm | 6.556 mm | 37.447 mm | 1,497 |
| 00035 | 29.214 mm | 8.646 mm | 22.770 mm | 2,816 |

Across all 36 frames, the report gives all-bone maximum outside distances of
27.050–47.848 mm, left-arm maxima of 4.463–10.769 mm, and vessel maxima of
21.450–45.988 mm. The exact extrema are retained in the JSON report; every
frame has a nonzero left-arm count above 1 mm. These are boundary excursions,
not a visual clearance decision.

At frame 00000 the whole-body view is upright and the arms are near the
sides. The left-elbow view shows a readable humerus-to-forearm arrangement,
with red and yellow structures crowded around the joint. The left-wrist view
shows the forearm and carpal chain with a thick red vessel immediately beside
the bones. The crop and depth ordering do not establish clearance.

At frame 00018 the whole body is in the sitting transition. The elbow view
shows the upper arm crossing the rib region and the forearm descending from a
visibly continuous joint; the vessel bundle follows the bend but crowds the
bone surfaces. The wrist view keeps the carpal structures visible, while the
nearby red bundle leaves no reliable depth judgment from the render alone.

At frame 00035 the body has returned toward an upright transition with one
lower limb still flexed. The elbow chain remains visually connected and red
vessels track along it. In the wrist view a thick red branch runs over the
distal forearm/hand region; the bones remain visible but the vessel corridor
is tight. These images demonstrate playback continuity only.

The second package's identity is compiled manifest SHA-256
`ceca8e440a34801054319d8dc128351d8460f62fefb3d518375fd99862a37a42`, motion
SHA-256 `9ea10483f9fdc94d6d4943b4bb660fdec916e6cc7382bdfd82bf179c1ee00988`,
and video SHA-256
`a1484574c635a8794340726cd43147aefd21072932db9faaf3347878d8038bd7`.
As with 213328, this frozen continuous playback remains an operational
evaluation and not an anatomical or publication pass.
