"""Detect the real B-mode region: four extrema of the sector, then its AABB.

Linear: left/right/top from the near-field bright-band corners, bottom from
the last textured row in that width. Convex: top of the probe-face arc,
bottom of the far-field arc, left and right vertices of the fan.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from us_framegrab.config import clamp_cbox


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return frame
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def longest_true_run(values: Sequence[bool] | np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best_end - best_start:
                best_start, best_end = start, index
            start = None
    if start is not None and len(values) - start > best_end - best_start:
        best_start, best_end = start, len(values)
    return best_start, best_end


def _count_true_runs(values: Sequence[bool] | np.ndarray) -> int:
    n = 0
    prev = False
    for value in values:
        flag = bool(value)
        if flag and not prev:
            n += 1
        prev = flag
    return n


def _mask_is_linear_rectangle(mask: np.ndarray, ext: dict[str, Any]) -> bool:
    """True if the body blob is a constant-width rectangle (linear probe)."""
    x0, x1, y0, y1 = (int(v) for v in ext["aabb"])
    height = max(y1 - y0, 1)
    width = max(x1 - x0, 1)

    def _width_at(frac: float) -> int:
        y = min(mask.shape[0] - 1, y0 + int(frac * (height - 1)))
        cols = np.flatnonzero(mask[y])
        if cols.size == 0:
            return 0
        return int(cols[-1] - cols[0] + 1)

    return (
        _width_at(0.08) >= 0.82 * width
        and _width_at(0.50) >= 0.82 * width
        and _width_at(0.92) >= 0.75 * width
    )


def _clip_init_cbox(grey_img: np.ndarray, init_cbox: Sequence[int]) -> tuple[int, int, int, int] | None:
    x0_i, x1_i, y0_i, y1_i = (int(v) for v in init_cbox)
    height, width = grey_img.shape[:2]
    x0_i = min(max(x0_i, 0), width)
    x1_i = min(max(x1_i, 0), width)
    y0_i = min(max(y0_i, 0), height)
    y1_i = min(max(y1_i, 0), height)
    if x1_i - x0_i < 4 or y1_i - y0_i < 4:
        return None
    return x0_i, x1_i, y0_i, y1_i


def _corner_black_level(roi: np.ndarray) -> float:
    """Bezel level from the four corners of the search window (not the fan)."""
    h, w = roi.shape
    ph, pw = max(8, h // 20), max(8, w // 20)
    samples = np.concatenate(
        [
            roi[:ph, :pw].ravel(),
            roi[:ph, -pw:].ravel(),
            roi[-ph:, :pw].ravel(),
            roi[-ph:, -pw:].ravel(),
        ]
    )
    return float(np.percentile(samples, 25))


def _local_std(roi: np.ndarray, k: int = 15) -> np.ndarray:
    import cv2

    f32 = roi.astype(np.float32)
    mean = cv2.blur(f32, (k, k))
    mean2 = cv2.blur(f32 * f32, (k, k))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def _sector_mask(roi: np.ndarray) -> np.ndarray | None:
    """Largest region that is not bezel: flood near-black from the ROI border."""
    import cv2

    gray = np.ascontiguousarray(roi)
    black = _corner_black_level(gray)
    delta = max(6.0, 0.05 * (255.0 - black))
    # Intensity only. Std-gated seeds treat CFM glow / icons as content.
    seed_u8 = (gray.astype(np.float32) <= black + delta).astype(np.uint8)
    _n, labels = cv2.connectedComponents(seed_u8, connectivity=8)
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return None
    bezel = np.isin(labels, border_ids)
    content = (~bezel).astype(np.uint8)
    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    if n2 < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    min_area = max(3000, int(gray.size * 0.03))
    if int(stats[idx, cv2.CC_STAT_AREA]) < min_area:
        return None
    mask = lab2 == idx
    std = _local_std(gray)
    speckle = (std > 2.8) & (gray.astype(np.float32) > black + 2.0)
    inner = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return mask | (speckle & inner)


def _full_width_band_run(
    gray_row: np.ndarray,
    hint_x0: int,
    hint_x1: int,
    bright_thresh: float,
) -> tuple[int, int] | None:
    """Longest bright run if it matches the linear B-mode width (not ticks / chrome)."""
    hint_w = max(hint_x1 - hint_x0, 1)
    xa = max(0, hint_x0 - 12)
    xb = min(int(gray_row.shape[0]), hint_x1 + 12)
    if xb - xa < 8:
        return None
    bright = gray_row[xa:xb] >= bright_thresh
    if _count_true_runs(bright) >= 4:
        return None
    start, end = longest_true_run(bright)
    run = end - start
    if run < 0.85 * hint_w or run > 1.15 * hint_w:
        return None
    return xa + start, xa + end


def _row_has_scan_texture(strip: np.ndarray, black: float) -> bool:
    """True if this row slice is B-mode speckle / overlay, not empty bezel."""
    if strip.size < 4:
        return False
    values = strip.astype(np.float32)
    med = float(np.median(values))
    std = float(np.std(values))
    frac_dark = float(np.mean(values <= black + 8.0))
    if frac_dark > 0.92 and std < 3.0:
        return False
    return std >= 2.2 or med > black + 10.0 or float(np.mean(values >= 80.0)) > 0.08


def _extend_linear_y1(gray: np.ndarray, x0: int, x1: int, y_from: int, black: float) -> int:
    last = y_from
    gap = 0
    for y in range(y_from + 1, gray.shape[0]):
        if _row_has_scan_texture(gray[y, x0:x1], black):
            last = y
            gap = 0
        else:
            gap += 1
            if gap >= 10:
                break
    return last + 1


def _linear_box_from_top_band(
    gray: np.ndarray,
    hint_aabb: Sequence[int],
    black: float,
) -> list[int] | None:
    """Linear crop from the near-field band corners, then scan down for y1."""
    hx0, hx1, hy0, hy1 = (int(v) for v in hint_aabb)
    hint_w = hx1 - hx0
    hint_h = hy1 - hy0
    if hint_w < 40 or hint_h < 4:
        return None
    # Stay near the flood top. Searching 40px up hits the HDMI header bar.
    y_lo = max(0, hy0 - 8)
    y_hi = min(gray.shape[0], hy0 + max(24, int(0.18 * hint_h)))
    for thresh in (max(black + 45.0, 70.0), max(black + 25.0, 50.0), max(black + 12.0, 36.0)):
        rows: list[tuple[int, int, int]] = []
        for y in range(y_lo, y_hi):
            run = _full_width_band_run(gray[y], hx0, hx1, thresh)
            if run is not None:
                rows.append((y, run[0], run[1]))
        if not rows:
            continue
        by_y = {y: (a, b) for y, a, b in rows}
        groups = _contiguous_groups([y for y, _a, _b in rows], gap=4)
        ranked = sorted(groups, key=lambda g: abs(g[0] - hy0))
        pack = ranked[0]
        if len(pack) < 2 and pack[0] < hy0 - 4 and len(ranked) > 1:
            pack = ranked[1]
        best = max((by_y[y] for y in pack), key=lambda ab: ab[1] - ab[0])
        x0, x1 = int(best[0]), int(best[1])
        y0 = int(pack[0])
        y1 = _extend_linear_y1(gray, x0, x1, int(pack[-1]), black)
        if y1 - y0 >= 8 and x1 - x0 >= 40:
            return [x0, x1, y0, y1]
    return None


def _linear_box_discover(gray: np.ndarray, black: float) -> list[int] | None:
    """Low-gain fallback: flood ate the dark body; recover from the top band."""
    height, width = gray.shape
    y_hi = min(height, max(80, int(0.35 * height)))
    for thresh in (max(black + 45.0, 70.0), max(black + 25.0, 50.0), max(black + 12.0, 36.0)):
        rows: list[tuple[int, int, int]] = []
        for y in range(0, y_hi):
            bright = gray[y] >= thresh
            if _count_true_runs(bright) >= 4:
                continue
            start, end = longest_true_run(bright)
            run = end - start
            # B-mode width, not a header bar and not an icon.
            if run < 0.28 * width or run > 0.72 * width:
                continue
            rows.append((y, start, end))
        if not rows:
            continue
        by_y = {y: (a, b) for y, a, b in rows}
        groups = _contiguous_groups([y for y, _a, _b in rows], gap=4)
        pack = max(groups, key=len)
        best = max((by_y[y] for y in pack), key=lambda ab: ab[1] - ab[0])
        x0, x1 = int(best[0]), int(best[1])
        y0 = int(pack[0])
        y1 = _extend_linear_y1(gray, x0, x1, int(pack[-1]), black)
        if y1 - y0 >= 80 and x1 - x0 >= 40:
            return [x0, x1, y0, y1]
    return None


def _row_ring_flags(
    gray_row: np.ndarray,
    axis_x: float,
    half_w: int,
    bright_thresh: float,
) -> tuple[bool, bool]:
    """Classify a row as a near-field ring.

    Returns ``(is_ring, is_two_lobe)``. A convex ring is a short centered
    crest *or* two bright lobes with a dark neck (horizontal chord of an
    arc). Soft-key / header bars are rejected as too wide for the fan.
    """
    # Imaging canvas is legal-range black; the gray title bar is not a ring.
    if float(np.median(gray_row)) > 28.0:
        return False, False
    w = gray_row.shape[0]
    full_a, full_b = longest_true_run(gray_row >= bright_thresh)
    full_run = full_b - full_a
    # Compare to the fan, not the 1920-wide HDMI frame: a 1100px header
    # bar is only ~0.57 of the frame but far wider than the sector neck.
    if full_run > 0.50 * w or full_run > 1.7 * (2 * half_w):
        return False, False
    xa = int(max(0, axis_x - half_w))
    xb = int(min(w, axis_x + half_w + 1))
    if xb - xa < 8:
        return False, False
    band = gray_row[xa:xb]
    bright = band >= bright_thresh
    n = int(np.count_nonzero(bright))
    if n < 4:
        return False, False
    # Ruler ticks are many short islands; a ring is one crest or two lobes.
    if _count_true_runs(bright) >= 4:
        return False, False
    a, b = longest_true_run(bright)
    run = b - a
    mid = xa + 0.5 * (a + b)
    crest = run >= 4 and abs(mid - axis_x) <= 0.60 * half_w
    axis_local = int(round(axis_x - xa))
    axis_local = min(max(axis_local, 0), band.size - 1)
    left_n = int(np.count_nonzero(bright[:axis_local]))
    right_n = int(np.count_nonzero(bright[axis_local + 1 :]))
    neck = bright[max(0, axis_local - 3) : axis_local + 4]
    neck_dark = int(np.count_nonzero(neck)) <= 2
    two_lobe = (
        left_n >= 3
        and right_n >= 3
        and neck_dark
        and min(left_n, right_n) >= 0.35 * max(left_n, right_n)
    )
    return (crest or two_lobe), two_lobe


def _contiguous_groups(ys: list[int], *, gap: int = 3) -> list[list[int]]:
    if not ys:
        return []
    ordered = sorted(ys)
    groups = [[ordered[0]]]
    for y in ordered[1:]:
        if y <= groups[-1][-1] + gap:
            groups[-1].append(y)
        else:
            groups.append([y])
    return groups


def _band_bright_span(
    gray_row: np.ndarray,
    axis_x: float,
    half_w: int,
    bright_thresh: float,
) -> int:
    xa = int(max(0, axis_x - half_w))
    xb = int(min(gray_row.shape[0], axis_x + half_w + 1))
    bright = gray_row[xa:xb] >= bright_thresh
    xs = np.flatnonzero(bright)
    if xs.size == 0:
        return 0
    return int(xs[-1] - xs[0] + 1)


def _apex_top_of_pack(
    rows: list[int],
    two_lobe: set[int],
    gray: np.ndarray,
    axis_x: float,
    half_w: int,
    bright_thresh: float,
) -> int:
    """Top of the probe-face pack, skipping a compact icon just above the arc."""
    ordered = sorted(rows)
    two = [y for y in ordered if y in two_lobe]
    if not two:
        return int(ordered[0])
    first_two = two[0]
    span_ref = _band_bright_span(gray[first_two], axis_x, half_w, bright_thresh)
    top = first_two
    for y in range(first_two - 1, ordered[0] - 1, -1):
        if y not in set(ordered):
            break
        span = _band_bright_span(gray[y], axis_x, half_w, bright_thresh)
        # Real crest shrinks smoothly; an "S" icon collapses to a blob.
        if span_ref > 0 and span < 0.40 * span_ref:
            break
        top = y
        span_ref = max(span, 1)
    return int(top)


def _collect_ring_rows(
    gray: np.ndarray,
    axis_x: float,
    half_w: int,
    y_start: int,
    y_stop: int,
    bright_thresh: float,
) -> tuple[list[int], set[int]]:
    rows: list[int] = []
    two_lobe: set[int] = set()
    for y in range(y_start, y_stop):
        is_ring, is_two = _row_ring_flags(gray[y], axis_x, half_w, bright_thresh)
        if is_ring:
            rows.append(y)
            if is_two:
                two_lobe.add(y)
    return rows, two_lobe


def _band_corner_points(
    gray: np.ndarray,
    rows: list[int],
    axis_x: float,
    half_w: int,
    bright_thresh: float,
) -> dict[str, tuple[int, int]] | None:
    """Left / right endpoints and crest of the upper ring band."""
    left: tuple[int, int] | None = None
    right: tuple[int, int] | None = None
    top: tuple[int, int] | None = None
    for y in rows:
        xa = int(max(0, axis_x - half_w))
        xb = int(min(gray.shape[1], axis_x + half_w + 1))
        xs = np.flatnonzero(gray[y, xa:xb] >= bright_thresh) + xa
        if xs.size == 0:
            continue
        p_l, p_r = (int(xs[0]), int(y)), (int(xs[-1]), int(y))
        if left is None or p_l[0] < left[0]:
            left = p_l
        if right is None or p_r[0] > right[0]:
            right = p_r
        if top is None or y < top[1]:
            top = (int(round(0.5 * (int(xs[0]) + int(xs[-1])))), int(y))
    if left is None or right is None or top is None:
        return None
    return {"left": left, "right": right, "top": top}


def _nearfield_apex_y(
    gray: np.ndarray,
    ext_full: dict[str, Any],
    black: float,
    *,
    init_y0: int,
) -> dict[str, Any] | None:
    """Find the probe-face ring-band corners on the black canvas.

    3C-A near-field is a thin concentric band just under the header. Tissue
    may start hundreds of pixels lower (empty near-field at 17 cm). Search
    the top ~32% of the frame on the fan axis — do not require the band to
    sit next to the body blob.
    """
    height = int(gray.shape[0])
    body_y0 = int(ext_full["aabb"][2])
    axis_x = 0.5 * float(ext_full["left"][0] + ext_full["right"][0])
    fan_half = 0.5 * float(ext_full["right"][0] - ext_full["left"][0])
    if fan_half < 20:
        return None
    half_w = int(np.clip(0.42 * fan_half, 80, 300))
    # Upper band lives in the top quarter of a 1080p US screen (y0 ≲ 0.25 H).
    y_hi = min(body_y0, max(int(0.32 * height), int(init_y0) + 40))
    y_start = max(0, min(int(init_y0) - 100, int(0.05 * height)))
    if y_start >= y_hi:
        return None

    def _pack(thresh: float) -> dict[str, Any] | None:
        rows, two = _collect_ring_rows(gray, axis_x, half_w, y_start, y_hi, thresh)
        if not rows:
            return None
        groups = _contiguous_groups(rows, gap=12)
        # Highest band on the black canvas (probe face), not a mid-fan echo.
        ranked = sorted(groups, key=lambda g: g[0])
        for group in ranked:
            two_in = {y for y in group if y in two}
            if len(group) < 5 and not two_in:
                continue
            apex_y = _apex_top_of_pack(group, two, gray, axis_x, half_w, thresh)
            corners = _band_corner_points(gray, group, axis_x, half_w, thresh)
            # Linear scan / ruler ticks span ~the full B-mode width.
            # Convex probe-face is a short arc (typically < 55% of fan width).
            body_w = 2.0 * fan_half
            if corners is not None:
                band_w = float(corners["right"][0] - corners["left"][0] + 1)
                if band_w > 0.58 * body_w:
                    continue
            return {"y0": int(apex_y), "corners": corners, "rows": group}
        return None

    found = _pack(max(black + 45.0, 70.0))
    if found is None:
        found = _pack(max(black + 25.0, 50.0))
    return found


def extrema_from_mask(mask: np.ndarray) -> dict[str, Any] | None:
    """True tangent points of a sector mask: leftmost, rightmost, top, bottom."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    i_left = int(np.argmin(xs))
    i_right = int(np.argmax(xs))
    i_top = int(np.argmin(ys))
    i_bottom = int(np.argmax(ys))
    left = (int(xs[i_left]), int(ys[i_left]))
    right = (int(xs[i_right]), int(ys[i_right]))
    top = (int(xs[i_top]), int(ys[i_top]))
    bottom = (int(xs[i_bottom]), int(ys[i_bottom]))
    aabb = [left[0], right[0] + 1, top[1], bottom[1] + 1]
    return {"left": left, "right": right, "top": top, "bottom": bottom, "aabb": aabb}


def detect_sector_extrema(
    grey_img: np.ndarray,
    init_cbox: Sequence[int],
) -> dict[str, Any] | None:
    """Full-frame extrema of the B-mode sector inside ``init_cbox``."""
    clipped = _clip_init_cbox(grey_img, init_cbox)
    if clipped is None:
        return None
    x0_i, x1_i, y0_i, y1_i = clipped
    roi = np.asarray(grey_img[y0_i:y1_i, x0_i:x1_i])
    if roi.size == 0:
        return None
    black = _corner_black_level(roi)
    mask = _sector_mask(roi)
    ext = extrema_from_mask(mask) if mask is not None else None
    is_linear = bool(mask is not None and ext is not None and _mask_is_linear_rectangle(mask, ext))
    if mask is None or ext is None:
        discovered = _linear_box_discover(roi, black)
        if discovered is None:
            return None
        x0_t, x1_t, y0_t, y1_t = discovered
        ext = {
            "left": (x0_t, y0_t),
            "right": (x1_t - 1, y0_t),
            "top": (int(round(0.5 * (x0_t + x1_t - 1))), y0_t),
            "bottom": (int(round(0.5 * (x0_t + x1_t - 1))), y1_t - 1),
            "aabb": discovered,
            "band_left": (x0_t, y0_t),
            "band_right": (x1_t - 1, y0_t),
        }
        is_linear = True
    elif is_linear:
        band_box = _linear_box_from_top_band(roi, ext["aabb"], black)
        if band_box is not None:
            x0_t, x1_t, y0_t, y1_t = band_box
            ext["aabb"] = band_box
            ext["left"] = (x0_t, y0_t)
            ext["right"] = (x1_t - 1, y0_t)
            ext["top"] = (int(round(0.5 * (x0_t + x1_t - 1))), y0_t)
            ext["bottom"] = (int(round(0.5 * (x0_t + x1_t - 1))), y1_t - 1)
            ext["band_left"] = (x0_t, y0_t)
            ext["band_right"] = (x1_t - 1, y0_t)
    x0, x1, y0, y1 = ext["aabb"]
    roi_h, roi_w = roi.shape
    if (x1 - x0) * (y1 - y0) > 0.92 * roi_h * roi_w:
        return None
    shifted = {
        "left": (ext["left"][0] + x0_i, ext["left"][1] + y0_i),
        "right": (ext["right"][0] + x0_i, ext["right"][1] + y0_i),
        "top": (ext["top"][0] + x0_i, ext["top"][1] + y0_i),
        "bottom": (ext["bottom"][0] + x0_i, ext["bottom"][1] + y0_i),
        "aabb": [x0_i + x0, x0_i + x1, y0_i + y0, y0_i + y1],
    }
    if "band_left" in ext and "band_right" in ext:
        shifted["band_left"] = (ext["band_left"][0] + x0_i, ext["band_left"][1] + y0_i)
        shifted["band_right"] = (ext["band_right"][0] + x0_i, ext["band_right"][1] + y0_i)
    apex = None
    if not is_linear:
        apex = _nearfield_apex_y(np.asarray(grey_img), shifted, black, init_y0=y0_i)
    if apex is not None and int(apex["y0"]) < shifted["aabb"][2]:
        corners = apex.get("corners") or {}
        top = corners.get("top")
        top_x = int(top[0]) if top is not None else int(
            round(0.5 * (shifted["left"][0] + shifted["right"][0]))
        )
        shifted["aabb"][2] = int(apex["y0"])
        shifted["top"] = (top_x, int(apex["y0"]))
        if corners.get("left") is not None:
            shifted["band_left"] = corners["left"]
        if corners.get("right") is not None:
            shifted["band_right"] = corners["right"]
    return shifted


def get_cropping_param(
    grey_img: np.ndarray,
    init_cbox: Sequence[int],
) -> tuple[bool, list[int] | None]:
    """Crop box from the four detected sector extrema (full-frame ``[x0,x1,y0,y1]``)."""
    ext = detect_sector_extrema(grey_img, init_cbox)
    if ext is None:
        return False, None
    return True, [int(v) for v in ext["aabb"]]


def apply_crop(
    frame: np.ndarray,
    cbox: Sequence[int],
    *,
    color: bool,
    hflip: bool,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, x1, y0, y1 = clamp_cbox(list(cbox), width, height)
    cropped = frame[y0:y1, x0:x1]
    if not color:
        cropped = to_gray(cropped)
    else:
        cropped = to_bgr(cropped)
    if hflip:
        cropped = np.flip(cropped, axis=1)
    return np.ascontiguousarray(cropped)
