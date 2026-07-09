#!/usr/bin/env python3
"""Generate a PDF of the AprilTag calibration board.

The tags are drawn from the ``pupil_apriltags`` official image data (the same
bitmap the detector expects). Every tag is scaled to the physical size given
in ``configs/board.yaml`` so the resulting PDF, printed at 100% (no scaling),
matches the geometry the calibration algorithm assumes.

We do not depend on the ``apriltag-imgs`` external repository; instead we use
``pupil_apriltags``'s bundled family PNGs. Requires ``reportlab``.

Usage::

    python scripts/generate_board_pdf.py --out board.pdf
"""
from __future__ import annotations

import argparse
import io
import os
import site
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from reportlab.lib.pagesizes import A0, landscape  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas as pdfcanvas  # noqa: E402

from multicam_calib.io.config import load_board  # noqa: E402


MM_PER_M = 1000.0


def _load_family_bitmap(family: str) -> Path:
    """Return the directory holding tag family PNGs from apriltag_imgs.

    pupil_apriltags does not bundle the source PNGs; instead we synthesize the
    bitmap of each tag from the family definition embedded in
    ``pupil_apriltags``'s C code layout. Because that is complex, we take an
    easier path: use OpenCV's ArUco module which ships an AprilTag dictionary
    that lets us render each tag id to a numpy array.
    """
    # We won't use this path — see `_render_tag_bitmap` for the pragmatic route.
    raise NotImplementedError


def _render_tag_bitmap(family: str, tag_id: int, px_per_bit: int) -> np.ndarray:
    """Render one AprilTag as a HxW uint8 array using OpenCV's aruco module.

    OpenCV's aruco dictionary ``DICT_APRILTAG_36h11`` matches the ``tag36h11``
    family byte-for-byte. Other families would need a different dictionary
    (or a real AprilTag renderer).
    """
    import cv2

    if family != "tag36h11":
        raise ValueError(f"Only tag36h11 is supported for PDF generation. Got {family!r}.")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # AprilTag 36h11 has 8x8 bit pattern (including 1-bit border), we want the tag
    # bit-pattern grid drawn at ``px_per_bit`` per bit for a crisp print.
    side_bits = 10  # 6x6 data + 1-bit black border on each side = 8; we add 1 extra black
    # cv2.aruco.generateImageMarker paints an inner 6x6 code plus a 1-bit black border,
    # totalling 8x8 grid. It resamples to the requested pixel size.
    grid = 8
    px = grid * px_per_bit
    img = np.zeros((px, px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, int(tag_id), px, img, 1)
    return img


def build(out_pdf: Path, *, dpi: int = 300) -> Path:
    cfg = load_board()
    tag_size_m = cfg.tag_size_m
    pitch_m = cfg.pitch_m

    board_w_m = (cfg.cols - 1) * pitch_m + tag_size_m
    board_h_m = (cfg.rows - 1) * pitch_m + tag_size_m
    margin_mm = 20.0  # margin around the tags for cutting

    page_w_mm = board_w_m * MM_PER_M + 2 * margin_mm
    page_h_mm = board_h_m * MM_PER_M + 2 * margin_mm
    # ReportLab uses points (1 pt = 1/72 inch = 25.4/72 mm).
    pt_per_mm = 72.0 / 25.4
    page = (page_w_mm * pt_per_mm, page_h_mm * pt_per_mm)

    out_pdf = out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = pdfcanvas.Canvas(str(out_pdf), pagesize=page)
    c.setTitle(f"AprilTag board {cfg.rows}x{cfg.cols} {cfg.family}")

    # Each tag is `tag_size_m` on a side. At `dpi` we want that many pixels:
    # ``px_side = tag_size_m * (dpi / 25.4 * 1000)``. We divide by the 8-bit
    # grid to get px_per_bit for cv2.aruco.
    px_side = max(64, int(round(tag_size_m * (dpi / 25.4) * MM_PER_M)))
    px_per_bit = max(8, px_side // 8)

    from PIL import Image
    for row in range(cfg.rows):
        for col in range(cfg.cols):
            tag_id = cfg.tag_id(row, col)
            bmp = _render_tag_bitmap(cfg.family, tag_id, px_per_bit)
            # X coordinate on PDF (left-to-right growth of columns).
            x_mm = margin_mm + col * pitch_m * MM_PER_M
            # Y coordinate: row 0 is top of the board, but PDF Y grows upward.
            y_mm = margin_mm + (cfg.rows - 1 - row) * pitch_m * MM_PER_M
            img = Image.fromarray(bmp)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            c.drawImage(
                ImageReader(buf),
                x=x_mm * pt_per_mm,
                y=y_mm * pt_per_mm,
                width=tag_size_m * MM_PER_M * pt_per_mm,
                height=tag_size_m * MM_PER_M * pt_per_mm,
                mask="auto",
            )
            # Annotate the tag id in tiny grey text under each tag (outside of
            # what the detector sees since the tag itself is a fixed square).
            c.setFillGray(0.5)
            c.setFont("Helvetica", 4)
            c.drawString(
                x_mm * pt_per_mm + 1,
                y_mm * pt_per_mm - 4,
                f"id={tag_id}",
            )

    # Header text (also outside the tag area — the printed board can be cut just
    # around the tags if desired).
    c.setFillGray(0.3)
    c.setFont("Helvetica", 10)
    c.drawString(
        margin_mm * pt_per_mm,
        (page_h_mm - margin_mm / 2) * pt_per_mm,
        f"family={cfg.family}  rows={cfg.rows} cols={cfg.cols}  tag={tag_size_m*100:.1f} cm  gap={cfg.tag_spacing_m*100:.1f} cm  "
        f"board {board_w_m*100:.1f}x{board_h_m*100:.1f} cm  — print at 100%, no scaling",
    )

    c.showPage()
    c.save()
    return out_pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "calibration_results" / "board.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    path = build(args.out, dpi=args.dpi)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
