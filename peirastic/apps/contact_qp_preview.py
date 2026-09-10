"""Read-only confidence preview for an image or a selected ultrasound/jpeg H5 row."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from peirastic.contact_qp.features import (FeatureExtractor, confidence_features,
                                          load_feature_config, welleweerd_config)


def read_image(path, frame=0, dataset="ultrasound/jpeg"):
    import cv2
    if path.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            if frame < 0 or frame >= len(f[dataset]):
                raise ValueError("frame index outside dataset")
            row = np.asarray(f[dataset][frame])
            image = (cv2.imdecode(row.astype(np.uint8), cv2.IMREAD_GRAYSCALE)
                     if row.ndim == 1 else row)
            if image is not None and image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dataset", default="ultrasound/jpeg")
    parser.add_argument("--feature-config")
    parser.add_argument("--output", required=True, type=Path, help="output prefix (PNG + JSON)")
    args = parser.parse_args(argv)
    cfg = load_feature_config(args.feature_config) if args.feature_config else welleweerd_config()
    im = read_image(args.input, args.frame, args.dataset)
    obs, confidence = FeatureExtractor(cfg).extract(im, frame_seq=args.frame, source_id=str(args.input),
                         capture_time_s=1., received_time_s=1., already_aligned=True)
    features = confidence_features(im, confidence, cfg)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.ndimage import zoom
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(im, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[0, 0].set_title("Original B-mode (display DN 0–255)")
    view = axes[0, 1].imshow(confidence, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    y0, y1 = features["top_roi_rows"]
    axes[0, 1].axhline(y0-.5, color="white", lw=1)
    axes[0, 1].axhline(y1-.5, color="white", lw=1)
    axes[0, 1].set_title("Raw random-walk confidence; fixed 0–1; ROI lines")
    fig.colorbar(view, ax=axes[0, 1])
    small = zoom(im, (cfg.height/im.shape[0], cfg.width/im.shape[1]), order=1, prefilter=False)
    axes[1, 0].imshow(small, cmap="gray", vmin=0, vmax=255, aspect="auto")
    labels = np.zeros(confidence.shape, dtype=int)
    # Show scanline decisions only within the declared top ROI.
    labels[y0:y1, np.array(features["low_confidence_columns"])] = 1
    labels[:, np.array(features["unknown_columns"])] = 2
    cmap = ListedColormap([(0, 0, 0, 0), (1, .1, .1, .65), (1, .65, 0, .65)])
    axes[1, 0].imshow(labels, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), aspect="auto")
    axes[1, 0].set_title("Red: low top-ROI confidence; orange: unknown scanline")
    curve = axes[1, 1]
    curve.plot(features["column_confidence"], label="Top-ROI column mean")
    curve.axhline(cfg.low_confidence_threshold, color="red", ls="--", label="Experimental threshold")
    for region in features["regions"]:
        curve.axvspan(region["x_start"]-.5, region["x_stop"]-.5,
                      color="orange" if region["label"] == "unknown" else "red", alpha=.18)
    curve.set(xlim=(0, cfg.width-1), ylim=(0, 1), xlabel="Processing image column (0-based)", ylabel="Confidence")
    curve.legend(loc="lower right")
    curve.set_title(f"L/C/R={np.round(obs.quality, 3)}; valid={obs.valid.tolist()}")
    fig.suptitle("Welleweerd-style features — experimental ROI/threshold, no physical contact calibration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    result = dict(source=str(args.input.resolve()), frame=args.frame, config=asdict(cfg),
                  window_version=cfg.window_version, observation=obs.to_dict(), features=features,
                  raw_confidence=confidence.tolist())
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(png=str(args.output.with_suffix('.png')), json=str(args.output.with_suffix('.json')),
                         quality=features['quality_lcr'], status=features['frame_status'])))


if __name__ == "__main__":
    main()
