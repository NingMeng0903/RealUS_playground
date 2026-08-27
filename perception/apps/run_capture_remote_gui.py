#!/usr/bin/env python3
"""Temporary full-screen-ish button to trigger capture + SMPL-X while lying on the bed.

Xbox Y on Window C (``python -m peirastic.apps.gamepad``) runs the same job.

Prerequisites (other terminals):
  - Cam publisher (:17356)
  - Genesis viewer G (optional but recommended for live mesh)

Writes debug artifacts under ``smplx_outputs/<timestamp>/moment_0000/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import messagebox
import tkinter as tk

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from perception.capture_flow import CaptureResult, try_start_smplx_capture  # noqa: E402


class CaptureRemoteGui:
    def __init__(self) -> None:
        self._running = False
        self._root = tk.Tk()
        self._root.title("RealUS Capture")
        self._root.minsize(520, 280)
        self._root.geometry("560x300")
        self._root.attributes("-topmost", True)

        frame = tk.Frame(self._root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)

        self._status = tk.StringVar(value="Ready — Cam :17356 · Xbox Y (Window C) or this button")
        tk.Label(
            frame,
            textvariable=self._status,
            wraplength=500,
            justify=tk.CENTER,
            font=("DejaVu Sans", 10),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self._btn = tk.Button(
            frame,
            text="CAPTURE\n& GENERATE SMPLX",
            font=("DejaVu Sans", 13, "bold"),
            height=4,
            command=self._on_click,
        )
        self._btn.grid(row=1, column=0, sticky="ew", pady=8)

        tk.Label(
            frame,
            text="Same job as Xbox Y on Window C.\n"
            "Output: smplx_outputs/<timestamp>/moment_0000/\n"
            "(images_raw · skeleton_2d · overlays)",
            font=("DejaVu Sans", 9),
            fg="#555",
            justify=tk.CENTER,
            wraplength=500,
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _on_click(self) -> None:
        if self._running:
            return
        start = try_start_smplx_capture(label="gui", on_done=self._on_done)
        if not start.started:
            self._status.set(f"Ignored — {start.reason}")
            return
        self._running = True
        self._btn.config(state=tk.DISABLED)
        self._status.set(f"Running … {start.run_name}")

    def _on_done(self, result: CaptureResult) -> None:
        self._root.after(0, lambda: self._finish(result))

    def _finish(self, result: CaptureResult) -> None:
        self._running = False
        self._btn.config(state=tk.NORMAL)
        moment_dir = result.moment_dir
        if result.ok:
            rel = f"smplx_outputs/{result.run_name}/moment_0000/"
            self._status.set(f"Done → {rel}")
            messagebox.showinfo(
                "Capture OK",
                "SMPL-X fit complete.\n\n"
                f"Folder:\n{moment_dir}\n\n"
                "images_raw/  — originals\n"
                "skeleton_2d/ — DWPose\n"
                "skeleton_fused/ — fused keypoints\n"
                "overlays/ — SMPL-X reprojection\n\n"
                "Genesis viewer should show orange mesh.",
            )
            return
        if result.quality_rejection is not None:
            quality = result.quality_rejection
            actual = quality.get("final_smplx_reprojection_error_px")
            limit = quality.get("final_smplx_reprojection_max_px")
            reasons: list[str] = []
            details: list[str] = []
            if not bool(quality.get("core_ok", True)):
                reasons.append("torso quality")
            if not bool(quality.get("foot_ok", True)):
                reasons.append("foot quality")
            if not bool(quality.get("reprojection_ok", True)):
                reasons.append("reprojection")
            if isinstance(actual, (int, float)):
                details.append(f"Final reprojection: {actual:.2f}px")
            if isinstance(limit, (int, float)):
                details.append(f"Publication limit: {limit:.2f}px")
            reason_text = ", ".join(reasons) if reasons else "final quality gate"
            self._status.set(f"Quality rejected — {reason_text}")
            messagebox.showwarning(
                "Capture quality rejected",
                "SMPL-X fitting completed and diagnostics were saved, but no high-precision mesh was published.\n\n"
                + "\n".join(details)
                + "\n\n"
                f"Inspect:\n{moment_dir / 'panels' / 'frame_000000'}\n"
                "This is a quality gate, not a camera or optimizer crash.",
            )
            return
        rel_log = f"smplx_outputs/{result.run_name}/capture_gui.log"
        self._status.set(f"Failed — see {rel_log}")
        messagebox.showerror("Capture failed", f"See log:\n{result.log_path}")

    def run(self) -> None:
        self._root.mainloop()


def main() -> int:
    os.chdir(_REPO)
    app = CaptureRemoteGui()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
