#!/usr/bin/env python3
"""Temporary full-screen-ish button to trigger capture + SMPL-X while lying on the bed.

Prerequisites (other terminals):
  - Cam publisher (:17356)
  - Genesis viewer G (optional but recommended for live mesh)

Writes debug artifacts under ``smplx_outputs/<timestamp>/moment_0000/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox


def _repo_root() -> Path:
    return Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def _smplx_output_root(repo: Path) -> Path:
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", repo / "smplx_outputs"))


def _capture_cmd(repo: Path, run_name: str) -> list[str]:
    py = os.environ.get("PY", sys.executable)
    out_root = _smplx_output_root(repo)
    return [
        py,
        str(repo / "perception/apps/run_smplx_capture.py"),
        "--config",
        str(repo / "configs/tracking/realus_dwpose_easymocap.yaml"),
        "--connect",
        "tcp://127.0.0.1:17356",
        "--output-root",
        str(out_root),
        "--run-name",
        run_name,
        "--write-debug-images",
        "--publish-genesis",
        "--publish-kind",
        "smplx_mesh",
    ]


class CaptureRemoteGui:
    def __init__(self) -> None:
        self._repo = _repo_root()
        self._running = False
        self._root = tk.Tk()
        self._root.title("RealUS Capture")
        self._root.minsize(520, 280)
        self._root.geometry("560x300")
        self._root.attributes("-topmost", True)

        frame = tk.Frame(self._root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)

        self._status = tk.StringVar(value="Ready — Cam publisher on :17356")
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
            text="Output:\nsmplx_outputs/<timestamp>/moment_0000/\n(images_raw · skeleton_2d · overlays)",
            font=("DejaVu Sans", 9),
            fg="#555",
            justify=tk.CENTER,
            wraplength=500,
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _on_click(self) -> None:
        if self._running:
            return
        self._running = True
        self._btn.config(state=tk.DISABLED)
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._status.set(f"Running … {run_name}")
        threading.Thread(target=self._run_capture, args=(run_name,), daemon=True).start()

    def _run_capture(self, run_name: str) -> None:
        out_root = _smplx_output_root(self._repo)
        moment_dir = out_root / run_name / "moment_0000"
        log_path = out_root / run_name / "capture_gui.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = _capture_cmd(self._repo, run_name)
        env = dict(os.environ)
        src = str((self._repo / "src").resolve())
        env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            log_path.write_text(
                f"cmd: {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n\n--- stderr ---\n{proc.stderr}\n",
                encoding="utf-8",
            )
            ok = proc.returncode == 0 and (moment_dir / "smplx_result.npz").is_file()
        except subprocess.TimeoutExpired:
            ok = False
            log_path.write_text("capture timed out after 600s\n", encoding="utf-8")
        except Exception as exc:
            ok = False
            log_path.write_text(f"capture failed: {exc}\n", encoding="utf-8")

        def _finish() -> None:
            self._running = False
            self._btn.config(state=tk.NORMAL)
            if ok:
                rel = f"smplx_outputs/{run_name}/moment_0000/"
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
            else:
                rel_log = f"smplx_outputs/{run_name}/capture_gui.log"
                self._status.set(f"Failed — see {rel_log}")
                messagebox.showerror("Capture failed", f"See log:\n{log_path}")

        self._root.after(0, _finish)

    def run(self) -> None:
        self._root.mainloop()


def main() -> int:
    os.chdir(_repo_root())
    app = CaptureRemoteGui()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
