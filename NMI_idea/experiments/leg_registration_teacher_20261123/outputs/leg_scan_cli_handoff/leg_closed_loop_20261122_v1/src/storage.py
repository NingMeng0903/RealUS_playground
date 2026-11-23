"""Atomic dataset shards and exact restart cursors."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from dataclasses import asdict, is_dataclass

import numpy as np


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, default=json_value, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=2) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), default=json_value,
                                    ensure_ascii=False, allow_nan=False) + "\n")


def read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def npz(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.asarray(value), allow_pickle=False)


def frame_stream(items):
    count = sum(len(item.frames) for _, _, item in items)
    echo = np.full((count, 7, 2), np.nan, dtype=np.float32)
    probabilities = np.zeros((count, 7, 3), dtype=np.float32)
    labels = np.full((count, 7), -128, dtype=np.int8)
    observation = np.empty((count, 3), dtype=np.float32)
    discrete = np.empty((count, 3), dtype=np.int16)
    offsets, cursor = [0], 0
    for _, _, item in items:
        if len(item.frames) != len(item.mixed):
            raise RuntimeError("Public anatomical explanation log lost a frame")
        for row, mixed in zip(item.frames, item.mixed):
            s, a, z, truth, q, r, target, unknown, t = row
            n = len(z)
            echo[cursor, :n] = z
            probabilities[cursor, :n, 0] = target
            probabilities[cursor, :n, 1] = unknown
            probabilities[cursor, :n, 2] = mixed
            labels[cursor, :n] = truth
            observation[cursor] = s, a, q
            discrete[cursor] = t, r, n
            cursor += 1
        offsets.append(cursor)
    offsets = np.asarray(offsets, dtype=np.int32)
    public = {"offsets": offsets, "pose_quality": observation, "tick_region_count": discrete,
              "echo_mm": echo, "target_unknown_mixed": probabilities}
    return public, {"offsets": offsets, "truth_labels": labels}


def step_stream(items):
    offsets = np.cumsum([0, *(len(item.steps) for _, _, item in items)], dtype=np.int32)
    steps = [row for _, _, item in items for row in item.steps]
    return {
        "offsets": offsets, "features": np.asarray([r["features"] for r in steps], dtype=np.float32),
        "next_features": np.asarray([r["next_features"] for r in steps], dtype=np.float32),
        "stage_skill_terminal": np.asarray([(r["stage"], r["skill"], r["done"]) for r in steps], dtype=np.int8),
        "legality_mask": np.asarray([r["mask"] for r in steps], dtype=bool),
        "latent_controls": np.asarray([r["latent"] for r in steps], dtype=np.float32),
        "sampled_controls": np.asarray([r["controls"] for r in steps], dtype=np.float32),
        "categorical_probs": np.asarray([r["probabilities"] for r in steps], dtype=np.float32),
        "reward_components": np.asarray([(r["reward"], r["identity_reward"], r["capture_reward"],
                                          r["terminal_reward"]) for r in steps], dtype=np.float32),
        "logp_and_value": np.asarray([(r["old_logp"], r["value"]) for r in steps], dtype=np.float32),
        "frame_tick_index": np.asarray([(r["first_frame"], r["frame_stop"], r["tick"], r["next_tick"],
                                         r["ticks"], r["tick_cap"]) for r in steps], dtype=np.int16),
        "pose_and_mm_cap": np.asarray([(r["station"], r["sector"], r["next_station"],
                                         r["next_sector"], r["mm_cap"]) for r in steps], dtype=np.float32),
    }


def stash(path, items, manifest, checkpoint=None, extra=None):
    """Commit a complete shard and its post-update weights in one rename."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        json_lines(temporary / "episodes.jsonl.gz",
                   [asdict(case) | {"arm": arm} | item.metrics for case, arm, item in items])
        steps = step_stream(items)
        npz(temporary / "steps.npz", steps)
        public, truth = frame_stream(items)
        npz(temporary / "public_frames.npz", public)
        npz(temporary / "evaluator_labels.npz", truth)
        layouts = [item.layout_centers for _, _, item in items]
        if any(layout is not None for layout in layouts):
            if not all(layout is not None for layout in layouts):
                raise RuntimeError("Cannot mix generated layouts and untouched holdout cases in a shard")
            npz(temporary / "synthetic_layouts.npz",
                {"centers_xz_mm": np.asarray(layouts, dtype=np.float64)})
        json_lines(temporary / "probes.jsonl.gz",
                   ({"episode": case.identifier, "arm": arm} | probe
                    for case, arm, item in items for probe in item.probes))
        json_lines(temporary / "stop_decisions.jsonl.gz",
                   ({"episode": case.identifier, "arm": arm, "step": i, "skill": step["skill"]}
                    | step["exit_check"]
                    for case, arm, item in items for i, step in enumerate(item.steps)
                    if "exit_check" in step))
        json_lines(temporary / "teacher.jsonl.gz",
                   ({"episode": case.identifier, "arm": arm} | item.teaching
                    for case, arm, item in items if item.teaching is not None))
        if extra and any(item.audit is not None for _, _, item in items):
            json_lines(temporary / "same_track.jsonl.gz",
                       ({"episode": case.identifier, "task": case.task, "profile": case.profile,
                         "source": case.source, "arm": arm, "regions": item.audit}
                        for case, arm, item in items if item.audit is not None))
        if checkpoint is not None:
            torch.save(checkpoint, temporary / "checkpoint.pt")
        manifest = manifest | {"episodes": len(items), "frames": int(public["offsets"][-1]),
                               "steps": int(steps["offsets"][-1])}
        atomic_json(temporary / "commit.json", manifest)
        for child in temporary.iterdir():
            with child.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def shards(directory):
    previous = -1
    for path in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9]")):
        if not (path / "commit.json").is_file() or int(path.name) != previous + 1:
            raise RuntimeError(f"Training shard gap at {path}")
        previous += 1
        yield path


def recount(directory, arm, expected_split="training"):
    seen, tape, layout_ids = set(), set(), set()
    frames = steps = rejects = 0
    for path in shards(directory):
        manifest = read_json(path / "commit.json")
        rows = list(read_lines(path / "episodes.jsonl.gz"))
        if len(rows) != manifest["episodes"]:
            raise ValueError(f"Episode shard has incomplete content {path}")
        with np.load(path / "steps.npz", allow_pickle=False) as values:
            indexed_steps = values["offsets"]
            terminal = values["stage_skill_terminal"][:, 2]
        with np.load(path / "public_frames.npz", allow_pickle=False) as values:
            indexed_frames = values["offsets"]
            frame_clock = values["tick_region_count"][:, 0]
        for i, row in enumerate(rows):
            key = row["identifier"]
            scenario = row["tape_seed"]
            valid = (row["split"] == expected_split and row["arm"] == arm and row["complete"]
                     and row["frames"] == row["end_tick"] and key not in seen and scenario not in tape
                     and indexed_steps[i + 1] - indexed_steps[i] == row["steps"]
                     and indexed_frames[i + 1] - indexed_frames[i] == row["frames"]
                     and terminal[indexed_steps[i + 1] - 1] == 1
                     and np.array_equal(frame_clock[indexed_frames[i]:indexed_frames[i + 1]],
                                        np.arange(1, row["frames"] + 1)))
            if not valid:
                rejects += 1
                continue
            seen.add(key)
            tape.add(scenario)
            if row.get("layout_id"):
                layout_ids.add(row["layout_id"])
            frames += row["frames"]
            steps += row["steps"]
    return {"complete_unique": len(seen), "excluded": rejects, "frames": frames, "steps": steps,
            "distinct_generated_layouts": len(layout_ids)}
