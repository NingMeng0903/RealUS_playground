"""Replay old scans alongside hard and recovered trajectories on CUDA."""

from collections import defaultdict
from pathlib import Path
import hashlib

import numpy as np

from evidence import ROOT
from study_v9 import atomic_json, digest
from recovery_reader import (_batch, _cpu_bundle, _fit_calibration, _loss, _make_model, _panel,
                             _parity, _plain, _prepare_rows, _role_index, _risk_summary,
                             _write_checkpoint, load_bundle, read_shard)
from fit_reader_v9 import open_shard


ARCHIVE = ROOT.parent / "leg_scan_reader_feedback_20261123_v4" / "run" / "fit"
V9_READER = ROOT.parent / "leg_scan_full_route_20261124_v9" / "run" / "reader_bundle.json"
MIX = (8, 7, 5)


def archive_rows(seed):
    paths = sorted(ARCHIVE.glob("*/data.npz"))
    if len(paths) < 8:
        raise RuntimeError("The normal-trajectory replay archive is incomplete")
    generator = np.random.default_rng(seed)
    indices = np.linspace(0, len(paths)-1, 8).astype(int)
    for i in indices:
        records, _, _ = open_shard(paths[i], terminal=True)
        for j in generator.choice(len(records), min(32, len(records)), replace=False):
            public, labels, instance, layout, route, _ = records[j]
            yield {"public": public, "labels": labels, "instance": instance, "layout_id": layout,
                   "route": route, "split": "train", "scope": "archive_normal", "complete": True}


def _by_class(rows):
    grouped = [defaultdict(list) for _ in MIX]
    for row in rows:
        scope = row.metadata.get("scope", "")
        group = {"archive_normal": 0, "hard_student_terminal": 1, "recovery_student_terminal": 2}.get(scope)
        if group is not None and len(row.labels):
            grouped[group][row.layout].append(row)
    if not all(grouped):
        raise RuntimeError("All three acquisition groups need at least one real layout")
    return grouped


def _update_references(rows, baseline):
    for row, _, margin, ids, _ in _panel(rows, baseline, use_history=True, collect_roles=False):
        from recovery_reader import _calibration, _score
        settings = _calibration(baseline, row.route, True)
        confidence = _score(margin, settings)
        confidence[ids == -99] = -1.
        row.base_ids = ids
        row.base_scores = confidence


def _summarize(rows, bundle):
    from recovery_reader import _calibration, _score

    summary = {}
    for route in ("femoral", "groin_to_distal_calf"):
        panel = _panel([r for r in rows if r.route == route], bundle, True, collect_roles=False)
        labels, accepted, visible = [], [], []
        for row, _, margin, ids, _ in panel:
            settings = _calibration(bundle, route, True)
            labels.append(ids)
            accepted.append((_score(margin, settings) >= settings["threshold"]) & (ids != -99))
            visible.append(row.correct_visible)
        summary[route] = _risk_summary(np.stack(labels), np.stack(accepted), np.stack(visible),
                                       np.ones(len(labels), dtype=bool)) | {"layouts": len(panel)}
    return summary


def fit(acquisition, initial, output, seed=551_826, epochs=6, batches=20):
    import torch
    from recovery_reader import HISTORY_LOGIT_CAP

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    base, prior = load_bundle(V9_READER), load_bundle(initial)
    fresh = list(sorted(acquisition.rglob("snapshots.npz")))
    combined = [*archive_rows(seed), *(r for shard in fresh for r in read_shard(shard))]
    deck, fingerprint = _prepare_rows(combined, base)
    assert all(row.metadata["complete"] for rows in deck.values() for row in rows)
    for rows in deck.values():
        _update_references(rows, prior)
    del combined
    groups = _by_class(deck["train"])
    limits = [len(x) for x in groups]
    if min(limits) < max(MIX):
        raise RuntimeError("Too few independent layouts in a replay stratum")
    model = _make_model(base).cuda()
    model.load_state_dict({key: torch.from_numpy(v) for key, v in prior["weights"].items()} |
                          {"center": torch.from_numpy(prior["center"]), "scale": torch.from_numpy(prior["scale"])})
    initial_weights = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=.00038, weight_decay=.004, fused=True)
    roles = np.bincount(np.concatenate([_role_index(r.labels) for r in deck["train"]]), minlength=8)
    weight = torch.as_tensor(np.clip((np.median(roles[roles > 0])/np.maximum(roles, 1))**.35, .55, 2.7),
                             dtype=torch.float32, device="cuda")
    lookup = torch.tensor([7, 6, 0, 1, 2, 3, 4, 5], device="cuda")
    settings = {"data_sha256": fingerprint, "input_reader_sha256": digest(initial), "mix_layouts_per_batch": MIX,
                "layout_ids_by_group": [sorted(g) for g in groups],
                "calibration_layout_ids": sorted({r.layout for r in deck["cal"]}),
                "development_layout_ids": sorted({r.layout for r in deck["dev"]}),
                "epochs": epochs, "steps_per_epoch": batches, "seed": seed, "cuda": torch.cuda.get_device_name(),
                "source_sha256": digest(Path(__file__))}
    atomic_json(output / "protocol.json", settings)
    incumbent = _summarize(deck["dev"], prior)
    best, trace = None, []
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for _ in range(batches):
            # A layout appears at most once within a minibatch, including
            # shows of the same person from the hard and recovered strata.
            chosen, batch = set(), []
            for g, count in zip(groups, MIX, strict=True):
                keys = sorted(set(g) - chosen)
                for key in rng.choice(keys, count, replace=False):
                    chosen.add(key)
                    batch.append(g[key][rng.integers(len(g[key]))])
            data = _batch(batch, lookup, prior, "cuda")
            loss, _ = _loss(model, data, weight, rng.random() > .20)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        bundle = _cpu_bundle(model)
        cal = {key: _fit_calibration(_panel(deck["cal"], bundle, enabled), prior)
               for key, enabled in (("history", True), ("history_off", False))}
        defaults = cal["history"]
        bundle |= {"calibration": cal, "thresholds": {k: v["threshold"] for k, v in defaults["routes"].items()},
                   "fallback_threshold": defaults["fallback"]["threshold"], "history_logit_cap": HISTORY_LOGIT_CAP,
                   "feature_order": prior["feature_order"], "input_reader_sha256": settings["input_reader_sha256"]}
        report = _summarize(deck["dev"], bundle)
        whole = report["groin_to_distal_calf"]
        old = incumbent["groin_to_distal_calf"]
        selector = (whole["wrong_branch_cases"] <= old["wrong_branch_cases"], whole["strict"],
                    -whole["wrong_branch_cases"], whole["correct_regions"])
        trace.append({"epoch": epoch, "loss": float(np.mean(losses)), "development": report, "key": selector})
        if best is None or selector > best:
            best = selector
            differences = {k: float((v.detach() - initial_weights[k]).abs().max()) for k, v in model.named_parameters()}
            assert all(x > 0. for x in differences.values())
            parity = _parity(model, bundle, deck["cal"][:2], base)[0]["absolute_max"]
            assert parity < 1.e-3
            _write_checkpoint(output / "best.pt", {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                                                   "numpy_rng": rng.bit_generator.state, "epoch": epoch}, torch)
            atomic_json(output / "reader_bundle.json", _plain(bundle))
            validated = {"epoch": epoch, "parameter_updates": differences, "cpu_cuda_max_error": parity,
                         "checkpoint_sha256": digest(output / "best.pt")}
        print(f"CUDA reader mix epoch {epoch}: whole {whole['strict']} strict, {whole['wrong_branch_cases']} wrong", flush=True)
    atomic_json(output / "results.json", {"baseline_v10": incumbent, "epochs": trace, "selected": validated,
                                         "mix": MIX, "cuda_optimizer_steps": epochs*batches,
                                         "bundle_sha256": digest(output / "reader_bundle.json")})
    return output / "reader_bundle.json"
