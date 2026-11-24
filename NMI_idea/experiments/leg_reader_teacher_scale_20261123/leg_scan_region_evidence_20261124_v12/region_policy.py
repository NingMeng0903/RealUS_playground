"""Joint region/program decisions, terminal continuation labels and GPU student fitting."""

from collections import Counter
from dataclasses import asdict

import numpy as np

import data_v12
from closed_scan import ACTOR_DIM, action_vector, execute_program, programs, snapshot
from evidence import fork_on_tape
from fit_v12 import _checkpoint, _norm, _torch_setup, bundle, hydrate, net, plain
from recovery_task import Live
from run_v11 import outcome
from study_v9 import atomic_json, digest, read_json
from track_state import linear


VALUE_WEIGHTS = np.array([1., .36, -.38, -.18], dtype=np.float32)
SEARCH = (182_715_831, 182_735_891)
VERIFY_TAPE = 182_815_979
MAX_MENU = 13
_REFERENCE = None


def terminal_value(grade):
    ids = np.asarray(grade["selected_ids"])
    return np.array([grade["strict"], (ids[3:] == 0).mean(), np.any(ids > 0),
                     np.isin(ids[3:], (-1, -2)).mean()], dtype=np.float32)


def paired_targets(matrices):
    returns = np.asarray(matrices, dtype=np.float32).mean(axis=0)
    return returns - returns[:1]


def rank(model, live, read, choices):
    x = np.stack([action_vector(live, read, a) for a in [None, *choices]])
    score = linear(model, x) @ VALUE_WEIGHTS
    return score[1:] - score[0]


def select(live, read, choices, model, mode, config):
    if not choices:
        return None
    if mode == "geometry_r5":
        late = [x for x in choices if x.region == 4 and x.name == "window"]
        return late[0] if late else None
    if mode == "sequential":
        available = {a.region for a in choices}
        region = max(available, key=lambda r: read.hypotheses[r].risk)
        choices = [p for p in choices if p.region == region]
    gain = rank(model, live, read, choices)
    gain = gain + config["r5_weight"] * np.array([x.region == 4 for x in choices])
    best = int(np.argmax(gain))
    return choices[best] if gain[best] > config["margin"] else None


def _memory(before, after, detail):
    pre = dict(zip(zip(before.public["frames"], before.public["slots"]), range(len(before.posterior))))
    post = dict(zip(zip(after.public["frames"], after.public["slots"]), range(len(after.posterior))))
    items = []
    for k in detail["keys"]:
        if k is None:
            continue
        i, j = pre[k], post[k]
        ids = after.decision["track_id"]
        # Treat all samples in the same station/sector cell as one support.
        supported = (ids == ids[j]) & (after.public["frames"] >= detail["start_tick"])
        views = set(zip(np.rint(after.public["stations"][supported]).astype(int),
                        np.rint(after.public["sectors"][supported]).astype(int) % 24))
        items.append({"key": k, "target_before": float(before.posterior[i, 0]),
                      "target_after": float(after.posterior[j, 0]),
                      "artery_before": float(before.posterior[i, 1]), "artery_after": float(after.posterior[j, 1]),
                      "new_supported_views": len(views),
                      "role_before": int(np.argmax(before.posterior[i])), "role_after": int(np.argmax(after.posterior[j]))})
    return items


def _read_after(engine, live, memory):
    return engine.read(live, original=not memory)


def roll(engine, live, model, mode, config, *, initial_seen=frozenset(), collect_prefix=False):
    """Return control after every executed program and every normal scan macro."""
    used, actions = set(initial_seen), []
    next_prefix = None
    got_new_evidence = bool(initial_seen)
    while not live.done:
        if live.frontier < live.ends[3]:
            engine.step(live)
            continue
        # Both learned arms use the same pre-probe reader. The feedback
        # ablation bypasses its post-probe identity state at subsequent calls.
        memory = mode != "no_identity_feedback" or not got_new_evidence
        read = _read_after(engine, live, memory)
        choices = programs(live, read, used)
        task = select(live, read, choices, model, mode, config)
        if task is None:
            if collect_prefix and next_prefix is None and live.frontier >= live.ends[4]:
                next_prefix = live.fork(), frozenset(used)
            engine.step(live)
            continue
        used.add(task.key)
        live, detail = execute_program(live, read, task)
        updated = _read_after(engine, live, memory)
        detail["objects"] = _memory(read, updated, detail)
        detail["before_end"] = live.frontier < live.scan.spec[2]
        detail["old_r5"] = read.hypotheses[4].keys[0] if 4 in read.hypotheses else None
        detail["new_r5"] = updated.hypotheses[4].keys[0] if 4 in updated.hypotheses else None
        actions.append(detail)
        got_new_evidence = True
        if collect_prefix and live.frontier >= live.ends[4]:
            return live, frozenset(used)
    if collect_prefix:
        if next_prefix is None:
            raise RuntimeError("The route did not reach the training prefix")
        return next_prefix
    read_at_end = engine.read(live)
    score = live.scan.grade(read_at_end.public, read_at_end.decision)[0]
    old = live.scan.grade(read_at_end.public, read_at_end.old)[0]
    return live, read_at_end, score, old, actions


def _terminal(engine, live, read, task, offset, follow, seen):
    if task is None:
        branch = fork_on_tape(live, offset)
        effects = None
    else:
        branch, effects = execute_program(live, read, task, offset)
    if follow is None:
        while not branch.done:
            engine.step(branch)
        at_end = engine.read(branch)
        grade = branch.scan.grade(at_end.public, at_end.decision)[0]
    else:
        # A reference action is the single real v10 continuation at this
        # decision. This prevents a resume label from retrying the same menu.
        if task is None:
            engine.step(branch)
        branch, public, grade, _, _ = roll(engine, branch, follow["model"], "joint", follow["config"],
                                          initial_seen=seen | ({task.key} if task else set()))
    if not grade["completed_legal"]:
        raise RuntimeError("An evaluated teacher branch did not actually complete the route")
    return grade, effects


def teacher_job(job):
    case, stage, follow = job
    engine = data_v12._ENGINE
    live = Live(case)
    ordinal = int(case.identifier.rsplit("_", 1)[-1])
    if follow is None:
        boundary = (3, 4, 4, 4, 5, 5)[ordinal % 6]
        engine.to_frontier(live, boundary)
        seen = frozenset()
    else:
        live, seen = roll(engine, live, follow["model"], "joint", follow["config"], collect_prefix=True)
    read = engine.read(live)
    choices = [None, *programs(live, read, seen)]
    matrices = []
    for offset in SEARCH:
        matrices.append([terminal_value(_terminal(engine, live, read, a, offset, follow, seen)[0]) for a in choices])
    delta = paired_targets(matrices)
    best = int(np.argmax(delta @ VALUE_WEIGHTS))
    proof, detail = _terminal(engine, live, read, choices[best], VERIFY_TAPE, follow, seen)
    ref, _ = _terminal(engine, live, read, None, VERIFY_TAPE, follow, seen)
    rows = np.stack([action_vector(live, read, a) for a in choices])
    if len(rows) > MAX_MENU:
        raise RuntimeError("A shared program escaped the registered action menu")
    labels = snapshot(live, read, "temperature")["labels"]
    claims = {r+1: [int(labels[i]) if i >= 0 else -99 for i in q.indices] for r, q in read.hypotheses.items()}
    caps = {
        "case": asdict(case), "layout_id": live.scan.tape.layout.identifier, "stage": stage,
        "prefix": [int(live.scan.node.station), int(live.scan.node.sector), int(live.scan.node.tick)],
        "frontier": int(live.frontier), "already_executed": sorted(seen),
        "claims": claims, "actions": [(a.region+1, a.name) if a else (0, "resume") for a in choices],
        "mean_delta": delta.tolist(), "choice_index": best, "proof_action": detail,
        "proof": outcome(proof), "proof_resume": outcome(ref),
    }
    return rows, delta, caps


def pack_teacher(collected):
    rows = np.zeros((len(collected), MAX_MENU, ACTOR_DIM), dtype=np.float32)
    values = np.zeros((len(collected), MAX_MENU, len(VALUE_WEIGHTS)), dtype=np.float32)
    mask = np.zeros(rows.shape[:2], dtype=bool)
    for i, (x, y, _) in enumerate(collected):
        rows[i, :len(x)], values[i, :len(x)], mask[i, :len(x)] = x, y, True
    return {"features": rows, "values": values, "mask": mask}


def fit_actor(fit, calibration, out, tag, seed, start=None, epochs=24):
    torch = _torch_setup(seed)
    from torch.nn import functional as F

    rng = np.random.default_rng(seed + 1)
    center, scale = _norm(fit["features"][fit["mask"]])
    model = net(ACTOR_DIM, (54, 36, len(VALUE_WEIGHTS)))
    if start is not None:
        center, scale = start["mean"], start["scale"]
        with torch.no_grad():
            for layer, w, b in zip((a for a in model if hasattr(a, "weight")), start["weight"], start["bias"], strict=True):
                layer.weight.copy_(torch.from_numpy(w))
                layer.bias.copy_(torch.from_numpy(b))
    opt = torch.optim.AdamW(model.parameters(), lr=.0014 if start is None else .00065, weight_decay=.006, fused=True)
    initial = [p.detach().clone() for p in model.parameters()]
    weights = torch.as_tensor(VALUE_WEIGHTS, device="cuda")

    def encode(group):
        return (torch.as_tensor(((group["features"] - center)/scale).clip(-6., 6.), device="cuda"),
                torch.as_tensor(group["values"], device="cuda"), torch.as_tensor(group["mask"], device="cuda"))

    tx, ty, tm = encode(fit)
    vx, vy, vm = encode(calibration)
    best, history, count = None, [], 0
    for epoch in range(epochs):
        for batch in np.array_split(rng.permutation(len(tx)), max(1, len(tx)//12)):
            ix = torch.as_tensor(batch, device="cuda")
            y, mask = ty[ix], tm[ix]
            z = model(tx[ix])
            z = z - z[:, :1]
            gain = z @ weights
            true = y @ weights
            logits = gain.masked_fill(~mask, -30.)
            goal = F.softmax(true.masked_fill(~mask, -30.)/.15, dim=1)
            regret = -(goal * F.log_softmax(logits/.15, dim=1)).sum(dim=1).mean()
            error = F.huber_loss(z[mask], y[mask], delta=.25)
            loss = .32*regret + error
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4., error_if_nonfinite=True)
            opt.step()
            count += 1
        with torch.no_grad():
            z = model(vx)
            z = z - z[:, :1]
            values = (z @ weights).masked_fill(~vm, -30.)
            pick = values.argmax(dim=1)
            gain = (vy[torch.arange(len(vx), device="cuda"), pick] @ weights).mean().item()
            reg = F.mse_loss(z[vm], vy[vm]).item()
        history.append({"epoch": epoch+1, "utility": gain, "mse": reg})
        key = (gain, -reg)
        if best is None or key > best["key"]:
            ck = _checkpoint(out/f"{tag}.pt", model, opt, rng, epoch=epoch+1, update=count)
            best = {"key": key, "epoch": epoch+1, "bundle": bundle(model, center, scale), "checkpoint_sha256": ck}
    if not all(torch.any(w != p).item() for w, p in zip(initial, model.parameters(), strict=True)):
        raise RuntimeError("The CUDA student did not update its parameters")
    atomic_json(out/f"{tag}.json", plain(best["bundle"]))
    atomic_json(out/f"{tag}_training.json", {"selected_epoch": best["epoch"], "training_layouts": len(tx),
                "calibration_layouts": len(vx), "checkpoint_sha256": best["checkpoint_sha256"], "optimizer_steps": count,
                "parameters": sum(p.numel() for p in model.parameters()), "dev": history,
                "bundle_sha256": digest(out/f"{tag}.json")})
    print("CUDA student", tag, best["epoch"], count, best["key"], flush=True)
    return best["bundle"]


def summarize_teacher(rows):
    by = Counter()
    for r in rows:
        score = np.asarray(r["mean_delta"]) @ VALUE_WEIGHTS
        truth, ref = r["proof"], r["proof_resume"]
        chosen = r["actions"][r["choice_index"]]
        by.update(prefixes=1, real_program_options=len(score)-1, positive_options=sum(score[1:]>1.e-5),
                  negative_options=sum(score[1:] < -1.e-5), neutral_options=sum(np.abs(score[1:]) <= 1.e-5),
                  positive_r5_options=sum(a[0] == 5 and s > 1.e-5 for a, s in zip(r["actions"], score, strict=True)),
                  r5_wrong_with_true_alternative=r["claims"].get(5, r["claims"].get("5", (0, -99)))[0] != 0 and
                      r["claims"].get(5, r["claims"].get("5", (0, -99)))[1] == 0,
                  proof_rescue=truth["strict"] and not ref["strict"],
                  proof_loss=ref["strict"] and not truth["strict"], proof_chosen_strict=truth["strict"],
                  proof_ref_strict=ref["strict"], proof_chosen_wrong=truth["wrong_branch"],
                  proof_ref_wrong=ref["wrong_branch"], selected_program=bool(chosen[0]),
                  selected_r5=chosen[0]==5, prefixes_after_student_program=bool(r["already_executed"]))
    return dict(by)
