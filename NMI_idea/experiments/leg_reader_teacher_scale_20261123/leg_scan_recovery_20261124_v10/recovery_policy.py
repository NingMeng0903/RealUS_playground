"""Goal-conditioned supervised skill initialization and residual PPO."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax

from recovery_task import FEATURE_DIM, options, stats, table, utility
from recovery_teacher import adapt


FORMAT = "public_material_goal_policy_v10"
TEMPERATURE = .78


def _predict(bundle, rows, valid):
    if bundle is None:
        zero = np.zeros(len(valid), dtype=float)
        return zero, np.eye(len(valid), dtype=float)[0], 0.
    weights = bundle["state"]
    x = np.clip((rows - bundle["center"]) / bundle["scale"], -7., 7.)
    z = x @ weights["enc.0.weight"].T + weights["enc.0.bias"]
    z = z * expit(z)
    z = z @ weights["enc.2.weight"].T + weights["enc.2.bias"]
    z = z * expit(z)
    logits = (z @ weights["action.weight"].T + weights["action.bias"])[:, 0] / TEMPERATURE
    joint = logits.copy()
    joint[1:] -= np.log(max(1, valid[1:].sum()))
    probability = softmax(np.where(valid, joint, -np.inf))
    critic = np.r_[z[0], z[valid].mean(axis=0)] @ weights["critic.weight"].T + weights["critic.bias"]
    return logits, probability, float(critic[0])


def read_bundle(path):
    d = json.loads(Path(path).read_text())
    if d["format"] != FORMAT:
        raise ValueError("Unknown recovery policy format")
    for k in ("center", "scale"):
        d[k] = np.asarray(d[k], dtype=np.float32)
    d["state"] = {k: np.asarray(v, dtype=np.float32) for k, v in d["state"].items()}
    return d


def rollout(live, bundle, seed, greedy=True, *, record=True, query=False, snapshot_split=None):
    prng = np.random.default_rng(seed)
    start_st = live.scan.node.station
    sequence, snapshots = [], []
    teacher = None
    before = utility(live.grade()) if record and not live.done else 0.
    while not live.done:
        choices = options(live) if bundle is not None else [(live.base_goal(), None, np.zeros(FEATURE_DIM))]
        rows, available = table(choices)
        logits, probability, value = _predict(bundle, rows, available)
        p = probability.astype(np.float64)
        if greedy:
            probes = np.flatnonzero(available[1:]) + 1
            action = int(probes[np.argmax(logits[probes])]) if len(probes) and logsumexp(
                logits[probes]) - np.log(len(probes)) > logits[0] else 0
        else:
            action = int(prng.choice(len(p), p=p / p.sum()))
        choice, plan, _ = choices[action]
        if query and teacher is None and len(choices) > 1 and (len(live.trace) + seed) % 3 != 0:
            teacher = adapt(live, choices, action, bundle, seed, rollout)
            if teacher is not None:
                teacher.update(step=len(sequence))
        if snapshot_split and choice.is_repair and len(snapshots) < 2:
            snapshots.append(live.snapshot(snapshot_split, "student_recovery_prefix"))
        if live.take(choice, plan) is None:
            raise RuntimeError("The chosen public material goal was not executed")
        if record:
            now = utility(live.grade())
            sequence.append({"x": rows, "mask": available, "action": action, "value": value,
                             "logp": float(np.log(max(1.e-20, probability[action]))), "reward": now - before})
            before = now
    grade = live.grade()
    if not grade["completed_legal"]:
        raise RuntimeError("A recovery policy did not complete under the common budget")
    if snapshot_split:
        scope = "student_terminal_pass" if grade["strict"] else (
            "student_terminal_reader_failure" if grade["acquired_ceiling"] else "student_terminal_capture_failure")
        snapshots.append(live.snapshot(snapshot_split, scope))
    return {"metrics": stats(live, grade), "experience": sequence, "teacher": teacher, "snapshots": snapshots,
            "start_station": int(start_st)}


def _network():
    import torch
    from torch import nn

    class RecoveryPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(FEATURE_DIM, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
            self.action = nn.Linear(32, 1)
            self.critic = nn.Linear(64, 1)
            self.register_buffer("center", torch.zeros(FEATURE_DIM))
            self.register_buffer("scale", torch.ones(FEATURE_DIM))

        def forward(self, x, mask):
            h = self.enc(((x - self.center) / self.scale).clamp(-7., 7.))
            score = self.action(h).squeeze(-1)
            pooled = (h * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            return score, self.critic(torch.cat((h[:, 0], pooled), dim=-1)).squeeze(-1)

    return RecoveryPolicy().cuda()


def _load(net, path):
    import torch
    x = read_bundle(path)
    state = x["state"] | {"center": x["center"], "scale": x["scale"]}
    net.load_state_dict({key: torch.as_tensor(value, device="cuda") for key, value in state.items()})


def _save(net, optimizer, target, info):
    import torch
    from study_v9 import atomic_json, digest
    target.mkdir(parents=True, exist_ok=True)
    state = net.state_dict()
    array = {key: value.detach().cpu().tolist() for key, value in state.items()}
    bundle = {"format": FORMAT, "center": array.pop("center"), "scale": array.pop("scale"), "state": array}
    atomic_json(target / "actor.json", bundle)
    temporary = target / "checkpoint.pending"
    torch.save({"state": state, "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "info": info}, temporary)
    temporary.replace(target / "checkpoint.pt")
    info["actor_sha256"] = digest(target / "actor.json")
    info["checkpoint_sha256"] = digest(target / "checkpoint.pt")
    atomic_json(target / "training.json", info)


def _delta(net, before):
    return float(sum((v.detach() - before[k]).square().sum().item()
                     for k, v in net.named_parameters() if not k.startswith("critic")).__pow__(.5))


def _tensor(rows, key, dtype, torch):
    return torch.as_tensor(np.stack([row[key] for row in rows]), dtype=dtype, device="cuda")


def _log_policy(scores, mask, torch):
    # Decide pending recovery as one group before choosing its goal. Adding
    # more viable coordinates must not amplify total exploration by itself.
    count = mask[:, 1:].sum(dim=1, keepdim=True).clamp_min(1)
    logits = torch.cat((scores[:, :1], scores[:, 1:] - count.log()), dim=1)
    return torch.log_softmax(logits.masked_fill(~mask, -1.e8), dim=1)


def fit_options(rows, target, seed, *, iterations=160, behavior=None):
    """Supervise conditional means and ordering across paired legal goals."""
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    net = _network()
    x = _tensor(rows, "x", torch.float32, torch)
    mask = _tensor(rows, "mask", torch.bool, torch)
    outcomes = _tensor(rows, "values", torch.float32, torch)
    if behavior is None:
        net.center.copy_(x[mask].mean(0))
        net.scale.copy_(x[mask].std(0).clamp_min(.04))
    else:
        _load(net, behavior)
    before = {k: v.detach().clone() for k, v in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=.0014, weight_decay=.0005)
    rank_target = _log_policy(2.7 * outcomes, mask, torch).exp()
    prng = np.random.default_rng(seed)
    losses = []
    for _ in range(iterations):
        idx = prng.choice(len(x), min(len(x), 40), replace=False)
        raw, _ = net(x[idx], mask[idx])
        relative = raw - raw[:, :1]
        goal = outcomes[idx] - outcomes[idx, :1]
        valid = mask[idx]
        error = F.smooth_l1_loss(relative[valid], goal[valid], reduction="none")
        quant = (error * (1 + .5 * goal[valid].abs().clamp_max(2.))).mean()
        logp = _log_policy(raw / TEMPERATURE, valid, torch)
        rank = -(rank_target[idx] * logp).sum(dim=1).mean()
        loss = quant + .30 * rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    out = {"stage": "paired_goal_supervision", "rows": len(rows), "goals": int(mask.sum()), "optimizer_steps": iterations,
           "loss_start": float(np.mean(losses[:12])), "loss_end": float(np.mean(losses[-12:])), "device": str(x.device),
           "actor_delta_l2": _delta(net, before), "parameters": sum(p.numel() for p in net.parameters())}
    if out["actor_delta_l2"] <= 0:
        raise RuntimeError("GPU skill training did not update parameters")
    _save(net, opt, target, out)
    return out


def update_online(rollouts, behavior, target, seed, *, iterations=64):
    """Use actually executed residual choices and paired adaptation labels."""
    import torch
    from torch.nn import functional as F

    net = _network()
    _load(net, behavior)
    before = {k: v.detach().clone() for k, v in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=.00032, weight_decay=.0005)
    rows, advantages, returns, targets, alternatives = [], [], [], [], []
    for ep in rollouts:
        seq = ep["experience"]
        adv = np.zeros(len(seq), dtype=float)
        running = 0.
        for i in reversed(range(len(seq))):
            next_value = seq[i + 1]["value"] if i + 1 < len(seq) else 0.
            running = seq[i]["reward"] + next_value - seq[i]["value"] + .92 * running
            adv[i] = running
        record = ep["teacher"]
        lookup = record["step"] if record is not None else -1
        for i, (row, estimate) in enumerate(zip(seq, adv, strict=True)):
            rows.append(row)
            advantages.append(estimate)
            returns.append(estimate + row["value"])
            targets.append(record["measured_gain"] if i == lookup else 0.)
            alternatives.append(record["alternative"] if i == lookup else -1)
    x = _tensor(rows, "x", torch.float32, torch)
    mask = _tensor(rows, "mask", torch.bool, torch)
    action = _tensor(rows, "action", torch.long, torch)
    old = _tensor(rows, "logp", torch.float32, torch)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device="cuda")
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(.003)
    returned = torch.as_tensor(returns, dtype=torch.float32, device="cuda")
    pairs = torch.as_tensor(targets, dtype=torch.float32, device="cuda")
    proposed = torch.as_tensor(alternatives, dtype=torch.long, device="cuda")
    prng = np.random.default_rng(seed)
    losses = []
    for _ in range(iterations):
        p = prng.choice(len(x), min(len(x), 96), replace=False)
        raw, value = net(x[p], mask[p])
        logp = _log_policy(raw / TEMPERATURE, mask[p], torch)
        chosen = logp.gather(1, action[p, None]).squeeze(1)
        ratio = (chosen - old[p]).exp()
        surrogate = torch.minimum(ratio * adv[p], ratio.clamp(.82, 1.18) * adv[p]).mean()
        entropy = -(logp.exp() * logp).sum(dim=1).mean()
        trained = proposed[p] >= 0
        other = raw.gather(1, proposed[p].clamp_min(0)[:, None]).squeeze(1) - raw[:, 0]
        imitation = (trained * F.smooth_l1_loss(other, pairs[p], reduction="none")).sum() / trained.sum().clamp_min(1)
        loss = -surrogate + .42 * F.smooth_l1_loss(value, returned[p]) - .012 * entropy + .22 * imitation
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    info = {"stage": "residual_ppo", "complete_routes": len(rollouts), "actual_macro_decisions": len(rows),
            "teacher_paired_visits": int((proposed >= 0).sum()), "optimizer_steps": iterations, "device": str(x.device),
            "mean_loss": float(np.mean(losses)), "actor_delta_l2": _delta(net, before)}
    if info["actor_delta_l2"] <= 0:
        raise RuntimeError("GPU residual PPO did not update parameters")
    _save(net, opt, target, info)
    return info
