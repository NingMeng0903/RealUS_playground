"""Rank with the deployed residual score and calibrate whole-history winners."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import trajectory_v13 as tr
from collect_v13 import SNAPSHOT_COLUMNS
from fit_v12 import _checkpoint, _norm, _torch_setup, bundle, hydrate, net, plain
from recovery_reader import ROLE_LABELS
from study_v9 import atomic_json, digest, freeze, read_json, verify_shard
from track_state import linear


RESIDUAL = 3.8
SCALES = (256, 512, 1024)
STEPS = 400
FRACTION = .70
REGIONS = (3, 4, 5)


def _semantic(labels):
    mapping = np.full(10, -1, dtype=np.int64)
    mapping[np.asarray(ROLE_LABELS)+2] = np.arange(len(ROLE_LABELS))
    return mapping[np.asarray(labels).astype(int)+2]


def load_data(out, stage):
    rows, checks = [], {}
    base = out/"history"/stage
    for marker in sorted(base.glob("*/commit.json")):
        path, cases = verify_shard(marker.parent)
        checks[str(marker.relative_to(out))] = digest(marker)
        with np.load(path, allow_pickle=False) as z:
            d = {k:z[k] for k in z.files}
        ix = pair = 0
        for meta in cases:
            views = {}
            for m in meta["snapshots"]:
                start, end = d["offsets"][ix:ix+2]
                v = {k:d[k][start:end] for k in SNAPSHOT_COLUMNS}
                v["semantic"] = _semantic(v["label"])
                views[m["scope"]] = m | v
                ix += 1
            end = pair + meta["pair_count"]
            between = {k[5:]: v[pair:end] for k,v in d.items() if k.startswith("pair_")}
            rows.append({"log":meta,"views":views,"pair":between})
            pair = end
        if ix+1 != len(d["offsets"]) or pair != len(d["pair_label"]):
            raise RuntimeError("Reader shard candidate or pair offsets disagree")
    return rows, checks


def _adjust(head, x, baseline):
    z = linear(head, x.astype(np.float32))
    return baseline + RESIDUAL*np.tanh(z[:,0])


def winner_features(region, y, baseline, track, best):
    value = y[best]
    competing = y[track != track[best]]
    rival = competing.max() if len(competing) else value - 3.
    return np.array([value, np.clip(value-rival, -6., 6.), np.log1p(len(y))/6.,
                     np.log1p(np.sum(track==track[best]))/4., baseline[best], float(region==5)], dtype=np.float32)


def winners(snapshot, adjusted):
    for region in REGIONS:
        ix = np.flatnonzero(snapshot["region"]==region)
        if len(ix):
            best = int(np.argmax(adjusted[ix]))
            row = int(ix[best])
            f = winner_features(region, adjusted[ix], snapshot["logit"][ix], snapshot["track"][ix], best)
            yield region,row,f


def logistic(model, x):
    z = np.asarray(x, dtype=np.float64)
    return expit((z-model["center"])/model["scale"] @ model["weights"] + model["bias"])


def fit_winners(head, cal):
    features, target, cells = [], [], []
    for row in cal:
        s = row["views"]["student"]
        for r,i,f in winners(s, _adjust(head, s["x"], s["logit"])):
            features.append(f)
            target.append(s["label"][i] == 0)
            cells.append((row["log"]["layout_id"], r, int(s["label"][i]), s["early_ok"]))
    x, y = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mean, std = _norm(x)
    ux = (x-mean)/std

    def objective(weights):
        w, b = weights[:-1], weights[-1]
        z = ux @ w + b
        diff = expit(z)-y
        penalty = .035
        return (np.mean(np.logaddexp(0.,z)-y*z) + .5*penalty*(w@w),
                np.r_[ux.T@diff/len(y) + penalty*w, diff.mean()])

    opt = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B")
    if not opt.success:
        raise RuntimeError("Whole-history winner calibration failed")
    fit = {"center":mean,"scale":std,"weights":opt.x[:-1],"bias":float(opt.x[-1])}
    v = logistic(fit,x)
    layouts = list(dict.fromkeys(k[0] for k in cells))
    thresholds = (.08,.20,.35,.50,.65)
    trial = []
    for cutoff in thresholds:
        labels = {k: [-99,-99,-99] for k in layouts}
        early = {}
        for (k,r,label,complete),confidence in zip(cells,v,strict=True):
            early[k] = complete
            labels[k][r-3] = label if confidence >= cutoff else -99
        strict = sum(all(z==0 for z in labels[k]) and early[k] for k in early)
        wrong = sum(any(z>0 for z in row) for row in labels.values())
        trial.append((int(strict),-int(wrong),cutoff))
    fit["threshold"] = max(trial)[2]
    fit["winner_brier"] = float(np.mean((v-y)**2))
    fit["winner_nll"] = float(np.mean(-y*np.log(v.clip(1.e-8,1.)) - (1.-y)*np.log((1.-v).clip(1.e-8,1.))))
    fit["calibrated_rows"] = len(y)
    fit["threshold_sweep"] = trial
    return fit


def select_snapshot(model, snap, mode):
    if mode in ("v10","v12"):
        scores = snap["old_winner" if mode=="v10" else "v12_winner"]
        selected = []
        for r in REGIONS:
            ix = np.flatnonzero(snap["region"]==r)
            selected.append(int(snap["label"][ix[np.argmax(scores[ix])]]) if len(ix) and scores[ix].max()>=0. else -99)
        return selected
    selected = [-99,-99,-99]
    for r,i,f in winners(snap,_adjust(model, snap["x"],snap["logit"])):
        if logistic(model["calibration"], f) >= model["calibration"]["threshold"]:
            selected[r-3] = int(snap["label"][i])
    return selected


def measure(model, rows, mode="new"):
    counts = Counter()
    for row in rows:
        snap = row["views"]["student"]
        y = select_snapshot(model,snap,mode)
        frozen = row["log"][mode] if mode in ("v10","v12") else row["log"]["v10"]
        ids = frozen["selected_ids"][:3]+y
        counts.update(layouts=1,strict=int(all(v==0 for v in ids)),wrong=int(any(v>0 for v in ids)),
                      r6_correct=int(ids[5]==0),r6_artery=int(ids[5]==1),r6_reject=int(ids[5]==-99))
    return counts


def deployment(read, model, mode, *, old_logits=None):
    if mode == "v10":
        return read.old
    if mode == "v12":
        return read.decision
    pub = read.public
    if not len(pub["frames"]):
        return read.old
    x, baseline = tr.observation(read), tr.old_logits(read)
    s = _adjust(model,x,baseline)
    if old_logits:
        for i,key in enumerate(zip(pub["frames"],pub["slots"],strict=True)):
            if key in old_logits:
                s[i] = old_logits[key]
    old = read.old
    score, peak, target = old["score"].copy(),old["winner_score"].copy(),old["target"].copy()
    late = pub["regions"]>=3
    score[late], peak[late], target[late] = -1., -1., expit(s[late])
    snapshot = {"logit":baseline,"region":pub["regions"],"track":read.decision["track_id"]}
    for r,i,f in winners(snapshot,s):
        c = float(logistic(model["calibration"], f))
        peak[i] = c
        if c >= model["calibration"]["threshold"]:
            score[i] = c
    return read.decision | {"score":score,"target":target,"winner_score":peak}


def choose_rows(rows, rng, mode, n):
    # A complete shuffled pass visits every layout before failed histories
    # are resampled. All conditions use the identical available layout roster.
    order = rng.permutation(n).tolist()
    weights = np.ones(n)
    if mode != "uniform":
        weights = np.array([2.75 if r["log"]["v12"]["selected_ids"][5]!=0 else
                            (1.30 if not r["log"]["v12"]["strict"] else .75) for r in rows])
    return order + rng.choice(n, STEPS*12-n, p=weights/weights.sum()).tolist()


def make_batch(rows, selected, rng, mode):
    groups = []
    pairs = {k:[] for k in ("before","after","b_logit","a_logit","label","evidence","region")}
    for i in selected:
        row = rows[i]
        extra = mode=="edited" and "failure_edit" in row["views"] and rng.random()<FRACTION
        s = row["views"]["failure_edit" if extra else "student"]
        eligible = [r for r in REGIONS if (s["region"]==r).any()]
        if not eligible:
            continue
        r = 5 if (5 in eligible and rng.random()<.57) else int(rng.choice(eligible))
        ix = s["region"]==r
        groups.append((s["x"][ix],s["logit"][ix],s["semantic"][ix],r))
        p = row["pair"]
        take = len(p["label"])
        if take:
            ix = rng.choice(take,min(10,take),replace=False)
            for k in pairs:
                pairs[k].append(p[k][ix])
    n = max(len(x) for x, *_ in groups)
    width = groups[0][0].shape[1]
    x = np.zeros((len(groups),n,width),dtype=np.float16)
    old = np.zeros(x.shape[:2],dtype=np.float32)
    y = np.full(old.shape,-100,dtype=np.int64)
    reg = []
    for i, (z,b,t,r) in enumerate(groups):
        k = len(z)
        x[i,:k],old[i,:k],y[i,:k] = z,b,t
        reg.append(r)
    p = {k: np.concatenate(v) for k,v in pairs.items() if v}
    return x,old,y,np.asarray(reg),p


def fit_variant(rows,cal,dev,out,variant,mode,seed):
    torch = _torch_setup(seed)
    from torch.nn import functional as F

    rng = np.random.default_rng(seed)
    raw = np.concatenate([r["views"]["student"]["x"][::17] for r in rows]).astype(np.float32)
    center, scale = _norm(raw)
    del raw
    net_ = net(len(center),(64,48,9))
    initial = [p.detach().clone() for p in net_.parameters()]
    opt = torch.optim.AdamW(net_.parameters(),lr=.00155,weight_decay=.005,fused=True)
    center_gpu,scale_gpu = [torch.as_tensor(a,device="cuda") for a in (center,scale)]

    def norm(a):
        x = torch.as_tensor(a,device="cuda",dtype=torch.float32)
        return ((x-center_gpu)/scale_gpu).clamp(-6.,6.)

    frozen = choose_rows(rows,rng,mode,len(rows))
    best, perf = None,[]
    for step in range(1,STEPS+1):
        batch = frozen[(step-1)*12:step*12]
        x,baseline,y,regions,p = make_batch(rows,batch,rng,mode)
        x = norm(x)
        base,truth = [torch.as_tensor(a,device="cuda") for a in (baseline,y)]
        reg = torch.as_tensor(np.where(regions==5,1.45,1.).astype(np.float32),device="cuda")
        z = net_(x)
        valid,good = truth!=-100,truth==0
        shifted = base+RESIDUAL*z[...,0].tanh()
        pool = shifted.masked_fill(~valid,-40.)
        null = torch.zeros_like(pool[:,:1])
        all_rank = torch.logsumexp(torch.cat((pool,null),dim=1),dim=1)
        correct = torch.logsumexp(shifted.masked_fill(~good,-40.),dim=1)
        rank_loss = (reg*(all_rank - torch.where(good.any(dim=1),correct,0.))).mean()
        bs = F.binary_cross_entropy_with_logits(shifted[valid],good[valid].float())
        role = F.cross_entropy(z[...,1:][valid],truth[valid])
        loss = rank_loss + .23*bs + .42*role
        if p:
            pair_y = torch.as_tensor(p["label"]==0,dtype=torch.float32,device="cuda")
            mask = torch.as_tensor(p["evidence"],dtype=torch.bool,device="cuda")
            before, after = [net_(norm(p[k]))[:,0] for k in ("before","after")]
            prior = torch.as_tensor(p["b_logit"],device="cuda") + RESIDUAL*before.tanh()
            revised = torch.as_tensor(p["a_logit"],device="cuda") + RESIDUAL*after.tanh()
            stability = F.huber_loss(revised,prior.detach(),reduction="none",delta=.40)
            correction = F.binary_cross_entropy_with_logits(revised,pair_y,reduction="none")
            loss += .25*torch.where(mask,correction,stability).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        if step%(STEPS//4)==0:
            head = bundle(net_,center,scale)
            head["calibration"] = fit_winners(head,cal)
            metrics = measure(head,dev)
            key = (metrics["strict"],-metrics["wrong"],metrics["r6_correct"])
            perf.append({"step":step,"loss":float(loss.detach()),"developer":dict(metrics)})
            if best is None or key>best["key"]:
                cpu_cuda = float(np.max(np.abs(linear(head,x[0,:5].detach().cpu().numpy()*scale+center) -
                                              net_(x[0,:5]).detach().cpu().numpy())))
                if cpu_cuda>.003:
                    raise RuntimeError("Exported reader disagrees with CUDA")
                saved = _checkpoint(out/f"{variant}.pt",net_,opt,rng,steps=step)
                best = {"key":key,"optim_step":step,"cpu_cuda_abs":cpu_cuda,"checkpoint_sha256":saved,"reader":head}
            print("CUDA reader",variant,step,dict(metrics),flush=True)
    if not all(torch.any(a!=b).item() for a,b in zip(initial,net_.parameters(),strict=True)):
        raise RuntimeError("Reader did not update its CUDA weights")
    return best,perf


def run(args):
    out = args.out
    directory = out/"models"
    directory.mkdir(parents=True,exist_ok=True)
    deck, ledger = {}, {}
    for stage in ("cal","reader_dev","fit"):
        deck[stage],checks = load_data(out,stage)
        ledger.update(checks)
    ids = [{r["log"]["layout_id"] for r in v} for v in deck.values()]
    if sum(map(len,ids))!=len(set.union(*ids)) or len(deck["fit"])<max(SCALES):
        raise RuntimeError("Reader cohorts overlap or collection is incomplete")
    permutation = np.random.default_rng(501_681).permutation(len(deck["fit"]))
    order = [deck["fit"][i] for i in permutation]
    scenarios = [(f"random_{n}",n,"uniform") for n in SCALES]
    scenarios += [("failure_512",512,"failure"),("edited_512",512,"edited")]
    head_lock = freeze(out,"reader_training_freeze.json",{"source_sha256":digest(Path(__file__)),
       "data_contract_sha256":digest(out/"data_freeze.json"),"shard_locks":ledger,"optimizer_steps":STEPS,
       "data_arms":scenarios,"r6_group_weight":1.45,"fit_seed":501_681,
       "layout_order":[v["log"]["layout_id"] for v in order]})
    cal,dev = deck["cal"],deck["reader_dev"]
    comparison = {m:dict(measure(None,dev,m)) for m in ("v10","v12")}
    chosen = max(comparison, key=lambda m:(comparison[m]["strict"],-comparison[m]["wrong"],comparison[m]["r6_correct"]))
    baseline = comparison[chosen]
    oldkey = (baseline["strict"],-baseline["wrong"],baseline["r6_correct"])
    best = None
    for name,n,mode in scenarios:
        f = directory/f"{name}.json"
        if f.exists():
            r = read_json(f)
            r["reader"] = hydrate(r["reader"])
            best_case = r
        else:
            m,trace = fit_variant(order[:n],cal,dev,directory,name,mode,9586)
            best_case = m | {"trace":trace,"layouts":n,"mode":mode}
            atomic_json(f,plain(best_case))
        score = dict(measure(best_case["reader"],dev))
        comparison[name] = score
        rank = (score["strict"],-score["wrong"],score["r6_correct"])
        if best is None or rank>best["key"]:
            best = {"name":name,"key":rank,"score":score}
    # A newly trained reader replaces the incumbent only on a disjoint
    # development criterion. Both frozen choices remain in final tests.
    terminal = "new" if best["key"]>oldkey else chosen
    freeze(out,"reader_freeze.json",{"training_sha256":head_lock,"candidate":best["name"],
           "candidate_sha256":digest(directory/f"{best['name']}.json"),"terminal_mode":terminal,
           "v10_dev":comparison["v10"],"v12_dev":comparison["v12"],"candidate_dev":best["score"]})
    atomic_json(out/"reader_comparison.json",{"candidate":best,"terminal_mode":terminal,"comparison":comparison})


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=tr.ROOT/"run")
    run(p.parse_args())
