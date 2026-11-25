"""Compact shared A/B scorer and fixed-trajectory selection on CUDA."""

from dataclasses import dataclass
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp, softmax

from relations_v14 import MAX_PAIRS


@dataclass(frozen=True)
class Rule:
    probability: float
    margin: float
    reject: float
    accept: float


BASELINE_RULE = Rule(1.01, 1.01, 1.01, 1.01)
# Relative constellations and track aggregation remain active in the ablation.
TOPOLOGY_COLUMNS = (17, 18, 19, 25, 26)


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(f)


def _digest(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
    return h.hexdigest()


def materialize(records):
    at = len(records)
    dim = next(np.asarray(r['state']).shape[-1] for r in records if len(r['state']))
    arrays = {'state': np.zeros((at, MAX_PAIRS, 2, dim), np.float32),
              'position': np.zeros((at, MAX_PAIRS, 2), np.float32),
              'motif': np.zeros((at, MAX_PAIRS, 2), np.float32),
              'valid': np.zeros((at, MAX_PAIRS, 2), bool),
              'truth': np.zeros((at, MAX_PAIRS, 3), bool),
              'rows': np.full((at, MAX_PAIRS, 2), -1, int),
              'label': np.full((at, MAX_PAIRS, 2), -99, int),
              'used': np.zeros((at, MAX_PAIRS), bool)}
    for i, rec in enumerate(records):
        count = len(rec['state'])
        for name in ('state', 'position', 'motif', 'valid', 'truth', 'rows', 'label'):
            if count:
                arrays[name][i, :count] = rec[name]
        arrays['used'][i, :count] = True
    arrays['truth'][:, :, 2] |= ~arrays['used']
    return arrays


def _layer(x, state, key):
    return x @ state[key + '.weight'].T + state[key + '.bias']


def _silu(x):
    return x*expit(x)


def numpy_logits(model, data):
    params = {k: np.asarray(v, np.float32) for k, v in model['weights'].items()}
    center = np.asarray(model['center'], np.float32)
    scale = np.asarray(model['scale'], np.float32)
    mask = np.asarray(model['mask'], np.float32)
    features = np.clip((data['state'] - center) / scale, -6., 6.) * mask
    v = data['valid']
    h = _silu(_layer(features, params, 'encode1'))
    h = _silu(_layer(h, params, 'encode2')) * v[..., None]
    delta = np.concatenate((h, h[..., ::-1, :]), axis=-1)
    relative = _layer(_silu(_layer(delta, params, 'candidate1')), params, 'candidate2')[..., 0]
    stem = np.concatenate((h.mean(axis=-2), np.abs(h[..., 0, :]-h[..., 1, :]),
                           v.sum(axis=-1, keepdims=True)*.5), axis=-1)
    unknown = _layer(_silu(_layer(stem, params, 'unknown1')), params, 'unknown2')
    motif = data['motif'] if model['variant'] == 'graph' else 0.
    candidate = 3.5*np.tanh(relative/2.) + data['position'] + 1.35*motif
    candidate = np.where(v, candidate, -1.e4)
    return np.concatenate((candidate, unknown), axis=-1)


def decide(p, records, data, rule):
    """Choose accepted A, a supported B, or defer. All decisions are readout."""
    changed = []
    for i, r in enumerate(records):
        base, take = r['baseline_row'], r['baseline_row']
        used = data['used'][i]
        if used.any():
            pp = p[i, used]
            rows, valid = data['rows'][i, used], data['valid'][i, used]
            gain = pp[:, 1] - np.maximum(pp[:, 0], pp[:, 2])
            proposals = np.flatnonzero(valid[:, 1] & (pp[:, 1] >= rule.probability) & (gain >= rule.margin))
            if len(proposals):
                take = int(rows[proposals[np.argmax(gain[proposals])], 1])
            elif pp[:, 2].min() >= rule.reject:
                take = -1
            elif base < 0 and pp[:, 0].max() >= rule.accept:
                take = int(rows[0, 0])
        changed.append(take)
    return changed


def outcome(records, chosen):
    rows = []
    for rec, idx in zip(records, chosen, strict=True):
        ids = list(rec['baseline_ids'])
        if idx != rec['baseline_row']:
            if idx < 0:
                ids[5] = -99
            else:
                labels = {k: v for p, ys in zip(rec['rows'], rec['label'], strict=True)
                          for k, v in zip(p, ys, strict=True) if k >= 0}
                ids[5] = labels[idx]
        rows.append({'case': rec['case'], 'band': rec['band'], 'labels': ids,
                     'strict': int(all(y == 0 for y in ids)), 'wrong': int(any(y > 0 for y in ids)),
                     'r6_correct': int(ids[5] == 0), 'r6_artery': int(ids[5] == 1),
                     'r6_wrong_branch': int(ids[5] > 0), 'r6_reject': int(ids[5] == -99),
                     'r6_ghost_or_blend': int(ids[5] in (-1, -2)),
                     'changed': int(idx != rec['baseline_row']), 'r6_row': idx,
                     'possible': int(rec['candidate_has_target']), 'captured': int(rec['captured'])})
    return rows


def totals(rows):
    keys = ('strict','wrong','r6_correct','r6_artery','r6_wrong_branch','r6_reject',
            'r6_ghost_or_blend','changed','possible','captured')
    return {k: sum(row[k] for row in rows) for k in keys}


def tune(p, records, arrays):
    best = BASELINE_RULE
    measures = totals(outcome(records, [r['baseline_row'] for r in records]))
    def criterion(m):
        return (m['strict'], -m['wrong'], m['r6_correct'], -m['r6_artery'], -m['changed'])
    for prob, margin, reject, accept in product((.48,.57,.66,.75,.84), (.025,.10,.22,.35),
                                               (.80,.91,1.01), (.58,.72,1.01)):
        rule = Rule(prob, margin, reject, accept)
        score = totals(outcome(records, decide(p, records, arrays, rule)))
        if criterion(score) > criterion(measures):
            best, measures = rule, score
    return best, measures, criterion(measures)


def _temperature(logits, data):
    used = data['used']
    y = data['truth'][used]
    z = logits[used]
    def loss(temp):
        s = z/temp
        return np.mean(logsumexp(s,axis=1) - logsumexp(np.where(y, s, -1.e4),axis=1))
    return float(minimize_scalar(loss, bounds=(.5, 3.5), method='bounded').x)


def train(splits, out, *, seed, epochs=48):
    import torch
    from torch import nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this fit')
    device = torch.device('cuda:0')
    data = {key: materialize(value) for key, value in splits.items()}
    train = data['fit']
    features = train['state'][train['valid']]
    center, scale = features.mean(0), features.std(0).clip(.04, None)
    dim = len(center)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode1 = nn.Linear(dim, 44)
            self.encode2 = nn.Linear(44, 24)
            self.candidate1 = nn.Linear(48, 24)
            self.candidate2 = nn.Linear(24, 1)
            self.unknown1 = nn.Linear(49, 24)
            self.unknown2 = nn.Linear(24, 1)

        def forward(self, x, position, motif, valid):
            h = F.silu(self.encode2(F.silu(self.encode1(x)))) * valid.unsqueeze(-1)
            pair = torch.cat((h, h.flip(-2)), dim=-1)
            relative = self.candidate2(F.silu(self.candidate1(pair))).squeeze(-1)
            z = (3.5*torch.tanh(relative/2.) + position + 1.35*motif).masked_fill(~valid, -1.e4)
            symmetric = torch.cat((h.mean(-2), (h[..., 0, :]-h[..., 1, :]).abs(),
                                   valid.sum(-1,keepdim=True)*.5),dim=-1)
            unknown = self.unknown2(F.silu(self.unknown1(symmetric)))
            return torch.cat((z,unknown),dim=-1)

    center_gpu = torch.as_tensor(center, device=device)
    scale_gpu = torch.as_tensor(scale, device=device)
    loaded = {k: torch.as_tensor(train[k], device=device) for k in
              ('state','position','motif','valid','truth','used')}
    recover = np.asarray([r['baseline_ids'][5] != 0 for r in splits['fit']])
    importance = torch.as_tensor(np.where(recover, 1.85, 1.), device=device, dtype=torch.float32)
    result = {}
    for name in ('relative','graph'):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Head().to(device)
        initial = [p.detach().clone() for p in net.parameters()]
        opt = torch.optim.AdamW(net.parameters(), lr=.0018, weight_decay=.006)
        mask = np.ones(dim, np.float32)
        if name == 'relative':
            mask[list(TOPOLOGY_COLUMNS)] = 0.
        m = torch.as_tensor(mask, device=device)
        best = None
        history = []
        for epoch in range(1, epochs+1):
            net.train()
            epoch_loss = 0.
            for ix in np.array_split(rng.permutation(len(recover)), max(1, len(recover)//48)):
                batch = torch.as_tensor(ix, device=device)
                x = ((loaded['state'][batch]-center_gpu)/scale_gpu).clamp(-6.,6.) * m
                pos = loaded['position'][batch]
                # Drop or perturb only the bounded unary on a fraction of
                # batches. Thus graphs remain usable under pose drift.
                perturb = torch.rand((len(ix),1,1),device=device)
                pos = torch.where(perturb < .20, torch.zeros_like(pos), pos)
                pos = torch.where(perturb > .77, (pos + .34*torch.randn_like(pos)).clamp(-1.15,1.15), pos)
                valid, yes, used = (loaded[k][batch] for k in ('valid','truth','used'))
                motif = loaded['motif'][batch] if name=='graph' else 0.
                logits = net(x,pos,motif,valid)
                error = logits.logsumexp(-1) - logits.masked_fill(~yes,-1.e4).logsumexp(-1)
                error = ((error * used).sum(-1) / used.sum(-1).clamp(min=1))
                loss = (error*importance[batch]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.)
                opt.step()
                epoch_loss += loss.detach().item()*len(ix)
            if epoch not in (1, 6, 12, 24, 36, epochs):
                continue
            net.eval()
            model = {'variant':name, 'center':center.tolist(), 'scale':scale.tolist(), 'mask':mask.tolist(),
                     'weights':{k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()}}
            model['temperature'] = _temperature(numpy_logits(model,data['cal']),data['cal'])
            pred = softmax(numpy_logits(model,data['dev'])/model['temperature'],axis=-1)
            rule, score, key = tune(pred,splits['dev'],data['dev'])
            history.append({'epoch':epoch, 'loss':epoch_loss/len(recover), 'dev':score,
                            'rule':rule.__dict__})
            print(name,epoch, history[-1],flush=True)
            if best is None or key > best['key']:
                with torch.no_grad():
                    take = data['dev']
                    example = {k: torch.as_tensor(take[k][:2],device=device) for k in ('state','position','motif','valid')}
                    values = ((example['state']-center_gpu)/scale_gpu).clamp(-6,6)*m
                    gpus = net(values,example['position'],example['motif'] if name=='graph' else 0.,example['valid'])
                    cpus = numpy_logits(model,{k:v[:2] for k,v in take.items()})
                    parity = float(np.max(abs(gpus.cpu().numpy()-cpus)))
                    if parity > .001:
                        raise RuntimeError('Export does not match CUDA')
                model['rule'], model['selected_epoch'] = rule.__dict__, epoch
                best = {'key': key, 'selected':score, 'parity_max':parity}
                save_json(out/f'{name}.json',model)
                file = out/f'{name}.pt'
                temp = file.with_suffix('.tmp')
                torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(), 'epoch':epoch,
                            'rng':rng.bit_generator.state, 'torch_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state(), 'seed':seed},temp)
                temp.replace(file)
        torch.cuda.synchronize()
        changed = all(torch.any(old!=new).item() for old,new in zip(initial,net.parameters(),strict=True))
        if not changed or not all(p.is_cuda for p in net.parameters()):
            raise RuntimeError('Failed to update actual CUDA parameters')
        result[name] = best | {'history':history,'parameters':sum(p.numel() for p in net.parameters()),
                               'updated_cuda_weights':True,'model_sha256':digest(out/f'{name}.json'),
                               'checkpoint_sha256':digest(out/f'{name}.pt')}
    result['gpu'] = torch.cuda.get_device_name(device)
    result['seed'] = seed
    save_json(out/'training.json',result)
    return result
