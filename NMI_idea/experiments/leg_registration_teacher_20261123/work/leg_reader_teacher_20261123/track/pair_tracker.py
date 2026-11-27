"""Public joint assignment smoother for target, accompanying artery and mixed echoes."""
from pathlib import Path
from types import SimpleNamespace
import sys

import joblib
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / 'leg_follow_condition'
sys.path.insert(0, str(PREVIOUS / 'learned_reader'))
from identify import FeatureReader

TARGET, ARTERY, MIXED, OTHER = range(4)
MAX_STATES = 64


@njit(cache=True)
def enumerate_states(count, unary, residual, pair_distance, missing_floor):
    """Form mutually exclusive ordered assignments, including pair fusion and misses."""
    n = len(count)
    slots = np.full((n, MAX_STATES, 3), -1, dtype=np.int64)
    weights = np.zeros((n, MAX_STATES))
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = count[i]
        missing_t = max(missing_floor, 1. - min(1., np.sum(unary[i, :m, TARGET])))
        missing_a = max(missing_floor, 1. - min(1., np.sum(unary[i, :m, ARTERY])))
        k = 0
        for a in range(-1, m):
            for b in range(-1, m):
                if a >= 0 and a == b:
                    continue
                wa = missing_t if a < 0 else unary[i, a, TARGET] / max(.012, 1. - unary[i,a,TARGET])
                wb = missing_a if b < 0 else unary[i, b, ARTERY] / max(.012, 1. - unary[i,b,ARTERY])
                slots[i,k,0] = a
                slots[i,k,1] = b
                # A simultaneous observation contains a directly visible pair vector.
                # The broad shape prior admits the two anatomies exchanging lateral order.
                if a >= 0 and b >= 0 and pair_distance[i] < 12.:
                    gap = residual[i,a,0] - residual[i,b,1]
                    wa *= .40 + .60*np.exp(-.5*np.sum(gap**2)/5.5**2)
                weights[i,k] = wa*wb
                k += 1
        if pair_distance[i] < 14.:
            for a in range(m):
                slots[i,k,2] = a
                weights[i,k] = .60 * unary[i,a,MIXED] / max(.025, 1.-unary[i,a,MIXED])
                k += 1
        total = max(1.e-12, np.sum(weights[i,:k]))
        weights[i,:k] /= total
        sizes[i] = k
    return slots, weights, sizes


@njit(cache=True, inline='always')
def phase_compatibility(previous, following, residual, gap, i, strength, sigma2, connected):
    """A continuous identity scores a same-role residual more highly than a jump."""
    # Contact pose is already removed by projecting to the material chart. The
    # extra variance in deep or nonadjacent views represents geometry interpolation.
    log_weight = 0.
    for r in range(2):
        a, b = previous[r], following[r]
        if a >= 0 and b >= 0 and connected[r]:
            dx = residual[i-1,a,r,0] - residual[i,b,r,0]
            dz = residual[i-1,a,r,1] - residual[i,b,r,1]
            level = min(2.7, (dx**2 + dz**2) / (2.*sigma2))
            log_weight += strength * (.70 - level)
    if previous[0] >= 0 and previous[1] >= 0 and following[0] >= 0 and following[1] >= 0:
        # Stable relative separation adds ordering evidence after a rigid shift.
        if connected[0] and connected[1]:
            dx = (residual[i-1,previous[0],0,0] - residual[i-1,previous[1],1,0]
                  - residual[i,following[0],0,0] + residual[i,following[1],1,0])
            dz = (residual[i-1,previous[0],0,1] - residual[i-1,previous[1],1,1]
                  - residual[i,following[0],0,1] + residual[i,following[1],1,1])
            log_weight += .38*strength*(.60-min(2.5,(dx**2 + dz**2)/(4.*sigma2)))
    if previous[2] >= 0 and following[2] >= 0 and connected[0] and connected[1]:
        dx = gap[i-1,previous[2],0] - gap[i,following[2],0]
        dz = gap[i-1,previous[2],1] - gap[i,following[2],1]
        log_weight += .50*strength*(.70-min(2.5,(dx**2 + dz**2)/(2.*sigma2)))
    return log_weight


@njit(cache=True)
def posterior(states, weights, sizes, count, ticks, stations, components, residual, gap, strength, rho):
    """Run exact finite-state forward/backward inference without locking a winning assignment."""
    n = len(count)
    forward = np.zeros_like(weights)
    backward = np.zeros_like(weights)
    forward[0] = weights[0]
    transitions = np.ones((n,MAX_STATES,MAX_STATES),dtype=np.float32)
    for i in range(1,n):
        dt = max(1, ticks[i]-ticks[i-1])
        ds = abs(stations[i]-stations[i-1])
        connected = (components[i]==components[i-1]) & (components[i]>=0)
        stationary = dt>30 or ds>7. or not np.any(connected)
        continuity = min(1., 3.5/dt) * np.exp(-.09*ds**2)
        variance = (1.2 + 1.35*(1.-rho**dt) + .37*ds)**2
        for a in range(sizes[i-1]):
            for b in range(sizes[i]):
                if not stationary:
                    transitions[i,a,b] = np.exp(strength*phase_compatibility(
                        states[i-1,a],states[i,b],residual,gap,i,continuity,variance,connected))
                forward[i,b] += forward[i-1,a]*transitions[i,a,b]
        forward[i] *= weights[i]
        forward[i] /= max(1.e-12,np.sum(forward[i]))
    backward[-1,:sizes[-1]] = 1.
    output = np.zeros((n,7,4))
    for i in range(n-1,-1,-1):
        if i<n-1:
            for a in range(sizes[i]):
                for b in range(sizes[i+1]):
                    backward[i,a] += transitions[i+1,a,b]*weights[i+1,b]*backward[i+1,b]
            backward[i] /= max(1.e-12,np.sum(backward[i]))
        state = forward[i]*backward[i]
        state /= max(1.e-12,np.sum(state))
        for j in range(sizes[i]):
            for r in range(3):
                k = states[i,j,r]
                if k>=0:
                    output[i,k,r] += state[j]
        for k in range(count[i]):
            output[i,k,OTHER] = max(0.,1.-np.sum(output[i,k,:3]))
    return output


def labels_to_pair(labels):
    return np.where(labels==0,TARGET,np.where(labels==1,ARTERY,np.where(labels==-2,MIXED,OTHER)))


def registered_classes(matrix):
    """Start the four classes from the previous public eight-origin reader."""
    return np.column_stack((matrix[:,0], matrix[:,1], matrix[:,6],
                            np.maximum(0.,1.-matrix[:,0]-matrix[:,1]-matrix[:,6])))


def contextual_classes(matrix, old):
    """Split the old reader's remaining mass using the public registration."""
    public = registered_classes(matrix).astype(np.float64)
    denominator = np.maximum(1.e-8, 1.-public[:,0])
    public[:,1:] *= ((1.-old)/denominator)[:,None]
    bad = denominator < 1.e-7
    public[bad,1:] = 0.
    public[bad,OTHER] = 1. - old[bad]
    public[:,0] = old
    return public


def group_arrays(prepared, start, indices, probabilities):
    counts = prepared.counts[start:]
    nominal = prepared.nominal[start:,:2]
    deltas = prepared.echoes[start:,:,None,:] - nominal[:,None]
    residual = np.einsum('nirc,nck->nirk',np.nan_to_num(deltas,nan=100.),prepared.transform[start:])
    gap = (residual[:,:,0] + residual[:,:,1])*.5
    distance = np.linalg.norm(nominal[:,0]-nominal[:,1],axis=-1)
    distance = np.nan_to_num(distance,nan=100.)
    cube = np.zeros((len(counts),7,4))
    cube[indices[:,0]-start,indices[:,1]] = probabilities
    cube[~np.isfinite(nominal[:,0,0]),:,TARGET] = 0.
    cube[~np.isfinite(nominal[:,1,0]),:,ARTERY] = 0.
    cube[:,:,OTHER] = np.maximum(0.,1.-cube[:,:,:3].sum(axis=-1))
    return cube, counts, residual, gap, distance


class PairReader:
    """Expose candidate probabilities for already-acquired sanitized frame groups."""

    def __init__(self, surface, model_path=None):
        self.extractor = FeatureReader(SimpleNamespace(surface=surface))
        old_record = joblib.load(PREVIOUS/'learned_reader/frozen/model.joblib')
        self.old = old_record['model']
        where = Path(model_path) if model_path is not None else HERE/'frozen.joblib'
        params = joblib.load(where)
        self.head = params['model']
        self.track_strength = params['track_strength']
        self.head_weight = params['head_weight']
        self.track_weight = params['track_weight']
        self.threshold = params['threshold']

    @staticmethod
    def sanitize(frames):
        return FeatureReader.public(frames)

    def paired_scores(self, public_frames, groups):
        blocks, snapshots = self.extract(public_frames,groups)
        if snapshots is None:
            return [(q,np.empty((0,4))) for q in blocks]
        matrix, ix, data, start = snapshots
        old = self.old.predict(matrix)
        head = self.head.predict_proba(matrix)
        result = infer(matrix, ix, data, start, old, head, self.head_weight,
                       self.track_weight, self.track_strength)
        return [(q,result[q[:,0]-start,q[:,1]]) for q in blocks]

    def scores(self, public_frames, groups):
        return [(ix,p[:,TARGET]) for ix,p in self.paired_scores(public_frames,groups)]

    def best(self, public_frames, groups):
        out=[]
        for q,p in self.scores(public_frames,groups):
            if len(p):
                k=int(np.argmax(p))
                out.append((int(q[k,0]),int(q[k,1]),float(p[k])))
            else:
                out.append(None)
        return out

    def extract(self, public_frames, groups):
        if any(f[k] is not None for f in public_frames for k in (3,6,7)):
            raise ValueError('Only sanitized public frame histories are accepted')
        groups = [list(g) for g in groups]
        if any(i < 0 or i >= len(public_frames) for g in groups for i in g):
            raise ValueError('Candidate frames must lie in the observed prefix')
        if not any(groups):
            return [np.empty((0,2),dtype=int) for _ in groups],None
        start = max(0,min(i for g in groups for i in g)-42)
        prepared = self.extractor.reader.prepare(public_frames)
        main = range(start,len(public_frames))
        matrix,ix = self.extractor.feature_sets(public_frames,[main])[0]
        counts = prepared.counts
        offsets = np.r_[0,np.cumsum(counts)]
        ids = [(np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in g])
                if g else np.empty(0,dtype=int)) for g in groups]
        blocks = [ix[q-offsets[start]] for q in ids]
        if not len(ix):
            return blocks,None
        return blocks,(matrix,ix,prepared,start)


def infer(matrix, ix, prepared, start, old, head, head_weight, track_weight, track_strength):
    context = contextual_classes(matrix,old)
    unary = head_weight*head + (1.-head_weight)*context
    cube,count,residual,gap,distance = group_arrays(prepared,start,ix,unary)
    if track_weight==0.:
        return cube
    states,weights,sizes = enumerate_states(count,cube,residual,distance,.15)
    good = prepared.quality[prepared.quality>.42]
    fraction = np.mean(good) if len(good) else .85
    rho = np.clip(.40 + 3.25*(.92-fraction),.4,.78)
    smoothed = posterior(states,weights,sizes,count,prepared.ticks[start:],prepared.stations[start:],
                         prepared.components[start:,:2],residual,gap,track_strength,rho)
    return (1.-track_weight)*cube + track_weight*smoothed
