from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from core import grade,History,pick,req,scan,smooth_suffix,terminal
from tracker import Reader


@lru_cache(maxsize=8)
def reader(env):
    return Reader(env)


def geometry_history(env,centers,frames):
    # The public or ablation anatomy is explicit. Both use identical nominal stress/noise assumptions.
    nominal=scan.Trial(2,0,"stress",n=1)
    nominal.layout=SimpleNamespace(centers=centers,links=env.scanner._anatomy.links[2],shared_vertices=())
    forecast=scan.ref.reference.GeometryForecast(env,nominal)
    h=History(env,nominal,forecast,oracle_geometry=True)
    h.append(frames)
    return h


def observer_outcomes(env,trial,item,extra):
    node,h,cut,spans=extra
    goal,entry,*_=item
    frames=node.reader.frames
    # These probabilities are recomputed with only the frames already acquired as of each evaluation time.
    public=reader(env)
    oracle=geometry_history(env,trial.layout.centers,[])
    out={}
    for mode in ("registered","known_geometry"):
        bands=[];probs=None
        for start,end in spans:
            if mode=="registered":
                probs=public.read(frames[:end])
                i0=0
            else:
                oracle.append(frames[:end]);i0=start;probs=smooth_suffix(oracle,start)
            ids=range(start,end)
            bands.append(grade(frames,pick(frames,probs,ids,i0),ids))
        allp=probs if mode=="registered" else oracle.smoothed()
        scores=terminal(node,h,entry,cut,bands,other=allp)
        out[mode]=scores
    return out
