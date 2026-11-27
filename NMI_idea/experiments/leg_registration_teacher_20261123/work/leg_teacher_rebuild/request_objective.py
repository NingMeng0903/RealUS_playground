"""Terminal request utility and paired teacher feedback. This module uses no simulator imports."""
import numpy as np


def utility(answer,penalty):
    if penalty<0:
        raise ValueError("The wrong-delivery cost must be nonnegative")
    cutoff=penalty/(1.+penalty)
    if answer["role"] is None or answer["p"]<cutoff:
        return 0.
    return 1. if answer["role"]==0 else -penalty


def two_view_score(answers,penalty=1.):
    # Two separately requested images have equal weight in this engineering check.
    if len(answers)!=2:
        raise ValueError("The benchmark requires an entry-site and a fresh downstream image")
    return sum(utility(a,penalty) for a in answers)/len(answers)


def advantage_feedback(model,penalty=1.,baseline="continue"):
    ref=model[baseline]
    result={}
    for name,option in model.items():
        if len(option)!=len(ref) or not ref:
            raise ValueError("Every branch must use the same indexed independent innovations")
        deltas=np.array([two_view_score(active["answers"],penalty)-
                         two_view_score(base["answers"],penalty)
                         for active,base in zip(option,ref)])
        proper=[active["delta_brier"]-base["delta_brier"] for active,base in zip(option,ref)
                if active["delta_brier"] is not None and base["delta_brier"] is not None]
        by_view={}
        for j,label in enumerate(("entry","fresh")):
            effect=np.array([utility(a["answers"][j],penalty)-utility(b["answers"][j],penalty)
                             for a,b in zip(option,ref)])
            by_view[label]={"mean":float(effect.mean()),
                            "sem":float(effect.std(ddof=1)/np.sqrt(len(effect))) if len(effect)>1 else None}
        elapsed=[a["ticks"]-b["ticks"] for a,b in zip(option,ref)]
        result[name]=dict(n=len(deltas),paired_advantage=float(deltas.mean()),
            sem=float(deltas.std(ddof=1)/np.sqrt(len(deltas))) if len(deltas)>1 else None,
            target_probability_of_gain=float(np.mean(deltas>0)),target_probability_of_loss=float(np.mean(deltas<0)),
            paired_brier_gain=float(np.mean(proper)) if proper else None,
            paired_view_advantages=by_view,extra_ticks=float(np.mean(elapsed)))
    return result
