"""Verify source locks, complete student visits, and layout-paired results."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from study_v8 import COHORTS, DEFAULT_PACK, ROOT, ROUTES, V6_RUN, V7, atomic_json, check_hashes, committed, digest, pinned, read_json, verify_shard
import motion_v8 as sim
from evidence_v8 import SCORE
from value_v8 import teacher_summary
import numpy as np


def verified_protocols(out, atlas):
    protocols={}
    for name in ("capture","reader","evidence","cycles"):
        path=out/f"{name}_protocol.json"
        data=read_json(path)
        check_hashes(data["sources"])
        if data["atlas_sha256"]!=digest(atlas):
            raise RuntimeError(f"Frozen atlas changed for {name}")
        protocols[name]=digest(path)
        if "reader_sha256" in data and data["reader_sha256"]!=digest(out/"reader_bundle.json"):
            raise RuntimeError(f"Frozen reader changed for {name}")
        if "edge_bundle_sha256" in data and data["edge_bundle_sha256"]!=digest(out/"edge_model_v8.json"):
            raise RuntimeError(f"Frozen edge scorer changed for {name}")
    return protocols


def recalc(rows, report, *, evidence=False):
    export=[]
    include=("strict","wrong_branch","joint","selected_ids","capture","old_pool_correct","suffix_valid","time_s","travel_mm","mm","longest_mm")
    for route in ROUTES:
        deck=[r for r in rows if r["case"]["task"]==route]
        ref=sim.REFERENCE if evidence else "mpc"
        index=np.random.default_rng(180_411_381).integers(0,len(deck),(3500,len(deck)))
        for m,summary in report[route]["arms"].items():
            contrast=summary["vs_mpc"]
            for field in ("strict","wrong_branch","joint"):
                control=np.array([[r["outcomes"][band][ref][field] for band in SCORE] for r in deck],dtype=float)
                trial=np.array([[r["outcomes"][band].get(m,r["outcomes"][band][ref])[field] for band in SCORE] for r in deck],dtype=float)
                d=trial-control; effect=d[:,1:].mean(axis=1); a=contrast[field]["actual"]
                assert a["added_or_increased"]==int((d[:,0]>0).sum()) and a["removed_or_decreased"]==int((d[:,0]<0).sum())
                interval=np.quantile(effect[index].mean(axis=1),[.025,.975]); q=contrast[field]["independent_tape_expected"]
                np.testing.assert_allclose([q["delta"],q["lower95"],q["upper95"]],[effect.mean(),*interval])
                assert summary["scores"][field]==trial[:,0].sum()
        for row in deck:
            scores={}
            for band in SCORE:
                for a,value in row["outcomes"][band].items():
                    assert value["completed_legal"] and value["layout_id"]==row["layout_id"]
                    scores.setdefault(a,{})[band]={k:value[k] for k in include}
            export.append({"route":route,"layout":row["layout_id"],"scores":scores})
    return export


def label_contract(group, phase, continuation):
    count=Counter(); actions=Counter()
    for r in group:
        a=r["actual_trace"]; init=r["origin"]; reach=r["reached"]; target=r["target"]
        assert r["actual_student"]["completed_legal"] and a["first"]["pending"]==a["second"]["pending"]
        assert reach["station"]==init["station"]+2 and reach["count"]>init["count"]
        expected=init if phase==1 else reach
        assert target["phase"]==phase and target["prefix_station"]==expected["station"] and target["prefix_frames"]==expected["count"]
        assert target["continuation_sha256"]==continuation
        assert r["layout_id"]==target["layout"]
        baseline={z["future_offset"]:z for z in target["physical_search"][sim.REFERENCE]}
        for name,samples in target["samples"].items():
            for delta,event in zip(samples,target["physical_search"][name],strict=True):
                ref=baseline[event["future_offset"]]; second=event["trace"]["second"]
                assert delta[1]==int(event["strict"])-int(ref["strict"]) and delta[2]==int(event["wrong_branch"])-int(ref["wrong_branch"])
                assert second["pending"]==a["first"]["pending"]
                if phase==1:
                    assert event["trace"]["first"]["path"][-1][0]==expected["station"]+2
                count["search_executions"]+=1
                if name!=sim.REFERENCE:
                    count[f"strict_sign_{delta[1]}"]+=1
                actions[name]+=1
    return dict(count) | {"search_programs":dict(actions)}


def prediction_export(rows):
    def compact(result):
        estimates={name:{k:record[k] for k in ("predicted","strict_lower","wrong_upper","score","gate","execution_legal") if k in record}
                   for name,record in result.get("estimates",{}).items()}
        return {"executed":result.get("executed"),"permitted":result.get("permitted"),"estimates":estimates}
    return [{"route":r["case"]["task"],"layout":r["layout_id"],"phase1":compact(r["student_phase1_predictions"]),
             "phase2":{band:compact(r["actors"][band]["student"]["second_prediction"]) for band in SCORE}}
            for r in rows]


def policy_boundary(out, atlas, readers):
    sim.initialize(atlas,readers,read_json(out/"reader_bundle.json"),read_json(out/"edge_model_v8.json"))
    checks=[]
    from study_v8 import roster
    for case in list(roster("phase2_fit"))[:4]:
        ctx=sim.setup(case); normal=sim.context_state(ctx,1)
        node=ctx["node"].fork(); altered=[]; n=0
        for frame in node.reader.frames:
            label=np.asarray(frame[3]); replacement=np.where(label>=0,(label+1)%6,label)
            altered.append((*frame[:3],replacement,*frame[4:])); n+=int(np.sum(label!=replacement))
        node.reader.frames=altered
        changed=sim.context_state(ctx | {"node":node},1)
        np.testing.assert_array_equal(normal["features"],changed["features"])
        assert [g.as_dict() for g in normal["plans"].values()]==[g.as_dict() for g in changed["plans"].values()]
        checks.append({"route":case.task,"altered_truth_labels":n,"same_public_scores_and_plan":True})
    short={sim.REFERENCE:[[0.,0.,0.]]*6,"test":[[0.,1.,0.]]*6}
    long={sim.REFERENCE:[[0.,0.,0.]]*32,"test":[[0.,1.,0.]]*6+[[0.,0.,0.]]*26}
    a,low=teacher_summary(short,sim.REFERENCE); b,high=teacher_summary(long,sim.REFERENCE)
    assert a==sim.REFERENCE and b=="test"
    return {"permuted_label_contracts":checks,"teacher_six_perfect_bands":low["estimates"]["test"]["gate"],
            "teacher_six_rescues_of_thirty_two":high["estimates"]["test"]["gate"]}


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"run");p.add_argument("--atlas-pack",type=Path,default=DEFAULT_PACK)
    a=p.parse_args(); out=a.out
    locks=verified_protocols(out,a.atlas_pack)
    readers=pinned(a.atlas_pack)
    sets={n:committed(out,n) for n in COHORTS}
    layouts=[r["layout_id"] for rows in sets.values() for r in rows]
    assert len(layouts)==len(set(layouts))
    assert all(r["case"]["source"] in (0,2) for rows in sets.values() for r in rows)
    for key in ("world_seed","tape_seed","action_seed"):
        assert len({r["case"][key] for rows in sets.values() for r in rows})==len(layouts)
    prior=set()
    for root in (V6_RUN,V7/"run"):
        for path in root.glob("*/*/commit.json"):
            _,rows=verify_shard(path.parent)
            prior.update(r["layout_id"] for r in rows)
    assert not set(layouts)&prior
    assessed=read_json(out/"reader_results.json")
    assert assessed["rules"]==locks["reader"] and assessed["frozen_reader_sha256"]==digest(out/"reader_bundle.json")
    for mode in ("phase2","phase1"):
        fit=read_json(out/f"{mode}_fit.json")
        assert fit["cuda_optimizer_step_verified"] and fit["best_validation_checkpoint_restored"] and fit["optimizer_steps"]>0
        assert fit["best_epoch"]==int(np.argmin(fit["validation_losses"]))+1
        assert fit["checkpoint_sha256"]==digest(Path(fit["checkpoint"])) and fit["bundle_sha256"]==digest(out/f"{mode}_bundle.json")
        split=fit["split"]
        groups=[set(split[k]) for k in ("fit_layouts","validation_layouts","calibration_layouts")]
        assert split["layout_disjoint"] and set.union(*groups)=={r["layout_id"] for r in sets[f"{mode}_fit"]}
    frozen=read_json(out/"student_freeze.json")
    p2=frozen["phase2_sha256"]
    assert p2==digest(out/"phase2_bundle.json") and frozen["phase1_sha256"]==digest(out/"phase1_bundle.json")
    effects={f"phase{which}":label_contract(sets[f"phase{which}_fit"],which,p2 if which==1 else None) for which in (1,2)}
    teacher_count=0
    for row in sets["student_test"]:
        assert row["teacher_search"]["continuation_sha256"]==p2
        teacher_count+=sum(len(v) for v in row["teacher_search"]["samples"].values())
    student=read_json(out/"cycles_results.json"); evidence=read_json(out/"evidence_results.json")
    assert student["protocol_sha256"]==locks["cycles"] and evidence["protocol_sha256"]==locks["evidence"]
    n=len(sets["phase1_fit"])+len(sets["phase2_fit"])
    assert n==student["real_student_training_episodes"]
    paired={"fixed_question":recalc(sets["evidence_test"],evidence["routes"],evidence=True),
            "two_stage_student":recalc(sets["student_test"],student["test"])}
    diagnostic=read_json(out/"ambiguity_results.json")
    rule=read_json(out/"ambiguity_protocol.json")
    check_hashes(rule["source"])
    assert diagnostic["protocol_sha256"]==digest(out/"ambiguity_protocol.json")
    assert rule["reader_sha256"]==digest(out/"reader_bundle.json") and rule["edge_sha256"]==digest(out/"edge_model_v8.json")
    d_ids={r["layout_id"] for r in diagnostic["rows"]}
    assert len(d_ids)==sum(rule["sizes"]) and not d_ids & (set(layouts)|prior)
    if diagnostic["comparisons"] is not None:
        found=[r for r in diagnostic["rows"] if r["balanced_found"]]
        paired["balanced_public_diagnostic"]=recalc(found,diagnostic["comparisons"],evidence=True)
    atomic_json(out/"paired_outcomes.json",paired)
    atomic_json(out/"student_test_predictions.json",prediction_export(sets["student_test"]))
    verified={"layout_counts":{k:len(v) for k,v in sets.items()},"all_distinct_v8_layouts":len(layouts),"excluded_prior_layouts":len(prior),
       "separate_balanced_public_diagnostic_layouts":len(d_ids),"balanced_public_diagnostic_found":diagnostic["balanced_counts"],
       "student_training_episodes":n,"teaching_development":effects,"teacher_heldout_search_executions":teacher_count,
       "fresh_evidence_legal_executions":sum(len(v) for r in sets["evidence_test"] for v in r["outcomes"].values()),
       "policy_scoring_legal_executions":len(sets["student_test"])*9,"all_frozen_hashes_checked":True,
       "bootstrap_recalculated_from_shards":True,"protocols":locks,"paired_sha256":digest(out/"paired_outcomes.json"),
       "student_test_predictions_sha256":digest(out/"student_test_predictions.json"),
       "contracts":policy_boundary(out,a.atlas_pack,readers)}
    atomic_json(out/"verification.json",verified)
    print({k:v for k,v in verified.items() if k not in ("protocols","contracts")},flush=True)


if __name__=="__main__":
    main()
