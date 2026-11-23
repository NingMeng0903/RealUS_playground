"""Regression fixtures and terminal readiness audit from the frozen v6 run."""

from argparse import ArgumentParser
from pathlib import Path
from study_v7 import DEFAULT_PACK, ROOT, V6_RUN, atomic_json, load_shard, pinned, read_json, roster
from causal_reader import choose
from goal_v7 import normalise_valid, self_test
from live_programs import prefix_setup
from reader_v7 import candidate_delivery, fuse
from role_fusion import infer_roles, split_roles
from value_v7 import teacher_choice
import motion_v7 as sim
import numpy as np


def core():
    scores=np.array([.62,.77]); retained=normalise_valid(scores,[True,True]); padded=normalise_valid([.62,.77,-np.inf,1.e8], [True,True,True,False])
    np.testing.assert_array_equal(retained,padded[:2]); assert np.all(np.isneginf(padded[2:]))
    p=np.array([[.9,.1,0.,0.,0.,0.,0.,0.],[.1,.9,0.,0.,0.,0.,0.,0.]])
    pub={"frames":np.array([0,1]),"slots":np.zeros(2,dtype=int),"stations":np.array([0.,1.]),
         "sectors":np.array([0,1]),"points":np.zeros((2,2)),"quality":np.zeros(2),"regions":np.zeros(2,dtype=int)}
    e=np.tile([1.,0.,0.,0.,0.,0.,0.,0.],(2,1)); links=np.ones((1,4,6),dtype=bool)
    graph={"edges":np.array([[0,1,.99]])}; old=infer_roles(pub,p,e,links,None,[0.,0.],graph=graph)
    fixed=fuse(pub,p,e,links,None,np.zeros(2),graph)
    np.testing.assert_array_equal(fixed["role"],fixed["local_role"])
    np.testing.assert_array_equal(fixed["object_role"],fixed["local_role"])
    assert np.all(fixed["support"]==0.) and np.all(fixed["view_counts"]==0.) and float(old["support"][0])>0.
    return {"legal_normalization_invariant":True,"normalised_valid":retained.tolist(),
            "old_zero_quality_support":old["support"].tolist(),"corrected_zero_quality_support":fixed["support"].tolist(),
            "corrected_identity_preserved":True,"planned_goal_toy":self_test()}


def object_rank():
    prior=read_json(V6_RUN/"reader_freeze.json")["calibration"]["groin_to_distal_calf"]["objects"]["joint"]
    p=np.array([[.00999,.00001,0.,0.,0.,0.,.99,0.],[.45,.45,0.,0.,0.,0.,.10,0.]])
    role=split_roles(p)["role"]
    pub={"frames":np.arange(2),"slots":np.zeros(2),"stations":np.ones(2),"sectors":np.array([1,3]),
         "points":np.array([[0.,0.],[4.,0.]]),"quality":np.array([.4,.8]),"regions":np.zeros(2,dtype=int)}
    g={"role":role,"object_role":role,"local_role":role,"unary":p,"usable":np.array([.01,.9]),"track_ids":np.arange(2),
       "view_counts":np.ones(2),"support":np.zeros(2)}
    hand,_,_=choose(pub,g,"objects",prior); joint=candidate_delivery(pub,g,prior); direct=int(joint.argmax())
    assert hand.tolist()==[0] and direct==1
    return {"handwritten_winner":int(hand[0]),"direct_joint_winner":direct,"joint_scores":joint.tolist(),"physical_mass":p[:,:6].sum(axis=1).tolist()}


def old_stop():
    events=[]
    for f in sorted((V6_RUN/"closed_test").glob("*/commit.json")):
        _, rows=load_shard(f.parent)
        for row in rows:
            for band, arms in row["outcomes"].items():
                action=arms["objects_stop"]; stop=action["completion_stop"]
                if stop is not None:
                    deploy=action["old_at_exit"]
                    events.append({"case":row["case"]["identifier"],"route":row["case"]["task"],"band":band,
                        "object_id":stop["old_candidate_id"],"object_readiness":stop["old_delivery_probability"],
                        "deployed":deploy,"same_id":deploy is not None and deploy["id"]==stop["old_candidate_id"]})
    actual=[r for r in events if r["band"]=="actual"]
    return {"stops_actual":len(actual),"deployed_would_reject_actual":sum(r["deployed"] is None or not r["deployed"]["submitted"] for r in actual),
            "different_winner_actual":sum(not r["same_id"] for r in actual),"events":events}


def small_n():
    data={"reference":[[0.,0.,0.]]*2,"candidate":[[1.,1.,0.]]*2}
    chosen,report=teacher_choice(data,"reference")
    return {"two_identical_rescues":report,"selected":chosen}


def public_only(out, atlas):
    sim.prepare(atlas, pinned(atlas), read_json(out/"observation_model.json"), read_json(out/"bridge_model.json"))
    checks = []
    for case in list(roster("student_wave0"))[:4]:
        reference = prefix_setup(case)
        node = reference["node"].fork()
        changed = []
        count = 0
        for frame in node.reader.frames:
            labels = np.asarray(frame[3]); shuffled = np.where(labels >= 0, (labels + 1) % 6, labels)
            count += int(np.sum(labels != shuffled))
            changed.append((*frame[:3], shuffled, *frame[4:]))
        node.reader.frames = changed
        altered = reference | {"node": node}
        np.testing.assert_array_equal(sim.actor_features(reference), sim.actor_features(altered))
        a, _ = sim.propose(reference, reference["node"], sim.make_reader(reference, True))
        b, _ = sim.propose(altered, node, sim.make_reader(altered, True))
        assert (a.as_dict() if a else None) == (b.as_dict() if b else None) and count > 0
        checks.append({"route": case.task, "hidden_candidate_labels_changed": count, "public_features_and_goal_identical": True})
    return checks


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    result={"mechanisms":core(),"object_selection":object_rank(),"v6_stopping_audit":old_stop(),"bounded_teacher":small_n()}
    if all((args.out/f"{m}_model.json").exists() for m in ("observation", "bridge")):
        result["public_only"] = public_only(args.out, args.atlas_pack)
    atomic_json(args.out/"contracts.json",result)
    print({k:(list(v) if isinstance(v,dict) else v) for k,v in result.items()},flush=True)
