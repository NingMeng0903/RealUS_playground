"""Shared closed-loop executor and revisable coverage for the recovery study."""

from dataclasses import dataclass, field, replace

import numpy as np

from recovery_world import FRAME_COUNT, REFERENCE_KNEE_DEG
from controller import (  # Loaded from the prior recovery experiment.
    BINS, CENTER_MM, EDGES, END, FRAMES_PER_BLOCK, IDENTITY_ACCEPT, MAX_TRAVEL,
    START, Candidate, Controller, Record, State, longitudinal, station_bin,
)
from primitive import BLOCK_STEPS, FRAME_STRIDE, SAMPLE_DT_S, Plan, admissible


QUALIFIED_EVIDENCE = "qualified"
LEGACY_EVIDENCE = "legacy"
VISIBLE_EVIDENCE = "visible"
EVIDENCE_MODES = (QUALIFIED_EVIDENCE, LEGACY_EVIDENCE, VISIBLE_EVIDENCE)
QUALITY_ACCEPT = IDENTITY_ACCEPT


@dataclass(frozen=True)
class Witness:
    frame: int
    slot: int
    station: float
    identity: np.ndarray  # P(slot was the target | static hypothesis, its own packet).
    centered: np.ndarray  # The geometry test under each of the same hypotheses.

    def quality(self, belief):
        return float(belief @ (self.identity * self.centered))


@dataclass
class ScanState(State):
    ledger: tuple = (None,) * BINS
    legacy_covered: int = 0
    image_candidates: tuple | None = None
    capture_support: tuple | None = None

    def query_bin(self):
        if self.capture_support is not None:
            for j in self.uncovered():
                if self.belief @ self.capture_support[j] < IDENTITY_ACCEPT:
                    return j
        return super().query_bin()


@dataclass
class ScanRecord(Record):
    detailed: bool = False
    certificates: list = field(default_factory=list)
    retractions: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def copy(self):
        return replace(self, packets=self.packets.copy(), motion=self.motion.copy(),
                       decisions=self.decisions.copy(), accepts=self.accepts.copy(),
                       blocks=self.blocks.copy(), certificates=self.certificates.copy(),
                       retractions=self.retractions.copy(), references=self.references.copy())


@dataclass(frozen=True)
class FreeCandidate(Candidate):
    request: tuple
    bow: float
    renderer: str = "waypoint"


def smoothstep(t):
    return t**3 * (10. + t * (-15. + 6. * t)), 30. * t**2 * (1. - t)**2


def waypoint(start, velocity, goal, bow):
    """Quintic boundary motion plus a displacement at the middle of the block."""
    t = np.linspace(0., 1., BLOCK_STEPS + 1)
    blend, slope = smoothstep(t)
    init = t - 6. * t**3 + 8. * t**4 - 3. * t**5
    init_slope = 1. - 18. * t**2 + 32. * t**3 - 15. * t**4
    shape = 64. * t**3 * (1. - t)**3
    shape_slope = 192. * t**2 * (1. - t)**2 * (1. - 2. * t)
    return Plan(0, start + (goal - start) * blend + velocity * init + bow * shape,
                (goal - start) * slope + velocity * init_slope + bow * shape_slope)


class AdaptiveController(Controller):
    def __init__(self, world, sections, coefficients, learned=None, *, evidence_mode=QUALIFIED_EVIDENCE):
        super().__init__(world, sections, coefficients)
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"unknown evidence mode: {evidence_mode}")
        self.learned = learned
        self.evidence_mode = evidence_mode

    def initial_scan(self, scene, *, detailed=False, start_s=START, start_x=None):
        center = self.world.nominal(start_s)[0] if start_x is None else start_x
        x = float(np.clip(center, -30., 0.))
        state = ScanState(self.world.prior.copy(), start_s, x, angle=REFERENCE_KNEE_DEG)
        if self.evidence_mode == VISIBLE_EVIDENCE:
            state.image_candidates = (((), np.empty((0, len(state.belief)))),) * BINS
            state.capture_support = (np.zeros_like(state.belief),) * BINS
        record = self.virtual_record(state, detailed=detailed)
        self.observe(scene, state, record, False)
        return state, record

    @staticmethod
    def virtual_record(state, *, detailed=False):
        return ScanRecord(detailed=detailed, motion=[[state.frame * .1, state.s, state.x,
                                                     state.vs, state.vx, state.angle]])

    def observe(self, scene, state, record, alias=False):
        packet, roles, truth = self.world.observe(scene, state.frame, state.s, state.x)
        mass, conditional = self.world.likelihood(packet, state.s, state.x, state.angle)
        posterior = state.belief * mass
        state.belief = posterior / posterior.sum()
        association = state.belief @ conditional
        slot = int(np.argmax(association)) if len(association) and max(association) >= IDENTITY_ACCEPT else -1
        state.slot, state.seen = slot, len(packet)
        state.residual = float(packet[slot, 0]) if slot >= 0 else 0.
        state.missed = 0 if slot >= 0 else state.missed + 1
        state.ambiguous = state.ambiguous + 1 if slot < 0 and len(packet) else 0

        row = {"frame": state.frame, "s": state.s, "x": state.x, "q": state.angle,
               "y": packet.tolist(), "association": association.tolist(), "accepted_slot": slot,
               "roles_truth": roles.tolist(), "target_truth": truth.tolist()}
        record.packets.append(row)
        if state.image_candidates is not None:
            self.retain_visible(state, record, conditional)
        else:
            self.retain_centered(state, record, conditional, roles, truth)
        query = state.query_bin()
        point = float(np.mean(EDGES[query:query + 2]))
        state.query_identity = self.world.identity_prob(state.belief, point, state.angle)
        if record.detailed:
            row["belief"] = state.belief.tolist()
            if state.image_candidates is not None:
                row["visible_ledger"] = [None if w is None else [w.frame, w.slot] for w in state.ledger]
            elif self.evidence_mode == QUALIFIED_EVIDENCE:
                row["qualified_coverage"] = state.covered

    def retain_visible(self, state, record, conditional, *, confirm_at_capture=False):
        bins = list(state.image_candidates)
        j = station_bin(state.s)
        slots = np.arange(conditional.shape[1])
        if confirm_at_capture:
            slots = slots[state.belief @ conditional >= IDENTITY_ACCEPT]
        if j >= 0 and len(slots):
            existing, weights = bins[j]
            centered = abs(self.world.target_grid(state.s, state.angle)[:, 0] - state.x) <= CENTER_MM
            retained = conditional[:, slots].T
            new = tuple(Witness(state.frame, int(slot), state.s, identity, centered)
                        for slot, identity in zip(slots, retained, strict=True))
            bins[j] = existing + new, np.vstack((weights, retained))
        state.image_candidates = tuple(bins)
        active = []
        support = []
        for j, (options, weights) in enumerate(bins):
            support.append(weights.max(axis=0, initial=0.))
            scores = weights @ state.belief
            winner = None
            if len(scores) and scores.max() >= IDENTITY_ACCEPT:
                best = int(np.flatnonzero(scores >= max(IDENTITY_ACCEPT, scores.max() - 1e-9))[0])
                winner = options[best]
            old = state.ledger[j]
            if winner is not old:
                if old is not None:
                    record.retractions.append({"frame": state.frame, "bin": j, "witness": old.frame,
                                               "slot": old.slot, "posterior": float(state.belief @ old.identity)})
                if winner is not None:
                    event = {"frame": winner.frame, "bin": j, "slot": winner.slot, "accepted_at": state.frame}
                    offset = winner.frame - record.packets[0]["frame"]
                    if offset >= 0:
                        source = record.packets[offset]
                        event["identity_ok"] = source["roles_truth"][winner.slot] == 0
                        event["center_ok"] = abs(source["target_truth"][0] - source["x"]) <= CENTER_MM
                    else:
                        event["predicted_identity"] = winner.identity
                    record.certificates.append(event)
            active.append(winner)
        state.capture_support = tuple(support)
        state.ledger = tuple(active)
        state.covered = sum(1 << j for j, w in enumerate(active) if w is not None)
        recent = max((w for w in active if w is not None), key=lambda w: w.frame, default=None)
        state.last_trusted = None if recent is None else recent.station

    def retain_centered(self, state, record, conditional, roles, truth):
        qualified = self.evidence_mode == QUALIFIED_EVIDENCE
        slot = state.slot
        active = list(state.ledger)
        retracted = False
        for j, witness in enumerate(active):
            if witness is not None:
                score = float(state.belief @ witness.identity)
                if score < IDENTITY_ACCEPT:
                    record.retractions.append({"frame": state.frame, "bin": j,
                                               "witness": witness.frame, "posterior": score})
                    active[j] = None
                    retracted = True
        j = station_bin(state.s)
        if slot >= 0 and abs(state.residual) <= CENTER_MM and j >= 0:
            if not qualified:
                state.last_trusted = state.s
            event = {"frame": state.frame, "bin": j, "slot": slot,
                     "identity_ok": bool(roles[slot] == 0), "center_ok": bool(abs(truth[0] - state.x) <= CENTER_MM)}
            if not state.legacy_covered & (1 << j):
                state.legacy_covered |= 1 << j
                record.accepts.append(event)
            if qualified or active[j] is None:
                means = self.world.target_grid(state.s, state.angle)
                incoming = Witness(state.frame, slot, state.s, conditional[:, slot].copy(),
                                   abs(means[:, 0] - state.x) <= CENTER_MM)
                current = active[j]
                if current is None:
                    active[j] = incoming
                    record.certificates.append(event)
                else:
                    old_quality, new_quality = current.quality(state.belief), incoming.quality(state.belief)
                    if new_quality > old_quality + 1e-9:
                        record.retractions.append({"frame": state.frame, "bin": j, "witness": current.frame,
                                                   "reason": "higher_centered_probability",
                                                   "old_quality": old_quality, "new_quality": new_quality})
                        active[j] = incoming
                        record.certificates.append(event)
        elif retracted and not qualified:
            recent = max((w for w in active if w is not None), key=lambda w: w.frame, default=None)
            state.last_trusted = None if recent is None else recent.station
        state.ledger = tuple(active)
        covered = {j: witness for j, witness in enumerate(active)
                   if witness is not None and (not qualified or witness.quality(state.belief) >= QUALITY_ACCEPT)}
        state.covered = sum(1 << j for j in covered)
        if qualified:
            recent = max(covered.values(), key=lambda w: w.frame, default=None)
            state.last_trusted = None if recent is None else recent.station

    def forecast(self, state, candidate, alias):
        return super().forecast(state, candidate, alias, earlier_evidence=state.capture_support)

    def adv_score(self, state, candidate, alias):
        return super().adv_score(state, candidate, alias, earlier_evidence=state.capture_support)

    def free_candidate(self, state, name, ds, dx, bow=0., longitudinal_bow=0., *, renderer="waypoint"):
        goal_s = float(np.clip(state.s + ds, START, END))
        goal_x = float(np.clip(state.x + dx, -30., 0.))
        if longitudinal_bow:
            progress = waypoint(state.s, state.vs, goal_s, longitudinal_bow)
            s, vs = progress.x, progress.v
        else:
            s, vs = longitudinal(state.s, state.vs, goal_s, 0.)
        if s.min() < START - 1e-7 or s.max() > END + 1e-7 or np.max(abs(vs)) > 16.:
            return None
        if np.max(abs(np.diff(vs))) / SAMPLE_DT_S > 75.:
            return None
        if renderer == "waypoint":
            lateral = waypoint(state.x, state.vx, goal_x, bow)
        elif renderer == "fit" and self.learned is not None:
            features = np.array([goal_x - state.x, bow])
            lateral = self.primitive.condition(0, BLOCK_STEPS, state.x, state.vx, goal_x, 0., features @ self.learned)
            # Permit only machine precision at an exactly constrained endpoint.
            lateral.x[[0, -1]], lateral.v[[0, -1]] = [state.x, goal_x], [state.vx, 0.]
        else:
            raise ValueError(f"unknown trajectory representation {renderer}")
        if not admissible(lateral):
            return None
        distance, turn = self.geometry.cost(s, lateral.x)
        if state.travel + distance > MAX_TRAVEL:
            return None
        center = self.mean(state, goal_s, False)[0]
        request = (ds, dx, bow, longitudinal_bow) if longitudinal_bow else (ds, dx, bow)
        return FreeCandidate(name, s, vs, lateral, distance, turn, float(lateral.goal - center), request, bow, renderer)

    def options_shared(self, state, *, expanded=True, renderer="waypoint"):
        menu = self.options(state, False, full=True)
        if not expanded:
            return menu
        # These relative goals are available to every policy. No mode uses
        # the hidden target, shadow lane or transition location.
        coast = float(.25 * state.vs)
        requests = {
            "side_left": (coast, -7., 0.), "side_right": (coast, 7., 0.),
            "back_left": (-4.5, -6., 0.), "back_right": (-4.5, 6., 0.),
            "back_center": (-4.5, 0., 0.),
            "forward_left": (5., -5.5, 0.), "forward_right": (5., 5.5, 0.),
            "look_left": (3., 0., -2.8), "look_right": (3., 0., 2.8),
        }
        for name, (ds, dx, bow) in requests.items():
            action = self.free_candidate(state, name, ds, dx, bow, renderer=renderer)
            if action is None:
                continue
            if not any(np.max(abs(other.s - action.s)) < .05
                       and np.max(abs(other.lateral.x - action.lateral.x)) < .05 for other in menu.values()):
                menu[name] = action
        return menu

    def choose_mpc(self, state, *, expanded, open_decision=False, renderer="waypoint"):
        here = station_bin(state.s)
        needs_centering = (self.evidence_mode == QUALIFIED_EVIDENCE and here >= 0
                           and state.ledger[here] is not None and not state.covered & (1 << here))
        active = open_decision or state.trigger() or needs_centering or state.s >= END - .30
        menu = self.options_shared(state, expanded=expanded and active, renderer=renderer) if active else (
            self.options(state, False, full=False))
        return self.mpc(state, menu, False, active=active, informative=True) if menu else None

    def execute(self, scene, state, record, candidate, alias=False):
        start_frame = state.frame
        plan = candidate.lateral
        motion_start = len(record.motion) - 1
        record.handoff = max(record.handoff, abs(plan.x[0] - state.x), abs(plan.v[0] - state.vx),
                             abs(candidate.s[0] - state.s), abs(candidate.vs[0] - state.vs))
        request = getattr(candidate, "request", None)
        record.decisions.append({"frame": start_frame, "action": candidate.name, "request": request,
                                 "nominal_goal": [float(candidate.s[-1]), plan.goal],
                                 "nominal_tcp_mm": candidate.distance})
        if record.detailed:
            record.references.append({"frame": start_frame, "tick": 0, "action": candidate.name,
                                      "nominal_s_vs": np.column_stack((candidate.s, candidate.vs)).tolist(),
                                      "nominal_x_vx": np.column_stack((plan.x, plan.v)).tolist()})
        for tick in range(FRAME_STRIDE, BLOCK_STEPS + 1, FRAME_STRIDE):
            for i in range(tick - FRAME_STRIDE + 1, tick + 1):
                j = i - plan.step
                record.motion.append([start_frame * .1 + i * SAMPLE_DT_S, float(candidate.s[i]),
                                      float(plan.x[j]), float(candidate.vs[i]), float(plan.v[j]), state.angle])
            state.s, state.x, state.vs, state.vx = record.motion[-1][1:5]
            state.frame += 1
            self.observe(scene, state, record)
            if tick == BLOCK_STEPS or state.slot < 0 or candidate.name == "wait":
                continue
            intended = self.mean(state, candidate.s[-1], False)[0] + candidate.goal_dx + .22 * state.residual
            if isinstance(candidate, FreeCandidate) or candidate.name in ("left", "right"):
                intended = np.clip(intended, candidate.lateral.goal - 1.4, candidate.lateral.goal + 1.4)
            intended = float(np.clip(intended, -30., 0.))
            for gain in (.45, .18):
                delta = gain * np.clip(intended - plan.goal, -3., 3.)
                if abs(delta) <= .025:
                    break
                remaining = BLOCK_STEPS - tick
                if isinstance(candidate, FreeCandidate):
                    # Same causal correction for waypoint and ProDMP-fit references.
                    blend, slope = smoothstep(np.linspace(0., 1., remaining + 1))
                    j = tick - plan.step
                    suffix = Plan(tick, plan.x[j:] + delta * blend,
                                  plan.v[j:] + delta * slope / (remaining * SAMPLE_DT_S))
                else:
                    goal = plan.goal + delta
                    shape = self.shape(candidate.name, state.x, state.vx, goal)
                    suffix = self.primitive.condition(tick, remaining, state.x, state.vx, goal, 0., shape)
                if not admissible(suffix):
                    continue
                past = np.asarray(record.motion[motion_start:])[:, 2]
                preview = np.r_[past, suffix.x[1:]]
                if state.travel + self.geometry.cost(candidate.s, preview)[0] > MAX_TRAVEL:
                    continue
                record.handoff = max(record.handoff, abs(suffix.x[0] - state.x), abs(suffix.v[0] - state.vx))
                record.revisions += 1
                if record.detailed:
                    record.references.append({"frame": state.frame, "tick": tick, "action": candidate.name,
                                              "nominal_x_vx": np.column_stack((suffix.x, suffix.v)).tolist()})
                plan = suffix
                break
        block = np.asarray(record.motion[motion_start:])
        tcp, turn = self.geometry.cost(block[:, 1], block[:, 2])
        state.travel += tcp
        record.decisions[-1].update(executed_goal=[state.s, state.x], executed_tcp_mm=tcp, executed_turn_deg=turn)
        record.blocks.append((candidate.name, motion_start, len(record.motion)))

    @staticmethod
    def can_continue(state):
        return state.frame + FRAMES_PER_BLOCK < FRAME_COUNT and not (
            state.s >= END - .065 and not state.uncovered())

    def tail(self, scene, state, record, *, expanded=True, open_blocks=0):
        while self.can_continue(state):
            action = self.choose_mpc(state, expanded=expanded, open_decision=open_blocks > 0)
            if action is None:
                break
            self.execute(scene, state, record, action)
            open_blocks -= 1
        return state, record


def identity_mistake(record, after_frame, sampled_latent=None):
    confirmed = {(p["frame"], p["accepted_slot"]): float(p["roles_truth"][p["accepted_slot"]] == 0)
                 for p in record.packets if p["frame"] > after_frame and p["accepted_slot"] >= 0}
    for event in record.certificates:
        if event.get("accepted_at", event["frame"]) <= after_frame:
            continue
        if "identity_ok" in event:
            valid = float(event["identity_ok"])
        else:
            valid = event["predicted_identity"][sampled_latent]
        confirmed[event["frame"], event["slot"]] = valid
    if len(confirmed) > len({f for f, _ in confirmed}):
        return 1.
    return 1. - float(np.prod(list(confirmed.values())))


def final_metrics(state, record, *, sampled_latent=None, root_frame=0, query_bin=None):
    """Score real witnesses, or integrate the identities of unseen past witnesses in model rollouts."""
    events = {(event["frame"], event["bin"], event["slot"]): event for event in record.certificates}
    visible = state.image_candidates is not None
    p_centered, p_no_wrong = 1., 1.
    centered_bins, correct_bins = 0, 0
    signed_errors = []
    for j, witness in enumerate(state.ledger):
        if witness is None:
            continue
        event = events.get((witness.frame, j, witness.slot))
        if sampled_latent is not None and (event is None or "identity_ok" not in event):
            correct = float(witness.identity[sampled_latent])
            centered = bool(witness.centered[sampled_latent])
        else:
            correct = float(event["identity_ok"])
            centered = event["center_ok"]
            if visible and sampled_latent is None and correct:
                source = record.packets[witness.frame - record.packets[0]["frame"]]
                signed_errors.append(source["target_truth"][0] - source["x"])
        p_centered *= correct * centered
        p_no_wrong *= correct
        centered_bins += int(correct == 1. and centered)
        correct_bins += int(correct == 1.)
    ready = state.s >= END - .28 and not state.uncovered()
    score = {"corrected_complete": float(ready) * (p_no_wrong if visible else p_centered),
             "final_wrong": 1. - p_no_wrong,
             "tcp_mm": state.travel, "seconds": state.frame * .1, "frames": state.frame + 1,
             "good_bins": centered_bins, "covered_mask": state.covered}
    if visible:
        score |= {"new_identity_mistake": identity_mistake(record, root_frame, sampled_latent),
                  "centered_complete": float(ready) * p_centered, "correct_visible_bins": correct_bins,
                  "centered_correct_bins": centered_bins,
                  "abs_center_error_sum_mm": float(np.abs(signed_errors).sum())}
    if sampled_latent is not None:
        return score
    old = record.details(state)
    wrong_certificates = [e for e in record.certificates if not e["identity_ok"]]
    revoked = {(r["witness"], r["bin"]): r["frame"] for r in record.retractions}
    wrong_at_root = [e["bin"] for e in wrong_certificates if e.get("accepted_at", e["frame"]) <= root_frame
                     < revoked.get((e["frame"], e["bin"]), FRAME_COUNT)]
    repaired = [j for j in wrong_at_root if state.ledger[j] is not None and state.ledger[j].frame > root_frame
                and events[(state.ledger[j].frame, j, state.ledger[j].slot)]["identity_ok"]
                and (visible or events[(state.ledger[j].frame, j, state.ledger[j].slot)]["center_ok"])]
    query = [r for r in record.packets if r["frame"] > root_frame and station_bin(r["s"]) == query_bin]
    if visible:
        fresh = any(e["bin"] == query_bin and e["frame"] > root_frame and e["identity_ok"]
                    for e in record.certificates)
    else:
        fresh = any(r["accepted_slot"] >= 0 and r["roles_truth"][r["accepted_slot"]] == 0
                    and abs(r["y"][r["accepted_slot"]][0]) <= CENTER_MM
                    and abs(r["target_truth"][0] - r["x"]) <= CENTER_MM for r in query)
    wrong_time = sum(max(0, revoked.get((e["frame"], e["bin"]), state.frame) - e["frame"])
                     for e in wrong_certificates) * .1
    return dict(score, old_strict_complete=old["complete"],
                ever_wrong=bool(identity_mistake(record, -1)) if visible else old["wrong"] or old["wrong_certification"],
                wrong_confident_frames=old["wrong_confident_frames"], wrong_card_seconds=wrong_time,
                wrong_certificates=len(wrong_certificates), at_root_wrong_bins=wrong_at_root,
                repaired_root_bins=repaired, retractions=len(record.retractions),
                target_reappeared=any(0 in r["roles_truth"] for r in query), fresh_query_confirmed=fresh,
                turn_degs=sum(d["executed_turn_deg"] for d in record.decisions),
                feedback_revisions=record.revisions, max_handoff_error=float(record.handoff))
