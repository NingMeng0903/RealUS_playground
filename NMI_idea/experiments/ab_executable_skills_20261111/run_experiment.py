"""Paired closed-loop execution of exact, observation-conditioned skills."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from skill_value import (
    ARMS, HERE, REFERENCE, Skill, Statistics, accepted, actual_statistics, plan,
    terminal_path, update,
)
from run_soft_stress import choose as legacy_choose, fixtures
from toy_world import Action, Fixture, latent_index, legal_actions, prior, sample_step


@dataclass(frozen=True)
class Case:
    case_id: str
    side: str
    permanent: bool
    fixture: Fixture
    stratum: str
    start: str


def cases() -> list[Case]:
    return [Case(f"{i:02d}-{start}", side, permanent, fixture, stratum, start)
            for i, (side, permanent, fixture, stratum) in enumerate(fixtures())
            for start in ("EL", "ER")]


def packet_draws(seed: int, case_index: int, replicate: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_index, replicate]))
    return rng.random((count, 3))


@dataclass
class Episode:
    case: Case
    arm: str
    draws: np.ndarray
    belief: np.ndarray = field(default_factory=prior)
    node: str = field(init=False)
    pending: str = field(init=False)
    private_state: int = field(init=False)
    visits: list[str] = field(init=False)
    outcome: str = "unresolved"
    actions: deque[Action] = field(default_factory=deque)
    skill: Skill | None = None
    waiting_for_probe_packet: bool = False
    events: list[dict] = field(default_factory=list)
    programs: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    returns: int = 0
    retreats: int = 0
    switches: int = 0
    missing: int = 0
    false_hits: int = 0
    missing_with_suffix: int = 0
    first_target_frame: int | None = None

    def __post_init__(self) -> None:
        self.node = self.case.start
        self.pending = self.node[-1]
        self.visits = [self.pending]
        self.private_state = latent_index(self.case.side, self.case.permanent, False)

    def start_skill(self) -> None:
        if self.arm == "A_legacy":
            # External reference only. Its historical one-step executive is
            # deliberately retained rather than pooled with the matched arms.
            action, side, _, _, _ = legacy_choose(
                "A_task", self.node, self.case.fixture, self.belief,
                self.pending, None, None)
            if action is None:
                raise AssertionError("connected graph has no legal legacy action")
            self.skill = Skill((action,), side or self.pending, (), Statistics(0., 0., 1.))
        else:
            self.skill = plan(self.arm, self.node, self.case.fixture, tuple(self.belief), self.pending)
        self.actions.extend(self.skill.path)
        self.waiting_for_probe_packet = bool(self.skill.goals_by_packet)
        self.programs.append(dict(
            frame=len(self.events), entry=self.node, pending=self.pending,
            belief=self.belief.tolist(), plan=self.skill.as_dict(),
            score_at_actual_exit=None if self.arm == "A_legacy" else
            actual_statistics(self.skill, self.case.fixture, self.belief).as_dict(),
        ))
        if self.skill.goal:
            self.pending = self.skill.goal

    def advance(self, until: int) -> None:
        """Advance to a recording boundary without changing policy or state."""
        if until > len(self.draws):
            raise ValueError("insufficient paired observation draws")
        while len(self.events) < until and self.outcome == "unresolved":
            if not self.actions:
                self.start_skill()
            action = self.actions.popleft()
            if action not in legal_actions(self.node, self.case.fixture):
                raise AssertionError(f"saved program tried illegal {self.node}:{action.name}")
            event = dict(program=len(self.programs) - 1, action=action.name,
                         entry=self.node, exit=action.dest)
            self.retreats += int(self.node != "C" and action.dest == "C")
            self.node = action.dest
            if self.node != "C" and self.node[-1] != self.visits[-1]:
                self.switches += 1
                self.returns += int(len(self.visits) >= 2 and self.visits[-2] == self.node[-1])
                self.visits.append(self.node[-1])
            self.private_state, packet, is_target = sample_step(
                self.private_state, action, self.case.fixture, *self.draws[len(self.events)])
            self.belief, p_target = update(self.belief, self.case.fixture, action, packet)
            stop = accepted(action, packet, p_target)
            if is_target and self.first_target_frame is None:
                self.first_target_frame = len(self.events) + 1
            self.missing += int(packet == 0)
            self.false_hits += int(packet != 0 and not is_target)
            if stop:
                self.outcome = "correct" if is_target else "wrong_stop"
                self.actions.clear()
                self.waiting_for_probe_packet = False
            elif self.waiting_for_probe_packet:
                assert self.skill is not None
                self.pending = self.skill.goals_by_packet[packet]
                tail = terminal_path(
                    self.node, self.pending, self.case.fixture,
                    already_captured_at_terminal=self.node == "T" + self.pending)
                self.actions.extend(tail)
                self.waiting_for_probe_packet = False
                event.update(selected_goal=self.pending, saved_suffix=[a.name for a in tail])
            self.missing_with_suffix += int(packet == 0 and bool(self.actions))
            self.events.append(dict(
                event, packet=packet, p_target=p_target, stop=stop,
                observed_target_for_evaluator=is_target,
                remaining_actions=[a.name for a in self.actions]))
        self.checkpoints.append(dict(
            requested=until, frames=len(self.events), outcome=self.outcome,
            actual_node=self.node, posterior=self.belief.tolist(),
            pending=self.pending, remaining_actions=[a.name for a in self.actions],
        ))

    def record(self, replicate: int) -> dict:
        return dict(
            arm=self.arm, case_id=self.case.case_id, replicate=replicate,
            stratum=self.case.stratum, fixture=dict(shadow=self.case.fixture.shadow,
                                                  narrow=self.case.fixture.narrow),
            start=self.case.start, true_side=self.case.side,
            oracle_possible=not self.case.permanent, outcome=self.outcome,
            frames=len(self.events), returns=self.returns, retreats=self.retreats,
            switches=self.switches, completed_bins=1 + int(self.outcome == "correct"),
            false_hits=self.false_hits, missing=self.missing,
            missing_with_suffix=self.missing_with_suffix,
            first_target_frame=self.first_target_frame,
            programs=self.programs, events=self.events, checkpoints=self.checkpoints,
        )


def source_manifest() -> dict[str, str]:
    sources = [HERE / "skill_value.py", HERE / "run_experiment.py",
               *[REFERENCE / s for s in ("toy_world.py", "program_value.py", "run_soft_stress.py")]]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=721621)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--guard", type=int, default=64)
    parser.add_argument("--case-limit", type=int, help="plumbing smoke only")
    args = parser.parse_args()
    if args.replicates < 1 or args.guard < 1:
        parser.error("replicates and guard must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "EPISODES.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite completed or partial run at {args.output}")
    selected_cases = cases()[:args.case_limit]
    checkpoint_frames = [n for n in (8, 16, 32) if n < args.guard] + [args.guard]
    sources = source_manifest()
    (args.output / "MANIFEST.json").write_text(json.dumps(dict(
        seed=args.seed, replicates=args.replicates, cases=[c.case_id for c in selected_cases],
        arms=ARMS, checkpoints=checkpoint_frames,
        oracle_contract="both terminal goals reachable in the reversible graph; permanent no true target",
        planner_guard_access=False, capture_noise_pairing="case, replicate, fresh packet index, channel",
        release_rule="complete saved program, immediately stop on public acceptance",
        source_sha256=sources,
    ), indent=2) + "\n")
    with (args.output / "EPISODES.jsonl").open("w") as stream:
        for i, case in enumerate(selected_cases):
            for rep in range(args.replicates):
                draws = packet_draws(args.seed, i, rep, args.guard)
                for arm in ARMS:
                    episode = Episode(case, arm, draws)
                    for checkpoint in checkpoint_frames:
                        episode.advance(checkpoint)
                        if episode.outcome != "unresolved":
                            break
                    stream.write(json.dumps(episode.record(rep), allow_nan=False) + "\n")
            stream.flush()
            print(f"{case.case_id} complete", flush=True)
    if source_manifest() != sources:
        raise RuntimeError("sources changed during the run")


if __name__ == "__main__":
    main()
