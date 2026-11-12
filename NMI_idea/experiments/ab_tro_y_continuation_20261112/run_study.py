"""Prospective three-arm moving-Y continuation study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
MAP_DIR = HERE.parent / 'ab_map_recheck_20261112'
sys.path.insert(0, str(MAP_DIR))
from closed_loop import MapWorld, map_errors, prepare
from belief import CONFIDENCE
from experiment import prefix_hash
from mechanics import (CAPTURE_DWELL, CHECK_PROGRESS, POSE_GRID, POSITION_TOL_M,
                       ROTATION_TOL_RAD, SAMPLE_DT, MotionModel)
from planning import (INNER_DRAWS, INNER_SEED, OUTER_DRAWS, OUTER_SEED,
                      TIME_COST_PER_S, TRAVEL_COST_PER_MM, WRONG_ASSOCIATION_COST,
                      choose_program, choose_real_tail, forecast)


ARMS = ('online_B', 'online_MPC', 'frozen_B')
RESERVES_MM = {'narrow': 10., 'wide': 26.}
PROBE = (3, 3)
FIRST_VIEWS = (2, 3, 4)
FINAL_SITE = 4
FINAL_VIEWS = (2, 3, 4)
PROBE_DURATION_S = 2.50
FIRST_DURATION_S = .65
TAIL_DURATION_S = .85
POST_CAP_MM = 60.
POST_CAP_DEG = 60.
HOLDOUT_SEED_BASE = 2026111217
SMOKE_SEED_BASE = 2026111290
HOLDOUT_N = 128
TIME_TOL = 1.e-9
WORLD = MODEL = PROBE_PATH = FIRST_PATHS = TAIL_PATHS = None


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def digest_json(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def frame_public(frame):
    return dict(site=frame.site, view=frame.view, t=frame.t, q=frame.q, velocity=frame.velocity,
                centroids_mm=frame.centroids.tolist())


def seeded(base, i):
    return int(np.random.SeedSequence([base, i]).generate_state(1)[0])


def initialize():
    global WORLD, MODEL, PROBE_PATH, FIRST_PATHS, TAIL_PATHS
    WORLD = MapWorld(mode='stress')
    MODEL = MotionModel(WORLD)
    if (WORLD.scan.central != PROBE[1] or len(WORLD.initial_route) != 28
            or not WORLD.identity_site(PROBE[0]) or not WORLD.identity_site(FINAL_SITE)):
        raise RuntimeError('unexpected geometry or first sweep layout')
    if not np.array_equal(WORLD.scan.offsets[list(FIRST_VIEWS)], [-8., 0., 8.]):
        raise RuntimeError('candidate strip changed in the source geometry')
    PROBE_PATH = MODEL.register(MODEL.departure, PROBE, PROBE_DURATION_S)
    if not PROBE_PATH.legal:
        raise RuntimeError(f'common probe failed preflight: {PROBE_PATH.reason}')
    FIRST_PATHS = tuple(MODEL.register(PROBE, (PROBE[0], view), FIRST_DURATION_S)
                        for view in FIRST_VIEWS)
    TAIL_PATHS = {move.goal[1]: tuple(MODEL.register(move.goal, (FINAL_SITE, view), TAIL_DURATION_S)
                                    for view in FINAL_VIEWS)
                  for move in FIRST_PATHS if move.legal}
    for reserve in RESERVES_MM.values():
        first, tails = legal_graph(reserve)
        if not any(len(tails[p.goal[1]]) >= 2 for p in first):
            raise RuntimeError('predeclared reserve has no genuine choice after any first action')
    # Geometry-only visibility preregistration: both atlas lumens in the acoustic
    # aperture and depth gate over the complete family of pose checks.
    lower, upper = np.array([-25., 3.]), np.array([25., 52.])
    projections = np.array([WORLD.projection(angle, *PROBE)[0] for angle in POSE_GRID])
    if not np.all((projections >= lower) & (projections <= upper)):
        raise RuntimeError('the common probe is not publicly branch-visible across the pose grid')
    return dict(fixed_rail_m=MODEL.rail_m, common_probe=asdict(PROBE_PATH),
                common_probe_nominal_beam_ranges_mm=dict(min=projections.min(axis=0).tolist(),
                                                          max=projections.max(axis=0).tolist()),
                first_paths=[asdict(path) for path in FIRST_PATHS],
                tail_paths={str(view): [asdict(path) for path in paths]
                            for view, paths in TAIL_PATHS.items()},
                legal_graph={name: {str(v): [p.id for p in tails] for v, tails in legal_graph(reserve)[1].items()}
                             for name, reserve in RESERVES_MM.items()},
                moving_knot_deg=WORLD.knot_angles.tolist(), model_joint_velocity_limits=MODEL.limits.tolist(),
                pose_grid_deg=POSE_GRID.tolist(), path_progress_grid=CHECK_PROGRESS.tolist(),
                registration_robot_from_atlas=MODEL.robot_from_atlas.tolist())


def legal_graph(reserve):
    first, tails = [], {}
    for move in FIRST_PATHS:
        if not move.legal or move.stroke_mm > reserve + TIME_TOL:
            continue
        eligible = tuple(path for path in TAIL_PATHS[move.goal[1]]
                         if path.legal and path.stroke_mm + move.stroke_mm <= reserve + TIME_TOL)
        if eligible:
            first.append(move)
            tails[move.goal[1]] = eligible
    return tuple(first), tails


def update(state, frame):
    return state.updated(WORLD, frame, retain=WORLD.identity_site(frame.site))


def world_capture(case, path, trace):
    # The image truth belongs to the environment. The low-level FK trajectory
    # reaches the same registered contact used by MapWorld at this same clock.
    frame = WORLD.observe(case, trace['t_end'], *path.goal)
    angle = trace['samples']['knee_deg'][-1]
    expected_atlas = WORLD.probe_world(angle, *path.goal)
    expected_robot = MODEL.chart_reference(path.start, path.goal, np.array([1.]), angle)[0]
    attained_robot = np.asarray(trace['samples']['robot_tcp_rotation'][-1])
    fk_error = float(np.linalg.norm(expected_atlas - trace['samples']['atlas_xyz_mm'][-1]))
    angular_error = float(Rotation.from_matrix(attained_robot.T @ expected_robot[:3, :3]).magnitude())
    trace['image_pose_fk_difference_mm'] = fk_error
    trace['image_pose_fk_difference_deg'] = float(np.rad2deg(angular_error))
    trace['image_nominal_contact_atlas_mm'] = expected_atlas.tolist()
    trace['image_pose_source'] = 'registered nominal moving-Y acoustic endpoint at realized capture clock'
    if fk_error > 1000. * POSITION_TOL_M + 1.e-8 or angular_error > ROTATION_TOL_RAD + 1.e-8:
        raise RuntimeError('synthetic acoustic pose and executed FK endpoint disagree')
    return frame


def target_claim_empty(event=30):
    return dict(event=event, site=FINAL_SITE, ordinal=None, confidence=0., confirmed=False)


def target_claim(state, event):
    candidates = [w for w in state.belief.witnesses if w.event == event and w.site == FINAL_SITE]
    if not candidates:
        return target_claim_empty(event)
    witness = max(candidates, key=lambda w: state.belief.probability(w)[1])
    confidence = float(state.belief.probability(witness)[1])
    return dict(event=event, site=witness.site, ordinal=witness.ordinal,
                centroid_mm=list(witness.coordinate), confidence=confidence,
                confirmed=bool(confidence >= CONFIDENCE),
                posterior_trunk_daughter_clutter=state.belief.probability(witness).tolist())


def evaluate_claim(claim, final_frame, completed):
    ordinal = claim['ordinal']
    label = (int(final_frame.labels[ordinal]) if final_frame is not None and ordinal is not None else None)
    confirmed = bool(claim['confirmed'])
    return dict(label_for_evaluation=label, correct_fresh=bool(confirmed and label == 1),
                strict=bool(completed and confirmed and label == 1),
                false_confirm=bool(confirmed and label != 1),
                false_confirm_trunk=bool(confirmed and label == 0),
                false_confirm_clutter=bool(confirmed and label == -1),
                target_present=bool(final_frame is not None and 1 in final_frame.labels))


def actual_move(case, path, start_t, start_q, cache):
    key = (path.id, start_t, None if start_q is None else np.asarray(start_q).tobytes())
    if key not in cache:
        pose_sensor = lambda t: case.true_angle_rate(t)[0]
        trace, q_out = MODEL.execute(path, start_t, pose_sensor, start_q)
        frame = world_capture(case, path, trace) if trace['legal_completed'] else None
        cache[key] = (trace, q_out, frame)
    return cache[key]


def rollout(case, prefix_marginal, start_state, start_q, name, reserve_name, forecast_values,
            graph, cache, common):
    reserve = RESERVES_MM[reserve_name]
    base = dict(seed=case.seed, index=common['index'], arm=name, reserve=reserve_name,
                reserve_mm=reserve, prefix_sha=common['prefix_sha'],
                common_probe_sha=common['common_probe_sha'],
                common_probe_visible_for_evaluation=common['common_probe_visible_for_evaluation'])
    first, tail_graph = graph
    start_t = common['probe_motion']['t_end']
    if start_state is None:
        return base | dict(strict=False, legal_completed=False, failure='common probe path did not complete',
                           stages=[], travel_mm=0., turn_deg=0., duration_s=0.,
                           false_confirm=False, false_confirm_trunk=False, false_confirm_clutter=False,
                           target_present=False, correct_fresh=False, first_view=None, actual_tail_view=None,
                           open_loop_tail_view=None, tail_switched=None, final_claim=target_claim_empty())
    planned, choices = choose_program(forecast_values, first, tail_graph, feedback=name != 'online_MPC')
    first_move = next(p for p in first if p.goal[1] == planned['first_view'])
    trace, mid_q, first_frame = actual_move(case, first_move, start_t, start_q, cache)
    state = start_state
    steps = [dict(motion=trace, frame=frame_public(first_frame) if first_frame is not None else None,
                  labels_for_evaluation=list(first_frame.labels) if first_frame is not None else None)]
    executed_tail = None
    final_frame = None
    tail_scores = None
    if first_frame is not None:
        state = update(state, first_frame)
        key = (name == 'frozen_B', first_move.goal[1], reserve_name)
        if key not in cache:
            cache[key] = choose_real_tail(WORLD, state.belief, tail_graph[first_move.goal[1]])
        executed_tail, tail_scores = cache[key]
        last_trace, _, final_frame = actual_move(case, executed_tail, first_frame.t, mid_q, cache)
        steps.append(dict(motion=last_trace, frame=frame_public(final_frame) if final_frame is not None else None,
                          labels_for_evaluation=list(final_frame.labels) if final_frame is not None else None))
        if final_frame is not None:
            state = update(state, final_frame)
    travel = sum(step['motion']['travel_mm'] for step in steps)
    turn = sum(step['motion']['turn_deg'] for step in steps)
    duration = sum(step['motion']['duration_s'] for step in steps)
    stroke = sum(step['motion']['stroke_mm'] for step in steps)
    complete = bool(final_frame is not None and travel <= POST_CAP_MM + TIME_TOL
                    and turn <= POST_CAP_DEG + TIME_TOL and stroke <= reserve + TIME_TOL
                    and duration <= FIRST_DURATION_S + TAIL_DURATION_S + TIME_TOL)
    claim = target_claim(state, 30)
    final_marginal = state.belief.weights.sum(axis=1)
    marginal_change = float(np.abs(final_marginal - prefix_marginal).sum())
    if name == 'frozen_B' and marginal_change > 1.e-10:
        raise RuntimeError('frozen static marginal changed')
    map_rmse, map_max = map_errors(WORLD, case, state.belief, state.belief.reading.t)
    return base | evaluate_claim(claim, final_frame, complete) | dict(
        legal_completed=complete, failure=None if complete else 'path or shared cap did not complete',
        first_view=first_move.goal[1], planning_candidates=choices, planned=planned,
        open_loop_tail_view=planned['open_loop_tail_view'],
        actual_tail_view=executed_tail.goal[1] if executed_tail is not None else None,
        tail_switched=(bool(executed_tail.goal[1] != planned['open_loop_tail_view'])
                       if executed_tail is not None else None), real_tail_scores=tail_scores,
        final_claim=claim, stages=steps, travel_mm=travel, turn_deg=turn, duration_s=duration,
        stroke_mm=stroke, map_marginal_change_l1=marginal_change,
        probe_map_marginal_change_l1=float(np.abs(start_state.belief.weights.sum(axis=1) - prefix_marginal).sum()),
        final_map_rmse_mm=map_rmse, final_map_max_mm=map_max)


def paired_case(item):
    index, seed = item
    case = WORLD.case(seed)
    prefix = prepare(WORLD, case)
    marginal = prefix.belief.weights.sum(axis=1)
    frozen = replace(prefix, belief=prefix.belief.copy().freeze_map())
    cache = {}
    probe_trace, q, probe_frame = actual_move(case, PROBE_PATH, prefix.previous.t, None, cache)
    public = frame_public(probe_frame) if probe_frame is not None else None
    common = dict(index=index, seed=seed, prefix_sha=prefix_hash(prefix),
                  prefix_last_t=prefix.previous.t,
                  prefix_frames=[frame_public(frame) | dict(probe_atlas_mm=WORLD.probe_world(frame.q, frame.site, frame.view).tolist())
                                 for frame in prefix.frames],
                  common_probe_sha=digest_json(public), common_probe_frame=public, probe_motion=probe_trace,
                  probe_labels_for_evaluation=list(probe_frame.labels) if probe_frame is not None else None,
                  common_probe_visible_for_evaluation=bool(probe_frame is not None and 1 in probe_frame.labels),
                  common_probe_trunk_visible_for_evaluation=bool(probe_frame is not None and 0 in probe_frame.labels),
                  common_probe_empty=bool(probe_frame is None or len(probe_frame.centroids) == 0),
                  common_probe_nominally_branch_visible=True)
    online_start = update(prefix, probe_frame) if probe_frame is not None else None
    frozen_start = update(frozen, probe_frame) if probe_frame is not None else None
    first, tails = legal_graph(max(RESERVES_MM.values()))
    tail_endpoints = tuple(next(p for paths in tails.values() for p in paths if p.goal[1] == v)
                           for v in FINAL_VIEWS if any(p.goal[1] == v for paths in tails.values() for p in paths))
    online_fc = forecast(WORLD, online_start.belief, first, tail_endpoints) if online_start else None
    frozen_fc = forecast(WORLD, frozen_start.belief, first, tail_endpoints) if frozen_start else None
    rows = []
    for name, reserve in RESERVES_MM.items():
        graph = legal_graph(reserve)
        for arm in ARMS:
            state, fc = (frozen_start, frozen_fc) if arm == 'frozen_B' else (online_start, online_fc)
            rows.append(rollout(case, marginal, state, q, arm, name, fc, graph, cache, common))
    return common, rows


def source_hashes():
    files = [HERE / x for x in ('run_study.py', 'planning.py', 'mechanics.py', 'PROTOCOL.md')]
    files += [MAP_DIR / 'map_model.py', MAP_DIR / 'closed_loop.py', WORLD.geometry_path]
    files += [HERE.parent / x for x in (
        'ab_knee_latent_rl_20261112/belief.py', 'ab_knee_latent_rl_20261112/knee_world.py',
        'ab_knee_latent_rl_20261112/experiment.py', 'ab_fisher_continuation_20261112/certification.py',
        'ab_leg_dynamic_20261112/dynamic_scan.py',
        'ab_leg_bifurcation_20261112/long_final_frozen/BRANCH_AND_GPS.json',
        'ab_leg_bifurcation_20261112/knee_rebake/atlas_right.npz',
        'ab_leg_bifurcation_20261112/knee_rebake/attachments_right.npz',
        'ab_leg_bifurcation_20261112/learned_prodmp.py',
        'ab_leg_bifurcation_20261112/long_leg_assets.py')]
    robot_manifest = MODEL.robot.manifest()
    mesh_hashes = robot_manifest['original_dird']['robot_contract']['collision_mesh_sha256']
    return {str(path): hash_file(path) for path in files} | robot_manifest['source_sha256'] | mesh_hashes


def manifest(preflight):
    return dict(schema='tro_moving_y_continuation_v1', sources=source_hashes(),
                protocol_sha256=hash_file(HERE / 'PROTOCOL.md'),
                holdout_seed_base=HOLDOUT_SEED_BASE, holdout_seed_list=[seeded(HOLDOUT_SEED_BASE, i) for i in range(HOLDOUT_N)],
                smoke_seed_base=SMOKE_SEED_BASE, cohort_n=HOLDOUT_N, arms=ARMS,
                reserve_mm=RESERVES_MM, sampled_dt_s=SAMPLE_DT, capture_dwell_s=CAPTURE_DWELL,
                shared_post_caps=dict(duration_s=FIRST_DURATION_S+TAIL_DURATION_S, travel_mm=POST_CAP_MM, turn_deg=POST_CAP_DEG),
                planning=dict(outer_nodes=OUTER_DRAWS, outer_seed=OUTER_SEED,
                              inner_nodes=INNER_DRAWS, inner_seed=INNER_SEED,
                              wrong_association_cost=WRONG_ASSOCIATION_COST, confidence_threshold=CONFIDENCE,
                              travel_cost_per_mm=TRAVEL_COST_PER_MM, time_cost_per_s=TIME_COST_PER_S),
                mechanic_preflight=preflight, python=sys.version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--holdout', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--smoke-n', type=int, default=1)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    preflight = initialize()
    frozen_file = HERE / 'FROZEN_MANIFEST.json'
    current = manifest(preflight)
    if args.freeze:
        with frozen_file.open('x') as handle:
            handle.write(json.dumps(current, sort_keys=True, indent=2, allow_nan=False) + '\n')
        print('Frozen manifest SHA-256', hash_file(frozen_file), flush=True)
        return
    if args.holdout:
        if json_text(json.loads(frozen_file.read_text())) != json_text(current):
            raise RuntimeError('frozen protocol, inputs or mechanics differ')
        seeds = current['holdout_seed_list']
    else:
        seeds = [seeded(SMOKE_SEED_BASE, i) for i in range(args.smoke_n)]
    if len(set(seeds)) != len(seeds) or (args.smoke and set(seeds) & set(current['holdout_seed_list'])):
        raise RuntimeError('duplicated or overlapping seeds')
    out = args.out or HERE / ('holdout' if args.holdout else 'smoke')
    out.mkdir(parents=True, exist_ok=False)
    current.update(split='holdout' if args.holdout else 'smoke', executed_seed_list=seeds,
                   frozen_manifest_sha256=hash_file(frozen_file) if args.holdout else None)
    (out / 'MANIFEST.json').write_text(json.dumps(current, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (out / 'COMMON.jsonl').open('w') as shared, (out / 'RUNS.jsonl').open('w') as runs:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('fork')) as pool:
            for index, (common, rows) in enumerate(pool.map(paired_case, enumerate(seeds))):
                shared.write(json_text(common) + '\n')
                runs.writelines(json_text(row) + '\n' for row in rows)
                shared.flush()
                runs.flush()
                print(f'{index + 1}/{len(seeds)} paired [{time.monotonic() - start:.1f} s]', flush=True)
    (out / 'COMPLETE.json').write_text(json.dumps(dict(
        common_rows=len(seeds), rollout_rows=len(seeds) * len(ARMS) * len(RESERVES_MM),
        common_sha256=hash_file(out / 'COMMON.jsonl'), rollouts_sha256=hash_file(out / 'RUNS.jsonl'),
        elapsed_wall_seconds=time.monotonic() - start), indent=2) + '\n')


if __name__ == '__main__':
    main()
