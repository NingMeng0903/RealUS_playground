"""Fit and audit direct paired full-completion differences from public inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from opportunity import BASE, MARGINS, cohorts, read_rows
from paired_metrics import summarize


RIDGE = 24.
PRICE_SLACK = .5
ARMS = (BASE, "B_cohort", "B_local", "fixed", "B_hard_cap")
PREDICTIONS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s", "rail_mm", "captures")
RESOURCE_CAPS = np.array([65., 90., 250., 6.])


def features(task, public, key, candidate):
    """Low-dimensional public belief and checked route descriptors."""
    depth, progress = public["depth"], public["progress"]
    parts = candidate["features"]
    first = parts[0]
    present_base = public["baseline_features"]
    base = present_base[0] if present_base else None
    pattern = (f'{first["site"] - public["target"]}/{first["tilt"]}/{first["rho"]}'
               f'/{len(parts)}')
    context = f"{task}/{depth}/{progress}"
    f = {"length": float(len(parts)), f"group:{context}": 1., f"route:{key}": 1.,
         f"route-context:{context}/{key}": 1., f"baseline:{public['baseline']}": 1.,
         f"start-node:{public['at']}": 1., f"pattern:{pattern}": 1.,
         f"pattern-context:{context}/{pattern}": 1., f"packet:{task}/{public['anchor']}": 1.,
         "progress": progress, "slots": public["slots"], "good": public["prior_good"],
         "ess": np.log1p(public["ess"]), "displacement": public["displacement_mean"],
         "positive": public["displacement_positive"], "x": public["last_x"], "z": public["last_z"],
         "detected": public["last_detected"], "multiple": public["last_multiple"],
         "baseline-available": int(base is not None)}
    for metric, r in enumerate(np.asarray(public["reserve"]) / RESOURCE_CAPS):
        f[f"reserve:{metric}"] = float(r)
    f[f"packet-pattern:{pattern}/detected"] = f["detected"]
    f[f"packet-pattern:{pattern}/multiple"] = f["multiple"]
    for name in ("displacement", "positive", "good", "x", "z"):
        f[f"evidence-action:{pattern}/{name}"] = f[name]
    for j, part in enumerate(parts):
        for name in ("tilt", "rho", "future", "target", "yes", "risk", "miss", "expected_correct_return"):
            f[f"action:{j}/{name}"] = float(part[name])
        for metric, raw in enumerate(np.asarray(part["cost"]) / RESOURCE_CAPS):
            f[f"action:{j}/cost{metric}"] = float(raw)
        f[f"action:{j}/relative-target"] = float(part["site"] - public["target"])
        f[f"action:{j}/relative-start"] = float(part["site"] - public["site"])
        f[f"action:{j}/yes-at-target"] = float(part["yes"] * part["target"])
        f[f"action:{j}/risk-at-target"] = float(part["risk"] * part["target"])
    if len(parts) == 2:
        second = parts[1]
        f["opposing-same-site"] = float(second["site"] == first["site"] and second["tilt"] != 0
                                          and second["tilt"] == -first["tilt"])
        f["same-acoustic-view"] = float(second["site"] == first["site"] and second["tilt"] == first["tilt"])
    if base is not None:
        f["first-equivalent-realization"] = float(first["site"] == base["site"] and first["tilt"] == base["tilt"]
                                                  and first["rho"] != base["rho"])
        for name in ("future", "target", "miss", "expected_correct_return"):
            f[f"first-minus-baseline:{name}"] = float(first[name] - base[name])
        f["first-minus-baseline:yes-at-target"] = first["yes"] * first["target"] - base["yes"] * base["target"]
        f["first-minus-baseline:risk-at-target"] = first["risk"] * first["target"] - base["risk"] * base["target"]
        for metric, raw in enumerate((np.asarray(first["cost"]) - base["cost"]) / RESOURCE_CAPS):
            f[f"first-minus-baseline:cost{metric}"] = float(raw)
    return f


def matrix(dicts, names):
    out = np.zeros((len(dicts), len(names)))
    index = {k: j for j, k in enumerate(names)}
    for i, values in enumerate(dicts):
        for k, value in values.items():
            if k in index:
                out[i, index[k]] = value
    return out


def train(rows):
    case_contexts = Counter(row["case_seed"] for row in rows if len(row["actions"]) > 1)
    x, y, weights = [], [], []
    for row in rows:
        outcomes = row["outcomes"]
        base = np.asarray(outcomes[BASE])[:, :len(PREDICTIONS)]
        for key, action in row["actions"].items():
            if key == BASE:
                continue
            x.append(features(row["task"], row["public"], key, action))
            y.append((np.asarray(outcomes[key])[:, :len(PREDICTIONS)] - base).mean(axis=0))
            weights.append(1. / ((len(row["actions"]) - 1) * case_contexts[row["case_seed"]]))
    names = sorted({k for values in x for k in values})
    x, y, w = matrix(x, names), np.asarray(y), np.asarray(weights)
    center = np.average(x, weights=w, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - center)**2, weights=w, axis=0)), .10)
    response = np.average(y, weights=w, axis=0)
    z = (x - center) / scale
    weighted = z * w[:, None]
    coefficients = np.linalg.solve(weighted.T @ z + RIDGE * np.eye(z.shape[1]), weighted.T @ (y - response))
    # Fold centering and scaling into the raw linear model for deployment.
    coefficients /= scale[:, None]
    intercept = response - center @ coefficients
    return dict(names=names, coefficients=coefficients.tolist(), intercept=intercept.tolist(), ridge=RIDGE,
                n_labeled_state_actions=len(x), n_latent_cases=len(case_contexts),
                training_field_names=PREDICTIONS)


def empirical_fixed(rows):
    table = {}
    for (task, depth), group in cohorts(rows).items():
        effects, total = defaultdict(list), defaultdict(lambda: np.zeros(len(PREDICTIONS)))
        for row in group:
            base = np.asarray(row["outcomes"][BASE])[:, :len(PREDICTIONS)]
            for key, outcomes in row["outcomes"].items():
                delta = (np.asarray(outcomes)[:, :len(PREDICTIONS)] - base).mean(axis=0)
                effects[key].append(delta)
                total[key] += delta / len(group)
        admitted = [BASE, *sorted(k for k, val in total.items() if k != BASE
                                   and np.all(np.abs(val[2:5]) <= MARGINS + 1.e-12))]
        best = max(admitted, key=lambda k: total[k][0])
        table[f"{task}/{depth}"] = dict(choice=best, fit_mean_delta=total[best].tolist(),
                                        action_means={k: np.asarray(v).mean(axis=0).tolist() for k, v in effects.items()})
    return table


class Predictor:
    def __init__(self, fit):
        self.weights = dict(zip(fit["names"], np.asarray(fit["coefficients"])))
        self.intercept = np.asarray(fit["intercept"])
        self.fixed = fit["fixed"]
        self.prices = fit.get("prices", {})

    def forecast(self, task, public, actions):
        values = {BASE: np.zeros(len(PREDICTIONS))}
        for key in sorted(actions):
            if key == BASE:
                continue
            f = features(task, public, key, actions[key])
            pred = self.intercept.copy()
            for name, value in f.items():
                if name in self.weights:
                    pred += value * self.weights[name]
            values[key] = pred
        return values

    def choose(self, task, depth, values, arm):
        group = f"{task}/{depth}"
        if arm == BASE:
            return BASE
        if arm == "fixed":
            fixed = self.fixed[group]["choice"]
            return fixed if fixed in values else BASE
        if arm == "B_hard_cap":
            return prefer_incumbent(values, lambda k: values[k][0])
        if arm == "B_local":
            feasible = [key for key, val in values.items() if np.all(np.abs(val[2:5]) <= MARGINS)]
            return prefer_incumbent(feasible, lambda k: values[k][0])
        if arm == "B_cohort":
            prices = np.asarray(self.prices[group]["normalized_price"])
            return prefer_incumbent(values, lambda k: values[k][0] - prices @ (values[k][2:5] / MARGINS))
        raise ValueError(arm)


def prefer_incumbent(choices, score):
    best = BASE
    best_value = score(BASE)
    for key in choices:
        val = score(key)
        if val > best_value + 1.e-12:
            best, best_value = key, val
    return best


def cost_prices(forecasts):
    """Dual cohort allocation with two-sided, normalized motion constraints."""
    n = len(forecasts)
    rows, cols, weights, bound = [], [], [], []
    for i, values in enumerate(forecasts):
        for val in values.values():
            cost = val[2:5] / MARGINS
            k = len(bound)
            rows.extend([k] * 7)
            cols.extend([0, 1, 2, 3, 4, 5, i + 6])
            weights.extend([*(-cost), *cost, -1.])
            bound.append(-val[0])
    matrix_ub = coo_matrix((weights, (rows, cols)), shape=(len(bound), n + 6)).tocsc()
    objective = np.full(n + 6, 1. / n)
    objective[:6] = PRICE_SLACK
    result = linprog(objective, A_ub=matrix_ub, b_ub=np.asarray(bound), bounds=(0., None), method="highs")
    if not result.success:
        raise ValueError(f"price fit failed: {result.message}")
    price = result.x[:3] - result.x[3:6]
    pick = [prefer_incumbent(values, lambda k: values[k][0] - price @ (values[k][2:5] / MARGINS))
            for values in forecasts]
    fitted = np.asarray([values[key] for values, key in zip(forecasts, pick)]).mean(axis=0)
    return dict(normalized_price=price.tolist(), dual_objective=float(result.fun),
                normalized_constraint=PRICE_SLACK, fitted_deterministic_delta=fitted.tolist())


def evaluate_development(rows, fit):
    model = Predictor(fit)
    output = {}
    for (task, depth), group in cohorts(rows).items():
        name = f"{task}/{depth}"
        predictions = [model.forecast(task, row["public"], row["actions"]) for row in group]
        selected = [{arm: model.choose(task, depth, val, arm) for arm in ARMS} for val in predictions]
        outcomes = [row["outcomes"] for row in group]
        base = [val[BASE] for val in outcomes]
        alternatives = Counter()
        scores = []
        predicted_choice_deltas = defaultdict(list)
        for val, data, choice in zip(predictions, outcomes, selected):
            reference = np.asarray(data[BASE])[:, 0]
            actual = {k: (np.asarray(v)[:, 0] - reference).mean() for k, v in data.items()}
            empirical = model.fixed[name]["action_means"]
            alternatives.update(choice.values())
            for arm, key in choice.items():
                predicted_choice_deltas[arm].append(val[key])
            valid = [k for k in val if k != BASE]
            if valid:
                scores.append(np.mean([[(val[k][0] - actual[k])**2, actual[k]**2,
                    (empirical.get(k, [0.])[0] - actual[k])**2] for k in valid], axis=0))
        output[name] = dict(mean_squared_delta_error=dict(zip(("public_ridge", "zero", "action_mean"),
                               np.asarray(scores).mean(axis=0).tolist())),
                            selected_route_counts=dict(alternatives),
                            policies={arm: dict(summarize([data[choice[arm]] for data, choice in zip(outcomes, selected)], base),
                                                predicted_mean_delta=np.mean(predicted_choice_deltas[arm], axis=0).tolist(),
                                                interventions=sum(choice[arm] != BASE for choice in selected))
                                      for arm in ARMS if arm != BASE})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    training, development = read_rows(args.train), read_rows(args.development)
    assert {r["case_seed"] for r in training}.isdisjoint({r["case_seed"] for r in development})
    fit = train(training)
    fit["fixed"] = empirical_fixed(training)
    model = Predictor(fit)
    fit["prices"] = {f"{task}/{depth}": cost_prices([model.forecast(task, row["public"], row["actions"]) for row in group])
                     for (task, depth), group in cohorts(development).items()}
    audit = evaluate_development(development, fit)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("predictor", fit), ("forecast_audit", audit)):
        (args.out / f"{name}.json").write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    for key, result in audit.items():
        print(key, result["mean_squared_delta_error"], {arm: round(row["delta"]["correct_completion"], 4)
                                                        for arm, row in result["policies"].items()})


if __name__ == "__main__":
    main()
