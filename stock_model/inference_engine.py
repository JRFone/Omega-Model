from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float
    transform: str = "bounded"
    estimated: bool = True
    phase: int = 1
    prior_mean: float | None = None
    prior_sd: float | None = None


@dataclass
class InferenceResult:
    parameters: dict[str, float]
    objective: float
    gradient: dict[str, float]
    maximum_gradient: float
    hessian: list[list[float]]
    covariance: list[list[float]]
    standard_errors: dict[str, float]
    correlations: list[list[float]]
    diagnostics: dict[str, Any]
    trace: list[dict[str, float | int]]


def to_unconstrained(value: float, spec: ParameterSpec) -> float:
    if spec.transform == "log":
        return log(max(value, 1e-300))
    if spec.transform == "identity":
        return float(value)
    width = max(spec.upper - spec.lower, 1e-12)
    p = np.clip((value - spec.lower) / width, 1e-12, 1 - 1e-12)
    return float(log(p / (1 - p)))


def from_unconstrained(value: float, spec: ParameterSpec) -> float:
    if spec.transform == "log":
        return float(np.clip(exp(np.clip(value, -700, 700)), spec.lower, spec.upper))
    if spec.transform == "identity":
        return float(np.clip(value, spec.lower, spec.upper))
    p = 1.0 / (1.0 + exp(-float(np.clip(value, -60, 60))))
    return float(spec.lower + (spec.upper - spec.lower) * p)


def parameter_dict(vector: Sequence[float], specs: Sequence[ParameterSpec]) -> dict[str, float]:
    estimated = [spec for spec in specs if spec.estimated]
    if len(vector) != len(estimated):
        raise ValueError("Parameter vector length does not match estimated parameter specifications.")
    result: dict[str, float] = {}
    cursor = 0
    for spec in specs:
        if spec.estimated:
            result[spec.name] = from_unconstrained(float(vector[cursor]), spec)
            cursor += 1
        else:
            result[spec.name] = float(spec.initial)
    return result


def prior_penalty(parameters: Mapping[str, float], specs: Sequence[ParameterSpec]) -> float:
    total = 0.0
    for spec in specs:
        if spec.prior_mean is None or spec.prior_sd is None:
            continue
        sd = max(float(spec.prior_sd), 1e-12)
        total += 0.5 * ((float(parameters[spec.name]) - float(spec.prior_mean)) / sd) ** 2 + log(sd)
    return float(total)


def finite_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        h = step * max(abs(x[i]), 1.0)
        plus = x.copy(); plus[i] += h
        minus = x.copy(); minus[i] -= h
        gradient[i] = (func(plus) - func(minus)) / (2.0 * h)
    return gradient


def finite_hessian(func: Callable[[np.ndarray], float], x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    hessian = np.zeros((n, n), dtype=float)
    f0 = func(x)
    steps = np.asarray([step * max(abs(v), 1.0) for v in x])
    for i in range(n):
        plus = x.copy(); plus[i] += steps[i]
        minus = x.copy(); minus[i] -= steps[i]
        hessian[i, i] = (func(plus) - 2.0 * f0 + func(minus)) / steps[i] ** 2
        for j in range(i + 1, n):
            pp = x.copy(); pp[i] += steps[i]; pp[j] += steps[j]
            pm = x.copy(); pm[i] += steps[i]; pm[j] -= steps[j]
            mp = x.copy(); mp[i] -= steps[i]; mp[j] += steps[j]
            mm = x.copy(); mm[i] -= steps[i]; mm[j] -= steps[j]
            value = (func(pp) - func(pm) - func(mp) + func(mm)) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _nelder_mead(func: Callable[[np.ndarray], float], start: np.ndarray, rounds: int = 500, tolerance: float = 1e-8) -> tuple[np.ndarray, float, list[dict[str, float | int]]]:
    start = np.asarray(start, dtype=float)
    n = len(start)
    simplex = [start.copy()]
    for i in range(n):
        point = start.copy(); point[i] += 0.2 * max(abs(start[i]), 1.0)
        simplex.append(point)
    values = [float(func(point)) for point in simplex]
    trace: list[dict[str, float | int]] = []
    for iteration in range(rounds):
        order = np.argsort(values)
        simplex = [simplex[int(i)] for i in order]
        values = [values[int(i)] for i in order]
        trace.append({"iteration": iteration, "objective": values[0], "simplex_spread": float(np.std(values))})
        if np.std(values) < tolerance:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        reflected_value = float(func(reflected))
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded_value = float(func(expanded))
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = centroid + 0.5 * (simplex[-1] - centroid)
        contracted_value = float(func(contracted))
        if contracted_value < values[-1]:
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (point - best) for point in simplex[1:]]
        values = [float(func(point)) for point in simplex]
    best_i = int(np.argmin(values))
    return simplex[best_i], values[best_i], trace


def fit_parameters(
    objective: Callable[[Mapping[str, float]], float],
    specs: Sequence[ParameterSpec],
    starts: int = 5,
    seed: int = 24601,
    rounds: int = 500,
) -> InferenceResult:
    estimated = [spec for spec in specs if spec.estimated]
    if not estimated:
        params = {spec.name: spec.initial for spec in specs}
        return InferenceResult(params, float(objective(params)), {}, 0.0, [], [], {}, [], {"converged": True, "parameters_estimated": 0}, [])
    rng = np.random.default_rng(seed)
    base = np.asarray([to_unconstrained(spec.initial, spec) for spec in estimated], dtype=float)

    def wrapped(vector: np.ndarray) -> float:
        params = parameter_dict(vector, specs)
        value = float(objective(params)) + prior_penalty(params, specs)
        return value if np.isfinite(value) else 1e100

    solutions = []
    all_trace: list[dict[str, float | int]] = []
    for start_index in range(max(int(starts), 1)):
        start = base.copy() if start_index == 0 else base + rng.normal(0.0, 1.0, size=len(base))
        solution, value, trace = _nelder_mead(wrapped, start, rounds=rounds)
        for row in trace:
            all_trace.append({**row, "start": start_index})
        solutions.append((value, solution))
    solutions.sort(key=lambda item: item[0])
    best_value, best_vector = solutions[0]
    gradient = finite_gradient(wrapped, best_vector)
    hessian = finite_hessian(wrapped, best_vector)
    eigenvalues = np.linalg.eigvalsh((hessian + hessian.T) / 2.0)
    positive_definite = bool(np.all(eigenvalues > 1e-10))
    covariance = np.linalg.pinv(hessian)
    covariance = (covariance + covariance.T) / 2.0
    standard_error_vector = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denom = np.outer(standard_error_vector, standard_error_vector)
    correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
    params = parameter_dict(best_vector, specs)
    gradient_dict = {spec.name: float(value) for spec, value in zip(estimated, gradient)}
    se_dict = {spec.name: float(value) for spec, value in zip(estimated, standard_error_vector)}
    near_boundary = []
    for spec in estimated:
        value = params[spec.name]
        span = max(spec.upper - spec.lower, 1e-12)
        if min(value - spec.lower, spec.upper - value) / span < 0.01:
            near_boundary.append(spec.name)
    diagnostics = {
        "converged": bool(np.max(np.abs(gradient)) < 1e-3),
        "maximum_gradient": float(np.max(np.abs(gradient))),
        "hessian_positive_definite": positive_definite,
        "hessian_min_eigenvalue": float(np.min(eigenvalues)),
        "hessian_max_eigenvalue": float(np.max(eigenvalues)),
        "hessian_condition_number": float(np.linalg.cond(hessian)),
        "near_boundary": near_boundary,
        "starts": len(solutions),
        "objective_spread": float(solutions[-1][0] - solutions[0][0]) if len(solutions) > 1 else 0.0,
        "parameter_specs": [asdict(spec) for spec in specs],
    }
    return InferenceResult(
        params,
        float(best_value),
        gradient_dict,
        diagnostics["maximum_gradient"],
        hessian.tolist(),
        covariance.tolist(),
        se_dict,
        correlation.tolist(),
        diagnostics,
        all_trace,
    )


def profile_parameter(
    objective: Callable[[Mapping[str, float]], float],
    specs: Sequence[ParameterSpec],
    parameter_name: str,
    values: Sequence[float],
    starts: int = 2,
    seed: int = 777,
) -> list[dict[str, Any]]:
    rows = []
    for index, value in enumerate(values):
        changed = []
        found = False
        for spec in specs:
            if spec.name == parameter_name:
                changed.append(ParameterSpec(**{**asdict(spec), "initial": float(value), "estimated": False}))
                found = True
            else:
                changed.append(spec)
        if not found:
            raise ValueError(f"Unknown profile parameter: {parameter_name}")
        fit = fit_parameters(objective, changed, starts=starts, seed=seed + index)
        rows.append({
            "parameter": parameter_name,
            "value": float(value),
            "objective": fit.objective,
            "maximum_gradient": fit.maximum_gradient,
            "converged": fit.diagnostics["converged"],
            "fitted_parameters": fit.parameters,
        })
    minimum = min(row["objective"] for row in rows) if rows else 0.0
    for row in rows:
        row["objective_delta"] = float(row["objective"] - minimum)
    return rows


def parametric_bootstrap(
    fit_function: Callable[[Any, int], Mapping[str, float]],
    simulation_function: Callable[[int], Any],
    parameter_names: Sequence[str],
    replicates: int = 100,
    seed: int = 4321,
) -> dict[str, Any]:
    rows = []
    failures = []
    for replicate in range(max(int(replicates), 1)):
        replicate_seed = seed + replicate * 1009
        try:
            data = simulation_function(replicate_seed)
            fitted = dict(fit_function(data, replicate_seed + 17))
            rows.append({"replicate": replicate, **{name: float(fitted[name]) for name in parameter_names}})
        except Exception as exc:  # bootstrap should report failures rather than abort the batch
            failures.append({"replicate": replicate, "error": str(exc)})
    summary = {}
    for name in parameter_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        summary[name] = {
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "lower_95": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
            "median": float(np.quantile(values, 0.5)) if len(values) else float("nan"),
            "upper_95": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
        }
    return {"replicates_requested": replicates, "successful": len(rows), "failed": len(failures), "rows": rows, "failures": failures, "summary": summary}


def random_walk_mcmc(
    log_posterior: Callable[[Mapping[str, float]], float],
    specs: Sequence[ParameterSpec],
    start: Mapping[str, float] | None = None,
    iterations: int = 5000,
    burn: int = 1000,
    thin: int = 5,
    proposal_scale: float = 0.12,
    seed: int = 98765,
) -> dict[str, Any]:
    estimated = [spec for spec in specs if spec.estimated]
    current = np.asarray([to_unconstrained((start or {}).get(spec.name, spec.initial), spec) for spec in estimated], dtype=float)
    rng = np.random.default_rng(seed)

    def lp(vector: np.ndarray) -> float:
        params = parameter_dict(vector, specs)
        return float(log_posterior(params)) - prior_penalty(params, specs)

    current_lp = lp(current)
    accepted = 0
    samples = []
    for iteration in range(max(int(iterations), 1)):
        proposal = current + rng.normal(0.0, proposal_scale, size=len(current))
        proposal_lp = lp(proposal)
        if np.isfinite(proposal_lp) and log(rng.uniform()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            accepted += 1
        if iteration >= burn and (iteration - burn) % max(int(thin), 1) == 0:
            samples.append({"iteration": iteration, "log_posterior": current_lp, **parameter_dict(current, specs)})
    summaries = {}
    for spec in estimated:
        values = np.asarray([row[spec.name] for row in samples], dtype=float)
        summaries[spec.name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return {
        "samples": samples,
        "summary": summaries,
        "acceptance_rate": accepted / max(int(iterations), 1),
        "iterations": iterations,
        "burn": burn,
        "thin": thin,
        "warning": "Random-walk MCMC is a functional baseline, not a substitute for HMC/NUTS diagnostics in high-dimensional assessments.",
    }
