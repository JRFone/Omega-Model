from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, log
from typing import Any, Callable

import numpy as np
import pandas as pd

from .core import (
    FitResult,
    ModelSettings,
    ProjectionSettings,
    _objective_breakdown,
    _production,
    _reference_points,
    fit,
    project,
)
from .data_io import StockDataset, read_stock_csv


PARAMETER_NAMES = [
    "k",
    "r",
    "initial_depletion",
    "sigma",
    "index_weight",
    "biomass_weight",
    "catch_multiplier",
    "pella_shape",
]


@dataclass(frozen=True)
class QuantOptimizerSettings:
    algorithm: str = "differential_evolution"
    population: int = 48
    generations: int = 35
    seed: int = 7241
    mutation: float = 0.75
    crossover: float = 0.80
    elite_fraction: float = 0.15
    local_refinement_rounds: int = 5


@dataclass(frozen=True)
class QuantLabSettings:
    model: str = "schaefer"
    search_draws: int = 300
    projection_years: int = 20
    projection_iterations: int = 300
    stress_draws: int = 180
    surface_points: int = 15
    sobol_samples: int = 128
    seed: int = 7241


def _parameter_bounds(dataset: StockDataset, base: ModelSettings) -> np.ndarray:
    catch = dataset.frame["catch"].to_numpy(dtype=float)
    max_catch = max(float(np.nanmax(catch)), 1.0)
    total_catch = max(float(np.nansum(catch)), max_catch)
    k_low = max(max_catch * 1.10, total_catch * 0.10)
    k_high = max(k_low * 8.0, total_catch * 20.0)
    biomass_present = bool(dataset.frame["biomass"].notna().any())
    return np.array(
        [
            [k_low, k_high],
            [0.0051, 1.15],
            [0.03, 0.99],
            [0.03, 1.20],
            [0.05, 5.0],
            [0.05, 5.0] if biomass_present else [base.biomass_weight, base.biomass_weight],
            [0.50, 1.60],
            [0.30, 4.00],
        ],
        dtype=float,
    )


def _decode(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + np.asarray(unit, dtype=float) * (bounds[:, 1] - bounds[:, 0])


def _encode(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
    return np.clip((np.asarray(values, dtype=float) - bounds[:, 0]) / span, 0.0, 1.0)


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return log(value / (1.0 - value))


def _candidate_objective(
    unit: np.ndarray,
    dataset: StockDataset,
    base: ModelSettings,
    bounds: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    values = _decode(unit, bounds)
    row = dict(zip(PARAMETER_NAMES, (float(v) for v in values)))
    settings = replace(
        base,
        index_weight=row["index_weight"],
        biomass_weight=row["biomass_weight"],
        catch_multiplier=row["catch_multiplier"],
        pella_shape=row["pella_shape"],
    )
    frame = dataset.frame
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float) * row["catch_multiplier"]
    index = frame["index"].to_numpy(dtype=float)
    biomass_obs = frame["biomass"].to_numpy(dtype=float)
    theta = np.array(
        [
            log(row["k"]),
            log(row["r"]),
            _logit(row["initial_depletion"]),
            log(row["sigma"]),
        ],
        dtype=float,
    )
    objective, pred_b, sigma, components = _objective_breakdown(theta, years, catches, index, biomass_obs, settings)
    reference = _reference_points(row["k"], row["r"], settings.model, row["pella_shape"])
    result = {
        **row,
        "objective": float(objective),
        "terminal_biomass": float(pred_b[-1]) if len(pred_b) else float("nan"),
        "terminal_depletion": float(pred_b[-1] / row["k"]) if len(pred_b) and row["k"] > 0 else float("nan"),
        "msy": reference["msy"],
        "bmsy": reference["bmsy"],
        "fmsy": reference["fmsy"],
        "effective_sigma": float(sigma),
        "objective_components": components,
    }
    return float(objective), result


def _coordinate_refine(
    start: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, dict[str, Any]]],
    rounds: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    current = np.clip(np.asarray(start, dtype=float), 0.0, 1.0)
    best_score, best_row = evaluate(current)
    step = np.full(len(current), 0.10, dtype=float)
    for _ in range(max(int(rounds), 0)):
        improved = False
        for index in range(len(current)):
            for direction in (-1.0, 1.0):
                trial = current.copy()
                trial[index] = np.clip(trial[index] + direction * step[index], 0.0, 1.0)
                score, row = evaluate(trial)
                if score < best_score:
                    current, best_score, best_row = trial, score, row
                    improved = True
        if not improved:
            step *= 0.5
    return current, float(best_score), best_row


def run_global_optimizer(
    dataset: StockDataset,
    base_settings: ModelSettings | None = None,
    optimizer_settings: QuantOptimizerSettings | None = None,
) -> dict[str, Any]:
    base = base_settings or ModelSettings()
    config = optimizer_settings or QuantOptimizerSettings()
    bounds = _parameter_bounds(dataset, base)
    rng = np.random.default_rng(config.seed)
    population_size = max(12, int(config.population))
    generations = max(1, int(config.generations))
    evaluate = lambda unit: _candidate_objective(unit, dataset, base, bounds)
    population = rng.random((population_size, len(PARAMETER_NAMES)))
    scores = np.empty(population_size, dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(population_size):
        scores[index], row = evaluate(population[index])
        rows.append(row)
    history: list[dict[str, float]] = []
    archive: list[dict[str, Any]] = []

    algorithm = str(config.algorithm or "differential_evolution").lower()
    for generation in range(generations):
        if algorithm in {"genetic", "genetic_algorithm", "ga"}:
            population, scores, rows = _genetic_generation(population, scores, rows, evaluate, rng, config)
        elif algorithm in {"cma_es", "cma-es", "covariance_es", "covariance_adaptation"}:
            population, scores, rows = _cma_generation(population, scores, rows, evaluate, rng, config)
        elif algorithm in {"nelder_mead", "nelder-mead", "simplex"}:
            population, scores, rows = _nelder_mead_generation(population, scores, rows, evaluate, rng, config)
        elif algorithm in {"random", "random_multistart"}:
            trial = rng.random(population.shape)
            for index in range(population_size):
                score, row = evaluate(trial[index])
                if score < scores[index]:
                    population[index], scores[index], rows[index] = trial[index], score, row
        else:
            population, scores, rows = _de_generation(population, scores, rows, evaluate, rng, config)
        order = np.argsort(scores)
        best_index = int(order[0])
        history.append(
            {
                "generation": float(generation + 1),
                "best_objective": float(scores[best_index]),
                "median_objective": float(np.median(scores)),
                "objective_spread": float(np.quantile(scores, 0.90) - np.quantile(scores, 0.10)),
            }
        )
        for index in order[: min(5, population_size)]:
            archive.append({"generation": generation + 1, **rows[int(index)]})

    best_index = int(np.argmin(scores))
    refined_unit, refined_score, refined_row = _coordinate_refine(
        population[best_index],
        evaluate,
        config.local_refinement_rounds,
    )
    if refined_score < scores[best_index]:
        population[best_index], scores[best_index], rows[best_index] = refined_unit, refined_score, refined_row

    order = np.argsort(scores)
    candidates = []
    for rank, index in enumerate(order, start=1):
        row = dict(rows[int(index)])
        row["rank"] = rank
        row["algorithm"] = algorithm
        candidates.append(row)
    diagnostics = high_dimensional_diagnostics(candidates, parameter_names=PARAMETER_NAMES)
    diagnostics["local_identifiability"] = local_identifiability_diagnostics(
        dataset,
        base,
        candidates[0],
        profile_points=9,
    )
    return {
        "summary": {
            "algorithm": algorithm,
            "population": population_size,
            "generations": generations,
            "evaluations_minimum": population_size * (generations + 1),
            "best_objective": float(candidates[0]["objective"]),
            "best_terminal_depletion": float(candidates[0]["terminal_depletion"]),
            "dimensions": len(PARAMETER_NAMES),
            "interpretation": "Global optimiser result. It identifies low-objective parameter sets but does not by itself prove identifiability or model correctness.",
        },
        "settings": asdict(config),
        "parameter_bounds": [
            {"parameter": name, "low": float(bounds[i, 0]), "high": float(bounds[i, 1])}
            for i, name in enumerate(PARAMETER_NAMES)
        ],
        "history": history,
        "candidates": candidates,
        "archive": sorted(archive, key=lambda row: row["objective"])[:100],
        "diagnostics": diagnostics,
    }


def _de_generation(population, scores, rows, evaluate, rng, config):
    size, dimensions = population.shape
    next_population = population.copy()
    next_scores = scores.copy()
    next_rows = list(rows)
    for index in range(size):
        choices = [i for i in range(size) if i != index]
        a, b, c = rng.choice(choices, size=3, replace=False)
        mutant = np.clip(population[a] + config.mutation * (population[b] - population[c]), 0.0, 1.0)
        mask = rng.random(dimensions) < config.crossover
        mask[int(rng.integers(0, dimensions))] = True
        trial = np.where(mask, mutant, population[index])
        score, row = evaluate(trial)
        if score < scores[index]:
            next_population[index], next_scores[index], next_rows[index] = trial, score, row
    return next_population, next_scores, next_rows


def _genetic_generation(population, scores, rows, evaluate, rng, config):
    size, dimensions = population.shape
    elite_count = max(1, min(size - 2, int(round(size * config.elite_fraction))))
    order = np.argsort(scores)
    new_population = [population[int(i)].copy() for i in order[:elite_count]]
    while len(new_population) < size:
        parent_a = population[_tournament(scores, rng)]
        parent_b = population[_tournament(scores, rng)]
        blend = rng.random(dimensions)
        child = blend * parent_a + (1.0 - blend) * parent_b
        mutation_mask = rng.random(dimensions) < (1.0 / dimensions)
        child += mutation_mask * rng.normal(0.0, max(config.mutation, 0.01) * 0.12, dimensions)
        new_population.append(np.clip(child, 0.0, 1.0))
    new_population_array = np.asarray(new_population, dtype=float)
    new_scores = np.empty(size, dtype=float)
    new_rows = []
    for index in range(size):
        score, row = evaluate(new_population_array[index])
        new_scores[index] = score
        new_rows.append(row)
    return new_population_array, new_scores, new_rows



def _cma_generation(population, scores, rows, evaluate, rng, config):
    """Covariance-adapting evolution step in normalized parameter space.

    This is dependency-free and follows the core CMA-ES concept of sampling
    from an elite-weighted mean and covariance. It is reported as an internal
    covariance-adaptation optimiser, not a reference CMA-ES implementation.
    """
    size, dimensions = population.shape
    elite_count = max(2, min(size, int(round(size * max(config.elite_fraction, 0.10)))))
    order = np.argsort(scores)
    elite = population[order[:elite_count]]
    ranks = np.arange(1, elite_count + 1, dtype=float)
    weights = np.log(elite_count + 0.5) - np.log(ranks)
    weights = weights / weights.sum()
    mean = np.sum(elite * weights[:, None], axis=0)
    centered = elite - mean
    covariance = centered.T @ (centered * weights[:, None])
    covariance += np.eye(dimensions) * 1e-4
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-7)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    spread = float(np.mean(np.sqrt(eigenvalues)))
    sigma = min(max(0.08 + 1.5 * spread, 0.05), 0.35)

    next_population = np.empty_like(population)
    next_scores = np.empty_like(scores)
    next_rows: list[dict[str, Any]] = []
    next_population[0] = population[int(order[0])]
    next_scores[0] = scores[int(order[0])]
    next_rows.append(rows[int(order[0])])
    for index in range(1, size):
        step = transform @ rng.normal(size=dimensions)
        candidate = np.clip(mean + sigma * step, 0.0, 1.0)
        score, row = evaluate(candidate)
        next_population[index] = candidate
        next_scores[index] = score
        next_rows.append(row)
    return next_population, next_scores, next_rows


def _nelder_mead_generation(population, scores, rows, evaluate, rng, config):
    """One bounded Nelder-Mead simplex update plus multi-start maintenance."""
    size, dimensions = population.shape
    order = np.argsort(scores)
    simplex_indices = order[: min(dimensions + 1, size)]
    simplex = population[simplex_indices].copy()
    simplex_scores = scores[simplex_indices].copy()
    simplex_rows = [rows[int(i)] for i in simplex_indices]
    local_order = np.argsort(simplex_scores)
    simplex = simplex[local_order]
    simplex_scores = simplex_scores[local_order]
    simplex_rows = [simplex_rows[int(i)] for i in local_order]

    best = simplex[0]
    worst = simplex[-1]
    second_worst_score = simplex_scores[-2] if len(simplex_scores) > 1 else simplex_scores[-1]
    centroid = np.mean(simplex[:-1], axis=0) if len(simplex) > 1 else best.copy()
    reflected = np.clip(centroid + (centroid - worst), 0.0, 1.0)
    reflected_score, reflected_row = evaluate(reflected)

    replacement = reflected
    replacement_score = reflected_score
    replacement_row = reflected_row
    if reflected_score < simplex_scores[0]:
        expanded = np.clip(centroid + 2.0 * (reflected - centroid), 0.0, 1.0)
        expanded_score, expanded_row = evaluate(expanded)
        if expanded_score < reflected_score:
            replacement, replacement_score, replacement_row = expanded, expanded_score, expanded_row
    elif reflected_score >= second_worst_score:
        contracted = np.clip(centroid + 0.5 * (worst - centroid), 0.0, 1.0)
        contracted_score, contracted_row = evaluate(contracted)
        if contracted_score < simplex_scores[-1]:
            replacement, replacement_score, replacement_row = contracted, contracted_score, contracted_row
        else:
            for index in range(1, len(simplex)):
                simplex[index] = np.clip(best + 0.5 * (simplex[index] - best), 0.0, 1.0)
                simplex_scores[index], simplex_rows[index] = evaluate(simplex[index])
            replacement = simplex[-1]
            replacement_score = simplex_scores[-1]
            replacement_row = simplex_rows[-1]

    simplex[-1] = replacement
    simplex_scores[-1] = replacement_score
    simplex_rows[-1] = replacement_row

    next_population = population.copy()
    next_scores = scores.copy()
    next_rows = list(rows)
    for slot, source in enumerate(simplex_indices):
        next_population[int(source)] = simplex[slot]
        next_scores[int(source)] = simplex_scores[slot]
        next_rows[int(source)] = simplex_rows[slot]

    for source in order[len(simplex_indices):]:
        if rng.random() < 0.55:
            candidate = np.clip(best + rng.normal(0.0, 0.08, dimensions), 0.0, 1.0)
        else:
            candidate = rng.random(dimensions)
        score, row = evaluate(candidate)
        if score < next_scores[int(source)]:
            next_population[int(source)] = candidate
            next_scores[int(source)] = score
            next_rows[int(source)] = row
    return next_population, next_scores, next_rows


def _tournament(scores: np.ndarray, rng: np.random.Generator, size: int = 3) -> int:
    choices = rng.integers(0, len(scores), size=max(2, size))
    return int(choices[int(np.argmin(scores[choices]))])


def high_dimensional_diagnostics(
    candidates: list[dict[str, Any]],
    parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    names = parameter_names or PARAMETER_NAMES
    valid = [row for row in candidates if all(np.isfinite(float(row.get(name, np.nan))) for name in names)]
    if not valid:
        return {"parameter_names": names, "parallel_coordinates": [], "correlations": [], "pca": [], "importance": []}
    matrix = np.array([[float(row[name]) for name in names] for row in valid], dtype=float)
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    span = np.where(high > low, high - low, 1.0)
    normalized = (matrix - low) / span
    objective = np.array([float(row.get("objective", np.nan)) for row in valid], dtype=float)
    terminal = np.array([float(row.get("terminal_depletion", np.nan)) for row in valid], dtype=float)
    parallel_rows = []
    for source, values in zip(valid[:200], normalized[:200]):
        parallel_rows.append(
            {
                "rank": int(source.get("rank", len(parallel_rows) + 1)),
                "objective": float(source.get("objective", np.nan)),
                "terminal_depletion": float(source.get("terminal_depletion", np.nan)),
                "values": {name: float(value) for name, value in zip(names, values)},
            }
        )

    correlations = []
    combined_names = names + ["objective", "terminal_depletion"]
    combined = np.column_stack([matrix, objective, terminal])
    corr = np.corrcoef(combined, rowvar=False) if len(valid) > 1 else np.eye(len(combined_names))
    corr = np.nan_to_num(corr)
    for i, left in enumerate(combined_names):
        for j in range(i + 1, len(combined_names)):
            correlations.append({"x": left, "y": combined_names[j], "correlation": float(corr[i, j])})

    centered = normalized - normalized.mean(axis=0)
    if len(valid) > 1:
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        variance_ratio = variance / max(float(variance.sum()), 1e-12)
    else:
        vt = np.eye(len(names))
        variance_ratio = np.zeros(len(names))
    pca = []
    for component in range(min(len(names), 8)):
        loadings = {name: float(vt[component, i]) for i, name in enumerate(names)}
        pca.append(
            {
                "component": component + 1,
                "variance_ratio": float(variance_ratio[component]) if component < len(variance_ratio) else 0.0,
                "loadings": loadings,
            }
        )

    importance = []
    objective_rank = _rank(objective)
    terminal_rank = _rank(terminal)
    for i, name in enumerate(names):
        parameter_rank = _rank(matrix[:, i])
        importance.append(
            {
                "parameter": name,
                "objective_rank_correlation": _safe_corr(parameter_rank, objective_rank),
                "terminal_depletion_rank_correlation": _safe_corr(parameter_rank, terminal_rank),
                "range_low": float(low[i]),
                "range_high": float(high[i]),
            }
        )
    importance.sort(key=lambda row: abs(row["terminal_depletion_rank_correlation"]), reverse=True)
    return {
        "parameter_names": names,
        "parallel_coordinates": parallel_rows,
        "correlations": correlations,
        "pca": pca,
        "importance": importance,
        "note": "Parallel-coordinate values are normalized within the evaluated candidate set. Rank correlations show association, not causation.",
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def objective_surface(
    dataset: StockDataset,
    base_settings: ModelSettings,
    best_candidate: dict[str, Any],
    points: int = 15,
) -> list[dict[str, float]]:
    bounds = _parameter_bounds(dataset, base_settings)
    points = max(5, min(int(points), 50))
    k_values = np.geomspace(max(bounds[0, 0], 1e-9), bounds[0, 1], points)
    r_values = np.geomspace(max(bounds[1, 0], 1e-9), bounds[1, 1], points)
    base_values = np.array([float(best_candidate[name]) for name in PARAMETER_NAMES], dtype=float)
    rows = []
    for k in k_values:
        for r in r_values:
            values = base_values.copy()
            values[0], values[1] = k, r
            unit = _encode(values, bounds)
            score, candidate = _candidate_objective(unit, dataset, base_settings, bounds)
            rows.append(
                {
                    "k": float(k),
                    "r": float(r),
                    "objective": float(score),
                    "objective_delta": float(score - float(best_candidate["objective"])),
                    "terminal_depletion": float(candidate["terminal_depletion"]),
                }
            )
    return rows



def local_identifiability_diagnostics(
    dataset: StockDataset,
    base_settings: ModelSettings,
    best_candidate: dict[str, Any],
    profile_points: int = 9,
    step: float = 0.025,
) -> dict[str, Any]:
    """Finite-difference local curvature and one-dimensional profile diagnostics."""
    bounds = _parameter_bounds(dataset, base_settings)
    values = np.array([float(best_candidate[name]) for name in PARAMETER_NAMES], dtype=float)
    center = _encode(values, bounds)
    evaluate = lambda unit: _candidate_objective(np.clip(unit, 0.0, 1.0), dataset, base_settings, bounds)[0]
    base_objective = float(evaluate(center))
    dimensions = len(center)
    h = max(float(step), 1e-4)
    hessian = np.zeros((dimensions, dimensions), dtype=float)

    for i in range(dimensions):
        plus = center.copy()
        minus = center.copy()
        plus[i] = min(1.0, center[i] + h)
        minus[i] = max(0.0, center[i] - h)
        hp = max(plus[i] - center[i], 1e-8)
        hm = max(center[i] - minus[i], 1e-8)
        fp = evaluate(plus)
        fm = evaluate(minus)
        hessian[i, i] = 2.0 * (
            fp / (hp * (hp + hm))
            - base_objective / (hp * hm)
            + fm / (hm * (hp + hm))
        )
        for j in range(i + 1, dimensions):
            pp = center.copy()
            pm = center.copy()
            mp = center.copy()
            mm = center.copy()
            pp[i], pp[j] = min(1.0, center[i] + h), min(1.0, center[j] + h)
            pm[i], pm[j] = min(1.0, center[i] + h), max(0.0, center[j] - h)
            mp[i], mp[j] = max(0.0, center[i] - h), min(1.0, center[j] + h)
            mm[i], mm[j] = max(0.0, center[i] - h), max(0.0, center[j] - h)
            di = max(pp[i] - mp[i], 1e-8)
            dj = max(pp[j] - pm[j], 1e-8)
            cross = (evaluate(pp) - evaluate(pm) - evaluate(mp) + evaluate(mm)) / (di * dj)
            hessian[i, j] = hessian[j, i] = cross

    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-12)
    positive = eigenvalues[eigenvalues > scale * 1e-8]
    condition_number = float(np.max(positive) / np.min(positive)) if len(positive) >= 2 else float("inf")
    effective_rank = int(len(positive))
    weak_directions = []
    for rank in range(min(4, dimensions)):
        vector = eigenvectors[:, rank]
        loading_order = np.argsort(np.abs(vector))[::-1][:4]
        weak_directions.append(
            {
                "direction": rank + 1,
                "curvature_eigenvalue": float(eigenvalues[rank]),
                "dominant_loadings": {
                    PARAMETER_NAMES[int(i)]: float(vector[int(i)])
                    for i in loading_order
                },
            }
        )

    points = max(5, min(int(profile_points), 31))
    profile_rows = []
    for dimension, name in enumerate(PARAMETER_NAMES):
        grid = np.linspace(0.0, 1.0, points)
        for unit_value in grid:
            trial = center.copy()
            trial[dimension] = unit_value
            decoded = _decode(trial, bounds)
            objective = float(evaluate(trial))
            profile_rows.append(
                {
                    "parameter": name,
                    "value": float(decoded[dimension]),
                    "unit_value": float(unit_value),
                    "objective": objective,
                    "objective_delta": objective - base_objective,
                }
            )

    status = "well_conditioned"
    if effective_rank < dimensions:
        status = "rank_deficient"
    elif not np.isfinite(condition_number) or condition_number > 1e8:
        status = "severely_ill_conditioned"
    elif condition_number > 1e5:
        status = "ill_conditioned"
    elif condition_number > 1e3:
        status = "weakly_conditioned"

    return {
        "status": status,
        "effective_rank": effective_rank,
        "dimensions": dimensions,
        "condition_number": condition_number,
        "eigenvalues": [float(value) for value in eigenvalues],
        "weak_directions": weak_directions,
        "profiles": profile_rows,
        "hessian": [
            {
                "row_parameter": PARAMETER_NAMES[i],
                **{PARAMETER_NAMES[j]: float(hessian[i, j]) for j in range(dimensions)},
            }
            for i in range(dimensions)
        ],
        "warning": (
            "Finite-difference curvature is local and depends on parameter scaling. "
            "Flat or negative-curvature directions indicate weak local identification, "
            "boundary effects or an imperfect optimum; they are not formal confidence intervals."
        ),
    }


def _projection_risk(projection: dict[str, Any]) -> dict[str, float]:
    rows = projection.get("projection") or []
    catches = np.array([float(row.get("catch_median", 0.0)) for row in rows], dtype=float)
    depletions = np.array([float(row.get("depletion_median", np.nan)) for row in rows], dtype=float)
    probabilities = np.array([float(row.get("prob_below_limit", 0.0)) for row in rows], dtype=float)
    shortfalls = np.array([float(row.get("expected_limit_shortfall", 0.0)) for row in rows], dtype=float)
    catch_volatility = float(np.std(np.diff(catches)) / max(np.mean(catches), 1e-9)) if len(catches) > 1 else 0.0
    return {
        "mean_catch": float(np.mean(catches)) if len(catches) else 0.0,
        "median_terminal_depletion": float(depletions[-1]) if len(depletions) else float("nan"),
        "maximum_probability_below_limit": float(np.max(probabilities)) if len(probabilities) else 0.0,
        "terminal_probability_below_limit": float(probabilities[-1]) if len(probabilities) else 0.0,
        "mean_expected_limit_shortfall": float(np.mean(shortfalls)) if len(shortfalls) else 0.0,
        "catch_volatility": catch_volatility,
    }


def run_hcr_genetic_optimization(
    fit_result: FitResult,
    years: int = 20,
    iterations: int = 250,
    population: int = 30,
    generations: int = 18,
    seed: int = 8831,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    population = max(12, int(population))
    generations = max(1, int(generations))
    # pstar, target depletion, limit fraction of target, maximum exploitation fraction
    genes = rng.random((population, 4))
    evaluated = [_evaluate_hcr_gene(gene, fit_result, years, iterations, seed + i) for i, gene in enumerate(genes)]
    history = []
    for generation in range(generations):
        fronts = _non_dominated_sort([row["objectives"] for row in evaluated])
        crowding = _crowding_by_front([row["objectives"] for row in evaluated], fronts)
        new_genes = []
        while len(new_genes) < population:
            a = _pareto_tournament(fronts, crowding, rng)
            b = _pareto_tournament(fronts, crowding, rng)
            blend = rng.random(4)
            child = blend * genes[a] + (1.0 - blend) * genes[b]
            child += (rng.random(4) < 0.25) * rng.normal(0.0, 0.08, 4)
            new_genes.append(np.clip(child, 0.0, 1.0))
        genes = np.asarray(new_genes)
        evaluated = [
            _evaluate_hcr_gene(gene, fit_result, years, iterations, seed + (generation + 1) * 1009 + i)
            for i, gene in enumerate(genes)
        ]
        fronts = _non_dominated_sort([row["objectives"] for row in evaluated])
        front_zero = [i for i, rank in enumerate(fronts) if rank == 0]
        history.append(
            {
                "generation": generation + 1,
                "pareto_solutions": len(front_zero),
                "highest_mean_catch": max(evaluated[i]["mean_catch"] for i in front_zero),
                "lowest_terminal_risk": min(evaluated[i]["terminal_probability_below_limit"] for i in front_zero),
            }
        )
    fronts = _non_dominated_sort([row["objectives"] for row in evaluated])
    pareto = [dict(evaluated[i], pareto_rank=int(fronts[i])) for i in range(population) if fronts[i] == 0]
    pareto.sort(key=lambda row: (row["terminal_probability_below_limit"], -row["mean_catch"]))
    all_rows = [dict(row, pareto_rank=int(fronts[i])) for i, row in enumerate(evaluated)]
    return {
        "summary": {
            "population": population,
            "generations": generations,
            "pareto_solutions": len(pareto),
            "objectives": ["maximize mean catch", "minimize limit risk", "minimize catch volatility", "minimize expected shortfall"],
            "note": "Pareto solutions expose trade-offs; there is no single scientifically correct optimum without management preferences.",
        },
        "history": history,
        "pareto": pareto,
        "all_strategies": sorted(all_rows, key=lambda row: (row["pareto_rank"], row["terminal_probability_below_limit"], -row["mean_catch"])),
    }


def _evaluate_hcr_gene(gene, fit_result, years, iterations, seed):
    pstar = 0.20 + gene[0] * 0.40
    target = 0.30 + gene[1] * 0.30
    limit = target * (0.15 + gene[2] * 0.60)
    maximum_exploitation = 0.35 + gene[3] * 0.55
    settings = ProjectionSettings(
        years=max(1, int(years)),
        iterations=max(40, int(iterations)),
        strategy="hcr_40_10",
        target_depletion=float(target),
        limit_depletion=float(min(limit, target - 0.01)),
        pstar=float(pstar),
        maximum_exploitation_fraction=float(maximum_exploitation),
        process_cv=float(fit_result.settings.get("process_cv", 0.12)),
        seed=int(seed),
    )
    projection = project(fit_result, settings)
    metrics = _projection_risk(projection)
    return {
        "pstar": float(pstar),
        "target_depletion": float(target),
        "limit_depletion": float(settings.limit_depletion),
        "maximum_exploitation_fraction": float(maximum_exploitation),
        **metrics,
        "objectives": [
            -metrics["mean_catch"],
            metrics["terminal_probability_below_limit"],
            metrics["catch_volatility"],
            metrics["mean_expected_limit_shortfall"],
        ],
    }


def _dominates(left, right) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _non_dominated_sort(objectives: list[list[float]]) -> list[int]:
    count = len(objectives)
    dominated = [set() for _ in range(count)]
    domination_count = [0] * count
    fronts: list[list[int]] = [[]]
    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated[p].add(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    ranks = [count] * count
    level = 0
    while level < len(fronts) and fronts[level]:
        next_front = []
        for p in fronts[level]:
            ranks[p] = level
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        level += 1
    return ranks


def _crowding_by_front(objectives: list[list[float]], ranks: list[int]) -> list[float]:
    crowding = [0.0] * len(objectives)
    for rank in sorted(set(ranks)):
        members = [i for i, value in enumerate(ranks) if value == rank]
        if len(members) <= 2:
            for i in members:
                crowding[i] = float("inf")
            continue
        for objective_index in range(len(objectives[0])):
            ordered = sorted(members, key=lambda i: objectives[i][objective_index])
            crowding[ordered[0]] = crowding[ordered[-1]] = float("inf")
            low = objectives[ordered[0]][objective_index]
            high = objectives[ordered[-1]][objective_index]
            span = max(high - low, 1e-12)
            for position in range(1, len(ordered) - 1):
                if np.isfinite(crowding[ordered[position]]):
                    crowding[ordered[position]] += (
                        objectives[ordered[position + 1]][objective_index]
                        - objectives[ordered[position - 1]][objective_index]
                    ) / span
    return crowding


def _pareto_tournament(ranks, crowding, rng) -> int:
    left, right = rng.integers(0, len(ranks), size=2)
    if ranks[left] < ranks[right]:
        return int(left)
    if ranks[right] < ranks[left]:
        return int(right)
    return int(left if crowding[left] >= crowding[right] else right)


def run_stress_tests(
    dataset: StockDataset,
    base_settings: ModelSettings | None = None,
    search_draws: int = 180,
    seed: int = 9631,
) -> dict[str, Any]:
    base = replace(base_settings or ModelSettings(), search_draws=max(120, int(search_draws)), seed=seed)
    baseline = fit(dataset, base)
    scenarios = _stress_scenarios(dataset, seed)
    rows = []
    fits = []
    for index, (name, description, stressed) in enumerate(scenarios):
        result = fit(stressed, replace(base, seed=seed + index + 1))
        rows.append(
            {
                "scenario": name,
                "description": description,
                "terminal_depletion": float(result.best["terminal_depletion"]),
                "change_terminal_depletion": float(result.best["terminal_depletion"] - baseline.best["terminal_depletion"]),
                "relative_change_terminal_depletion": float(
                    (result.best["terminal_depletion"] - baseline.best["terminal_depletion"])
                    / max(abs(baseline.best["terminal_depletion"]), 1e-9)
                ),
                "msy": float(result.best["msy"]),
                "change_msy": float(result.best["msy"] - baseline.best["msy"]),
                "objective": float(result.best["objective"]),
            }
        )
        fits.append({"scenario": name, "best": result.best, "history": result.history, "settings": result.settings})
    rows.sort(key=lambda row: abs(row["relative_change_terminal_depletion"]), reverse=True)
    return {
        "summary": {
            "baseline_terminal_depletion": float(baseline.best["terminal_depletion"]),
            "scenarios": len(rows),
            "largest_absolute_relative_change": float(max(abs(row["relative_change_terminal_depletion"]) for row in rows)),
            "note": "Stress tests measure model sensitivity to controlled data distortions. They do not prove which distorted scenario is true.",
        },
        "baseline": {"best": baseline.best, "history": baseline.history, "settings": baseline.settings},
        "stress_tests": rows,
        "fits": fits,
    }


def _stress_scenarios(dataset: StockDataset, seed: int):
    frame = dataset.frame.copy()
    scenarios = []

    def add(name, description, changed):
        scenarios.append((name, description, StockDataset(name=name, frame=changed, provenance=dataset.provenance, transformations=dataset.transformations, warnings=dataset.warnings, raw_columns=dataset.raw_columns, index_columns=dataset.index_columns)))

    for multiplier in (0.80, 1.20):
        changed = frame.copy()
        changed["catch"] *= multiplier
        add(f"catch_{multiplier:.2f}", f"Multiply all catches/removals by {multiplier:.2f}.", changed)
    if frame["index"].notna().any():
        for multiplier in (0.80, 1.20):
            changed = frame.copy()
            changed["index"] *= multiplier
            add(f"index_level_{multiplier:.2f}", f"Multiply all CPUE/index observations by {multiplier:.2f}.", changed)
        positive = frame["index"].dropna()
        scale = max(float(positive.median()), 1e-9)
        for exponent, label in ((0.70, "hyperstable"), (1.30, "hyperdepleted")):
            changed = frame.copy()
            changed["index"] = scale * (changed["index"] / scale) ** exponent
            add(f"index_{label}", f"Apply an index exponent of {exponent:.2f} while preserving the median scale.", changed)
        changed = frame.copy()
        midpoint = len(changed) // 2
        changed.loc[changed.index[midpoint:], "index"] *= 1.25
        add("index_regime_shift_plus25", "Increase second-half index catchability by 25%.", changed)
        changed = frame.copy()
        changed.loc[changed.index[::4], "index"] = np.nan
        add("index_missing_every_fourth_year", "Remove every fourth CPUE/index observation.", changed)
        rng = np.random.default_rng(seed)
        changed = frame.copy()
        mask = changed["index"].notna()
        changed.loc[mask, "index"] *= rng.lognormal(-0.5 * 0.20**2, 0.20, int(mask.sum()))
        add("index_observation_noise_cv20", "Apply reproducible lognormal index noise with CV about 20%.", changed)
    if frame["biomass"].notna().any():
        for multiplier in (0.80, 1.20):
            changed = frame.copy()
            changed["biomass"] *= multiplier
            add(f"biomass_{multiplier:.2f}", f"Multiply biomass observations by {multiplier:.2f}.", changed)
    return scenarios


def detect_index_regime_shift(dataset: StockDataset, minimum_segment: int = 4) -> dict[str, Any]:
    frame = dataset.frame.loc[dataset.frame["index"].notna(), ["year", "index"]].copy()
    if len(frame) < minimum_segment * 2:
        return {"status": "insufficient_data", "points": int(len(frame)), "minimum_required": minimum_segment * 2}
    values = np.log(frame["index"].to_numpy(dtype=float))
    years = frame["year"].to_numpy(dtype=int)
    total_sse = float(np.sum((values - values.mean()) ** 2))
    candidates = []
    for split in range(minimum_segment, len(values) - minimum_segment + 1):
        left, right = values[:split], values[split:]
        sse = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        candidates.append(
            {
                "split_year": int(years[split]),
                "sse": sse,
                "sse_improvement_fraction": float((total_sse - sse) / max(total_sse, 1e-12)),
                "mean_log_index_before": float(left.mean()),
                "mean_log_index_after": float(right.mean()),
                "level_ratio_after_before": float(exp(right.mean() - left.mean())),
            }
        )
    best = max(candidates, key=lambda row: row["sse_improvement_fraction"])
    return {
        "status": "candidate_change_point",
        "points": int(len(frame)),
        "best": best,
        "candidates": candidates,
        "warning": "This is an exploratory structural-break screen. A change can reflect abundance, catchability, targeting, reporting, gear, management or sampling changes.",
    }


def sobol_projection_screen(
    fit_result: FitResult,
    years: int = 20,
    samples: int = 128,
    seed: int = 4421,
) -> dict[str, Any]:
    names = ["r_multiplier", "k_multiplier", "initial_depletion_multiplier", "catch_multiplier", "pstar"]
    bounds = np.array([[0.6, 1.4], [0.7, 1.3], [0.7, 1.3], [0.5, 1.5], [0.25, 0.60]], dtype=float)
    samples = max(32, min(int(samples), 2048))
    rng = np.random.default_rng(seed)
    a = rng.random((samples, len(names)))
    b = rng.random((samples, len(names)))
    a_values = bounds[:, 0] + a * (bounds[:, 1] - bounds[:, 0])
    b_values = bounds[:, 0] + b * (bounds[:, 1] - bounds[:, 0])
    evaluator = lambda values: _deterministic_terminal_depletion(fit_result, values, years)
    ya = np.array([evaluator(row) for row in a_values])
    yb = np.array([evaluator(row) for row in b_values])
    variance = float(np.var(np.concatenate([ya, yb]), ddof=1))
    rows = []
    for dimension, name in enumerate(names):
        ab = a_values.copy()
        ab[:, dimension] = b_values[:, dimension]
        yab = np.array([evaluator(row) for row in ab])
        first = float(np.mean(yb * (yab - ya)) / max(variance, 1e-12))
        total = float(0.5 * np.mean((ya - yab) ** 2) / max(variance, 1e-12))
        rows.append(
            {
                "parameter": name,
                "first_order_index": first,
                "total_order_index": total,
                "interaction_contribution": total - first,
                "low": float(bounds[dimension, 0]),
                "high": float(bounds[dimension, 1]),
            }
        )
    rows.sort(key=lambda row: row["total_order_index"], reverse=True)
    return {
        "summary": {
            "samples_per_matrix": samples,
            "model_evaluations": samples * (2 + len(names)),
            "output_variance": variance,
            "method": "Saltelli-style first and total-order screening on a deterministic projection emulator",
        },
        "sensitivity": rows,
        "warning": "This screen varies a reduced projection emulator around the fitted result. It is not a substitute for full assessment-model uncertainty propagation.",
    }


def _deterministic_terminal_depletion(fit_result: FitResult, values: np.ndarray, years: int) -> float:
    r_mult, k_mult, initial_mult, catch_mult, pstar = (float(v) for v in values)
    model = fit_result.settings.get("model", "schaefer")
    pella_shape = float(fit_result.settings.get("pella_shape", 1.35))
    base_k = max(float(fit_result.best["k_b0"]), 1e-9)
    k = base_k * k_mult
    r = max(float(fit_result.best["r"]) * r_mult, 1e-9)
    depletion = min(max(float(fit_result.best["terminal_depletion"]) * initial_mult, 1e-6), 1.8)
    biomass = depletion * k
    reference = _reference_points(k, r, model, pella_shape)
    target, limit = 0.40, 0.10
    for _ in range(max(1, int(years))):
        dep = biomass / k
        ramp = min(1.0, max(0.0, (dep - limit) / (target - limit)))
        catch = min(reference["msy"] * ramp * pstar / 0.5 * catch_mult, biomass * 0.85)
        biomass = max(1e-6 * k, biomass + _production(biomass, k, r, model, pella_shape) - catch)
    return float(biomass / k)


def run_quant_lab(
    csv_text: str,
    name: str = "Quant Lab dataset",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    config = QuantLabSettings(
        model=str(options.get("model") or "schaefer"),
        search_draws=int(options.get("search_draws") or 300),
        projection_years=int(options.get("projection_years") or 20),
        projection_iterations=int(options.get("projection_iterations") or 300),
        stress_draws=int(options.get("stress_draws") or 180),
        surface_points=int(options.get("surface_points") or 15),
        sobol_samples=int(options.get("sobol_samples") or 128),
        seed=int(options.get("seed") or 7241),
    )
    dataset = read_stock_csv(csv_text, name=name)
    base_settings = ModelSettings(
        model=config.model,
        search_draws=max(120, config.search_draws),
        seed=config.seed,
        r_prior_median=float(options.get("r_prior_median") or 0.18),
        r_prior_cv=float(options.get("r_prior_cv") or 0.75),
        obs_cv=float(options.get("obs_cv") or 0.22),
        process_cv=float(options.get("process_cv") or 0.12),
        index_weight=float(options.get("index_weight") or 1.0),
        biomass_weight=float(options.get("biomass_weight") or 1.0),
        catch_multiplier=float(options.get("catch_multiplier") or 1.0),
        pella_shape=float(options.get("pella_shape") or 1.35),
    )
    optimizer_config = QuantOptimizerSettings(
        algorithm=str(options.get("algorithm") or "differential_evolution"),
        population=int(options.get("population") or 48),
        generations=int(options.get("generations") or 35),
        seed=config.seed,
        mutation=float(options.get("mutation") or 0.75),
        crossover=float(options.get("crossover") or 0.80),
        local_refinement_rounds=int(options.get("local_refinement_rounds") or 5),
    )
    optimizer = run_global_optimizer(dataset, base_settings, optimizer_config)
    best_candidate = optimizer["candidates"][0]
    surface = objective_surface(dataset, base_settings, best_candidate, config.surface_points)
    fitted = fit(dataset, base_settings)
    risk = run_hcr_genetic_optimization(
        fitted,
        years=config.projection_years,
        iterations=config.projection_iterations,
        population=int(options.get("hcr_population") or 30),
        generations=int(options.get("hcr_generations") or 18),
        seed=config.seed + 1,
    )
    stresses = run_stress_tests(dataset, base_settings, config.stress_draws, config.seed + 2)
    regime = detect_index_regime_shift(dataset)
    sobol = sobol_projection_screen(fitted, config.projection_years, config.sobol_samples, config.seed + 3)

    # Imported locally to avoid a module-level cycle: quant_validation reuses
    # the optimiser and core functions defined in this module.
    from .quant_validation import (
        EnsembleSettings,
        OptimizerAgreementSettings,
        WalkForwardSettings,
        run_model_ensemble,
        run_optimizer_agreement,
        run_walk_forward_validation,
    )

    walk_forward = run_walk_forward_validation(
        dataset,
        base_settings,
        WalkForwardSettings(
            minimum_training_years=int(options.get("minimum_training_years") or 6),
            holdout_years=int(options.get("holdout_years") or 1),
            search_draws=int(options.get("walk_forward_search_draws") or 150),
            seed=config.seed + 4,
        ),
    )
    optimizer_agreement = run_optimizer_agreement(
        dataset,
        base_settings,
        OptimizerAgreementSettings(
            population=int(options.get("agreement_population") or 16),
            generations=int(options.get("agreement_generations") or 4),
            seed=config.seed + 5,
        ),
    )
    model_ensemble = run_model_ensemble(
        dataset,
        base_settings,
        EnsembleSettings(
            search_draws=int(options.get("ensemble_search_draws") or 160),
            projection_years=config.projection_years,
            projection_iterations=max(80, int(options.get("ensemble_projection_iterations") or config.projection_iterations)),
            seed=config.seed + 6,
        ),
    )
    return {
        "summary": {
            "dataset": dataset.name,
            "rows": int(len(dataset.frame)),
            "first_year": int(dataset.frame["year"].min()),
            "last_year": int(dataset.frame["year"].max()),
            "dimensions": len(PARAMETER_NAMES),
            "modules": [
                "global parameter optimisation",
                "8-dimensional diagnostics",
                "3-dimensional r-K-objective surface",
                "genetic multi-objective HCR optimisation",
                "data stress tests",
                "regime-shift screening",
                "Saltelli-style projection sensitivity",
                "rolling walk-forward predictive validation",
                "five-optimizer agreement testing",
                "cross-model ensemble and structural disagreement",
                "finite-difference 8D curvature and profile diagnostics",
            ],
        },
        "settings": asdict(config),
        "baseline_fit": {
            "settings": fitted.settings,
            "best": fitted.best,
            "diagnostics": fitted.diagnostics,
            "history": fitted.history,
        },
        "optimizer": optimizer,
        "surface_3d": surface,
        "risk_frontier": risk,
        "stress_tests": stresses,
        "regime_shift": regime,
        "sobol": sobol,
        "walk_forward": walk_forward,
        "optimizer_agreement": optimizer_agreement,
        "model_ensemble": model_ensemble,
    }
