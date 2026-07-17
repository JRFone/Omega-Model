from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import FitResult, ModelSettings, _production, _reference_points, _simulate, fit
from .data_io import StockDataset, normalise_frame
from .state_space_biomass import StateSpaceBiomassSettings, fit_state_space_biomass

_EPS = 1e-12


@dataclass(frozen=True)
class BiomassTruthSettings:
    """Configuration for evidence-weighted biomass synthesis.

    The engine deliberately uses the term *best-supported* rather than *true*
    for real data. A known truth only exists in simulation. The output keeps a
    separate truth-gap/identifiability assessment so a precise-looking curve is
    not mistaken for direct observation.
    """

    models: tuple[str, ...] = ("schaefer", "fox", "pella", "state_space_schaefer")
    holdout_years: int = 4
    search_draws: int = 500
    samples: int = 1200
    process_cv: float = 0.10
    maximum_single_model_weight: float = 0.75
    include_individual_indices: bool = True
    include_composite_index: bool = True
    robust_weight_temperature: float = 1.0
    state_space_particles: int = 160
    state_space_candidates: int = 24
    seed: int = 48177


@dataclass
class BiomassTruthResult:
    name: str
    settings: dict[str, Any]
    summary: dict[str, Any]
    trajectory: list[dict[str, float]]
    candidates: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    samples: dict[str, Any]


@dataclass
class _Candidate:
    label: str
    model: str
    index_column: str
    dataset: StockDataset
    fit: FitResult
    predictive_loss: float
    predictive_points: int
    full_data_loss: float
    weight: float = 0.0


def _index_columns(dataset: StockDataset) -> list[str]:
    columns = [column for column in dataset.frame.columns if column == "index" or str(column).startswith("index_")]
    return [column for column in columns if int(dataset.frame[column].notna().sum()) >= 3]


def _composite_index(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.full(len(frame), np.nan, dtype=float)
    standardised: list[np.ndarray] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        transformed = np.full(len(values), np.nan, dtype=float)
        if valid.any():
            centre = exp(float(np.mean(np.log(values[valid]))))
            transformed[valid] = values[valid] / max(centre, _EPS)
        standardised.append(transformed)
    matrix = np.vstack(standardised)
    output = np.full(matrix.shape[1], np.nan, dtype=float)
    for position in range(matrix.shape[1]):
        values = matrix[:, position]
        valid = np.isfinite(values) & (values > 0)
        if valid.any():
            output[position] = exp(float(np.mean(np.log(values[valid]))))
    return output


def _candidate_dataset(dataset: StockDataset, index_values: np.ndarray, label: str) -> StockDataset:
    frame = dataset.frame.copy()
    frame["index"] = np.asarray(index_values, dtype=float)
    return StockDataset(
        name=f"{dataset.name} — {label}",
        frame=frame,
        provenance=dataset.provenance,
        transformations=[*dataset.transformations, {"operation": "biomass_truth_candidate", "details": {"index": label}}],
        warnings=list(dataset.warnings),
        raw_columns=list(dataset.raw_columns),
        index_columns=["index"],
    )


def _fit_candidate(
    dataset: StockDataset,
    model: str,
    search_draws: int,
    seed: int,
    *,
    state_space_particles: int = 160,
    state_space_candidates: int = 24,
) -> FitResult:
    if str(model).strip().lower() == "state_space_schaefer":
        # Scale the compact random search with the requested workload while
        # retaining bounded defaults for responsive desktop use.
        candidates = max(8, min(int(state_space_candidates), max(8, int(search_draws) // 8)))
        return fit_state_space_biomass(
            dataset,
            StateSpaceBiomassSettings(
                particles=max(int(state_space_particles), 64),
                candidates=candidates,
                seed=int(seed),
            ),
        )
    return fit(
        dataset,
        ModelSettings(
            model=model,
            search_draws=max(int(search_draws), 120),
            seed=int(seed),
            process_cv=0.10,
        ),
    )


def _predictive_loss(candidate_dataset: StockDataset, model: str, settings: BiomassTruthSettings, seed: int) -> tuple[float, int]:
    frame = candidate_dataset.frame.reset_index(drop=True)
    holdout = min(max(int(settings.holdout_years), 0), max(len(frame) // 3, 0))
    if holdout < 2 or len(frame) - holdout < 5:
        return float("nan"), 0
    train_frame = frame.iloc[:-holdout].copy()
    train = StockDataset(
        name=f"{candidate_dataset.name} holdout train",
        frame=train_frame,
        provenance=candidate_dataset.provenance,
        transformations=candidate_dataset.transformations,
        warnings=candidate_dataset.warnings,
        raw_columns=candidate_dataset.raw_columns,
        index_columns=["index"],
    )
    fitted = _fit_candidate(
        train,
        model,
        max(settings.search_draws // 2, 140),
        seed,
        state_space_particles=max(settings.state_space_particles // 2, 64),
        state_space_candidates=max(settings.state_space_candidates // 2, 8),
    )
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float)
    k = float(fitted.best["k_b0"])
    r = float(fitted.best["r"])
    b0 = float(fitted.best["initial_depletion"])
    predicted = _simulate(years, catches, k, r, b0, model, float(fitted.settings.get("pella_shape", 1.35)))
    train_index = train_frame["index"].to_numpy(dtype=float)
    train_pred = predicted[:-holdout]
    valid_train = np.isfinite(train_index) & (train_index > 0) & np.isfinite(train_pred) & (train_pred > 0)
    terms: list[float] = []
    if valid_train.any():
        q = exp(float(np.mean(np.log(train_index[valid_train]) - np.log(train_pred[valid_train]))))
        obs = frame["index"].to_numpy(dtype=float)[-holdout:]
        pred = q * predicted[-holdout:]
        valid = np.isfinite(obs) & (obs > 0) & np.isfinite(pred) & (pred > 0)
        terms.extend(np.square(np.log(obs[valid]) - np.log(pred[valid])).tolist())
    biomass_obs = frame["biomass"].to_numpy(dtype=float)[-holdout:]
    biomass_pred = predicted[-holdout:]
    valid_bio = np.isfinite(biomass_obs) & (biomass_obs > 0) & np.isfinite(biomass_pred) & (biomass_pred > 0)
    terms.extend(np.square(np.log(biomass_obs[valid_bio]) - np.log(biomass_pred[valid_bio])).tolist())
    if not terms:
        return float("nan"), 0
    return float(np.mean(terms)), len(terms)


def _capped_normalise(raw: np.ndarray, cap: float) -> np.ndarray:
    values = np.maximum(np.asarray(raw, dtype=float), 0.0)
    if not np.isfinite(values).all() or float(values.sum()) <= 0:
        values = np.ones_like(values)
    values /= values.sum()
    cap = float(np.clip(cap, 1.0 / max(len(values), 1), 1.0))
    # Iterative water-filling preserves relative weights among uncapped models.
    active = np.ones(len(values), dtype=bool)
    result = np.zeros(len(values), dtype=float)
    remaining = 1.0
    source = values.copy()
    for _ in range(len(values) + 2):
        if not active.any():
            break
        allocation = source[active]
        allocation = allocation / max(float(allocation.sum()), _EPS) * remaining
        active_indices = np.where(active)[0]
        exceeded = allocation > cap + 1e-15
        if not exceeded.any():
            result[active_indices] = allocation
            remaining = 0.0
            break
        for index, amount, too_high in zip(active_indices, allocation, exceeded):
            if too_high:
                result[index] = cap
                remaining -= cap
                active[index] = False
        if remaining <= _EPS:
            break
    if remaining > 1e-10 and active.any():
        active_indices = np.where(active)[0]
        result[active_indices] += remaining / len(active_indices)
    result = np.maximum(result, 0.0)
    return result / max(float(result.sum()), _EPS)


def _candidate_weights(candidates: Sequence[_Candidate], settings: BiomassTruthSettings) -> np.ndarray:
    predictive = np.array([candidate.predictive_loss for candidate in candidates], dtype=float)
    full = np.array([candidate.full_data_loss for candidate in candidates], dtype=float)
    if np.isfinite(predictive).any():
        fallback = np.nanmedian(predictive[np.isfinite(predictive)]) + np.nanstd(predictive[np.isfinite(predictive)])
        score = np.where(np.isfinite(predictive), predictive, fallback)
    else:
        score = full.copy()
    finite = np.isfinite(score)
    if not finite.any():
        return np.ones(len(candidates), dtype=float) / max(len(candidates), 1)
    score = np.where(finite, score, np.nanmax(score[finite]) + 10.0)
    scale = max(float(np.median(np.abs(score - np.median(score)))), 1e-4)
    temperature = max(float(settings.robust_weight_temperature), 0.05)
    raw = np.exp(-0.5 * np.clip((score - np.min(score)) / (scale * temperature), 0.0, 700.0))
    return _capped_normalise(raw, settings.maximum_single_model_weight)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cumulative = np.cumsum(sorted_weights) / max(float(sorted_weights.sum()), _EPS)
    return float(np.interp(float(probability), cumulative, sorted_values))


def _sample_candidate_paths(candidate: _Candidate, count: int, rng: np.random.Generator, process_cv: float) -> tuple[np.ndarray, np.ndarray]:
    frame = candidate.dataset.frame
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float)
    ensemble = candidate.fit.ensemble or [
        {
            "weight": 1.0,
            "k": candidate.fit.best["k_b0"],
            "r": candidate.fit.best["r"],
            "b0_frac": candidate.fit.best["initial_depletion"],
            "sigma": candidate.fit.best["sigma"],
        }
    ]
    member_weights = np.asarray([max(float(row.get("weight", 0.0)), 0.0) for row in ensemble], dtype=float)
    if float(member_weights.sum()) <= 0:
        member_weights[:] = 1.0
    member_weights /= member_weights.sum()
    paths = np.empty((count, len(years)), dtype=float)
    capacities = np.empty(count, dtype=float)
    model = candidate.model
    pella_shape = float(candidate.fit.settings.get("pella_shape", 1.35))
    sigma_process = np.sqrt(np.log1p(max(float(process_cv), 0.0) ** 2))
    for sample in range(count):
        member = ensemble[int(rng.choice(len(ensemble), p=member_weights))]
        k = max(float(member.get("k", candidate.fit.best["k_b0"])), _EPS)
        r = max(float(member.get("r", candidate.fit.best["r"])), _EPS)
        b0 = float(member.get("b0_frac", candidate.fit.best["initial_depletion"]))
        deterministic = _simulate(years, catches, k, r, b0, model, pella_shape)
        path = np.empty_like(deterministic)
        path[0] = deterministic[0]
        member_process_cv = max(float(member.get("process_cv", process_cv)), 0.0)
        member_process_sigma = np.sqrt(np.log1p(member_process_cv**2))
        for year_index in range(1, len(years)):
            previous = path[year_index - 1]
            expected = max(k * 1e-6, previous + _production(previous, k, r, model, pella_shape) - catches[year_index - 1])
            innovation = (
                rng.lognormal(-0.5 * member_process_sigma**2, member_process_sigma)
                if member_process_sigma > 0
                else 1.0
            )
            path[year_index] = max(k * 1e-6, expected * innovation)
        paths[sample] = path
        capacities[sample] = k
    return paths, capacities


def _identifiability_grade(
    dataset: StockDataset,
    candidates: Sequence[_Candidate],
    terminal_samples: np.ndarray,
    terminal_weights: np.ndarray,
) -> dict[str, Any]:
    frame = dataset.frame
    biomass_points = int(frame["biomass"].notna().sum())
    index_columns = _index_columns(dataset)
    index_points = int(sum(frame[column].notna().sum() for column in index_columns))
    median = _weighted_quantile(terminal_samples, terminal_weights, 0.50)
    p10 = _weighted_quantile(terminal_samples, terminal_weights, 0.10)
    p90 = _weighted_quantile(terminal_samples, terminal_weights, 0.90)
    relative_spread = float((p90 - p10) / max(abs(median), _EPS))
    predictive = [candidate.predictive_loss for candidate in candidates if np.isfinite(candidate.predictive_loss)]
    predictive_rmse = float(np.sqrt(np.mean(predictive))) if predictive else float("nan")
    model_terminal = np.asarray([candidate.fit.best["terminal_biomass"] for candidate in candidates], dtype=float)
    model_disagreement = float((np.quantile(model_terminal, 0.90) - np.quantile(model_terminal, 0.10)) / max(np.median(model_terminal), _EPS))

    score = 0
    score += 2 if biomass_points >= 5 else 1 if biomass_points >= 1 else 0
    score += 2 if len(index_columns) >= 2 else 1 if index_points >= 5 else 0
    score += 2 if relative_spread <= 0.35 else 1 if relative_spread <= 0.75 else 0
    score += 2 if model_disagreement <= 0.30 else 1 if model_disagreement <= 0.70 else 0
    score += 2 if np.isfinite(predictive_rmse) and predictive_rmse <= 0.25 else 1 if np.isfinite(predictive_rmse) and predictive_rmse <= 0.50 else 0
    if score >= 9:
        grade, label = "A", "strongly identified for the supplied evidence"
    elif score >= 7:
        grade, label = "B", "reasonably identified but still model dependent"
    elif score >= 5:
        grade, label = "C", "moderately identified with material uncertainty"
    elif score >= 3:
        grade, label = "D", "weakly identified"
    else:
        grade, label = "E", "not identified well enough to call a true biomass estimate"
    return {
        "grade": grade,
        "label": label,
        "score": score,
        "maximum_score": 10,
        "absolute_biomass_observations": biomass_points,
        "index_columns": index_columns,
        "index_observations": index_points,
        "terminal_relative_interval_width_p10_p90": relative_spread,
        "cross_model_terminal_disagreement": model_disagreement,
        "holdout_log_rmse": predictive_rmse,
        "absolute_scale_warning": biomass_points == 0,
        "interpretation": (
            "Real data do not reveal an assumption-free true biomass. This grade describes how strongly the supplied data, model structures and predictive checks support the estimated scale and trajectory."
        ),
    }


def estimate_best_supported_biomass(
    dataset: StockDataset,
    settings: BiomassTruthSettings | None = None,
) -> BiomassTruthResult:
    config = settings or BiomassTruthSettings()
    frame = dataset.frame.reset_index(drop=True)
    indices = _index_columns(dataset)
    index_variants: list[tuple[str, np.ndarray]] = []
    if config.include_composite_index and len(indices) > 1:
        index_variants.append(("composite_index", _composite_index(frame, indices)))
    if config.include_individual_indices:
        for column in indices:
            index_variants.append((str(column), frame[column].to_numpy(dtype=float)))
    if not index_variants:
        index_variants.append(("no_index", np.full(len(frame), np.nan, dtype=float)))

    candidates: list[_Candidate] = []
    candidate_counter = 0
    for model in config.models:
        for index_label, index_values in index_variants:
            candidate_counter += 1
            candidate_data = _candidate_dataset(dataset, index_values, index_label)
            fitted = _fit_candidate(
                candidate_data,
                model,
                config.search_draws,
                config.seed + candidate_counter * 101,
                state_space_particles=config.state_space_particles,
                state_space_candidates=config.state_space_candidates,
            )
            predictive_loss, predictive_points = _predictive_loss(candidate_data, model, config, config.seed + candidate_counter * 1009)
            n_obs = max(
                int(candidate_data.frame["index"].notna().sum()) + int(candidate_data.frame["biomass"].notna().sum()),
                1,
            )
            full_data_loss = float(fitted.best["objective"] / n_obs)
            candidates.append(
                _Candidate(
                    label=f"{model}:{index_label}",
                    model=model,
                    index_column=index_label,
                    dataset=candidate_data,
                    fit=fitted,
                    predictive_loss=predictive_loss,
                    predictive_points=predictive_points,
                    full_data_loss=full_data_loss,
                )
            )

    weights = _candidate_weights(candidates, config)
    for candidate, weight in zip(candidates, weights):
        candidate.weight = float(weight)

    rng = np.random.default_rng(config.seed)
    allocations = np.maximum(1, np.floor(weights * max(config.samples, len(candidates))).astype(int))
    # Adjust allocation to target total while retaining at least one sample/model.
    target = max(int(config.samples), len(candidates))
    while int(allocations.sum()) < target:
        allocations[int(np.argmax(weights - allocations / max(allocations.sum(), 1)))] += 1
    while int(allocations.sum()) > target and int(allocations.max()) > 1:
        index = int(np.argmax(allocations))
        allocations[index] -= 1

    path_blocks: list[np.ndarray] = []
    capacity_blocks: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    sample_candidate: list[str] = []
    for candidate, count, candidate_weight in zip(candidates, allocations, weights):
        paths, capacities = _sample_candidate_paths(candidate, int(count), rng, config.process_cv)
        path_blocks.append(paths)
        capacity_blocks.append(capacities)
        sample_weights.append(np.full(int(count), float(candidate_weight) / max(int(count), 1), dtype=float))
        sample_candidate.extend([candidate.label] * int(count))
    paths = np.vstack(path_blocks)
    capacities = np.concatenate(capacity_blocks)
    path_weights = np.concatenate(sample_weights)
    path_weights /= max(float(path_weights.sum()), _EPS)
    depletion = paths / np.maximum(capacities[:, None], _EPS)

    years = frame["year"].to_numpy(dtype=int)
    trajectory: list[dict[str, float]] = []
    for year_index, year in enumerate(years):
        values = paths[:, year_index]
        dep_values = depletion[:, year_index]
        trajectory.append(
            {
                "year": int(year),
                "biomass_p05": _weighted_quantile(values, path_weights, 0.05),
                "biomass_p10": _weighted_quantile(values, path_weights, 0.10),
                "biomass_median": _weighted_quantile(values, path_weights, 0.50),
                "biomass_p90": _weighted_quantile(values, path_weights, 0.90),
                "biomass_p95": _weighted_quantile(values, path_weights, 0.95),
                "depletion_p10": _weighted_quantile(dep_values, path_weights, 0.10),
                "depletion_median": _weighted_quantile(dep_values, path_weights, 0.50),
                "depletion_p90": _weighted_quantile(dep_values, path_weights, 0.90),
            }
        )

    identifiability = _identifiability_grade(dataset, candidates, paths[:, -1], path_weights)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.weight, reverse=True):
        candidate_rows.append(
            {
                "candidate": candidate.label,
                "model": candidate.model,
                "index": candidate.index_column,
                "weight": candidate.weight,
                "predictive_loss": candidate.predictive_loss,
                "predictive_points": candidate.predictive_points,
                "full_data_loss_per_observation": candidate.full_data_loss,
                "objective": float(candidate.fit.best["objective"]),
                "terminal_biomass": float(candidate.fit.best["terminal_biomass"]),
                "terminal_depletion": float(candidate.fit.best["terminal_depletion"]),
                "k": float(candidate.fit.best["k_b0"]),
                "r": float(candidate.fit.best["r"]),
                "initial_depletion": float(candidate.fit.best["initial_depletion"]),
                "backend": (
                    "bootstrap_particle_filter"
                    if candidate.fit.diagnostics.get("state_space")
                    else candidate.fit.diagnostics.get("refinement_backend")
                ),
                "state_space": bool(candidate.fit.diagnostics.get("state_space", False)),
            }
        )

    terminal = trajectory[-1]
    summary = {
        "status": "BEST_SUPPORTED_ESTIMATE",
        "terminal_biomass_median": terminal["biomass_median"],
        "terminal_biomass_p10": terminal["biomass_p10"],
        "terminal_biomass_p90": terminal["biomass_p90"],
        "terminal_depletion_median": terminal["depletion_median"],
        "terminal_depletion_p10": terminal["depletion_p10"],
        "terminal_depletion_p90": terminal["depletion_p90"],
        "candidate_models": len(candidates),
        "identifiability_grade": identifiability["grade"],
        "identifiability_label": identifiability["label"],
        "statement": "This is the best-supported evidence-weighted estimate for the supplied data and assumptions, not a directly observed or assumption-free true biomass.",
    }
    diagnostics = {
        "identifiability": identifiability,
        "model_weight_entropy": float(-np.sum(weights * np.log(np.maximum(weights, _EPS))) / max(log(max(len(weights), 2)), _EPS)),
        "dominant_candidate": candidate_rows[0]["candidate"] if candidate_rows else None,
        "dominant_weight": candidate_rows[0]["weight"] if candidate_rows else None,
        "candidate_weight_cap": config.maximum_single_model_weight,
        "multiple_index_conflict": _multiple_index_conflict(frame, indices),
        "data_warnings": list(dataset.warnings),
    }
    sample_summary = {
        "count": int(len(paths)),
        "candidate_labels": sample_candidate,
        "weights": path_weights.tolist(),
        "terminal_biomass": paths[:, -1].tolist(),
        "terminal_depletion": depletion[:, -1].tolist(),
    }
    return BiomassTruthResult(dataset.name, asdict(config), summary, trajectory, candidate_rows, diagnostics, sample_summary)


def _multiple_index_conflict(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(columns):
        for second in columns[first_index + 1 :]:
            x = pd.to_numeric(frame[first], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[second], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if int(valid.sum()) < 4:
                continue
            correlation = float(np.corrcoef(np.log(x[valid]), np.log(y[valid]))[0, 1])
            rows.append({"first": first, "second": second, "overlap": int(valid.sum()), "log_correlation": correlation})
    worst = min((row["log_correlation"] for row in rows), default=float("nan"))
    return {
        "pairs": rows,
        "minimum_log_correlation": worst,
        "status": "CONFLICT" if np.isfinite(worst) and worst < 0.30 else "CAUTION" if np.isfinite(worst) and worst < 0.70 else "PASS" if rows else "NOT_TESTED",
    }


__all__ = [
    "BiomassTruthSettings",
    "BiomassTruthResult",
    "estimate_best_supported_biomass",
]
