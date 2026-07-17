from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log, pi
from typing import Any

import numpy as np

from .core import FitResult, ModelSettings, _reference_points, fit
from .data_io import StockDataset

_EPS = 1e-12


@dataclass(frozen=True)
class StateSpaceBiomassSettings:
    particles: int = 192
    candidates: int = 32
    seed: int = 90441
    process_cv_low: float = 0.03
    process_cv_high: float = 0.35
    observation_cv_low: float = 0.05
    observation_cv_high: float = 0.60
    resample_ess_fraction: float = 0.50


def _lognormal_logpdf(observation: float, prediction: np.ndarray, cv: float) -> np.ndarray:
    if not np.isfinite(observation) or observation <= 0:
        return np.zeros_like(prediction, dtype=float)
    sigma = np.sqrt(np.log1p(max(float(cv), 0.01) ** 2))
    values = np.maximum(np.asarray(prediction, dtype=float), _EPS)
    residual = (np.log(observation) - np.log(values)) / sigma
    return -0.5 * residual**2 - np.log(sigma) - np.log(max(observation, _EPS)) - 0.5 * log(2.0 * pi)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.uniform() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    indices = np.empty(n, dtype=int)
    i = j = 0
    while i < n:
        if positions[i] < cumulative[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / max(float(sorted_weights.sum()), _EPS)
    return float(np.interp(probability, cumulative, sorted_values))


def _particle_filter(
    years: np.ndarray,
    catches: np.ndarray,
    index: np.ndarray,
    biomass_obs: np.ndarray,
    *,
    k: float,
    r: float,
    b0_frac: float,
    q: float,
    process_cv: float,
    observation_cv: float,
    particles: int,
    seed: int,
    resample_ess_fraction: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_particles = max(int(particles), 32)
    process_sigma = np.sqrt(np.log1p(max(float(process_cv), 0.001) ** 2))
    initial_sigma = max(process_sigma, 0.05)
    states = k * b0_frac * rng.lognormal(-0.5 * initial_sigma**2, initial_sigma, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    log_likelihood = 0.0
    rows: list[dict[str, float]] = []

    for time_index, year in enumerate(years):
        if time_index > 0:
            production = np.maximum(0.0, r * states * (1.0 - states / max(k, _EPS)))
            expected = np.maximum(k * 1e-6, states + production - catches[time_index - 1])
            states = expected * rng.lognormal(-0.5 * process_sigma**2, process_sigma, n_particles)
            states = np.clip(states, k * 1e-6, k * 2.5)
        log_weights = np.log(np.maximum(weights, _EPS))
        if np.isfinite(index[time_index]) and index[time_index] > 0:
            log_weights += _lognormal_logpdf(float(index[time_index]), q * states, observation_cv)
        if np.isfinite(biomass_obs[time_index]) and biomass_obs[time_index] > 0:
            log_weights += _lognormal_logpdf(float(biomass_obs[time_index]), states, max(observation_cv, 0.10))
        maximum = float(np.max(log_weights))
        unnormalised = np.exp(log_weights - maximum)
        normaliser = max(float(np.sum(unnormalised)), _EPS)
        log_likelihood += maximum + log(normaliser)
        weights = unnormalised / normaliser
        rows.append(
            {
                "year": int(year),
                "catch": float(catches[time_index]),
                "biomass": _weighted_quantile(states, weights, 0.50),
                "biomass_p10": _weighted_quantile(states, weights, 0.10),
                "biomass_p90": _weighted_quantile(states, weights, 0.90),
                "depletion": _weighted_quantile(states / max(k, _EPS), weights, 0.50),
                "effective_sample_size": float(1.0 / max(float(np.sum(weights**2)), _EPS)),
            }
        )
        ess = 1.0 / max(float(np.sum(weights**2)), _EPS)
        if ess < max(float(resample_ess_fraction), 0.05) * n_particles:
            indices = _systematic_resample(weights, rng)
            states = states[indices]
            weights.fill(1.0 / n_particles)
    return {"log_likelihood": float(log_likelihood), "history": rows}


def fit_state_space_biomass(
    dataset: StockDataset,
    settings: StateSpaceBiomassSettings | None = None,
    initial_fit: FitResult | None = None,
) -> FitResult:
    config = settings or StateSpaceBiomassSettings()
    deterministic = initial_fit or fit(dataset, ModelSettings(model="schaefer", search_draws=160, seed=config.seed + 1))
    frame = dataset.frame
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float)
    index = frame["index"].to_numpy(dtype=float)
    biomass_obs = frame["biomass"].to_numpy(dtype=float)
    base_k = max(float(deterministic.best["k_b0"]), _EPS)
    base_r = max(float(deterministic.best["r"]), _EPS)
    base_b0 = float(np.clip(deterministic.best["initial_depletion"], 0.02, 1.20))
    deterministic_biomass = np.asarray([row["biomass"] for row in deterministic.history], dtype=float)
    valid = np.isfinite(index) & (index > 0) & (deterministic_biomass > 0)
    base_q = exp(float(np.mean(np.log(index[valid]) - np.log(deterministic_biomass[valid])))) if valid.any() else 1.0 / base_k
    rng = np.random.default_rng(config.seed)
    candidates: list[dict[str, Any]] = []
    for candidate_index in range(max(int(config.candidates), 8)):
        if candidate_index == 0:
            k, r, b0, q = base_k, base_r, base_b0, base_q
            process_cv, observation_cv = 0.10, max(float(deterministic.best["sigma"]), 0.10)
        else:
            k = base_k * rng.lognormal(0.0, 0.30)
            r = base_r * rng.lognormal(0.0, 0.25)
            b0 = float(np.clip(base_b0 + rng.normal(0.0, 0.12), 0.03, 1.20))
            q = base_q * rng.lognormal(0.0, 0.20)
            process_cv = rng.uniform(config.process_cv_low, config.process_cv_high)
            observation_cv = rng.uniform(config.observation_cv_low, config.observation_cv_high)
        output = _particle_filter(
            years,
            catches,
            index,
            biomass_obs,
            k=k,
            r=r,
            b0_frac=b0,
            q=q,
            process_cv=process_cv,
            observation_cv=observation_cv,
            particles=config.particles,
            seed=config.seed + 1000 + candidate_index * 7919,
            resample_ess_fraction=config.resample_ess_fraction,
        )
        # Weak regularisation prevents implausible random-search extremes from
        # winning solely because of particle noise.
        penalty = 0.5 * (log(k / base_k) / 0.70) ** 2 + 0.5 * (log(r / base_r) / 0.70) ** 2
        objective = -float(output["log_likelihood"]) + penalty
        candidates.append(
            {
                "objective": float(objective),
                "k": float(k),
                "r": float(r),
                "b0_frac": float(b0),
                "q": float(q),
                "process_cv": float(process_cv),
                "observation_cv": float(observation_cv),
                "history": output["history"],
            }
        )
    candidates.sort(key=lambda row: row["objective"])
    best_candidate = candidates[0]
    delta = np.asarray([row["objective"] - best_candidate["objective"] for row in candidates[: min(20, len(candidates))]], dtype=float)
    weights = np.exp(-0.5 * np.clip(delta, 0.0, 700.0))
    weights /= max(float(weights.sum()), _EPS)
    ensemble = []
    for weight, row in zip(weights, candidates):
        reference = _reference_points(row["k"], row["r"], "schaefer")
        ensemble.append(
            {
                "weight": float(weight),
                "nll": float(row["objective"]),
                "k": float(row["k"]),
                "r": float(row["r"]),
                "b0_frac": float(row["b0_frac"]),
                "sigma": float(row["observation_cv"]),
                "terminal_biomass": float(row["history"][-1]["biomass"]),
                "terminal_depletion": float(row["history"][-1]["depletion"]),
                "msy": reference["msy"],
                "bmsy": reference["bmsy"],
                "fmsy": reference["fmsy"],
                "process_cv": float(row["process_cv"]),
            }
        )
    reference = _reference_points(best_candidate["k"], best_candidate["r"], "schaefer")
    history = [
        {"year": row["year"], "catch": row["catch"], "biomass": row["biomass"], "depletion": row["depletion"]}
        for row in best_candidate["history"]
    ]
    best = {
        "k_b0": float(best_candidate["k"]),
        "r": float(best_candidate["r"]),
        "initial_depletion": float(best_candidate["b0_frac"]),
        "initial_biomass": float(history[0]["biomass"]),
        "terminal_biomass": float(history[-1]["biomass"]),
        "terminal_depletion": float(history[-1]["depletion"]),
        "msy": reference["msy"],
        "bmsy": reference["bmsy"],
        "fmsy": reference["fmsy"],
        "sigma": float(best_candidate["observation_cv"]),
        "process_cv": float(best_candidate["process_cv"]),
        "objective": float(best_candidate["objective"]),
    }
    diagnostics = {
        "state_space": True,
        "filter": "bootstrap particle filter",
        "particles": config.particles,
        "parameter_candidates": config.candidates,
        "q": float(best_candidate["q"]),
        "filtered_interval_available": True,
        "history_intervals": best_candidate["history"],
        "uncertainty_method": "particle-filter latent-state distribution plus candidate-model spread",
        "warning": "This compact particle-filter implementation is an additional structural candidate, not a replacement for a fully conditioned random-effects assessment.",
    }
    fit_settings = asdict(config) | {"model": "state_space_schaefer", "pella_shape": 1.35}
    return FitResult(dataset.name, fit_settings, best, diagnostics, history, ensemble)


__all__ = ["StateSpaceBiomassSettings", "fit_state_space_biomass"]
