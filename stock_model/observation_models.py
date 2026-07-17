from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi
from typing import Any, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class LikelihoodResult:
    name: str
    nll: float
    points: int
    residuals: tuple[float, ...]
    diagnostics: dict[str, Any]


def _mask(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.isfinite(obs) & np.isfinite(pred)


def normal_nll(observed: Sequence[float], predicted: Sequence[float], sd: float, name: str = "normal") -> LikelihoodResult:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = _mask(obs, pred)
    sigma = max(float(sd), 1e-9)
    residual = obs[mask] - pred[mask]
    nll = float(np.sum(0.5 * (residual / sigma) ** 2 + log(sigma) + 0.5 * log(2 * pi)))
    return LikelihoodResult(name, nll, int(mask.sum()), tuple(float(v) for v in residual), {"sd": sigma})


def lognormal_nll(observed: Sequence[float], predicted: Sequence[float], cv: float, name: str = "lognormal") -> LikelihoodResult:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = _mask(obs, pred) & (obs > 0) & (pred > 0)
    sigma = max(float(np.sqrt(np.log1p(max(cv, 1e-9) ** 2))), 1e-9)
    residual = np.log(obs[mask]) - np.log(pred[mask])
    nll = float(np.sum(0.5 * (residual / sigma) ** 2 + log(sigma) + 0.5 * log(2 * pi)))
    return LikelihoodResult(name, nll, int(mask.sum()), tuple(float(v) for v in residual), {"cv": float(cv), "log_sd": sigma})


def student_t_nll(observed: Sequence[float], predicted: Sequence[float], scale: float, df: float = 4.0, name: str = "student_t") -> LikelihoodResult:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = _mask(obs, pred)
    nu = max(float(df), 1.01)
    sigma = max(float(scale), 1e-9)
    residual = (obs[mask] - pred[mask]) / sigma
    constant = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi) - log(sigma)
    log_density = constant - 0.5 * (nu + 1) * np.log1p(residual**2 / nu)
    nll = float(-np.sum(log_density))
    return LikelihoodResult(name, nll, int(mask.sum()), tuple(float(v) for v in residual), {"scale": sigma, "df": nu})


def multinomial_nll(observed_counts: Sequence[float], predicted_proportions: Sequence[float], name: str = "multinomial") -> LikelihoodResult:
    counts = np.asarray(observed_counts, dtype=float)
    probs = np.asarray(predicted_proportions, dtype=float)
    valid = np.isfinite(counts) & np.isfinite(probs) & (counts >= 0)
    counts = counts[valid]
    probs = np.maximum(probs[valid], _EPS)
    probs = probs / max(float(probs.sum()), _EPS)
    total = float(counts.sum())
    nll = -(lgamma(total + 1) - float(np.sum([lgamma(v + 1) for v in counts])) + float(np.dot(counts, np.log(probs))))
    obs_prop = counts / max(total, _EPS)
    residual = obs_prop - probs
    return LikelihoodResult(name, float(nll), len(counts), tuple(float(v) for v in residual), {"sample_size": total})


def dirichlet_multinomial_nll(
    observed_counts: Sequence[float],
    predicted_proportions: Sequence[float],
    concentration: float,
    name: str = "dirichlet_multinomial",
) -> LikelihoodResult:
    counts = np.asarray(observed_counts, dtype=float)
    probs = np.asarray(predicted_proportions, dtype=float)
    valid = np.isfinite(counts) & np.isfinite(probs) & (counts >= 0)
    counts = counts[valid]
    probs = np.maximum(probs[valid], _EPS)
    probs /= max(float(probs.sum()), _EPS)
    theta = max(float(concentration), 1e-6)
    alpha = np.maximum(theta * probs, _EPS)
    n = float(counts.sum())
    log_like = lgamma(n + 1) - float(np.sum([lgamma(v + 1) for v in counts]))
    log_like += lgamma(theta) - lgamma(theta + n)
    log_like += float(np.sum([lgamma(a + x) - lgamma(a) for a, x in zip(alpha, counts)]))
    obs_prop = counts / max(n, _EPS)
    residual = obs_prop - probs
    eff_n = n * (theta + 1.0) / max(theta + n, _EPS)
    return LikelihoodResult(
        name,
        float(-log_like),
        len(counts),
        tuple(float(v) for v in residual),
        {"sample_size": n, "concentration": theta, "approx_effective_sample_size": float(eff_n)},
    )


def logistic_normal_nll(
    observed_proportions: Sequence[float],
    predicted_proportions: Sequence[float],
    sd: float,
    reference_bin: int = -1,
    name: str = "logistic_normal",
) -> LikelihoodResult:
    obs = np.asarray(observed_proportions, dtype=float)
    pred = np.asarray(predicted_proportions, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs = np.maximum(obs[valid], _EPS)
    pred = np.maximum(pred[valid], _EPS)
    obs /= obs.sum()
    pred /= pred.sum()
    ref = reference_bin if reference_bin >= 0 else len(obs) - 1
    if ref < 0 or ref >= len(obs):
        raise ValueError("reference_bin is outside the composition vector.")
    keep = np.arange(len(obs)) != ref
    residual = np.log(obs[keep] / obs[ref]) - np.log(pred[keep] / pred[ref])
    sigma = max(float(sd), 1e-9)
    nll = float(np.sum(0.5 * (residual / sigma) ** 2 + log(sigma) + 0.5 * log(2 * pi)))
    return LikelihoodResult(name, nll, len(residual), tuple(float(v) for v in residual), {"sd": sigma, "reference_bin": ref})


def ageing_error_matrix(max_age: int, sd_years: float, bias_years: float = 0.0, plus_group: bool = True) -> np.ndarray:
    if max_age < 1:
        raise ValueError("max_age must be at least 1.")
    sigma = max(float(sd_years), 1e-6)
    ages = np.arange(max_age + 1, dtype=float)
    matrix = np.zeros((max_age + 1, max_age + 1), dtype=float)
    for true_age in range(max_age + 1):
        mean = true_age + float(bias_years)
        density = np.exp(-0.5 * ((ages - mean) / sigma) ** 2)
        if plus_group and true_age == max_age:
            density[-1] += np.sum(np.exp(-0.5 * (((np.arange(max_age + 1, max_age + 11)) - mean) / sigma) ** 2))
        matrix[true_age] = density / max(float(density.sum()), _EPS)
    return matrix


def apply_ageing_error(true_composition: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    comp = np.asarray(true_composition, dtype=float)
    value = np.asarray(matrix, dtype=float)
    if value.shape != (len(comp), len(comp)):
        raise ValueError("Ageing-error matrix dimensions must match the age composition.")
    if np.any(value < 0) or not np.allclose(value.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Ageing-error matrix rows must be non-negative and sum to one.")
    observed = comp @ value
    return observed / max(float(observed.sum()), _EPS)


def francis_weight(observed: np.ndarray, predicted: np.ndarray, sample_sizes: Sequence[float]) -> dict[str, float]:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    if obs.shape != pred.shape or obs.ndim != 2 or len(n) != obs.shape[0]:
        raise ValueError("Francis weighting needs year-by-bin observed and predicted matrices plus one sample size per year.")
    bins = np.arange(obs.shape[1], dtype=float)
    obs_mean = obs @ bins
    pred_mean = pred @ bins
    pred_var = np.sum(pred * (bins[None, :] - pred_mean[:, None]) ** 2, axis=1)
    se = np.sqrt(np.maximum(pred_var / np.maximum(n, 1.0), _EPS))
    standardized = (obs_mean - pred_mean) / se
    variance = float(np.var(standardized, ddof=1)) if len(standardized) > 1 else float(standardized[0] ** 2)
    multiplier = 1.0 / max(variance, _EPS)
    return {
        "francis_multiplier": float(multiplier),
        "standardized_residual_variance": variance,
        "mean_standardized_residual": float(np.mean(standardized)),
        "years": int(len(n)),
    }


def estimate_dirichlet_concentration(
    observed_counts: np.ndarray,
    predicted_proportions: np.ndarray,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    obs = np.asarray(observed_counts, dtype=float)
    pred = np.asarray(predicted_proportions, dtype=float)
    if obs.shape != pred.shape or obs.ndim != 2:
        raise ValueError("Observed and predicted composition matrices must have the same year-by-bin shape.")
    values = np.asarray(grid if grid is not None else np.geomspace(0.5, 1000.0, 80), dtype=float)
    profile = []
    best = None
    for concentration in values:
        total = 0.0
        for row_obs, row_pred in zip(obs, pred):
            total += dirichlet_multinomial_nll(row_obs, row_pred, float(concentration)).nll
        profile.append({"concentration": float(concentration), "nll": float(total)})
        if best is None or total < best[0]:
            best = (total, float(concentration))
    assert best is not None
    return {"concentration": best[1], "nll": float(best[0]), "profile": profile}


def combine_likelihoods(components: Sequence[LikelihoodResult], weights: Sequence[float] | None = None) -> dict[str, Any]:
    if weights is None:
        weights = [1.0] * len(components)
    if len(weights) != len(components):
        raise ValueError("Likelihood weights must match the number of components.")
    rows = []
    total = 0.0
    for result, weight in zip(components, weights):
        contribution = max(float(weight), 0.0) * result.nll
        total += contribution
        rows.append({
            "component": result.name,
            "raw_nll": result.nll,
            "weight": float(weight),
            "weighted_nll": contribution,
            "points": result.points,
            **result.diagnostics,
        })
    return {"total_objective": float(total), "components": rows}
