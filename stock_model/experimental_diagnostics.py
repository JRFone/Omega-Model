from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, log
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import FitResult, ModelSettings, _objective_breakdown, fit
from .data_io import StockDataset
from .diagnostics_suite import lag1, runs_test
from .inference_engine import finite_hessian

_EPS = 1e-12


@dataclass(frozen=True)
class ExperimentalDiagnosticSettings:
    seed: int = 77231
    posterior_predictive_replicates: int = 400
    mutual_information_bins: int = 6
    mutual_information_permutations: int = 250
    change_point_max: int = 3
    adversarial_catch_fraction: float = 0.10
    adversarial_index_drift: float = 0.03
    data_clone_factors: tuple[int, ...] = (1, 2, 4, 8)
    search_draws: int = 220


def _best_theta(fitted: FitResult) -> np.ndarray:
    depletion = float(np.clip(fitted.best["initial_depletion"], 1e-8, 1.0 - 1e-8))
    return np.array(
        [
            log(max(float(fitted.best["k_b0"]), _EPS)),
            log(max(float(fitted.best["r"]), _EPS)),
            log(depletion / (1.0 - depletion)),
            log(max(float(fitted.best["sigma"]), 0.03)),
        ],
        dtype=float,
    )


def _index_residuals(dataset: StockDataset, fitted: FitResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = dataset.frame
    index = frame["index"].to_numpy(dtype=float)
    biomass = np.asarray([row["biomass"] for row in fitted.history], dtype=float)
    years = frame["year"].to_numpy(dtype=int)
    valid = np.isfinite(index) & (index > 0) & np.isfinite(biomass) & (biomass > 0)
    residual = np.full(len(frame), np.nan, dtype=float)
    if valid.any():
        q = exp(float(np.mean(np.log(index[valid]) - np.log(biomass[valid]))))
        residual[valid] = np.log(index[valid]) - np.log(q * biomass[valid])
    return years, residual, valid


def _single_change_point(values: np.ndarray, years: np.ndarray, maximum: int = 3) -> dict[str, Any]:
    valid = np.isfinite(values)
    x = values[valid]
    y = years[valid]
    if len(x) < 8:
        return {"status": "NOT_TESTED", "reason": "At least eight residual observations are required.", "change_points": []}

    def segment_cost(start: int, end: int) -> float:
        segment = x[start:end]
        return float(np.sum((segment - np.mean(segment)) ** 2))

    n = len(x)
    baseline_rss = max(segment_cost(0, n), _EPS)
    baseline_bic = n * log(baseline_rss / n) + log(n)
    best = {"bic": baseline_bic, "breaks": [], "rss": baseline_rss}
    # Exhaustive search is intentionally limited to three breaks and short annual series.
    candidates = list(range(3, n - 3))
    for count in range(1, min(maximum, 3) + 1):
        if count == 1:
            combinations = ((a,) for a in candidates)
        elif count == 2:
            combinations = ((a, b) for a in candidates for b in candidates if a + 3 <= b)
        else:
            combinations = ((a, b, c) for a in candidates for b in candidates for c in candidates if a + 3 <= b and b + 3 <= c)
        for breaks in combinations:
            bounds = (0, *breaks, n)
            rss = sum(segment_cost(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
            parameters = len(breaks) + 1
            bic = n * log(max(rss, _EPS) / n) + parameters * log(n)
            if bic < best["bic"]:
                best = {"bic": float(bic), "breaks": list(breaks), "rss": float(rss)}
    improvement = float(baseline_bic - best["bic"])
    years_break = [int(y[index]) for index in best["breaks"]]
    return {
        "status": "FLAG" if improvement > 6.0 else "CAUTION" if improvement > 2.0 else "PASS",
        "change_points": years_break,
        "bic_improvement": improvement,
        "baseline_bic": float(baseline_bic),
        "best_bic": float(best["bic"]),
        "interpretation": "Large BIC improvement indicates a residual mean shift that may reflect changing catchability, survey design, selectivity or unmodelled biology.",
    }


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 6:
        return float("nan")
    histogram, _, _ = np.histogram2d(x, y, bins=max(int(bins), 3))
    probability = histogram / max(float(histogram.sum()), _EPS)
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = probability > 0
    return float(np.sum(probability[mask] * np.log(probability[mask] / np.maximum(expected[mask], _EPS))))


def _nonlinear_memory(residual: np.ndarray, settings: ExperimentalDiagnosticSettings) -> dict[str, Any]:
    values = residual[np.isfinite(residual)]
    if len(values) < 8:
        return {"status": "NOT_TESTED", "mutual_information": float("nan"), "permutation_p": float("nan")}
    observed = _mutual_information(values[:-1], values[1:], settings.mutual_information_bins)
    rng = np.random.default_rng(settings.seed)
    null = np.empty(max(int(settings.mutual_information_permutations), 20), dtype=float)
    for index in range(len(null)):
        permuted = rng.permutation(values[1:])
        null[index] = _mutual_information(values[:-1], permuted, settings.mutual_information_bins)
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return {
        "status": "FLAG" if p_value < 0.05 else "CAUTION" if p_value < 0.15 else "PASS",
        "mutual_information": float(observed),
        "permutation_p": p_value,
        "null_p95": float(np.quantile(null, 0.95)),
        "interpretation": "Detects nonlinear year-to-year residual dependence that ordinary autocorrelation can miss.",
    }


def _spectral_residual_test(residual: np.ndarray) -> dict[str, Any]:
    values = residual[np.isfinite(residual)]
    if len(values) < 10:
        return {"status": "NOT_TESTED", "dominant_period": float("nan"), "spectral_concentration": float("nan")}
    centred = values - np.mean(values)
    power = np.abs(np.fft.rfft(centred)) ** 2
    frequencies = np.fft.rfftfreq(len(centred), d=1.0)
    if len(power) <= 1 or float(power[1:].sum()) <= _EPS:
        return {"status": "PASS", "dominant_period": float("inf"), "spectral_concentration": 0.0}
    index = int(np.argmax(power[1:]) + 1)
    concentration = float(power[index] / max(float(power[1:].sum()), _EPS))
    period = float(1.0 / max(frequencies[index], _EPS))
    return {
        "status": "FLAG" if concentration > 0.55 else "CAUTION" if concentration > 0.35 else "PASS",
        "dominant_period": period,
        "spectral_concentration": concentration,
        "interpretation": "Strong periodic residual power can indicate cyclic recruitment, survey timing effects or unmodelled temporal structure.",
    }


def _catchability_hyperstability(dataset: StockDataset, fitted: FitResult) -> dict[str, Any]:
    frame = dataset.frame
    index = frame["index"].to_numpy(dtype=float)
    biomass = np.asarray([row["biomass"] for row in fitted.history], dtype=float)
    years = frame["year"].to_numpy(dtype=float)
    valid = np.isfinite(index) & (index > 0) & np.isfinite(biomass) & (biomass > 0)
    if valid.sum() < 6 or np.std(np.log(biomass[valid])) <= 1e-8:
        return {"status": "NOT_TESTED", "beta": float("nan"), "q_drift_per_year": float("nan")}
    design = np.column_stack([np.ones(valid.sum()), np.log(biomass[valid]), years[valid] - years[valid].mean()])
    response = np.log(index[valid])
    coefficient, *_ = np.linalg.lstsq(design, response, rcond=None)
    beta = float(coefficient[1])
    drift = float(coefficient[2])
    if beta < 0.70:
        status = "FLAG"
    elif beta < 0.90 or abs(drift) > 0.03:
        status = "CAUTION"
    else:
        status = "PASS"
    return {
        "status": status,
        "beta": beta,
        "q_drift_per_year": drift,
        "hyperstability_index": float(max(0.0, 1.0 - beta)),
        "interpretation": "beta below one means the index changes more slowly than model biomass; that is consistent with hyperstability but is not proof of its cause.",
    }


def _sloppiness(dataset: StockDataset, fitted: FitResult) -> dict[str, Any]:
    frame = dataset.frame
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float)
    index = frame["index"].to_numpy(dtype=float)
    biomass = frame["biomass"].to_numpy(dtype=float)
    settings = ModelSettings(**{key: value for key, value in fitted.settings.items() if key in ModelSettings.__dataclass_fields__})
    theta = _best_theta(fitted)

    def objective(value: np.ndarray) -> float:
        return _objective_breakdown(value, years, catches, index, biomass, settings)[0]

    hessian = finite_hessian(objective, theta, step=3e-4)
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = eigenvalues[eigenvalues > 1e-8]
    condition = float(np.max(positive) / max(np.min(positive), _EPS)) if len(positive) else float("inf")
    weakest = eigenvectors[:, int(np.argmin(eigenvalues))]
    parameter_names = ["log_K", "log_r", "logit_initial_depletion", "log_sigma"]
    weak_vector = sorted(
        ({"parameter": name, "loading": float(value)} for name, value in zip(parameter_names, weakest)),
        key=lambda row: abs(row["loading"]),
        reverse=True,
    )
    return {
        "status": "FLAG" if condition > 1e8 or np.min(eigenvalues) <= 0 else "CAUTION" if condition > 1e5 else "PASS",
        "condition_number": condition,
        "eigenvalues": eigenvalues.tolist(),
        "weakest_direction": weak_vector,
        "positive_definite": bool(np.all(eigenvalues > 0)),
        "interpretation": "A very large condition number identifies parameter combinations that the data cannot separate cleanly, even when the optimiser converges.",
    }


def _posterior_predictive(dataset: StockDataset, fitted: FitResult, settings: ExperimentalDiagnosticSettings) -> dict[str, Any]:
    years, residual, valid = _index_residuals(dataset, fitted)
    observed = residual[valid]
    if len(observed) < 6:
        return {"status": "NOT_TESTED", "checks": []}
    rng = np.random.default_rng(settings.seed + 11)
    sigma = max(float(fitted.best["sigma"]), 0.03)
    count = max(int(settings.posterior_predictive_replicates), 50)
    simulated = rng.normal(0.0, sigma, size=(count, len(observed)))

    def summaries(values: np.ndarray) -> np.ndarray:
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0
        return np.array([np.std(values), lag1(values), slope, np.max(np.abs(values))], dtype=float)

    observed_summary = summaries(observed)
    simulated_summary = np.vstack([summaries(row) for row in simulated])
    names = ["residual_sd", "lag1", "time_slope", "maximum_absolute_residual"]
    checks = []
    failures = 0
    for position, name in enumerate(names):
        values = simulated_summary[:, position]
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))
        passed = lower <= observed_summary[position] <= upper
        failures += int(not passed)
        checks.append({"summary": name, "observed": float(observed_summary[position]), "p025": lower, "p975": upper, "passed": bool(passed)})
    return {
        "status": "FLAG" if failures >= 2 else "CAUTION" if failures == 1 else "PASS",
        "checks": checks,
        "failed_checks": failures,
        "replicates": count,
        "interpretation": "Posterior-predictive-style checks compare residual patterns with simulations from the fitted observation model. They test adequacy, not model truth.",
    }


def _clone_dataset(dataset: StockDataset) -> StockDataset:
    return StockDataset(
        name=dataset.name,
        frame=dataset.frame.copy(),
        provenance=dataset.provenance,
        transformations=dataset.transformations,
        warnings=dataset.warnings,
        raw_columns=dataset.raw_columns,
        index_columns=dataset.index_columns,
    )


def _data_cloning(dataset: StockDataset, fitted: FitResult, settings: ExperimentalDiagnosticSettings) -> dict[str, Any]:
    base_settings = ModelSettings(**{key: value for key, value in fitted.settings.items() if key in ModelSettings.__dataclass_fields__})
    rows: list[dict[str, Any]] = []
    parameter_names = ["k_b0", "r", "initial_depletion", "terminal_depletion"]
    for clone in settings.data_clone_factors:
        clone_settings = replace(
            base_settings,
            search_draws=max(settings.search_draws, 120),
            seed=settings.seed + int(clone) * 43,
            index_weight=base_settings.index_weight * clone,
            biomass_weight=base_settings.biomass_weight * clone,
        )
        result = fit(_clone_dataset(dataset), clone_settings)
        spread = {}
        for parameter in ("k", "r", "b0_frac", "terminal_depletion"):
            values = np.array([row.get(parameter, np.nan) for row in result.ensemble], dtype=float)
            spread[parameter] = float(np.nanstd(values)) if np.isfinite(values).any() else float("nan")
        rows.append(
            {
                "clone_factor": int(clone),
                **{name: float(result.best[name]) for name in parameter_names},
                "spread_k": spread["k"],
                "spread_r": spread["r"],
                "spread_initial_depletion": spread["b0_frac"],
                "spread_terminal_depletion": spread["terminal_depletion"],
            }
        )
    slopes = {}
    clone_values = np.log(np.asarray([row["clone_factor"] for row in rows], dtype=float))
    for column in ("spread_k", "spread_r", "spread_initial_depletion", "spread_terminal_depletion"):
        values = np.asarray([row[column] for row in rows], dtype=float)
        valid = np.isfinite(values) & (values > 0)
        slopes[column] = float(np.polyfit(clone_values[valid], np.log(values[valid]), 1)[0]) if valid.sum() >= 2 else float("nan")
    worst = max((value for value in slopes.values() if np.isfinite(value)), default=float("nan"))
    return {
        "status": "FLAG" if np.isfinite(worst) and worst > -0.10 else "CAUTION" if np.isfinite(worst) and worst > -0.35 else "PASS" if np.isfinite(worst) else "NOT_TESTED",
        "rows": rows,
        "log_spread_slopes": slopes,
        "expected_identifiable_slope": -0.5,
        "interpretation": "With identifiable parameters, uncertainty should contract as the likelihood is cloned. Flat contraction suggests structural non-identifiability. Ensemble spread is used here as an approximation, so this remains experimental.",
    }


def _perturbed_dataset(dataset: StockDataset, *, catch_multiplier: float = 1.0, index_drift: float = 0.0, drop_terminal: bool = False, index_multiplier: float = 1.0) -> StockDataset:
    frame = dataset.frame.copy().reset_index(drop=True)
    frame["catch"] = frame["catch"] * catch_multiplier
    positions = np.arange(len(frame), dtype=float)
    frame["index"] = frame["index"] * index_multiplier * np.exp(index_drift * positions)
    if drop_terminal and len(frame) > 5:
        frame.loc[len(frame) - 1, ["index", "biomass"]] = np.nan
    return StockDataset(
        name=f"{dataset.name} perturbation",
        frame=frame,
        provenance=dataset.provenance,
        transformations=dataset.transformations,
        warnings=dataset.warnings,
        raw_columns=dataset.raw_columns,
        index_columns=dataset.index_columns,
    )


def _adversarial_stress(dataset: StockDataset, fitted: FitResult, settings: ExperimentalDiagnosticSettings) -> dict[str, Any]:
    base_settings = ModelSettings(**{key: value for key, value in fitted.settings.items() if key in ModelSettings.__dataclass_fields__})
    base_settings = replace(base_settings, search_draws=max(settings.search_draws, 120))
    perturbations = [
        ("catch_minus", {"catch_multiplier": 1.0 - settings.adversarial_catch_fraction}),
        ("catch_plus", {"catch_multiplier": 1.0 + settings.adversarial_catch_fraction}),
        ("index_upward_drift", {"index_drift": settings.adversarial_index_drift}),
        ("index_downward_drift", {"index_drift": -settings.adversarial_index_drift}),
        ("index_scale_half", {"index_multiplier": 0.5}),
        ("index_scale_double", {"index_multiplier": 2.0}),
        ("terminal_data_missing", {"drop_terminal": True}),
    ]
    base_depletion = float(fitted.best["terminal_depletion"])
    rows: list[dict[str, Any]] = []
    for position, (name, kwargs) in enumerate(perturbations):
        result = fit(_perturbed_dataset(dataset, **kwargs), replace(base_settings, seed=settings.seed + 500 + position))
        terminal = float(result.best["terminal_depletion"])
        rows.append(
            {
                "perturbation": name,
                "terminal_depletion": terminal,
                "absolute_change": terminal - base_depletion,
                "relative_change": (terminal - base_depletion) / max(abs(base_depletion), _EPS),
            }
        )
    maximum = max(abs(float(row["relative_change"])) for row in rows)
    return {
        "status": "FLAG" if maximum > 0.50 else "CAUTION" if maximum > 0.20 else "PASS",
        "maximum_absolute_relative_change": float(maximum),
        "rows": rows,
        "interpretation": "Adversarial stress testing measures how much the stock status moves under small but plausible data errors. It does not assign probabilities to those errors.",
    }


def run_experimental_diagnostics(
    dataset: StockDataset,
    fitted: FitResult | None = None,
    settings: ExperimentalDiagnosticSettings | None = None,
) -> dict[str, Any]:
    config = settings or ExperimentalDiagnosticSettings()
    model_settings = ModelSettings(search_draws=max(config.search_draws, 120), seed=config.seed)
    fitted = fitted or fit(dataset, model_settings)
    years, residual, valid = _index_residuals(dataset, fitted)
    simple_runs = runs_test(residual[np.isfinite(residual)]) if np.isfinite(residual).sum() >= 4 else {"z": float("nan"), "runs": float("nan"), "expected_runs": float("nan")}
    results = {
        "catchability_hyperstability": _catchability_hyperstability(dataset, fitted),
        "residual_change_points": _single_change_point(residual, years, config.change_point_max),
        "nonlinear_residual_memory": _nonlinear_memory(residual, config),
        "residual_spectrum": _spectral_residual_test(residual),
        "parameter_sloppiness": _sloppiness(dataset, fitted),
        "posterior_predictive_checks": _posterior_predictive(dataset, fitted, config),
        "data_cloning": _data_cloning(dataset, fitted, config),
        "adversarial_stress": _adversarial_stress(dataset, fitted, config),
        "simple_residual_checks": {
            "lag1": lag1(residual[np.isfinite(residual)]),
            "runs_test": simple_runs,
            "status": "CAUTION" if np.isfinite(lag1(residual[np.isfinite(residual)])) and abs(lag1(residual[np.isfinite(residual)])) > 0.35 else "PASS",
        },
    }
    statuses = [value.get("status") for value in results.values() if isinstance(value, Mapping)]
    flags = statuses.count("FLAG")
    cautions = statuses.count("CAUTION")
    tested = sum(status not in {None, "NOT_TESTED"} for status in statuses)
    if flags >= 3:
        grade, label = "F", "multiple experimental diagnostics found major instability"
    elif flags >= 1:
        grade, label = "D", "at least one major experimental warning requires investigation"
    elif cautions >= 3:
        grade, label = "C", "several weak signals require targeted testing"
    elif tested >= 6:
        grade, label = "B", "no major experimental failure detected"
    else:
        grade, label = "INCOMPLETE", "insufficient data for the experimental suite"
    return {
        "summary": {
            "status": "COMPLETE",
            "grade": grade,
            "label": label,
            "diagnostics_tested": tested,
            "flags": flags,
            "cautions": cautions,
            "boundary": "Experimental diagnostics are hypothesis generators. A flag identifies a pattern to investigate; it does not prove a particular biological cause.",
        },
        "settings": asdict(config),
        "fit_summary": fitted.best,
        "diagnostics": results,
    }


__all__ = ["ExperimentalDiagnosticSettings", "run_experimental_diagnostics"]
