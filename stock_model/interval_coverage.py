from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .core import FitResult, ModelSettings, _production, fit
from .data_io import StockDataset
from .likelihood_profiles import ProfileSettings, profile_likelihood
from .native_backend import get_native_engine

_EPS = 1e-12


@dataclass(frozen=True)
class CoverageSettings:
    replicates: int = 100
    confidence_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    methods: tuple[str, ...] = ("hessian",)
    workers: int = 1
    seed: int = 39191
    search_draws: int = 160
    bootstrap_replicates: int = 40
    profile_points: int = 13
    profile_multistarts: int = 2
    process_cv: float | None = None
    observation_cv: float | None = None
    biomass_limit: float = 0.20
    hessian_step: float = 2e-4
    include_time_series: bool = True
    native_threads_per_worker: int = 1


def _copy_dataset(dataset: StockDataset, frame: pd.DataFrame, name: str) -> StockDataset:
    return StockDataset(
        name=name,
        frame=frame.reset_index(drop=True),
        provenance=dict(dataset.provenance),
        transformations=list(dataset.transformations),
        warnings=list(dataset.warnings),
        raw_columns=list(dataset.raw_columns),
        index_columns=list(dataset.index_columns),
    )


def _encode_fit(fitted: FitResult) -> np.ndarray:
    initial = min(max(float(fitted.best["initial_depletion"]), 1e-9), 1.0 - 1e-9)
    return np.asarray(
        [
            math.log(max(float(fitted.best["k_b0"]), _EPS)),
            math.log(max(float(fitted.best["r"]), _EPS)),
            math.log(initial / (1.0 - initial)),
            math.log(max(float(fitted.best["sigma"]), 0.03)),
        ],
        dtype=float,
    )


def _decode_theta(theta: Sequence[float]) -> dict[str, float]:
    values = np.asarray(theta, dtype=float)
    return {
        "k": float(math.exp(values[0])),
        "r": float(math.exp(values[1])),
        "initial_depletion": float(1.0 / (1.0 + math.exp(-values[2]))),
        "sigma": float(np.clip(math.exp(values[3]), 0.03, 1.5)),
    }


def _model_arrays(dataset: StockDataset, settings: ModelSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    return (
        frame["year"].to_numpy(dtype=int),
        frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0),
        frame["index"].to_numpy(dtype=float),
        frame["biomass"].to_numpy(dtype=float),
    )


def _numerical_hessian_from_ad_gradient(dataset: StockDataset, settings: ModelSettings, theta: np.ndarray, step: float) -> tuple[np.ndarray, dict[str, Any]]:
    years, catches, index, biomass = _model_arrays(dataset, settings)
    engine = get_native_engine()
    dimension = len(theta)
    hessian = np.empty((dimension, dimension), dtype=float)
    steps = np.asarray([max(float(step), abs(float(value)) * float(step)) for value in theta], dtype=float)
    for column in range(dimension):
        plus = theta.copy(); plus[column] += steps[column]
        minus = theta.copy(); minus[column] -= steps[column]
        gradient_plus = engine.objective_gradient(plus, years, catches, index, biomass, settings).gradient
        gradient_minus = engine.objective_gradient(minus, years, catches, index, biomass, settings).gradient
        hessian[:, column] = (gradient_plus - gradient_minus) / (2.0 * steps[column])
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = bool(np.all(eigenvalues > 1e-10))
    try:
        covariance = np.linalg.inv(hessian) if positive else np.linalg.pinv(hessian, rcond=1e-10)
    except np.linalg.LinAlgError:
        covariance = np.full_like(hessian, np.nan)
    condition = float(np.linalg.cond(hessian)) if np.all(np.isfinite(hessian)) else float("inf")
    return covariance, {
        "hessian": hessian.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "positive_definite": positive,
        "condition_number": condition,
        "method": "central differences of the native automatic-differentiation gradient",
        "backend": engine.status().backend,
    }


def _derived_vector(theta: np.ndarray, dataset: StockDataset, settings: ModelSettings) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    years, catches, index, biomass = _model_arrays(dataset, settings)
    result = get_native_engine().objective_gradient(theta, years, catches, index, biomass, settings)
    decoded = _decode_theta(theta)
    depletions = np.asarray(result.biomass, dtype=float) / max(decoded["k"], _EPS)
    quantities = {
        "k": decoded["k"],
        "r": decoded["r"],
        "initial_depletion": decoded["initial_depletion"],
        "sigma": decoded["sigma"],
        "terminal_biomass": float(result.biomass[-1]),
        "terminal_depletion": float(depletions[-1]),
    }
    return quantities, np.asarray(result.biomass, dtype=float), depletions


def _derived_jacobian(theta: np.ndarray, dataset: StockDataset, settings: ModelSettings, step: float) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = ["k", "r", "initial_depletion", "sigma", "terminal_biomass", "terminal_depletion"]
    center_quantities, center_biomass, center_depletion = _derived_vector(theta, dataset, settings)
    jacobian = np.empty((len(names), len(theta)), dtype=float)
    biomass_jacobian = np.empty((len(center_biomass), len(theta)), dtype=float)
    depletion_jacobian = np.empty((len(center_depletion), len(theta)), dtype=float)
    for column in range(len(theta)):
        h = max(float(step), abs(float(theta[column])) * float(step))
        plus = theta.copy(); plus[column] += h
        minus = theta.copy(); minus[column] -= h
        q_plus, b_plus, d_plus = _derived_vector(plus, dataset, settings)
        q_minus, b_minus, d_minus = _derived_vector(minus, dataset, settings)
        for row, name in enumerate(names):
            jacobian[row, column] = (q_plus[name] - q_minus[name]) / (2.0 * h)
        biomass_jacobian[:, column] = (b_plus - b_minus) / (2.0 * h)
        depletion_jacobian[:, column] = (d_plus - d_minus) / (2.0 * h)
    return names, jacobian, biomass_jacobian, depletion_jacobian


def _wald_intervals(
    fitted_dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    levels: Sequence[float],
    step: float,
) -> dict[str, Any]:
    theta = _encode_fit(fitted)
    covariance, hessian = _numerical_hessian_from_ad_gradient(fitted_dataset, settings, theta, step)
    names, jacobian, biomass_jacobian, depletion_jacobian = _derived_jacobian(theta, fitted_dataset, settings, step)
    quantities, biomass, depletion = _derived_vector(theta, fitted_dataset, settings)
    variance = np.diag(jacobian @ covariance @ jacobian.T)
    biomass_variance = np.diag(biomass_jacobian @ covariance @ biomass_jacobian.T)
    depletion_variance = np.diag(depletion_jacobian @ covariance @ depletion_jacobian.T)
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for row, name in enumerate(names):
        intervals[name] = {}
        se = math.sqrt(max(float(variance[row]), 0.0)) if np.isfinite(variance[row]) else float("nan")
        estimate = float(quantities[name])
        for level in levels:
            z = float(norm.ppf(0.5 + float(level) / 2.0))
            low = estimate - z * se
            high = estimate + z * se
            if name in {"k", "r", "sigma", "terminal_biomass"}:
                low = max(low, 0.0)
            if name in {"initial_depletion", "terminal_depletion"}:
                low = max(low, 0.0); high = min(high, 2.0)
            intervals[name][str(float(level))] = {"low": float(low), "high": float(high), "width": float(high - low), "standard_error": se}

    time_series = []
    years = fitted_dataset.frame.sort_values("year")["year"].to_numpy(dtype=int)
    for position, year in enumerate(years):
        bio_se = math.sqrt(max(float(biomass_variance[position]), 0.0)) if np.isfinite(biomass_variance[position]) else float("nan")
        dep_se = math.sqrt(max(float(depletion_variance[position]), 0.0)) if np.isfinite(depletion_variance[position]) else float("nan")
        for level in levels:
            z = float(norm.ppf(0.5 + float(level) / 2.0))
            time_series.append(
                {
                    "year": int(year),
                    "confidence_level": float(level),
                    "biomass_low": float(max(0.0, biomass[position] - z * bio_se)),
                    "biomass_high": float(biomass[position] + z * bio_se),
                    "depletion_low": float(max(0.0, depletion[position] - z * dep_se)),
                    "depletion_high": float(min(2.0, depletion[position] + z * dep_se)),
                }
            )
    return {"intervals": intervals, "time_series": time_series, "hessian": hessian}


def _simulate_from_parameters(
    dataset: StockDataset,
    settings: ModelSettings,
    k: float,
    r: float,
    initial_depletion: float,
    sigma: float,
    seed: int,
    process_cv: float,
    observation_cv: float,
) -> tuple[StockDataset, dict[str, Any]]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)
    rng = np.random.default_rng(seed)
    process_sigma = math.sqrt(math.log1p(max(process_cv, 0.0) ** 2))
    observation_sigma = math.sqrt(math.log1p(max(observation_cv, 0.0) ** 2))
    biomass = np.empty(len(years), dtype=float)
    biomass[0] = k * initial_depletion
    for position in range(1, len(years)):
        expected = max(
            k * 1e-6,
            biomass[position - 1]
            + _production(biomass[position - 1], k, r, settings.model, settings.pella_shape)
            - catches[position - 1],
        )
        biomass[position] = expected * rng.lognormal(-0.5 * process_sigma**2, process_sigma)
    source_index = frame["index"].to_numpy(dtype=float)
    valid_index = np.isfinite(source_index) & (source_index > 0)
    if valid_index.any():
        q = float(np.exp(np.mean(np.log(source_index[valid_index]) - np.log(np.maximum(biomass[valid_index], _EPS)))))
    else:
        q = 1.0
    index = q * biomass * rng.lognormal(-0.5 * observation_sigma**2, observation_sigma, len(years))
    direct = np.full(len(years), np.nan, dtype=float)
    source_biomass = frame["biomass"].to_numpy(dtype=float)
    direct_mask = np.isfinite(source_biomass) & (source_biomass > 0)
    if direct_mask.any():
        direct[direct_mask] = biomass[direct_mask] * rng.lognormal(-0.5 * observation_sigma**2, observation_sigma, int(direct_mask.sum()))
    simulated_frame = pd.DataFrame(
        {
            "year": years,
            "catch": frame["catch"].to_numpy(dtype=float),
            "index": index,
            "biomass": direct,
        }
    )
    simulated = _copy_dataset(dataset, simulated_frame, f"{dataset.name} coverage simulation {seed}")
    truth = {
        "k": float(k),
        "r": float(r),
        "initial_depletion": float(initial_depletion),
        "sigma": float(sigma),
        "terminal_biomass": float(biomass[-1]),
        "terminal_depletion": float(biomass[-1] / max(k, _EPS)),
        "biomass": biomass.tolist(),
        "depletion": (biomass / max(k, _EPS)).tolist(),
        "years": years.tolist(),
    }
    return simulated, truth


def _bootstrap_intervals(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    levels: Sequence[float],
    replicates: int,
    seed: int,
    search_draws: int,
    process_cv: float,
    observation_cv: float,
) -> dict[str, Any]:
    rows = []
    failures = []
    for bootstrap in range(max(4, int(replicates))):
        try:
            simulated, _truth = _simulate_from_parameters(
                dataset,
                settings,
                float(fitted.best["k_b0"]),
                float(fitted.best["r"]),
                float(fitted.best["initial_depletion"]),
                float(fitted.best["sigma"]),
                seed + bootstrap * 15485863,
                process_cv,
                observation_cv,
            )
            result = fit(simulated, replace(settings, seed=seed + bootstrap * 32452843, search_draws=max(120, int(search_draws))))
            rows.append(
                {
                    "k": float(result.best["k_b0"]),
                    "r": float(result.best["r"]),
                    "initial_depletion": float(result.best["initial_depletion"]),
                    "sigma": float(result.best["sigma"]),
                    "terminal_biomass": float(result.best["terminal_biomass"]),
                    "terminal_depletion": float(result.best["terminal_depletion"]),
                    "biomass": [float(row["biomass"]) for row in result.history],
                    "depletion": [float(row["depletion"]) for row in result.history],
                }
            )
        except Exception as exc:
            failures.append({"replicate": bootstrap + 1, "error": f"{type(exc).__name__}: {exc}"})
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    names = ("k", "r", "initial_depletion", "sigma", "terminal_biomass", "terminal_depletion")
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        intervals[name] = {}
        for level in levels:
            tail = (1.0 - float(level)) / 2.0
            low, high = np.quantile(values, [tail, 1.0 - tail]) if len(values) else (float("nan"), float("nan"))
            intervals[name][str(float(level))] = {"low": float(low), "high": float(high), "width": float(high - low)}
    time_series = []
    if rows:
        years = dataset.frame.sort_values("year")["year"].to_numpy(dtype=int)
        biomass_matrix = np.asarray([row["biomass"] for row in rows], dtype=float)
        depletion_matrix = np.asarray([row["depletion"] for row in rows], dtype=float)
        for position, year in enumerate(years):
            for level in levels:
                tail = (1.0 - float(level)) / 2.0
                b_low, b_high = np.quantile(biomass_matrix[:, position], [tail, 1.0 - tail])
                d_low, d_high = np.quantile(depletion_matrix[:, position], [tail, 1.0 - tail])
                time_series.append(
                    {
                        "year": int(year),
                        "confidence_level": float(level),
                        "biomass_low": float(b_low),
                        "biomass_high": float(b_high),
                        "depletion_low": float(d_low),
                        "depletion_high": float(d_high),
                    }
                )
    return {"intervals": intervals, "time_series": time_series, "successful_replicates": len(rows), "failures": failures}


def _profile_intervals(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    levels: Sequence[float],
    points: int,
    multistarts: int,
    seed: int,
) -> dict[str, Any]:
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    diagnostics = {}
    for position, parameter in enumerate(("k", "r", "initial_depletion", "sigma")):
        result = profile_likelihood(
            dataset,
            settings,
            fitted,
            parameter,
            ProfileSettings(
                points=max(7, int(points)),
                confidence_levels=tuple(float(level) for level in levels),
                workers=1,
                multistarts=max(1, int(multistarts)),
                seed=seed + position * 104729,
                use_cache=False,
            ),
        )
        intervals[parameter] = {
            str(float(row["confidence_level"])): {
                "low": float(row["low"]) if row["low"] is not None else float("nan"),
                "high": float(row["high"]) if row["high"] is not None else float("nan"),
                "width": float(row["high"] - row["low"]) if row["low"] is not None and row["high"] is not None else float("nan"),
                "complete": bool(row["complete"]),
            }
            for row in result["intervals"]
        }
        diagnostics[parameter] = result["summary"]
    return {"intervals": intervals, "time_series": [], "diagnostics": diagnostics}


def _coverage_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    # Outer coverage replicates are parallel. Limit OpenMP inside each process to
    # prevent nested-process/thread oversubscription and fork/OpenMP stalls.
    get_native_engine().set_threads(max(1, int(task.get("native_threads_per_worker", 1))))
    frame = pd.DataFrame(task["frame"])
    dataset = StockDataset(task["dataset_name"], frame, index_columns=[column for column in frame if column == "index" or str(column).startswith("index_")])
    settings = ModelSettings(**task["model_settings"])
    config = CoverageSettings(**task["coverage_settings"])
    replicate = int(task["replicate"])
    truth_parameters = task["truth_parameters"]
    process_cv = float(config.process_cv if config.process_cv is not None else settings.process_cv)
    observation_cv = float(config.observation_cv if config.observation_cv is not None else settings.obs_cv)
    simulated, truth = _simulate_from_parameters(
        dataset,
        settings,
        float(truth_parameters["k"]),
        float(truth_parameters["r"]),
        float(truth_parameters["initial_depletion"]),
        float(truth_parameters["sigma"]),
        int(config.seed) + replicate * 104729,
        process_cv,
        observation_cv,
    )
    try:
        fitted = fit(simulated, replace(settings, seed=int(config.seed) + replicate * 15485863, search_draws=max(120, int(config.search_draws))))
    except Exception as exc:
        return {"replicate": replicate + 1, "fit_success": False, "error": f"{type(exc).__name__}: {exc}", "truth": truth, "methods": {}}

    method_outputs: dict[str, Any] = {}
    methods = tuple(str(method).lower() for method in config.methods)
    try:
        if "hessian" in methods:
            method_outputs["hessian"] = _wald_intervals(simulated, settings, fitted, config.confidence_levels, config.hessian_step)
    except Exception as exc:
        method_outputs["hessian"] = {"error": f"{type(exc).__name__}: {exc}", "intervals": {}, "time_series": []}
    try:
        if "profile" in methods:
            method_outputs["profile"] = _profile_intervals(
                simulated,
                settings,
                fitted,
                config.confidence_levels,
                config.profile_points,
                config.profile_multistarts,
                int(config.seed) + replicate * 32452843,
            )
    except Exception as exc:
        method_outputs["profile"] = {"error": f"{type(exc).__name__}: {exc}", "intervals": {}, "time_series": []}
    try:
        if "parametric_bootstrap" in methods or "bootstrap" in methods:
            method_outputs["parametric_bootstrap"] = _bootstrap_intervals(
                simulated,
                settings,
                fitted,
                config.confidence_levels,
                config.bootstrap_replicates,
                int(config.seed) + replicate * 49979687,
                config.search_draws,
                process_cv,
                observation_cv,
            )
    except Exception as exc:
        method_outputs["parametric_bootstrap"] = {"error": f"{type(exc).__name__}: {exc}", "intervals": {}, "time_series": []}

    estimates = {
        "k": float(fitted.best["k_b0"]),
        "r": float(fitted.best["r"]),
        "initial_depletion": float(fitted.best["initial_depletion"]),
        "sigma": float(fitted.best["sigma"]),
        "terminal_biomass": float(fitted.best["terminal_biomass"]),
        "terminal_depletion": float(fitted.best["terminal_depletion"]),
    }
    return {
        "replicate": replicate + 1,
        "fit_success": True,
        "truth": truth,
        "estimates": estimates,
        "objective": float(fitted.best["objective"]),
        "backend": fitted.diagnostics.get("refinement_backend"),
        "methods": method_outputs,
    }


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def run_interval_coverage(
    dataset: StockDataset,
    settings: ModelSettings,
    truth_fit: FitResult,
    coverage_settings: CoverageSettings | None = None,
) -> dict[str, Any]:
    """Run repeated known-truth frequentist coverage experiments.

    Every outer replicate simulates a known population, refits Omega, constructs
    uncertainty intervals with the requested method, and checks whether the known
    truth is inside each interval. Failed fits and incomplete intervals remain in
    the denominator and are reported explicitly.
    """

    config = coverage_settings or CoverageSettings()
    truth_parameters = {
        "k": float(truth_fit.best["k_b0"]),
        "r": float(truth_fit.best["r"]),
        "initial_depletion": float(truth_fit.best["initial_depletion"]),
        "sigma": float(truth_fit.best["sigma"]),
    }
    frame_payload = dataset.frame.sort_values("year").reset_index(drop=True).to_dict(orient="list")
    tasks = [
        {
            "frame": frame_payload,
            "dataset_name": dataset.name,
            "model_settings": asdict(settings),
            "coverage_settings": asdict(config),
            "truth_parameters": truth_parameters,
            "replicate": replicate,
            "native_threads_per_worker": max(1, int(config.native_threads_per_worker)),
        }
        for replicate in range(max(2, int(config.replicates)))
    ]

    rows: list[dict[str, Any]] = []
    if int(config.workers) > 1:
        with ProcessPoolExecutor(
            max_workers=min(int(config.workers), len(tasks)),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {executor.submit(_coverage_worker, task): task["replicate"] for task in tasks}
            for future in as_completed(futures):
                replicate = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append({"replicate": replicate + 1, "fit_success": False, "error": f"{type(exc).__name__}: {exc}", "truth": {}, "methods": {}})
    else:
        rows = [_coverage_worker(task) for task in tasks]
    rows.sort(key=lambda row: int(row["replicate"]))

    successful = [row for row in rows if row.get("fit_success")]
    failed_fits = len(rows) - len(successful)
    method_names = sorted({method for row in successful for method in (row.get("methods") or {})})
    quantities = ("k", "r", "initial_depletion", "sigma", "terminal_biomass", "terminal_depletion")
    coverage_rows = []
    for method in method_names:
        for quantity in quantities:
            for level in config.confidence_levels:
                attempted = len(rows)
                complete = 0
                covered = 0
                widths = []
                for row in successful:
                    interval = (((row.get("methods") or {}).get(method) or {}).get("intervals") or {}).get(quantity, {}).get(str(float(level)))
                    if not interval:
                        continue
                    low = float(interval.get("low", float("nan")))
                    high = float(interval.get("high", float("nan")))
                    if not (np.isfinite(low) and np.isfinite(high)):
                        continue
                    complete += 1
                    truth = float(row["truth"][quantity])
                    covered += int(low <= truth <= high)
                    widths.append(high - low)
                empirical = covered / attempted if attempted else float("nan")
                mc_low, mc_high = _wilson_interval(covered, attempted)
                coverage_rows.append(
                    {
                        "method": method,
                        "parameter": quantity,
                        "nominal": float(level),
                        "empirical": float(empirical),
                        "monte_carlo_low": float(mc_low),
                        "monte_carlo_high": float(mc_high),
                        "attempted_replicates": attempted,
                        "complete_intervals": complete,
                        "covered": covered,
                        "incomplete_or_failed": attempted - complete,
                        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
                        "absolute_coverage_error": abs(float(empirical) - float(level)) if np.isfinite(empirical) else float("nan"),
                    }
                )

    recovery = []
    for quantity in quantities:
        truth_values = np.asarray([row["truth"][quantity] for row in successful], dtype=float)
        estimates = np.asarray([row["estimates"][quantity] for row in successful], dtype=float)
        if not len(estimates):
            continue
        errors = estimates - truth_values
        relative = errors / np.maximum(np.abs(truth_values), _EPS)
        recovery.append(
            {
                "parameter": quantity,
                "mean_bias": float(np.mean(errors)),
                "mean_relative_bias": float(np.mean(relative)),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "median_relative_error": float(np.median(relative)),
            }
        )

    time_series_coverage = []
    if config.include_time_series:
        years = dataset.frame.sort_values("year")["year"].to_numpy(dtype=int)
        for method in method_names:
            for position, year in enumerate(years):
                for level in config.confidence_levels:
                    for quantity, low_key, high_key in (
                        ("biomass", "biomass_low", "biomass_high"),
                        ("depletion", "depletion_low", "depletion_high"),
                    ):
                        complete = 0; covered = 0
                        for row in successful:
                            entries = ((row.get("methods") or {}).get(method) or {}).get("time_series") or []
                            match = next((entry for entry in entries if int(entry["year"]) == int(year) and float(entry["confidence_level"]) == float(level)), None)
                            if not match:
                                continue
                            low = float(match[low_key]); high = float(match[high_key])
                            if not (np.isfinite(low) and np.isfinite(high)):
                                continue
                            complete += 1
                            truth = float(row["truth"][quantity][position])
                            covered += int(low <= truth <= high)
                        attempted = len(rows)
                        empirical = covered / attempted if attempted else float("nan")
                        time_series_coverage.append(
                            {
                                "method": method,
                                "quantity": quantity,
                                "year": int(year),
                                "nominal": float(level),
                                "empirical": float(empirical),
                                "attempted_replicates": attempted,
                                "complete_intervals": complete,
                            }
                        )

    limit = float(config.biomass_limit)
    false_overfished = sum(
        float(row["truth"]["terminal_depletion"]) >= limit and float(row["estimates"]["terminal_depletion"]) < limit
        for row in successful
    )
    false_healthy = sum(
        float(row["truth"]["terminal_depletion"]) < limit and float(row["estimates"]["terminal_depletion"]) >= limit
        for row in successful
    )
    valid_errors = [float(row["absolute_coverage_error"]) for row in coverage_rows if np.isfinite(float(row["absolute_coverage_error"]))]
    maximum_coverage_error = max(valid_errors, default=float("nan"))
    maximum_bias = max((abs(float(row["mean_relative_bias"])) for row in recovery), default=float("nan"))
    failure_fraction = failed_fits / len(rows) if rows else 1.0
    status = (
        "PASS"
        if np.isfinite(maximum_coverage_error) and maximum_coverage_error <= 0.10 and maximum_bias <= 0.10 and failure_fraction <= 0.05
        else "WARN"
        if np.isfinite(maximum_coverage_error) and maximum_coverage_error <= 0.20 and maximum_bias <= 0.25 and failure_fraction <= 0.15
        else "FAIL"
    )
    return {
        "summary": {
            "status": status,
            "attempted_replicates": len(rows),
            "successful_fits": len(successful),
            "failed_fits": failed_fits,
            "failure_fraction": float(failure_fraction),
            "methods": method_names,
            "maximum_absolute_coverage_error": maximum_coverage_error,
            "maximum_absolute_mean_relative_bias": maximum_bias,
            "formal_known_truth_testing": True,
            "failed_and_incomplete_intervals_count_against_coverage": True,
            "false_overfished_classifications": int(false_overfished),
            "false_healthy_classifications": int(false_healthy),
            "classification_limit": limit,
        },
        "coverage": coverage_rows,
        "recovery": recovery,
        "time_series_coverage": time_series_coverage,
        "replicates": rows,
        "configuration": asdict(config),
        "interpretation": (
            "Coverage is empirical repeated-sampling performance against known simulated truth. A nominal 95% interval is calibrated only when "
            "approximately 95% of attempted replicates contain the truth, with failed fits and incomplete intervals retained as failures."
        ),
    }
