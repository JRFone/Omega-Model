from __future__ import annotations

"""Automated and exploratory expert stock-assessment workflow.

The workflow combines the diagnostic classes commonly assembled from SS3,
r4ss, ss3diags, ss3sim, SSMSE and bespoke scripts. It does not lock the analyst
out of alternative configurations: exploration-mode overrides are allowed and
recorded, while automatic mode runs all available gates and reports failures.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from math import exp, log
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .closed_loop_mse import (
    MSESettings,
    ManagementProcedure,
    OperatingModelSettings,
    pareto_front,
    run_closed_loop_mse,
)
from .core import FitResult, ModelSettings, _objective_breakdown, _production, fit
from .data_io import StockDataset
from .age_structured import AgeFitSettings, AgeStructuredSettings, fit_age_structured
from .aspm_diagnostic import ASPMSettings, run_age_structured_aspm
from .interval_coverage import CoverageSettings, run_interval_coverage
from .likelihood_profiles import ProfileSettings, profile_likelihood
from .native_backend import get_native_engine, native_status
from .diagnostics_suite import data_conflict_matrix, reliability_grade, retrospective_metrics
from .quant_lab import (
    PARAMETER_NAMES,
    QuantOptimizerSettings,
    local_identifiability_diagnostics,
    run_global_optimizer,
)
from .biomass_truth_engine import BiomassTruthSettings, estimate_best_supported_biomass
from .experimental_diagnostics import ExperimentalDiagnosticSettings, run_experimental_diagnostics
from .advanced_mse import (
    AdvancedMSESettings,
    MSEAssessmentSettings,
    MSEManagementProcedure,
    MSEObservationSettings,
    default_operating_scenarios,
    run_advanced_mse,
)
from .quant_validation import (
    EnsembleSettings,
    OptimizerAgreementSettings,
    WalkForwardSettings,
    run_model_ensemble,
    run_optimizer_agreement,
    run_walk_forward_validation,
)


_EPS = 1e-12


@dataclass(frozen=True)
class WorkflowOverride:
    setting: str
    value: Any
    reason: str
    author: str = "analyst"


@dataclass(frozen=True)
class ExpertWorkflowSettings:
    mode: str = "automatic"  # automatic or exploration
    speed: str = "quick"  # quick, standard, deep
    seed: int = 54131
    workers: int = 0
    peels: int | None = None
    jitter_runs: int | None = None
    simulation_replicates: int | None = None
    optimizer_population: int | None = None
    optimizer_generations: int | None = None
    mse_simulations: int | None = None
    mse_years: int | None = None
    skipped_steps: tuple[str, ...] = ()
    overrides: tuple[WorkflowOverride, ...] = ()
    continue_after_failure: bool = True
    cache_directory: str | None = None

    def resolved(self) -> "ResolvedWorkflowSettings":
        speed = str(self.speed).strip().lower()
        if speed not in {"quick", "standard", "deep"}:
            speed = "quick"
        mode = str(self.mode).strip().lower()
        if mode not in {"automatic", "exploration"}:
            mode = "automatic"
        defaults = {
            "quick": dict(peels=2, jitter_runs=4, simulation_replicates=4, optimizer_population=12, optimizer_generations=1, mse_simulations=50, mse_years=10, search_draws=120),
            "standard": dict(peels=5, jitter_runs=16, simulation_replicates=40, optimizer_population=24, optimizer_generations=6, mse_simulations=300, mse_years=25, search_draws=220),
            "deep": dict(peels=8, jitter_runs=40, simulation_replicates=200, optimizer_population=48, optimizer_generations=14, mse_simulations=1000, mse_years=35, search_draws=500),
        }[speed]
        return ResolvedWorkflowSettings(
            mode=mode,
            speed=speed,
            seed=int(self.seed),
            workers=max(1, int(self.workers or min(8, max(2, (os_cpu_count() or 2) - 1)))),
            peels=max(1, int(self.peels if self.peels is not None else defaults["peels"])),
            jitter_runs=max(2, int(self.jitter_runs if self.jitter_runs is not None else defaults["jitter_runs"])),
            simulation_replicates=max(2, int(self.simulation_replicates if self.simulation_replicates is not None else defaults["simulation_replicates"])),
            optimizer_population=max(12, int(self.optimizer_population if self.optimizer_population is not None else defaults["optimizer_population"])),
            optimizer_generations=max(1, int(self.optimizer_generations if self.optimizer_generations is not None else defaults["optimizer_generations"])),
            mse_simulations=max(20, int(self.mse_simulations if self.mse_simulations is not None else defaults["mse_simulations"])),
            mse_years=max(5, int(self.mse_years if self.mse_years is not None else defaults["mse_years"])),
            search_draws=max(120, int(defaults["search_draws"])),
            skipped_steps=tuple(str(value) for value in self.skipped_steps),
            overrides=tuple(self.overrides),
            continue_after_failure=bool(self.continue_after_failure),
            cache_directory=self.cache_directory,
        )


@dataclass(frozen=True)
class ResolvedWorkflowSettings:
    mode: str
    speed: str
    seed: int
    workers: int
    peels: int
    jitter_runs: int
    simulation_replicates: int
    optimizer_population: int
    optimizer_generations: int
    mse_simulations: int
    mse_years: int
    search_draws: int
    skipped_steps: tuple[str, ...]
    overrides: tuple[WorkflowOverride, ...]
    continue_after_failure: bool
    cache_directory: str | None


@dataclass
class WorkflowStep:
    name: str
    status: str
    required: bool
    message: str
    result_key: str | None = None
    error: str | None = None


class WorkflowRunError(RuntimeError):
    pass


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:  # pragma: no cover
        return None


def mean_absolute_scaled_error(observed: Sequence[float], predicted: Sequence[float], seasonality: int = 1) -> float:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if int(mask.sum()) == 0:
        return float("nan")
    numerator = float(np.mean(np.abs(obs[mask] - pred[mask])))
    finite_obs = obs[np.isfinite(obs)]
    seasonality = max(1, int(seasonality))
    if len(finite_obs) <= seasonality:
        return float("nan")
    denominator = float(np.mean(np.abs(finite_obs[seasonality:] - finite_obs[:-seasonality])))
    return numerator / max(denominator, _EPS)


def _copy_dataset(dataset: StockDataset, frame: pd.DataFrame, suffix: str) -> StockDataset:
    return StockDataset(
        name=f"{dataset.name} {suffix}",
        frame=frame.reset_index(drop=True).copy(),
        provenance=dict(dataset.provenance),
        transformations=list(dataset.transformations),
        warnings=list(dataset.warnings),
        raw_columns=list(dataset.raw_columns),
        index_columns=list(dataset.index_columns),
    )


def run_jitter_diagnostic(
    dataset: StockDataset,
    settings: ModelSettings,
    *,
    runs: int = 12,
    seed: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    runs = max(2, int(runs))

    def execute(index: int) -> dict[str, Any]:
        fitted = fit(dataset, replace(settings, seed=int(seed) + index * 7919))
        return {
            "run": index + 1,
            "seed": int(seed) + index * 7919,
            "optimizer": "Omega random multistart + coordinate refine",
            "objective": float(fitted.best["objective"]),
            "terminal_depletion": float(fitted.best["terminal_depletion"]),
            "k": float(fitted.best["k_b0"]),
            "r": float(fitted.best["r"]),
            "initial_depletion": float(fitted.best["initial_depletion"]),
            "sigma": float(fitted.best["sigma"]),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), runs))) as executor:
        futures = {executor.submit(execute, index): index for index in range(runs)}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["run"])
    objectives = np.asarray([row["objective"] for row in rows], dtype=float)
    depletion = np.asarray([row["terminal_depletion"] for row in rows], dtype=float)
    best = float(np.min(objectives))
    for row in rows:
        row["objective_delta"] = float(row["objective"] - best)
    near_best = depletion[objectives <= best + 2.0]
    spread = float(np.ptp(near_best)) if len(near_best) else float(np.ptp(depletion))
    status = "PASS" if spread <= 0.03 else "WARN" if spread <= 0.08 else "FAIL"
    return {
        "summary": {
            "status": status,
            "runs": runs,
            "best_objective": best,
            "objective_range": float(np.ptp(objectives)),
            "near_best_terminal_depletion_spread": spread,
            "near_best_runs": int(len(near_best)),
            "interpretation": "Near-equivalent objective values should not imply materially different stock status.",
        },
        "runs": rows,
    }


def parameter_boundary_diagnostic(dataset: StockDataset, fitted: FitResult) -> dict[str, Any]:
    frame = dataset.frame
    catches = frame["catch"].to_numpy(dtype=float) * float(fitted.settings.get("catch_multiplier", 1.0))
    max_catch = max(float(np.nanmax(catches)), 1.0)
    total_catch = max(float(np.nansum(catches)), max_catch)
    k_low = max(max_catch * 3.0, total_catch * 0.25)
    k_high = max(k_low * 5.0, total_catch * 12.0)
    bounds = {
        "k": (k_low, k_high, float(fitted.best["k_b0"])),
        "r": (0.025, 0.65, float(fitted.best["r"])),
        "initial_depletion": (0.0, 1.0, float(fitted.best["initial_depletion"])),
        "sigma": (0.08, 0.75, float(fitted.best["sigma"])),
    }
    rows = []
    near = []
    for name, (low, high, value) in bounds.items():
        fraction = (value - low) / max(high - low, _EPS)
        is_near = fraction <= 0.03 or fraction >= 0.97
        if is_near:
            near.append(name)
        rows.append({
            "parameter": name,
            "value": value,
            "lower": low,
            "upper": high,
            "fraction_through_bounds": fraction,
            "near_boundary": is_near,
        })
    return {"summary": {"status": "PASS" if not near else "WARN", "near_boundary": near}, "parameters": rows}



def finite_difference_gradient_diagnostic(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    *,
    step: float = 1e-5,
) -> dict[str, Any]:
    """Evaluate the exact native AD gradient, with parity-check finite differences.

    The function name is retained for API compatibility with Omega 1.2. The
    primary result is now the compiled C++ forward-mode automatic derivative.
    """

    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)
    index = frame["index"].to_numpy(dtype=float)
    biomass_obs = frame["biomass"].to_numpy(dtype=float)
    b0 = min(max(float(fitted.best["initial_depletion"]), 1e-8), 1.0 - 1e-8)
    theta = np.asarray(
        [
            log(float(fitted.best["k_b0"])),
            log(float(fitted.best["r"])),
            log(b0 / (1.0 - b0)),
            log(float(fitted.best["sigma"])),
        ],
        dtype=float,
    )
    names = ("log_k", "log_r", "logit_initial_depletion", "log_sigma")
    max_catch = max(float(np.nanmax(catches)), 1.0)
    total_catch = max(float(np.nansum(catches)), max_catch)
    k_low = max(max_catch * 3.0, total_catch * 0.25)
    k_high = max(k_low * 5.0, total_catch * 12.0)
    transformed_bounds = (
        (log(k_low), log(k_high)),
        (log(0.0051), log(1.199)),
        (-6.0, 6.0),
        (log(0.03), log(1.5)),
    )
    engine = get_native_engine()
    native = engine.objective_gradient(theta, years, catches, index, biomass_obs, settings)
    h = max(float(step), 1e-7)
    rows = []
    parity_errors = []
    for position, name in enumerate(names):
        local_step = h * max(1.0, abs(float(theta[position])))
        automatic = float(native.gradient[position])
        lower, upper = transformed_bounds[position]
        boundary_tolerance = max(1e-8, local_step * 1.1)
        active_bound = "lower" if theta[position] - lower <= boundary_tolerance else "upper" if upper - theta[position] <= boundary_tolerance else None
        projected = automatic
        if (active_bound == "lower" and automatic >= 0.0) or (active_bound == "upper" and automatic <= 0.0):
            projected = 0.0
        finite: float | None = None
        error: float | None = None
        parity_note: str | None = None
        if active_bound is None:
            plus = theta.copy(); minus = theta.copy()
            plus[position] += local_step; minus[position] -= local_step
            f_plus = _objective_breakdown(plus, years, catches, index, biomass_obs, settings)[0]
            f_minus = _objective_breakdown(minus, years, catches, index, biomass_obs, settings)[0]
            finite = float((f_plus - f_minus) / (2.0 * local_step))
            error = abs(automatic - finite)
            parity_errors.append(error)
        else:
            parity_note = f"Central finite-difference parity is not defined at the active {active_bound} transform bound."
        rows.append(
            {
                "parameter": name,
                "automatic_gradient": automatic,
                "finite_difference_gradient": finite,
                "absolute_gradient": abs(automatic),
                "projected_absolute_gradient": abs(projected),
                "parity_error": error,
                "parity_tested": active_bound is None,
                "active_bound": active_bound,
                "parity_note": parity_note,
            }
        )
    maximum = max((row["projected_absolute_gradient"] for row in rows), default=float("nan"))
    maximum_raw = max((row["absolute_gradient"] for row in rows), default=float("nan"))
    maximum_parity_error = max(parity_errors, default=float("nan"))
    stationary_status = "PASS" if np.isfinite(maximum) and maximum <= 1e-3 else "WARN" if np.isfinite(maximum) and maximum <= 1e-2 else "FAIL"
    parity_status = "NOT TESTED" if not parity_errors else "PASS" if np.isfinite(maximum_parity_error) and maximum_parity_error <= 1e-5 else "WARN" if maximum_parity_error <= 1e-3 else "FAIL"
    status = "FAIL" if "FAIL" in {stationary_status, parity_status} else "WARN" if "WARN" in {stationary_status, parity_status} else "PASS"
    return {
        "summary": {
            "status": status,
            "stationary_status": stationary_status,
            "ad_parity_status": parity_status,
            "maximum_gradient": maximum,
            "maximum_raw_gradient": maximum_raw,
            "maximum_ad_finite_difference_error": maximum_parity_error,
            "parity_parameters_tested": len(parity_errors),
            "parity_parameters_not_tested_at_active_bounds": len(rows) - len(parity_errors),
            "objective": float(native.objective),
            "method": "compiled C++ forward-mode automatic differentiation, checked against central finite differences away from active transform bounds; stationarity uses projected gradients",
            "backend": native.backend,
            "criterion": "maximum projected gradient ≤0.001 preferred; AD parity error ≤1e-5 where central differences are defined",
        },
        "gradients": rows,
    }


def likelihood_component_profiles(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    *,
    points: int = 9,
) -> dict[str, Any]:
    """One-dimensional fixed-other-parameter profiles split by objective component.

    These profiles show which likelihood or penalty component pulls a parameter in
    which direction. They are intentionally labelled fixed-other-parameter profiles;
    the separate identifiability workflow performs refitted/local profiles.
    """

    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)
    index = frame["index"].to_numpy(dtype=float)
    biomass_obs = frame["biomass"].to_numpy(dtype=float)
    b0 = min(max(float(fitted.best["initial_depletion"]), 1e-8), 1.0 - 1e-8)
    center = np.asarray([
        log(float(fitted.best["k_b0"])),
        log(float(fitted.best["r"])),
        log(b0 / (1.0 - b0)),
        log(float(fitted.best["sigma"])),
    ], dtype=float)
    definitions = (
        ("k", 0, 0.70, lambda value: exp(value)),
        ("r", 1, 0.70, lambda value: exp(value)),
        ("initial_depletion", 2, 1.50, lambda value: 1.0 / (1.0 + exp(-value))),
        ("sigma", 3, 0.70, lambda value: exp(value)),
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    points = max(5, min(int(points), 31))
    for parameter, position, half_width, decode in definitions:
        parameter_rows = []
        for transformed in np.linspace(center[position] - half_width, center[position] + half_width, points):
            theta = center.copy(); theta[position] = transformed
            objective, _prediction, _sigma, components = _objective_breakdown(theta, years, catches, index, biomass_obs, settings)
            row = {"parameter": parameter, "value": float(decode(float(transformed))), "objective": float(objective), **{key: float(value) for key, value in components.items()}}
            rows.append(row); parameter_rows.append(row)
        component_names = [key for key in parameter_rows[0] if key not in {"parameter", "value", "objective"}]
        preferences = []
        for component in component_names:
            component_values = np.asarray([float(row[component]) for row in parameter_rows], dtype=float)
            component_range = float(np.ptp(component_values))
            tolerance = max(1e-8, 1e-6 * max(float(np.max(np.abs(component_values))), 1.0))
            if component_range <= tolerance:
                continue
            best = min(parameter_rows, key=lambda row: float(row[component]))
            preferences.append({
                "component": component,
                "preferred_value": float(best["value"]),
                "minimum_component_objective": float(best[component]),
                "component_objective_range": component_range,
            })
        values = np.asarray([row["preferred_value"] for row in preferences], dtype=float)
        scale = max(float(np.max([row["value"] for row in parameter_rows]) - np.min([row["value"] for row in parameter_rows])), _EPS)
        conflict_index = float(np.ptp(values) / scale) if len(values) >= 2 else 0.0
        summaries.append({
            "parameter": parameter,
            "component_preference_conflict_0_1": conflict_index,
            "status": "PASS" if conflict_index <= 0.25 else "WARN" if conflict_index <= 0.55 else "FAIL",
            "preferences": preferences,
        })
    maximum = max((row["component_preference_conflict_0_1"] for row in summaries), default=0.0)
    return {
        "summary": {
            "status": "PASS" if maximum <= 0.25 else "WARN" if maximum <= 0.55 else "FAIL",
            "maximum_component_preference_conflict_0_1": maximum,
            "method": "fixed-other-parameter component profiles",
        },
        "parameters": summaries,
        "profiles": rows,
    }


def run_composition_reweighting_comparison(
    dataset: StockDataset,
    age_composition: pd.DataFrame | None,
    length_composition: pd.DataFrame | None,
    *,
    population: int = 12,
    generations: int = 2,
    seed: int = 1,
) -> dict[str, Any]:
    if age_composition is None and length_composition is None:
        return {
            "summary": {
                "status": "NOT TESTED",
                "reason": "No age_composition.csv or length_composition.csv was supplied.",
            },
            "scenarios": [],
        }
    scenarios: list[tuple[str, float, float]] = [("Base composition weights", 1.0, 1.0)]
    if age_composition is not None:
        scenarios.extend((("Age composition low", 0.25, 1.0), ("Age composition high", 4.0, 1.0)))
    if length_composition is not None:
        scenarios.extend((("Length composition low", 1.0, 0.25), ("Length composition high", 1.0, 4.0)))
    rows = []
    for index, (label, age_weight, length_weight) in enumerate(scenarios):
        result = fit_age_structured(
            dataset,
            AgeStructuredSettings(age_comp_weight=age_weight, length_comp_weight=length_weight),
            AgeFitSettings(
                population=max(12, int(population)),
                generations=max(1, int(generations)),
                seed=int(seed) + index * 3571,
                local_rounds=1,
                estimate_recruitment_sigma=False,
            ),
            age_composition=age_composition,
            length_composition=length_composition,
        )
        components = result.diagnostics.get("objective_components") or {}
        rows.append({
            "scenario": label,
            "age_comp_weight": age_weight,
            "length_comp_weight": length_weight,
            "objective": float(result.best["objective"]),
            "terminal_depletion": float(result.best["terminal_depletion"]),
            "natural_mortality": float(result.best["natural_mortality"]),
            "steepness": float(result.best["steepness"]),
            "age_composition_deviance": float(components.get("age_composition", components.get("age_composition_deviance", 0.0)) or 0.0),
            "length_composition_deviance": float(components.get("length_composition", components.get("length_composition_deviance", 0.0)) or 0.0),
        })
    spread = float(np.ptp([row["terminal_depletion"] for row in rows])) if rows else 0.0
    return {
        "summary": {
            "status": "PASS" if spread <= 0.05 else "WARN" if spread <= 0.15 else "FAIL",
            "terminal_depletion_spread": spread,
            "scenarios": len(rows),
            "interpretation": "Large changes show that composition weighting materially controls stock status or resolves conflict.",
        },
        "scenarios": rows,
    }


def residual_diagnostics(dataset: StockDataset, fitted: FitResult) -> dict[str, Any]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    biomass = np.asarray([row["biomass"] for row in fitted.history], dtype=float)
    observed_index = frame["index"].to_numpy(dtype=float)
    observed_biomass = frame["biomass"].to_numpy(dtype=float)
    index_mask = np.isfinite(observed_index) & (observed_index > 0) & np.isfinite(biomass) & (biomass > 0)
    q = float(exp(np.mean(np.log(observed_index[index_mask]) - np.log(biomass[index_mask])))) if index_mask.any() else float("nan")
    predicted_index = q * biomass if np.isfinite(q) else np.full_like(biomass, np.nan)
    rows = []
    index_residuals = []
    biomass_residuals = []
    for position, year in enumerate(frame["year"].to_numpy(dtype=int)):
        index_residual = (
            float(log(observed_index[position]) - log(predicted_index[position]))
            if np.isfinite(observed_index[position]) and observed_index[position] > 0 and np.isfinite(predicted_index[position]) and predicted_index[position] > 0
            else float("nan")
        )
        biomass_residual = (
            float(log(observed_biomass[position]) - log(biomass[position]))
            if np.isfinite(observed_biomass[position]) and observed_biomass[position] > 0 and biomass[position] > 0
            else float("nan")
        )
        if np.isfinite(index_residual):
            index_residuals.append(index_residual)
        if np.isfinite(biomass_residual):
            biomass_residuals.append(biomass_residual)
        rows.append({
            "year": int(year),
            "index_observed": float(observed_index[position]) if np.isfinite(observed_index[position]) else None,
            "index_predicted": float(predicted_index[position]) if np.isfinite(predicted_index[position]) else None,
            "index_log_residual": index_residual if np.isfinite(index_residual) else None,
            "biomass_observed": float(observed_biomass[position]) if np.isfinite(observed_biomass[position]) else None,
            "biomass_predicted": float(biomass[position]),
            "biomass_log_residual": biomass_residual if np.isfinite(biomass_residual) else None,
        })
    matrix = [
        [row["index_log_residual"] if row["index_log_residual"] is not None else np.nan for row in rows],
        [row["biomass_log_residual"] if row["biomass_log_residual"] is not None else np.nan for row in rows],
    ]
    return {
        "summary": {
            "index_log_rmse": float(np.sqrt(np.mean(np.square(index_residuals)))) if index_residuals else None,
            "index_log_bias": float(np.mean(index_residuals)) if index_residuals else None,
            "biomass_log_rmse": float(np.sqrt(np.mean(np.square(biomass_residuals)))) if biomass_residuals else None,
            "biomass_log_bias": float(np.mean(biomass_residuals)) if biomass_residuals else None,
            "q": q if np.isfinite(q) else None,
        },
        "rows": rows,
        "heatmap": {"matrix": np.asarray(matrix, dtype=float).tolist(), "row_labels": ["Index", "Biomass"], "column_labels": frame["year"].astype(int).tolist()},
    }


def run_retrospective_diagnostic(
    dataset: StockDataset,
    settings: ModelSettings,
    *,
    peels: int = 5,
    search_draws: int = 120,
    seed: int = 1,
) -> dict[str, Any]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    maximum_peels = max(0, len(frame) - 5)
    peels = min(max(1, int(peels)), maximum_peels)
    full_fit = fit(dataset, replace(settings, search_draws=max(120, int(search_draws)), seed=seed))
    full_series = {int(row["year"]): float(row["depletion"]) for row in full_fit.history}
    peel_series = []
    peel_rows = []
    for peel in range(1, peels + 1):
        subset = frame.iloc[:-peel].copy()
        peeled = fit(
            _copy_dataset(dataset, subset, f"retrospective peel {peel}"),
            replace(settings, search_draws=max(120, int(search_draws)), seed=seed + peel * 1237),
        )
        series = {int(row["year"]): float(row["depletion"]) for row in peeled.history}
        peel_series.append(series)
        peel_rows.append({
            "peel": peel,
            "terminal_year": int(peeled.history[-1]["year"]),
            "terminal_depletion": float(peeled.best["terminal_depletion"]),
            "objective": float(peeled.best["objective"]),
            "series": series,
        })
    metrics = retrospective_metrics(full_series, peel_series)
    rho = metrics["mohn_rho"]
    status = "PASS" if np.isfinite(rho) and abs(rho) <= 0.15 else "WARN" if np.isfinite(rho) and abs(rho) <= 0.30 else "FAIL"
    return {
        "summary": {"status": status, **{key: value for key, value in metrics.items() if key != "peels"}},
        "full": full_series,
        "peels": peel_rows,
        "mohn_rows": metrics["peels"],
    }


def add_mase_to_walk_forward(dataset: StockDataset, result: dict[str, Any]) -> dict[str, Any]:
    predictions = result.get("predictions") or []
    observed = [row.get("observed_index") for row in predictions]
    predicted = [row.get("predicted_index") for row in predictions]
    mase = mean_absolute_scaled_error(
        [float(value) if value is not None else np.nan for value in observed],
        [float(value) if value is not None else np.nan for value in predicted],
    )
    result = dict(result)
    summary = dict(result.get("summary") or {})
    summary["index_mase"] = mase if np.isfinite(mase) else None
    summary["mase_status"] = "PASS" if np.isfinite(mase) and mase < 1.0 else "WARN" if np.isfinite(mase) and mase < 1.5 else "FAIL"
    summary["mase_interpretation"] = "MASE below 1 predicts better than a naïve persistence/change benchmark."
    result["summary"] = summary
    chart_rows = []
    for row in predictions:
        chart_rows.append({
            "year": row.get("prediction_year"),
            "observed": row.get("observed_index"),
            "predicted": row.get("predicted_index"),
            "fold": row.get("fold"),
        })
    result["chart_rows"] = chart_rows
    return result


def run_aspm_diagnostic(dataset: StockDataset, base_settings: ModelSettings, full_fit: FitResult) -> dict[str, Any]:
    aspm_settings = replace(base_settings, biomass_weight=0.0, target_depletion=None)
    aspm = fit(dataset, aspm_settings)
    full = np.asarray([row["depletion"] for row in full_fit.history], dtype=float)
    stripped = np.asarray([row["depletion"] for row in aspm.history], dtype=float)
    correlation = float(np.corrcoef(full, stripped)[0, 1]) if len(full) > 2 and np.std(full) > 0 and np.std(stripped) > 0 else 0.0
    terminal_difference = float(aspm.best["terminal_depletion"] - full_fit.best["terminal_depletion"])
    status = "PASS" if correlation >= 0.8 and abs(terminal_difference) <= 0.08 else "WARN" if correlation >= 0.5 and abs(terminal_difference) <= 0.18 else "FAIL"
    return {
        "summary": {
            "status": status,
            "trajectory_correlation": correlation,
            "terminal_depletion_difference": terminal_difference,
            "full_terminal_depletion": float(full_fit.best["terminal_depletion"]),
            "aspm_terminal_depletion": float(aspm.best["terminal_depletion"]),
            "interpretation": (
                "This strips direct biomass observations and asks whether catch plus abundance indices support the broad trend. "
                "For a production-model base this is an ASPM-style diagnostic, not a full SS3 age-composition removal run."
            ),
        },
        "full": full_fit.history,
        "aspm": aspm.history,
    }


def run_data_removal_influence(dataset: StockDataset, settings: ModelSettings, base_fit: FitResult) -> dict[str, Any]:
    frame = dataset.frame.copy()
    scenarios: list[tuple[str, pd.DataFrame, ModelSettings]] = []
    no_index = frame.copy(); no_index["index"] = np.nan
    scenarios.append(("Remove abundance index", no_index, replace(settings, index_weight=0.0)))
    no_biomass = frame.copy(); no_biomass["biomass"] = np.nan
    scenarios.append(("Remove direct biomass", no_biomass, replace(settings, biomass_weight=0.0, target_depletion=None)))
    if len(frame) >= 10:
        cut = max(1, len(frame) // 5)
        scenarios.append(("Remove earliest 20%", frame.iloc[cut:].copy(), settings))
        scenarios.append(("Remove latest 20%", frame.iloc[:-cut].copy(), settings))
    rows = []
    for index, (label, scenario_frame, scenario_settings) in enumerate(scenarios):
        fitted = fit(_copy_dataset(dataset, scenario_frame, label), replace(scenario_settings, seed=int(settings.seed) + (index + 1) * 3137))
        terminal = float(fitted.best["terminal_depletion"])
        rows.append({
            "omitted": label,
            "terminal_depletion": terminal,
            "absolute_change": abs(terminal - float(base_fit.best["terminal_depletion"])),
            "relative_change": (terminal - float(base_fit.best["terminal_depletion"])) / max(abs(float(base_fit.best["terminal_depletion"])), _EPS),
            "objective": float(fitted.best["objective"]),
        })
    rows.sort(key=lambda row: row["absolute_change"], reverse=True)
    maximum = float(rows[0]["absolute_change"]) if rows else 0.0
    return {"summary": {"status": "PASS" if maximum <= 0.05 else "WARN" if maximum <= 0.15 else "FAIL", "maximum_absolute_change": maximum}, "scenarios": rows}


def run_weighting_comparison(dataset: StockDataset, settings: ModelSettings) -> dict[str, Any]:
    scenarios = [
        ("Index low", 0.25, settings.biomass_weight),
        ("Index base", 1.0, settings.biomass_weight),
        ("Index high", 4.0, settings.biomass_weight),
        ("Biomass low", settings.index_weight, 0.25),
        ("Biomass base", settings.index_weight, 1.0),
        ("Biomass high", settings.index_weight, 4.0),
    ]
    rows = []
    for index, (label, index_weight, biomass_weight) in enumerate(scenarios):
        fitted = fit(dataset, replace(settings, index_weight=index_weight, biomass_weight=biomass_weight, seed=int(settings.seed) + index * 1907))
        rows.append({
            "scenario": label,
            "index_weight": float(index_weight),
            "biomass_weight": float(biomass_weight),
            "terminal_depletion": float(fitted.best["terminal_depletion"]),
            "objective": float(fitted.best["objective"]),
            "k": float(fitted.best["k_b0"]),
            "r": float(fitted.best["r"]),
        })
    values = np.asarray([row["terminal_depletion"] for row in rows], dtype=float)
    spread = float(np.ptp(values))
    return {
        "summary": {
            "status": "PASS" if spread <= 0.05 else "WARN" if spread <= 0.15 else "FAIL",
            "terminal_depletion_spread": spread,
            "scope": "Index and biomass weights for the current production-model route. Age/length composition weights are handled by the age-structured workspace when composition files are supplied.",
        },
        "scenarios": rows,
    }


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], probability: float) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[mask]; w = w[mask]
    if not len(x):
        return float("nan")
    if float(w.sum()) <= 0:
        w = np.ones(len(x), dtype=float)
    order = np.argsort(x)
    x = x[order]; w = w[order]
    cumulative = np.cumsum(w) / np.sum(w)
    return float(np.interp(float(probability), cumulative, x))


def run_simulation_recovery(
    dataset: StockDataset,
    settings: ModelSettings,
    truth_fit: FitResult,
    *,
    replicates: int = 20,
    search_draws: int = 120,
    seed: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float) * float(settings.catch_multiplier)
    true_k = float(truth_fit.best["k_b0"])
    true_r = float(truth_fit.best["r"])
    true_b0 = float(truth_fit.best["initial_depletion"])
    model = str(settings.model)
    pella_shape = float(settings.pella_shape)
    observed_index = frame["index"].to_numpy(dtype=float)
    true_biomass_history = np.asarray([row["biomass"] for row in truth_fit.history], dtype=float)
    mask = np.isfinite(observed_index) & (observed_index > 0)
    q = float(exp(np.mean(np.log(observed_index[mask]) - np.log(true_biomass_history[mask])))) if mask.any() else 1.0
    process_sigma = np.sqrt(np.log1p(max(float(settings.process_cv), 0.0) ** 2))
    observation_sigma = np.sqrt(np.log1p(max(float(settings.obs_cv), 0.0) ** 2))
    levels = (0.50, 0.80, 0.90, 0.95)

    def execute(rep: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed) + rep * 104729)
        biomass = np.empty(len(years), dtype=float)
        biomass[0] = true_k * true_b0
        for position in range(1, len(years)):
            expected = max(
                true_k * 1e-6,
                biomass[position - 1]
                + _production(biomass[position - 1], true_k, true_r, model, pella_shape)
                - catches[position - 1],
            )
            biomass[position] = expected * rng.lognormal(-0.5 * process_sigma**2, process_sigma)
        index = q * biomass * rng.lognormal(-0.5 * observation_sigma**2, observation_sigma, len(years))
        direct_biomass = np.full(len(years), np.nan)
        source_biomass = frame["biomass"].to_numpy(dtype=float)
        direct_mask = np.isfinite(source_biomass) & (source_biomass > 0)
        if direct_mask.any():
            direct_biomass[direct_mask] = biomass[direct_mask] * rng.lognormal(-0.5 * observation_sigma**2, observation_sigma, int(direct_mask.sum()))
        simulated_frame = pd.DataFrame({"year": years, "catch": frame["catch"].to_numpy(dtype=float), "index": index, "biomass": direct_biomass})
        simulated = _copy_dataset(dataset, simulated_frame, f"simulation {rep + 1}")
        fitted = fit(simulated, replace(settings, seed=int(seed) + rep * 15485863, search_draws=max(120, int(search_draws))))
        ensemble = fitted.ensemble or []
        weights = [float(row.get("weight", 1.0)) for row in ensemble]
        intervals = {}
        quantities = {
            "k": (true_k, [float(row["k"]) for row in ensemble]),
            "r": (true_r, [float(row["r"]) for row in ensemble]),
            "terminal_depletion": (float(biomass[-1] / true_k), [float(row["terminal_depletion"]) for row in ensemble]),
        }
        for quantity, (truth, values) in quantities.items():
            intervals[quantity] = {}
            for level in levels:
                tail = (1.0 - level) / 2.0
                low = _weighted_quantile(values, weights, tail)
                high = _weighted_quantile(values, weights, 1.0 - tail)
                intervals[quantity][str(level)] = {"low": low, "high": high, "covered": bool(low <= truth <= high)}
        return {
            "replicate": rep + 1,
            "truth_k": true_k,
            "estimated_k": float(fitted.best["k_b0"]),
            "truth_r": true_r,
            "estimated_r": float(fitted.best["r"]),
            "truth_terminal_depletion": float(biomass[-1] / true_k),
            "estimated_terminal_depletion": float(fitted.best["terminal_depletion"]),
            "objective": float(fitted.best["objective"]),
            "intervals": intervals,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), int(replicates)))) as executor:
        futures = {executor.submit(execute, rep): rep for rep in range(max(2, int(replicates)))}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["replicate"])

    recovery = []
    for quantity, truth_key, estimate_key in (
        ("k", "truth_k", "estimated_k"),
        ("r", "truth_r", "estimated_r"),
        ("terminal_depletion", "truth_terminal_depletion", "estimated_terminal_depletion"),
    ):
        truth = np.asarray([row[truth_key] for row in rows], dtype=float)
        estimate = np.asarray([row[estimate_key] for row in rows], dtype=float)
        relative = (estimate - truth) / np.maximum(np.abs(truth), _EPS)
        recovery.append({
            "quantity": quantity,
            "mean_bias": float(np.mean(estimate - truth)),
            "mean_relative_bias": float(np.mean(relative)),
            "rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
            "median_relative_error": float(np.median(relative)),
        })

    coverage = []
    for quantity in ("k", "r", "terminal_depletion"):
        for level in levels:
            covered = [bool(row["intervals"][quantity][str(level)]["covered"]) for row in rows]
            coverage.append({"parameter": quantity, "nominal": level, "empirical": float(np.mean(covered)), "replicates": len(rows)})
    maximum_bias = max(abs(float(row["mean_relative_bias"])) for row in recovery)
    coverage_error = max(abs(float(row["empirical"]) - float(row["nominal"])) for row in coverage)
    status = "PASS" if maximum_bias <= 0.10 and coverage_error <= 0.15 else "WARN" if maximum_bias <= 0.25 and coverage_error <= 0.30 else "FAIL"
    return {
        "summary": {
            "status": status,
            "replicates": len(rows),
            "maximum_absolute_mean_relative_bias": maximum_bias,
            "maximum_absolute_coverage_error": coverage_error,
            "warning": "Coverage uses Omega candidate-ensemble intervals. It is a diagnostic of the current uncertainty approximation, not proof of frequentist coverage for a final assessment.",
        },
        "recovery": recovery,
        "coverage": coverage,
        "replicates": rows,
    }


def run_mse_workflow(fitted: FitResult, *, simulations: int, years: int, seed: int) -> dict[str, Any]:
    operating = OperatingModelSettings(
        k=float(fitted.best["k_b0"]),
        r=float(fitted.best["r"]),
        initial_depletion=float(fitted.best["terminal_depletion"]),
        process_cv=float(fitted.settings.get("process_cv", 0.12)),
        observation_cv=float(fitted.settings.get("obs_cv", 0.20)),
        implementation_cv=0.10,
    )
    procedures = [
        ManagementProcedure("Conservative 40-10", target_f_fraction=0.70, maximum_catch_change=0.15, pstar=0.40),
        ManagementProcedure("Balanced 40-10", target_f_fraction=1.00, maximum_catch_change=0.20, pstar=0.45),
        ManagementProcedure("Stable catch", target_f_fraction=0.85, maximum_catch_change=0.10, pstar=0.45),
        ManagementProcedure("High yield stress test", target_f_fraction=1.25, maximum_catch_change=0.30, pstar=0.50),
    ]
    result = run_closed_loop_mse(
        operating,
        procedures,
        MSESettings(years=max(5, int(years)), simulations=max(20, int(simulations)), seed=int(seed), initial_catch=float(fitted.best["msy"]) * 0.75),
    )
    result["pareto_front"] = pareto_front(result.get("strategies") or [])
    return result


def _component_rows(fitted: FitResult) -> list[dict[str, Any]]:
    components = fitted.diagnostics.get("objective_components") or {}
    total = max(sum(float(value) for value in components.values()), _EPS)
    return [
        {
            "component": str(name),
            "objective": float(value),
            "share": float(value) / total,
            "delta_nll": float(value),
        }
        for name, value in sorted(components.items(), key=lambda item: float(item[1]), reverse=True)
    ]


def _gradient_at_candidate(dataset: StockDataset, settings: ModelSettings, candidate: Mapping[str, Any]) -> float:
    # Reuse the public optimizer as a stable route to the candidate objective and
    # report its normalized local slope. This is a finite-difference diagnostic,
    # not ADMB's exact maximum gradient.
    output = run_global_optimizer(
        dataset,
        settings,
        QuantOptimizerSettings(algorithm="nelder_mead", population=12, generations=1, seed=int(settings.seed) + 9901, local_refinement_rounds=1),
    )
    local = output.get("diagnostics", {}).get("local_identifiability") or {}
    hessian = np.asarray(local.get("hessian") or [], dtype=object)
    # No gradient is returned by the current optimizer. Objective spread among the
    # best candidates is used only as a conservative proxy.
    candidates = output.get("candidates") or []
    values = [float(row["objective"]) for row in candidates[:5] if np.isfinite(float(row.get("objective", np.nan)))]
    return float(np.ptp(values)) / max(len(values), 1) if values else float("nan")


def run_expert_workflow(
    dataset: StockDataset,
    model_settings: ModelSettings | None = None,
    workflow_settings: ExpertWorkflowSettings | None = None,
    progress: Callable[[str], None] | None = None,
    age_composition: pd.DataFrame | None = None,
    length_composition: pd.DataFrame | None = None,
) -> dict[str, Any]:
    base = model_settings or ModelSettings()
    config = (workflow_settings or ExpertWorkflowSettings()).resolved()
    cache_path: Path | None = None
    if config.cache_directory:
        cache_material = {
            "schema": "omega-expert-workflow-1.4",
            "dataset": dataset.frame.to_csv(index=False),
            "model_settings": asdict(base),
            "workflow_settings": asdict(config),
            "age_composition": age_composition.to_csv(index=False) if age_composition is not None else None,
            "length_composition": length_composition.to_csv(index=False) if length_composition is not None else None,
        }
        digest = hashlib.sha256(json.dumps(cache_material, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        cache_path = Path(config.cache_directory) / f"{digest}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached.setdefault("summary", {})["cache_hit"] = True
            return cached
    results: dict[str, Any] = {}
    steps: list[WorkflowStep] = []

    def emit(message: str) -> None:
        if progress:
            progress(message)

    skip_aliases = {
        "Native automatic-differentiation gradient check": {"Finite-difference gradient check"},
        "Genuine age-structured ASPM and ASPM-R diagnostic": {"ASPM-style catch and index diagnostic"},
        "Formal known-truth interval-coverage testing": {"Simulation-recovery and interval-coverage testing"},
    }

    def execute(name: str, key: str, function: Callable[[], Any], *, required: bool = True) -> Any:
        requested_skips = set(config.skipped_steps)
        aliases = skip_aliases.get(name, set())
        if name in requested_skips or bool(aliases & requested_skips):
            steps.append(WorkflowStep(name, "SKIPPED", required, "Skipped by recorded exploration override.", key))
            return None
        emit(name)
        try:
            value = function()
            results[key] = value
            status = "PASS"
            if isinstance(value, Mapping):
                summary_value = value.get("summary")
                if isinstance(summary_value, Mapping):
                    status = str(summary_value.get("status", "PASS"))
            status_key = status.strip().lower()
            status_map = {
                "completed": "PASS",
                "strong_agreement": "PASS",
                "well_conditioned": "PASS",
                "pass": "PASS",
                "moderate_agreement": "WARN",
                "weakly_conditioned": "WARN",
                "ill_conditioned": "WARN",
                "warn": "WARN",
                "warning": "WARN",
                "weak_agreement": "FAIL",
                "rank_deficient": "FAIL",
                "severely_ill_conditioned": "FAIL",
                "fail": "FAIL",
                "failed": "FAIL",
                "insufficient_data": "NOT TESTED",
                "not_tested": "NOT TESTED",
            }
            normalized_status = status_map.get(status_key, status.upper())
            steps.append(WorkflowStep(name, normalized_status, required, "Completed.", key))
            return value
        except Exception as exc:
            steps.append(WorkflowStep(name, "FAIL", required, "Step failed; failure retained in the evidence record.", key, f"{type(exc).__name__}: {exc}"))
            if required and not config.continue_after_failure:
                raise WorkflowRunError(f"{name} failed: {exc}") from exc
            return None

    base_fit: FitResult = execute(
        "Fit base assessment",
        "base_fit",
        lambda: fit(dataset, replace(base, search_draws=config.search_draws, seed=config.seed)),
    )
    if base_fit is None:
        return {
            "summary": {"status": "FAILED", "reason": "Base fit failed"},
            "steps": [asdict(step) for step in steps],
            "results": results,
            "settings": asdict(config),
        }

    biomass_evidence = execute(
        "Evidence-weighted best-supported biomass synthesis",
        "biomass_evidence",
        lambda: asdict(
            estimate_best_supported_biomass(
                dataset,
                BiomassTruthSettings(
                    search_draws=config.search_draws,
                    samples=180 if config.speed == "quick" else 600 if config.speed == "standard" else 1800,
                    holdout_years=3 if config.speed == "quick" else 5,
                    seed=config.seed + 25,
                ),
            )
        ),
        required=False,
    )

    residuals = execute("Residual diagnostics and heatmap", "residuals", lambda: residual_diagnostics(dataset, base_fit))
    experimental = execute(
        "Experimental diagnostic triangulation",
        "experimental_diagnostics",
        lambda: run_experimental_diagnostics(
            dataset,
            base_fit,
            ExperimentalDiagnosticSettings(
                search_draws=config.search_draws,
                posterior_predictive_replicates=60 if config.speed == "quick" else 300 if config.speed == "standard" else 1000,
                mutual_information_permutations=20 if config.speed == "quick" else 150 if config.speed == "standard" else 500,
                data_clone_factors=(1, 2) if config.speed == "quick" else (1, 2, 4, 8),
                seed=config.seed + 50,
            ),
        ),
        required=False,
    )
    boundaries = execute("Parameter-bound checks", "boundaries", lambda: parameter_boundary_diagnostic(dataset, base_fit))
    gradient = execute("Native automatic-differentiation gradient check", "gradient", lambda: finite_difference_gradient_diagnostic(dataset, base, base_fit))
    jitter = execute(
        "Jitter and multi-start distribution",
        "jitter",
        lambda: run_jitter_diagnostic(dataset, replace(base, search_draws=config.search_draws), runs=config.jitter_runs, seed=config.seed + 100, workers=config.workers),
    )
    optimizers = execute(
        "Multi-optimizer agreement",
        "optimizer_agreement",
        lambda: run_optimizer_agreement(
            dataset,
            replace(base, search_draws=config.search_draws),
            OptimizerAgreementSettings(population=config.optimizer_population, generations=config.optimizer_generations, seed=config.seed + 200),
        ),
    )

    def profile_step() -> dict[str, Any]:
        point_count = 7 if config.speed == "quick" else 13 if config.speed == "standard" else 21
        multistarts = 2 if config.speed == "quick" else 3 if config.speed == "standard" else 5
        outputs: dict[str, Any] = {}
        for position, parameter in enumerate(("k", "r", "initial_depletion", "sigma")):
            outputs[parameter] = profile_likelihood(
                dataset,
                replace(base, search_draws=config.search_draws),
                base_fit,
                parameter,
                ProfileSettings(
                    points=point_count,
                    workers=max(1, min(config.workers, point_count)),
                    multistarts=multistarts,
                    seed=config.seed + 300 + position * 104729,
                    cache_dir=config.cache_directory,
                    use_cache=bool(config.cache_directory),
                ),
            )
        statuses = [str(value.get("summary", {}).get("status", "FAIL")) for value in outputs.values()]
        status = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        failed_points = sum(int(value.get("summary", {}).get("failed_points", 0)) for value in outputs.values())
        nonconverged = sum(int(value.get("summary", {}).get("nonconverged_points", 0)) for value in outputs.values())
        return {
            "summary": {
                "status": status,
                "parameters_profiled": len(outputs),
                "all_other_active_parameters_refitted": True,
                "failed_points": failed_points,
                "nonconverged_points": nonconverged,
                "backend": native_status(),
            },
            "parameters": outputs,
        }

    profiles = execute("Likelihood profiles and local identifiability", "profiles", profile_step)
    component_profiles = execute(
        "Likelihood-component conflict profiles",
        "likelihood_component_profiles",
        lambda: likelihood_component_profiles(dataset, base, base_fit, points=9 if config.speed == "quick" else 15),
    )
    retrospective = execute(
        "Retrospective analysis and Mohn's rho",
        "retrospective",
        lambda: run_retrospective_diagnostic(dataset, base, peels=config.peels, search_draws=config.search_draws, seed=config.seed + 400),
    )
    walk_forward = execute(
        "Hindcast prediction and MASE",
        "hindcast",
        lambda: add_mase_to_walk_forward(
            dataset,
            run_walk_forward_validation(
                dataset,
                replace(base, search_draws=config.search_draws),
                WalkForwardSettings(minimum_training_years=max(5, min(8, len(dataset.frame) // 2)), holdout_years=1, search_draws=config.search_draws, seed=config.seed + 500),
            ),
        ),
    )
    aspm = execute(
        "Genuine age-structured ASPM and ASPM-R diagnostic",
        "aspm",
        lambda: run_age_structured_aspm(
            dataset,
            age_composition=age_composition,
            length_composition=length_composition,
            settings=ASPMSettings(
                multistarts=2 if config.speed == "quick" else 4 if config.speed == "standard" else 6,
                seed=config.seed + 575,
                max_iterations=250 if config.speed == "quick" else 600 if config.speed == "standard" else 1000,
                full_fit_population=12 if config.speed == "quick" else 24 if config.speed == "standard" else 40,
                full_fit_generations=1 if config.speed == "quick" else 6 if config.speed == "standard" else 14,
            ),
        ),
    )
    removal = execute("Data-removal influence analysis", "data_removal", lambda: run_data_removal_influence(dataset, base, base_fit))
    weighting = execute("Data-weighting comparison", "weighting", lambda: run_weighting_comparison(dataset, base))
    composition_weighting = execute(
        "Composition reweighting comparison",
        "composition_weighting",
        lambda: run_composition_reweighting_comparison(
            dataset,
            age_composition,
            length_composition,
            population=12 if config.speed == "quick" else 24 if config.speed == "standard" else 40,
            generations=1 if config.speed == "quick" else 6 if config.speed == "standard" else 14,
            seed=config.seed + 550,
        ),
        required=False,
    )
    conflict = execute(
        "Data-conflict matrix",
        "data_conflict",
        lambda: data_conflict_matrix(
            {
                key: dataset.frame[key].to_numpy(dtype=float)
                for key in ("catch", "index", "biomass")
                if key in dataset.frame and np.isfinite(dataset.frame[key].to_numpy(dtype=float)).sum() >= 4
            }
        ),
    )
    ensemble = execute(
        "Structural model ensemble",
        "ensemble",
        lambda: run_model_ensemble(
            dataset,
            replace(base, search_draws=config.search_draws),
            EnsembleSettings(search_draws=config.search_draws, projection_years=20, projection_iterations=120 if config.speed == "quick" else 300, seed=config.seed + 600),
        ),
    )
    simulation = execute(
        "Formal known-truth interval-coverage testing",
        "simulation_recovery",
        lambda: run_interval_coverage(
            dataset,
            replace(base, search_draws=config.search_draws),
            base_fit,
            CoverageSettings(
                replicates=config.simulation_replicates,
                confidence_levels=(0.50, 0.80, 0.90, 0.95),
                methods=("hessian",) if config.speed in {"quick", "standard"} else ("hessian", "profile"),
                workers=config.workers,
                seed=config.seed + 700,
                search_draws=config.search_draws,
                profile_points=9 if config.speed == "deep" else 7,
                profile_multistarts=2,
                include_time_series=True,
            ),
        ),
    )

    mse = execute(
        "Closed-loop management strategy evaluation",
        "mse",
        lambda: run_mse_workflow(base_fit, simulations=config.mse_simulations, years=config.mse_years, seed=config.seed + 800),
    )

    def advanced_mse_step() -> dict[str, Any]:
        age_base = fit_age_structured(
            dataset,
            AgeStructuredSettings(),
            AgeFitSettings(
                population=12 if config.speed == "quick" else 18 if config.speed == "standard" else 28,
                generations=1 if config.speed == "quick" else 4 if config.speed == "standard" else 8,
                local_rounds=1,
                seed=config.seed + 825,
                estimate_recruitment_sigma=False,
            ),
            age_composition=age_composition,
            length_composition=length_composition,
        )
        procedures = [
            MSEManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.20, fishing_fraction_of_fmsy=0.55),
            MSEManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.75),
            MSEManagementProcedure("Yield focused", target_depletion=0.35, limit_depletion=0.10, fishing_fraction_of_fmsy=0.95),
        ]
        scenarios = default_operating_scenarios()[: 2 if config.speed == "quick" else 4 if config.speed == "standard" else 8]
        return run_advanced_mse(
            age_base,
            procedures,
            scenarios=scenarios,
            observation=MSEObservationSettings(),
            assessment=MSEAssessmentSettings(mode="fast_filter", assessment_interval=3, data_lag_years=1),
            settings=AdvancedMSESettings(
                years=min(config.mse_years, 6 if config.speed == "quick" else 12 if config.speed == "standard" else 20),
                simulations_per_scenario=1 if config.speed == "quick" else 3 if config.speed == "standard" else 10,
                workers=max(1, min(config.workers, 4)),
                sample_trajectories_per_cell=1,
                seed=config.seed + 850,
            ),
        )

    advanced_mse = execute(
        "Separate-truth age-structured MSE",
        "advanced_mse",
        advanced_mse_step,
        required=False,
    )

    profile_summary = (profiles or {}).get("summary") or {}
    hessian_positive = None
    diagnostic_input = {
        "maximum_gradient": ((gradient or {}).get("summary") or {}).get("maximum_gradient"),
        "hessian_positive_definite": hessian_positive,
        "hessian_condition_number": None,
        "mohn_rho": ((retrospective or {}).get("summary") or {}).get("mohn_rho"),
        "holdout_relative_error": ((walk_forward or {}).get("summary") or {}).get("biomass_relative_rmse") or ((walk_forward or {}).get("summary") or {}).get("index_log_rmse"),
        "conflict_score_0_100": (conflict or {}).get("conflict_score_0_100"),
        "near_boundary": ((boundaries or {}).get("summary") or {}).get("near_boundary") or [],
        "optimizer_terminal_depletion_spread": next(
            (row.get("range") for row in ((optimizers or {}).get("agreement") or []) if row.get("quantity") == "terminal_depletion"),
            None,
        ),
    }
    grade = execute("Automatic evidence-based reliability grade", "reliability", lambda: reliability_grade(diagnostic_input))
    results["likelihood_components"] = _component_rows(base_fit)
    convergence_metrics = {
        "maximum_gradient": ((gradient or {}).get("summary") or {}).get("maximum_gradient"),
        "near_boundary_parameters": ((boundaries or {}).get("summary") or {}).get("near_boundary") or [],
        "jitter_terminal_depletion_spread": ((jitter or {}).get("summary") or {}).get("near_best_terminal_depletion_spread"),
        "optimizer_objective_spread": ((optimizers or {}).get("summary") or {}).get("objective_spread"),
        "optimizer_terminal_depletion_cv": ((optimizers or {}).get("summary") or {}).get("terminal_depletion_cv"),
        "profile_failed_points": ((profiles or {}).get("summary") or {}).get("failed_points"),
        "profile_nonconverged_points": ((profiles or {}).get("summary") or {}).get("nonconverged_points"),
    }
    convergence_failures = sum(
        step.status == "FAIL"
        for step in steps
        if step.result_key in {"gradient", "boundaries", "jitter", "optimizer_agreement", "profiles"}
    )
    results["convergence_dashboard"] = {
        "summary": {
            "status": "PASS" if convergence_failures == 0 else "WARN" if convergence_failures == 1 else "FAIL",
            "failed_convergence_gates": convergence_failures,
            **convergence_metrics,
        },
        "metrics": [{"metric": key, "value": value} for key, value in convergence_metrics.items()],
    }

    failed_required = sum(step.required and step.status == "FAIL" for step in steps)
    required_untested = sum(step.required and step.status == "NOT TESTED" for step in steps)
    warned = sum(step.status == "WARN" for step in steps)
    skipped = sum(step.status == "SKIPPED" for step in steps)
    untested = sum(step.status == "NOT TESTED" for step in steps)
    complete = sum(step.status in {"PASS", "WARN"} for step in steps)
    overall = "READY_FOR_EXPERT_REVIEW"
    if failed_required >= 3:
        overall = "NOT_RELIABLE"
    elif failed_required:
        overall = "CONDITIONAL"
    elif required_untested or (skipped and config.mode == "automatic"):
        overall = "INCOMPLETE"
    elif warned:
        overall = "READY_WITH_WARNINGS"

    override_rows = [asdict(value) for value in config.overrides]
    if config.skipped_steps:
        override_rows.extend({"setting": "skip_step", "value": value, "reason": "Exploration-mode skip", "author": "analyst"} for value in config.skipped_steps)

    summary = {
        "status": overall,
        "mode": config.mode,
        "speed": config.speed,
        "steps": len(steps),
        "steps_completed": complete,
        "required_failures": failed_required,
        "required_untested": required_untested,
        "warnings": warned,
        "skipped": skipped,
        "untested": untested,
        "reliability_grade": (grade or {}).get("grade"),
        "terminal_depletion": float(base_fit.best["terminal_depletion"]),
        "objective": float(base_fit.best["objective"]),
        "exploration_policy": (
            "Alternative ideas and overrides are allowed. Automatic checks are not hidden; skipped or failed checks remain visible in the evidence record."
        ),
        "scientific_scope": (
            "This workflow automates implemented Omega diagnostics. It does not replace independent review, complete SS3 numerical parity, or stock-specific data validation."
        ),
    }
    payload = {
        "summary": {**summary, "cache_hit": False},
        "settings": asdict(config),
        "overrides": override_rows,
        "steps": [asdict(step) for step in steps],
        "base": {
            "best": dict(base_fit.best),
            "settings": dict(base_fit.settings),
            "diagnostics": dict(base_fit.diagnostics),
            "history": list(base_fit.history),
            "ensemble": list(base_fit.ensemble),
        },
        "results": {key: value for key, value in results.items() if key != "base_fit"},
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


__all__ = [
    "WorkflowOverride",
    "ExpertWorkflowSettings",
    "ResolvedWorkflowSettings",
    "WorkflowStep",
    "mean_absolute_scaled_error",
    "run_jitter_diagnostic",
    "parameter_boundary_diagnostic",
    "finite_difference_gradient_diagnostic",
    "likelihood_component_profiles",
    "run_composition_reweighting_comparison",
    "residual_diagnostics",
    "run_retrospective_diagnostic",
    "add_mase_to_walk_forward",
    "run_aspm_diagnostic",
    "run_data_removal_influence",
    "run_weighting_comparison",
    "run_simulation_recovery",
    "run_mse_workflow",
    "run_expert_workflow",
]
