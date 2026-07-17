from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .age_structured import (
    AgeFitSettings,
    AgeStructuredResult,
    AgeStructuredSettings,
    _settings_from_result,
    fit_age_structured,
    simulate_age_structured,
)
from .data_io import StockDataset

_EPS = 1e-12


@dataclass(frozen=True)
class ASPMSettings:
    multistarts: int = 4
    seed: int = 28111
    max_iterations: int = 800
    estimate_recruitment_deviations: bool = True
    recruitment_penalty_sigma: float | None = None
    include_direct_biomass: bool = True
    run_no_index: bool = True
    run_index_influence: bool = True
    full_fit_population: int = 24
    full_fit_generations: int = 10


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def _copy_dataset(dataset: StockDataset, frame: pd.DataFrame, name: str) -> StockDataset:
    return StockDataset(
        name=name,
        frame=frame.reset_index(drop=True),
        provenance=dict(dataset.provenance),
        transformations=list(dataset.transformations),
        warnings=list(dataset.warnings),
        raw_columns=list(dataset.raw_columns),
        index_columns=[column for column in frame.columns if column == "index" or str(column).startswith("index_")],
    )


def _lognormal_profiled_nll(observed: np.ndarray, predicted: np.ndarray, sigma: float) -> tuple[float, float, np.ndarray]:
    mask = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0) & (predicted > 0)
    residuals = np.full(len(observed), np.nan, dtype=float)
    if not mask.any():
        return 0.0, float("nan"), residuals
    log_q = float(np.mean(np.log(observed[mask]) - np.log(predicted[mask])))
    q = math.exp(log_q)
    residuals[mask] = np.log(observed[mask]) - np.log(q * predicted[mask])
    sd = max(float(sigma), 0.03)
    nll = float(np.sum(0.5 * np.square(residuals[mask] / sd) + math.log(sd) + 0.5 * math.log(2.0 * math.pi)))
    return nll, q, residuals


def _aspm_objective(
    theta: np.ndarray,
    dataset: StockDataset,
    base: AgeStructuredSettings,
    *,
    allow_recruitment_deviations: bool,
    include_index: bool,
    include_direct_biomass: bool,
    recruitment_penalty_sigma: float,
) -> tuple[float, dict[str, Any]]:
    frame = dataset.frame.reset_index(drop=True)
    r0 = math.exp(float(theta[0]))
    initial_depletion = _sigmoid(float(theta[1]))
    candidate = replace(base, r0=r0, initial_depletion=initial_depletion)

    if allow_recruitment_deviations:
        raw_deviations = np.asarray(theta[2:], dtype=float)
        # The weak sum-to-zero convention separates average recruitment from R0.
        deviations = raw_deviations - float(np.mean(raw_deviations)) if len(raw_deviations) else np.zeros(len(frame), dtype=float)
        multipliers = np.exp(np.clip(deviations - 0.5 * recruitment_penalty_sigma**2, -8.0, 8.0))
    else:
        deviations = np.zeros(len(frame), dtype=float)
        multipliers = np.ones(len(frame), dtype=float)

    simulation = simulate_age_structured(dataset, candidate, recruitment_multipliers=multipliers)
    history = simulation["history"]
    predicted_index = np.asarray([row["survey_biomass"] for row in history], dtype=float)
    predicted_biomass = np.asarray([row["total_biomass"] for row in history], dtype=float)
    observed_index = frame["index"].to_numpy(dtype=float) if include_index else np.full(len(frame), np.nan)
    observed_biomass = frame["biomass"].to_numpy(dtype=float) if include_direct_biomass else np.full(len(frame), np.nan)

    index_nll, q_index, index_residuals = _lognormal_profiled_nll(observed_index, predicted_index, candidate.index_cv)
    biomass_nll, q_biomass, biomass_residuals = _lognormal_profiled_nll(observed_biomass, predicted_biomass, candidate.biomass_cv)
    r0_prior_sd = 1.5
    r0_prior = 0.5 * ((math.log(r0) - math.log(max(base.r0, _EPS))) / r0_prior_sd) ** 2
    depletion_prior = 0.5 * ((initial_depletion - base.initial_depletion_prior) / max(base.initial_depletion_prior_sd, 0.05)) ** 2
    recruitment_penalty = (
        0.5 * float(np.sum(np.square(deviations / max(recruitment_penalty_sigma, 0.05))))
        if allow_recruitment_deviations
        else 0.0
    )
    catch_penalty = 0.5 * (
        simulation["catch_mismatch_total"] / max(float(frame["catch"].sum()), 1.0) / 0.001
    ) ** 2
    components = {
        "index_likelihood": float(index_nll),
        "biomass_likelihood": float(biomass_nll),
        "recruitment_deviation_penalty": float(recruitment_penalty),
        "r0_prior": float(r0_prior),
        "initial_depletion_prior": float(depletion_prior),
        "catch_reconstruction_penalty": float(catch_penalty),
    }
    objective = float(sum(components.values()))
    return objective, {
        "settings": candidate,
        "simulation": simulation,
        "components": components,
        "q_index": q_index,
        "q_biomass": q_biomass,
        "index_residuals": index_residuals,
        "biomass_residuals": biomass_residuals,
        "recruitment_deviations": deviations,
        "recruitment_multipliers": multipliers,
    }


def _fit_variant(
    dataset: StockDataset,
    base: AgeStructuredSettings,
    *,
    name: str,
    allow_recruitment_deviations: bool,
    include_index: bool,
    include_direct_biomass: bool,
    config: ASPMSettings,
    seed_offset: int,
) -> dict[str, Any]:
    n_years = len(dataset.frame)
    sigma = float(config.recruitment_penalty_sigma or base.recruitment_sigma)
    center = np.asarray([math.log(base.r0), _logit(base.initial_depletion)], dtype=float)
    if allow_recruitment_deviations:
        center = np.concatenate([center, np.zeros(n_years, dtype=float)])
    bounds = [(math.log(max(base.r0 / 100.0, 1.0)), math.log(base.r0 * 100.0)), (-7.0, 7.0)]
    if allow_recruitment_deviations:
        bounds.extend([(-4.0, 4.0)] * n_years)

    rng = np.random.default_rng(int(config.seed) + int(seed_offset))
    starts = [center]
    for _ in range(max(0, int(config.multistarts) - 1)):
        start = center.copy()
        start[0] += rng.normal(0.0, 0.7)
        start[1] += rng.normal(0.0, 0.8)
        if allow_recruitment_deviations:
            start[2:] = rng.normal(0.0, min(sigma, 0.8), n_years)
        for index, (low, high) in enumerate(bounds):
            start[index] = np.clip(start[index], low, high)
        starts.append(start)

    attempts: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, dict[str, Any], Any] | None = None
    for start_number, start in enumerate(starts, 1):
        try:
            result = minimize(
                lambda values: _aspm_objective(
                    values,
                    dataset,
                    base,
                    allow_recruitment_deviations=allow_recruitment_deviations,
                    include_index=include_index,
                    include_direct_biomass=include_direct_biomass,
                    recruitment_penalty_sigma=sigma,
                )[0],
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max(100, int(config.max_iterations)), "ftol": 1e-11, "gtol": 1e-6, "maxls": 50},
            )
            objective, details = _aspm_objective(
                np.asarray(result.x, dtype=float),
                dataset,
                base,
                allow_recruitment_deviations=allow_recruitment_deviations,
                include_index=include_index,
                include_direct_biomass=include_direct_biomass,
                recruitment_penalty_sigma=sigma,
            )
            attempt = {
                "start": start_number,
                "objective": float(objective),
                "success": bool(result.success and np.isfinite(objective)),
                "optimizer_success": bool(result.success),
                "iterations": int(getattr(result, "nit", 0)),
                "evaluations": int(getattr(result, "nfev", 0)),
                "message": str(result.message),
            }
            if best is None or objective < best[0]:
                best = (float(objective), np.asarray(result.x, dtype=float), details, result)
        except Exception as exc:
            attempt = {
                "start": start_number,
                "objective": float("inf"),
                "success": False,
                "optimizer_success": False,
                "iterations": 0,
                "evaluations": 0,
                "message": f"{type(exc).__name__}: {exc}",
            }
        attempts.append(attempt)

    if best is None:
        return {
            "name": name,
            "status": "FAIL",
            "success": False,
            "message": "All ASPM optimisation attempts failed.",
            "attempts": attempts,
            "history": [],
            "components": {},
        }

    objective, theta, details, result = best
    candidate: AgeStructuredSettings = details["settings"]
    history = details["simulation"]["history"]
    residual_rows = []
    for row, index_residual, biomass_residual in zip(history, details["index_residuals"], details["biomass_residuals"]):
        residual_rows.append(
            {
                "year": int(row["year"]),
                "index_log_residual": float(index_residual) if np.isfinite(index_residual) else float("nan"),
                "biomass_log_residual": float(biomass_residual) if np.isfinite(biomass_residual) else float("nan"),
            }
        )
    identifiability = "identified by catch and abundance information" if include_index or include_direct_biomass else "not identified by catch alone; shown as an assumption-driven stress test"
    return {
        "name": name,
        "status": "PASS" if bool(result.success) else "WARN",
        "success": bool(result.success),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "objective": float(objective),
        "r0": float(candidate.r0),
        "initial_depletion": float(candidate.initial_depletion),
        "terminal_biomass": float(history[-1]["total_biomass"]),
        "terminal_spawning_biomass": float(history[-1]["spawning_biomass"]),
        "terminal_depletion": float(history[-1]["depletion"]),
        "terminal_f": float(history[-1]["f_scalar"]),
        "q_index": details["q_index"],
        "q_biomass": details["q_biomass"],
        "components": details["components"],
        "history": history,
        "residuals": residual_rows,
        "recruitment_deviations": np.asarray(details["recruitment_deviations"], dtype=float).tolist(),
        "recruitment_multipliers": np.asarray(details["recruitment_multipliers"], dtype=float).tolist(),
        "attempts": attempts,
        "multistart_objective_spread": float(np.ptp([row["objective"] for row in attempts if np.isfinite(row["objective"])])) if any(np.isfinite(row["objective"]) for row in attempts) else float("nan"),
        "biology_fixed_from_full_model": True,
        "selectivity_fixed_from_full_model": True,
        "composition_likelihoods_removed": True,
        "recruitment_deviations_estimated": allow_recruitment_deviations,
        "include_index": include_index,
        "include_direct_biomass": include_direct_biomass,
        "identifiability": identifiability,
    }


def run_age_structured_aspm(
    dataset: StockDataset,
    *,
    full_result: AgeStructuredResult | None = None,
    base_settings: AgeStructuredSettings | None = None,
    age_composition: pd.DataFrame | None = None,
    length_composition: pd.DataFrame | None = None,
    settings: ASPMSettings | None = None,
) -> dict[str, Any]:
    """Run a genuine age-structured production-model diagnostic.

    The ASPM variants retain the full model's ages, natural mortality, growth,
    maturity, weight-at-age, fleet selectivity, retention and catch history while
    removing age/length composition likelihoods. This distinguishes the routine
    from a surplus-production approximation.
    """

    config = settings or ASPMSettings()
    if full_result is None:
        full_result = fit_age_structured(
            dataset,
            base_settings or AgeStructuredSettings(),
            AgeFitSettings(
                population=max(12, int(config.full_fit_population)),
                generations=max(1, int(config.full_fit_generations)),
                seed=int(config.seed),
                local_rounds=2,
                estimate_recruitment_sigma=False,
            ),
            age_composition=age_composition,
            length_composition=length_composition,
        )
    full_settings = _settings_from_result(full_result)

    variants: list[dict[str, Any]] = []
    variants.append(
        _fit_variant(
            dataset,
            full_settings,
            name="ASPM",
            allow_recruitment_deviations=False,
            include_index=True,
            include_direct_biomass=bool(config.include_direct_biomass),
            config=config,
            seed_offset=101,
        )
    )
    if config.estimate_recruitment_deviations:
        variants.append(
            _fit_variant(
                dataset,
                full_settings,
                name="ASPM-R",
                allow_recruitment_deviations=True,
                include_index=True,
                include_direct_biomass=bool(config.include_direct_biomass),
                config=config,
                seed_offset=211,
            )
        )
    if config.run_no_index:
        variants.append(
            _fit_variant(
                dataset,
                full_settings,
                name="ASPM no-index",
                allow_recruitment_deviations=False,
                include_index=False,
                include_direct_biomass=False,
                config=config,
                seed_offset=307,
            )
        )

    index_influence: list[dict[str, Any]] = []
    if config.run_index_influence:
        index_columns = list(dataset.index_columns or ["index"])
        for position, column in enumerate(index_columns):
            if column not in dataset.frame:
                continue
            frame = dataset.frame.copy()
            frame["index"] = pd.to_numeric(frame[column], errors="coerce")
            other_columns = [candidate for candidate in index_columns if candidate != column and candidate in frame]
            for other in other_columns:
                frame[other] = np.nan
            index_dataset = _copy_dataset(dataset, frame, f"{dataset.name} — {column}")
            row = _fit_variant(
                index_dataset,
                full_settings,
                name=f"Index-only influence: {column}",
                allow_recruitment_deviations=False,
                include_index=True,
                include_direct_biomass=False,
                config=replace(config, multistarts=max(2, min(config.multistarts, 3))),
                seed_offset=401 + position * 17,
            )
            row["index_column"] = column
            index_influence.append(row)

    full_history = full_result.history
    full_terminal = float(full_result.best["terminal_depletion"])
    comparison = []
    for variant in variants + index_influence:
        if not variant.get("history"):
            continue
        delta = float(variant["terminal_depletion"] - full_terminal)
        full_series = np.asarray([row["depletion"] for row in full_history], dtype=float)
        variant_series = np.asarray([row["depletion"] for row in variant["history"]], dtype=float)
        n = min(len(full_series), len(variant_series))
        rmse = float(np.sqrt(np.mean(np.square(variant_series[:n] - full_series[:n])))) if n else float("nan")
        comparison.append(
            {
                "variant": variant["name"],
                "full_terminal_depletion": full_terminal,
                "variant_terminal_depletion": float(variant["terminal_depletion"]),
                "terminal_difference": delta,
                "trajectory_rmse": rmse,
                "objective": float(variant.get("objective", float("nan"))),
                "status": "PASS" if abs(delta) <= 0.05 and rmse <= 0.08 else "WARN" if abs(delta) <= 0.15 and rmse <= 0.20 else "FAIL",
            }
        )

    informative = [row for row in comparison if row["variant"] != "ASPM no-index"]
    maximum_difference = max((abs(float(row["terminal_difference"])) for row in informative), default=float("nan"))
    status = "PASS" if np.isfinite(maximum_difference) and maximum_difference <= 0.05 else "WARN" if np.isfinite(maximum_difference) and maximum_difference <= 0.15 else "FAIL"
    return {
        "summary": {
            "status": status,
            "full_terminal_depletion": full_terminal,
            "maximum_informative_terminal_difference": maximum_difference,
            "full_model_objective": float(full_result.best["objective"]),
            "age_structured": True,
            "catch_history_retained": True,
            "biology_and_selectivity_fixed_from_full_model": True,
            "composition_likelihoods_removed": True,
            "variants": len(variants),
            "index_influence_runs": len(index_influence),
        },
        "full_model": {
            "best": dict(full_result.best),
            "history": full_history,
            "objective_components": full_result.diagnostics.get("objective_components") or {},
        },
        "variants": variants,
        "index_influence": index_influence,
        "comparison": comparison,
        "configuration": asdict(config),
        "interpretation": (
            "A large full-model versus ASPM difference means the detailed integrated trajectory depends materially on composition information, "
            "recruitment deviations or other structural data. It is a driver diagnostic, not automatic proof that either trajectory is correct."
        ),
    }
