from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from math import exp, log, sqrt
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .age_structured import (
    AgeFitSettings,
    AgeStructuredResult,
    AgeStructuredSettings,
    SectorSettings,
    _advance_numbers,
    _annual_catch,
    _beverton_holt,
    _settings_from_result,
    _solve_f_for_catch,
    _spawning_biomass,
    _survey_biomass,
    _total_biomass,
    _unfished_numbers,
    equilibrium_reference_points,
    fit_age_structured,
    life_history_arrays,
    sector_curves,
)
from .biomass_truth_engine import BiomassTruthSettings, estimate_best_supported_biomass
from .data_io import StockDataset, normalise_frame

_EPS = 1e-12


@dataclass(frozen=True)
class MSEOperatingScenario:
    name: str = "base"
    weight: float = 1.0
    natural_mortality_multiplier: float = 1.0
    steepness_shift: float = 0.0
    recruitment_sigma_multiplier: float = 1.0
    recruitment_mean_multiplier: float = 1.0
    recruitment_regime_probability: float = 0.0
    poor_regime_multiplier: float = 0.60
    catchability_drift_per_year: float = 0.0
    hyperstability_beta: float = 1.0
    discard_mortality_multiplier: float = 1.0
    selectivity_age_shift: float = 0.0


@dataclass(frozen=True)
class MSEObservationSettings:
    index_cv: float = 0.20
    catch_reporting_cv: float = 0.08
    catch_reporting_bias: float = 0.0
    direct_biomass_cv: float = 0.25
    direct_biomass_interval: int = 0
    survey_interval: int = 1
    age_composition_interval: int = 3
    age_sample_size: int = 150
    missing_observation_probability: float = 0.0
    ageing_sd: float = 0.5


@dataclass(frozen=True)
class MSEAssessmentSettings:
    mode: str = "fast_filter"  # fast_filter, biomass_ensemble, full_age_structured
    assessment_interval: int = 3
    data_lag_years: int = 1
    minimum_years: int = 6
    assumed_index_beta: float = 1.0
    initial_biomass_cv: float = 0.30
    filter_process_cv: float = 0.16
    filter_observation_cv: float = 0.25
    ensemble_search_draws: int = 180
    ensemble_samples: int = 300
    age_fit_population: int = 14
    age_fit_generations: int = 5
    assessment_failure_bias: float = 0.0


@dataclass(frozen=True)
class MSEManagementProcedure:
    name: str
    rule: str = "hcr"  # hcr, fixed_f, fixed_catch
    target_depletion: float = 0.40
    limit_depletion: float = 0.10
    fishing_fraction_of_fmsy: float = 0.80
    fixed_f: float = 0.05
    fixed_catch: float = 0.0
    pstar: float = 0.45
    maximum_catch_change: float = 0.20
    minimum_catch: float = 0.0
    maximum_catch: float = float("inf")
    closure_below_limit: bool = True
    sector_allocations: tuple[tuple[str, float], ...] = (
        ("commercial", 0.50),
        ("charter", 0.15),
        ("recreational", 0.35),
    )
    seasonal_closure_fraction: float = 0.0
    spatial_closure_fraction: float = 0.0
    bag_limit_effort_multiplier: float = 1.0
    effort_limit_multiplier: float = 1.0
    implementation_cv: float = 0.10
    compliance_fraction: float = 0.95


@dataclass(frozen=True)
class AdvancedMSESettings:
    years: int = 30
    simulations_per_scenario: int = 250
    seed: int = 91821
    workers: int = 1
    sample_trajectories_per_cell: int = 2
    economic_price_per_tonne: float = 1.0
    economic_cost_per_f: float = 0.0
    risk_aversion: float = 2.0
    minimum_probability_above_limit: float = 0.90


def default_operating_scenarios() -> tuple[MSEOperatingScenario, ...]:
    return (
        MSEOperatingScenario("base", 0.25),
        MSEOperatingScenario("low_M", 0.10, natural_mortality_multiplier=0.75),
        MSEOperatingScenario("high_M", 0.10, natural_mortality_multiplier=1.30),
        MSEOperatingScenario("low_recruitment", 0.15, recruitment_mean_multiplier=0.65, recruitment_sigma_multiplier=1.20),
        MSEOperatingScenario("regime_shift", 0.10, recruitment_regime_probability=0.08, poor_regime_multiplier=0.45),
        MSEOperatingScenario("hyperstable_CPUE", 0.10, hyperstability_beta=0.55, catchability_drift_per_year=0.01),
        MSEOperatingScenario("high_release_mortality", 0.10, discard_mortality_multiplier=1.50),
        MSEOperatingScenario("selectivity_shift", 0.10, selectivity_age_shift=-1.0),
    )


def generate_management_grid() -> list[MSEManagementProcedure]:
    procedures: list[MSEManagementProcedure] = []
    for target in (0.35, 0.40, 0.45, 0.50):
        for limit in (0.10, 0.15, 0.20):
            if limit >= target:
                continue
            for fraction in (0.50, 0.70, 0.90, 1.00):
                procedures.append(
                    MSEManagementProcedure(
                        name=f"HCR_T{target:.2f}_L{limit:.2f}_F{fraction:.2f}",
                        target_depletion=target,
                        limit_depletion=limit,
                        fishing_fraction_of_fmsy=fraction,
                    )
                )
    procedures.extend(
        [
            MSEManagementProcedure("Conservative closure", target_depletion=0.45, limit_depletion=0.20, fishing_fraction_of_fmsy=0.55, maximum_catch_change=0.10),
            MSEManagementProcedure("Seasonal closure", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, seasonal_closure_fraction=0.25),
            MSEManagementProcedure("Recreational effort reduction", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, bag_limit_effort_multiplier=0.70),
        ]
    )
    return procedures


def _scenario_settings(base: AgeStructuredSettings, scenario: MSEOperatingScenario) -> AgeStructuredSettings:
    sectors: list[SectorSettings] = []
    for sector in base.sectors:
        sectors.append(
            replace(
                sector,
                selectivity_a50=sector.selectivity_a50 + scenario.selectivity_age_shift,
                discard_mortality=float(np.clip(sector.discard_mortality * scenario.discard_mortality_multiplier, 0.0, 1.0)),
            )
        )
    return replace(
        base,
        natural_mortality=max(base.natural_mortality * scenario.natural_mortality_multiplier, 1e-4),
        steepness=float(np.clip(base.steepness + scenario.steepness_shift, 0.2001, 0.999)),
        recruitment_sigma=max(base.recruitment_sigma * scenario.recruitment_sigma_multiplier, 0.0),
        sectors=tuple(sectors),
    )


def _normal_lognormal_multiplier(rng: np.random.Generator, cv: float, size: int | None = None) -> np.ndarray | float:
    sigma = sqrt(log(1.0 + max(float(cv), 0.0) ** 2))
    if sigma <= 0:
        return np.ones(size, dtype=float) if size is not None else 1.0
    return rng.lognormal(-0.5 * sigma**2, sigma, size=size)


def _allocation(procedure: MSEManagementProcedure, sectors: Sequence[SectorSettings]) -> dict[str, float]:
    supplied = {str(name): max(float(value), 0.0) for name, value in procedure.sector_allocations}
    values = np.array([supplied.get(sector.name, max(sector.catch_share, 0.0)) for sector in sectors], dtype=float)
    if float(values.sum()) <= 0:
        values[:] = 1.0
    values /= values.sum()
    return {sector.name: float(value) for sector, value in zip(sectors, values)}


def _management_action(
    estimate: Mapping[str, float],
    previous_target_catch: float,
    procedure: MSEManagementProcedure,
    fmsy: float,
    estimated_biomass: float,
) -> dict[str, float]:
    depletion = max(float(estimate.get("estimated_depletion", 0.0)), 0.0)
    if procedure.closure_below_limit and depletion <= procedure.limit_depletion:
        target_f = 0.0
        raw_catch = 0.0
    elif procedure.rule == "fixed_catch":
        target_f = float("nan")
        raw_catch = max(procedure.fixed_catch, 0.0)
    elif procedure.rule == "fixed_f":
        target_f = max(procedure.fixed_f, 0.0)
        raw_catch = target_f * max(estimated_biomass, 0.0)
    else:
        ramp = float(
            np.clip(
                (depletion - procedure.limit_depletion) / max(procedure.target_depletion - procedure.limit_depletion, _EPS),
                0.0,
                1.0,
            )
        )
        target_f = max(fmsy, 0.0) * max(procedure.fishing_fraction_of_fmsy, 0.0) * ramp * max(procedure.pstar, 0.0) / 0.5
        raw_catch = target_f * max(estimated_biomass, 0.0)
    management_scalar = (
        max(0.0, 1.0 - procedure.seasonal_closure_fraction)
        * max(0.0, 1.0 - procedure.spatial_closure_fraction)
        * max(procedure.bag_limit_effort_multiplier, 0.0)
        * max(procedure.effort_limit_multiplier, 0.0)
    )
    raw_catch *= management_scalar
    if previous_target_catch > 0:
        low = previous_target_catch * (1.0 - max(procedure.maximum_catch_change, 0.0))
        high = previous_target_catch * (1.0 + max(procedure.maximum_catch_change, 0.0))
        raw_catch = float(np.clip(raw_catch, low, high))
    raw_catch = float(np.clip(raw_catch, procedure.minimum_catch, procedure.maximum_catch))
    return {"target_catch": raw_catch, "target_f": float(target_f), "management_scalar": float(management_scalar)}


def _fast_filter_assessment(
    records: list[dict[str, float]],
    previous: Mapping[str, float],
    b0: float,
    settings: MSEAssessmentSettings,
) -> dict[str, float]:
    latest = records[-1]
    previous_biomass = max(float(previous.get("estimated_biomass", b0 * 0.5)), _EPS)
    previous_catch = max(float(latest.get("catch", 0.0)), 0.0)
    projected = max(b0 * 1e-6, previous_biomass * (1.0 + 0.08 * (1.0 - previous_biomass / max(b0, _EPS))) - previous_catch)
    index_values = np.array([row.get("index", np.nan) for row in records], dtype=float)
    valid = np.isfinite(index_values) & (index_values > 0)
    if valid.sum() >= 2:
        first_index = float(index_values[np.where(valid)[0][0]])
        first_biomass = max(float(previous.get("initial_estimated_biomass", previous_biomass)), _EPS)
        beta = max(float(settings.assumed_index_beta), 0.05)
        q = first_index / max(first_biomass**beta, _EPS)
        observed = max(float(index_values[np.where(valid)[0][-1]]), _EPS)
        observation_estimate = (observed / max(q, _EPS)) ** (1.0 / beta)
        process_var = max(settings.filter_process_cv, 0.02) ** 2
        observation_var = max(settings.filter_observation_cv, 0.02) ** 2
        log_estimate = (
            log(projected) / process_var + log(max(observation_estimate, _EPS)) / observation_var
        ) / (1.0 / process_var + 1.0 / observation_var)
        estimated_biomass = exp(log_estimate)
        uncertainty_cv = sqrt(1.0 / (1.0 / process_var + 1.0 / observation_var))
    else:
        estimated_biomass = projected
        uncertainty_cv = max(settings.filter_process_cv, 0.10)
    bias = 1.0 + settings.assessment_failure_bias
    estimated_biomass = max(estimated_biomass * bias, b0 * 1e-6)
    initial_biomass = float(previous.get("initial_estimated_biomass", estimated_biomass))
    initial_depletion = float(previous.get("initial_estimated_depletion", previous.get("estimated_depletion", 0.5)))
    estimated_depletion = initial_depletion * estimated_biomass / max(initial_biomass, _EPS)
    return {
        "estimated_biomass": float(estimated_biomass),
        "estimated_depletion": float(max(estimated_depletion, 0.0)),
        "estimated_cv": float(uncertainty_cv),
        "initial_estimated_biomass": initial_biomass,
        "initial_estimated_depletion": initial_depletion,
        "assessment_mode": "fast_filter",
        "assessment_success": 1.0,
    }


def _records_dataset(records: Sequence[Mapping[str, float]], name: str) -> StockDataset:
    frame = pd.DataFrame(records)
    for column in ("year", "catch", "index", "biomass"):
        if column not in frame:
            frame[column] = np.nan
    return normalise_frame(frame[["year", "catch", "index", "biomass"]], name=name)


def _assessment(
    records: list[dict[str, float]],
    age_rows: list[dict[str, float]],
    previous: Mapping[str, float],
    b0: float,
    base_result: AgeStructuredResult,
    settings: MSEAssessmentSettings,
    seed: int,
) -> dict[str, float]:
    lag = max(int(settings.data_lag_years), 0)
    available = records[:-lag] if lag > 0 and len(records) > lag else records
    if len(available) < max(int(settings.minimum_years), 5):
        return _fast_filter_assessment(records, previous, b0, settings)
    try:
        if settings.mode == "biomass_ensemble":
            dataset = _records_dataset(available, "MSE assessment history")
            result = estimate_best_supported_biomass(
                dataset,
                BiomassTruthSettings(
                    search_draws=settings.ensemble_search_draws,
                    samples=settings.ensemble_samples,
                    holdout_years=min(3, max(len(available) // 5, 0)),
                    seed=seed,
                ),
            )
            estimated_biomass = float(result.summary["terminal_biomass_median"])
            initial_biomass = float(previous.get("initial_estimated_biomass", result.trajectory[0]["biomass_median"]))
            initial_depletion = float(previous.get("initial_estimated_depletion", previous.get("estimated_depletion", result.trajectory[0]["depletion_median"])))
            return {
                "estimated_biomass": estimated_biomass,
                "estimated_depletion": float(max(initial_depletion * estimated_biomass / max(initial_biomass, _EPS), 0.0)),
                "estimated_cv": float(
                    (result.summary["terminal_biomass_p90"] - result.summary["terminal_biomass_p10"])
                    / max(2.56 * result.summary["terminal_biomass_median"], _EPS)
                ),
                "initial_estimated_biomass": initial_biomass,
                "initial_estimated_depletion": initial_depletion,
                "assessment_mode": "biomass_ensemble",
                "assessment_success": 1.0,
            }
        if settings.mode == "full_age_structured":
            dataset = _records_dataset(available, "MSE full age assessment")
            age_frame = pd.DataFrame(age_rows)
            if not age_frame.empty:
                age_frame = age_frame[age_frame["year"] <= int(available[-1]["year"])].copy()
            base_settings = _settings_from_result(base_result)
            fitted = fit_age_structured(
                dataset,
                base_settings,
                AgeFitSettings(
                    population=settings.age_fit_population,
                    generations=settings.age_fit_generations,
                    seed=seed,
                    local_rounds=2,
                    estimate_recruitment_sigma=False,
                ),
                age_composition=age_frame if not age_frame.empty else None,
            )
            return {
                "estimated_biomass": float(fitted.best["terminal_biomass"]),
                "estimated_depletion": float(fitted.best["terminal_depletion"]),
                "estimated_cv": float(np.std([row["terminal_depletion"] for row in fitted.ensemble]) / max(fitted.best["terminal_depletion"], _EPS)) if fitted.ensemble else 0.30,
                "initial_estimated_biomass": float(previous.get("initial_estimated_biomass", fitted.history[0]["total_biomass"])),
                "initial_estimated_depletion": float(previous.get("initial_estimated_depletion", fitted.history[0]["depletion"])),
                "assessment_mode": "full_age_structured",
                "assessment_success": 1.0,
            }
    except Exception:
        fallback = _fast_filter_assessment(records, previous, b0, settings)
        fallback["assessment_success"] = 0.0
        fallback["assessment_mode"] = f"{settings.mode}_FAILED_FALLBACK"
        return fallback
    return _fast_filter_assessment(records, previous, b0, settings)


def _single_simulation(
    base_result: AgeStructuredResult,
    procedure: MSEManagementProcedure,
    scenario: MSEOperatingScenario,
    observation: MSEObservationSettings,
    assessment_settings: MSEAssessmentSettings,
    settings: AdvancedMSESettings,
    simulation_index: int,
) -> dict[str, Any]:
    stable = int.from_bytes(hashlib.blake2b(f"{procedure.name}|{scenario.name}".encode("utf-8"), digest_size=8).digest(), "little") % 100000
    rng = np.random.default_rng(settings.seed + simulation_index * 10007 + stable)
    base_model_settings = _settings_from_result(base_result)
    truth_settings = _scenario_settings(base_model_settings, scenario)
    life = life_history_arrays(truth_settings)
    curves = sector_curves(truth_settings, life)
    b0 = _spawning_biomass(_unfished_numbers(truth_settings), life, truth_settings)
    numbers = np.asarray(base_result.state["final_numbers"], dtype=float).copy()
    # Preserve initial depletion while allowing scenario-specific life history.
    current_ssb = _spawning_biomass(numbers, life, truth_settings)
    references = equilibrium_reference_points(truth_settings)
    allocation = _allocation(procedure, truth_settings.sectors)
    q0 = 1.0 / max(_survey_biomass(numbers, life) ** max(scenario.hyperstability_beta, 0.05), _EPS)
    records: list[dict[str, float]] = []
    age_rows: list[dict[str, float]] = []
    estimate: dict[str, float] = {
        "estimated_biomass": float(_total_biomass(numbers, life) * _normal_lognormal_multiplier(rng, assessment_settings.initial_biomass_cv)),
        "estimated_depletion": float(current_ssb / max(b0, _EPS)),
        "estimated_cv": assessment_settings.initial_biomass_cv,
        "assessment_mode": "initial",
        "assessment_success": 1.0,
    }
    estimate["initial_estimated_biomass"] = estimate["estimated_biomass"]
    estimate["initial_estimated_depletion"] = estimate["estimated_depletion"]
    previous_target_catch = max(float(base_result.history[-1].get("observed_catch", 0.0)), 0.0)
    previous_implemented_catch = previous_target_catch
    rec_dev = 0.0
    poor_regime = False
    trajectories: list[dict[str, float]] = []
    catches: list[float] = []
    depletions: list[float] = []
    estimated_depletions: list[float] = []
    assessment_errors: list[float] = []
    closure_years = 0
    assessment_failures = 0
    cumulative_economic = 0.0
    start_year = int(base_result.state["last_year"]) + 1

    for year_offset in range(max(int(settings.years), 1)):
        year = start_year + year_offset
        true_biomass_before = _total_biomass(numbers, life)
        true_ssb_before = _spawning_biomass(numbers, life, truth_settings)
        true_depletion_before = true_ssb_before / max(b0, _EPS)
        survey_biomass = _survey_biomass(numbers, life)
        observation_available = rng.uniform() >= max(observation.missing_observation_probability, 0.0)
        index = np.nan
        if observation_available and year_offset % max(observation.survey_interval, 1) == 0:
            beta = max(scenario.hyperstability_beta, 0.05)
            q = q0 * exp(scenario.catchability_drift_per_year * year_offset)
            index = q * max(survey_biomass, _EPS) ** beta * float(_normal_lognormal_multiplier(rng, observation.index_cv))
        direct_biomass = np.nan
        if observation.direct_biomass_interval > 0 and year_offset % observation.direct_biomass_interval == 0:
            direct_biomass = true_biomass_before * float(_normal_lognormal_multiplier(rng, observation.direct_biomass_cv))
        reported_catch = previous_implemented_catch * (1.0 + observation.catch_reporting_bias) * float(_normal_lognormal_multiplier(rng, observation.catch_reporting_cv))
        records.append({"year": float(year), "catch": float(max(reported_catch, 0.0)), "index": float(index), "biomass": float(direct_biomass)})

        if year_offset % max(assessment_settings.assessment_interval, 1) == 0:
            estimate = _assessment(
                records,
                age_rows,
                estimate,
                b0,
                base_result,
                assessment_settings,
                settings.seed + simulation_index * 97 + year_offset,
            )
            if float(estimate.get("assessment_success", 1.0)) < 0.5:
                assessment_failures += 1
        action = _management_action(estimate, previous_target_catch, procedure, references["fmsy"], estimate["estimated_biomass"])
        target_catch = action["target_catch"]
        sector_targets = {name: target_catch * share for name, share in allocation.items()}
        implementation = float(_normal_lognormal_multiplier(rng, procedure.implementation_cv))
        implementation *= float(np.clip(procedure.compliance_fraction, 0.0, 1.5))
        implemented_sector = {name: max(value * implementation, 0.0) for name, value in sector_targets.items()}
        f_scalar, outcome, _mismatch = _solve_f_for_catch(numbers, implemented_sector, truth_settings, life, curves)
        implemented_catch = float(outcome["total_landed_biomass"])
        if implemented_catch <= 1e-9:
            closure_years += 1

        if observation.age_composition_interval > 0 and year_offset % observation.age_composition_interval == 0:
            landed = np.sum(np.vstack(list(outcome["landed_numbers_at_age"].values())), axis=0)
            probability = landed / max(float(landed.sum()), _EPS)
            sample_size = max(int(observation.age_sample_size), 1)
            sampled = rng.multinomial(sample_size, probability)
            for age, value in enumerate(sampled):
                age_rows.append({"year": year, "sector": "all", "age": age, "proportion": float(value / sample_size), "sample_size": float(sample_size)})

        if scenario.recruitment_regime_probability > 0 and rng.uniform() < scenario.recruitment_regime_probability:
            poor_regime = not poor_regime
        mean_multiplier = scenario.recruitment_mean_multiplier * (scenario.poor_regime_multiplier if poor_regime else 1.0)
        expected_recruitment = _beverton_holt(true_ssb_before, b0, truth_settings) * max(mean_multiplier, 0.01)
        innovation = rng.normal(0.0, truth_settings.recruitment_sigma)
        rec_dev = truth_settings.recruitment_rho * rec_dev + sqrt(max(1.0 - truth_settings.recruitment_rho**2, 0.0)) * innovation
        recruitment = expected_recruitment * exp(rec_dev - 0.5 * truth_settings.recruitment_sigma**2)
        numbers = _advance_numbers(outcome["survivors"], recruitment)

        true_biomass_after = _total_biomass(numbers, life)
        true_ssb_after = _spawning_biomass(numbers, life, truth_settings)
        true_depletion_after = true_ssb_after / max(b0, _EPS)
        catches.append(implemented_catch)
        depletions.append(true_depletion_after)
        estimated_depletions.append(float(estimate["estimated_depletion"]))
        assessment_errors.append(float(estimate["estimated_depletion"] - true_depletion_before))
        cumulative_economic += implemented_catch * settings.economic_price_per_tonne - f_scalar * settings.economic_cost_per_f
        previous_target_catch = target_catch
        previous_implemented_catch = implemented_catch
        if simulation_index < settings.sample_trajectories_per_cell:
            trajectories.append(
                {
                    "year": year,
                    "true_biomass": float(true_biomass_after),
                    "true_spawning_biomass": float(true_ssb_after),
                    "true_depletion": float(true_depletion_after),
                    "estimated_depletion": float(estimate["estimated_depletion"]),
                    "target_catch": float(target_catch),
                    "implemented_catch": float(implemented_catch),
                    "f_scalar": float(f_scalar),
                    "recruitment": float(recruitment),
                    "assessment_success": float(estimate.get("assessment_success", 1.0)),
                }
            )

    catch_array = np.asarray(catches, dtype=float)
    depletion_array = np.asarray(depletions, dtype=float)
    error_array = np.asarray(assessment_errors, dtype=float)
    estimated_array = np.asarray(estimated_depletions, dtype=float)
    limit = procedure.limit_depletion
    target = procedure.target_depletion
    false_healthy = np.mean((estimated_array >= target) & (depletion_array < limit))
    false_overfished = np.mean((estimated_array < limit) & (depletion_array >= target))
    return {
        "procedure": procedure.name,
        "scenario": scenario.name,
        "simulation": simulation_index,
        "terminal_depletion": float(depletion_array[-1]),
        "minimum_depletion": float(np.min(depletion_array)),
        "mean_depletion": float(np.mean(depletion_array)),
        "mean_catch": float(np.mean(catch_array)),
        "catch_cv": float(np.std(catch_array) / max(float(np.mean(catch_array)), _EPS)),
        "probability_year_below_limit": float(np.mean(depletion_array < limit)),
        "terminal_above_target": float(depletion_array[-1] >= target),
        "ever_below_limit": float(np.min(depletion_array) < limit),
        "closure_frequency": float(closure_years / max(settings.years, 1)),
        "assessment_bias": float(np.mean(error_array)),
        "assessment_rmse": float(sqrt(np.mean(np.square(error_array)))),
        "assessment_failures": float(assessment_failures),
        "false_healthy_rate": float(false_healthy),
        "false_overfished_rate": float(false_overfished),
        "cumulative_economic_value": float(cumulative_economic),
        "trajectories": trajectories,
    }


def _aggregate_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    return {
        "procedure": rows[0]["procedure"],
        "scenario": rows[0]["scenario"],
        "simulations": len(rows),
        "prob_terminal_above_target": float(np.mean(values("terminal_above_target"))),
        "prob_ever_below_limit": float(np.mean(values("ever_below_limit"))),
        "median_terminal_depletion": float(np.median(values("terminal_depletion"))),
        "terminal_depletion_p10": float(np.quantile(values("terminal_depletion"), 0.10)),
        "median_annual_catch": float(np.median(values("mean_catch"))),
        "median_catch_cv": float(np.median(values("catch_cv"))),
        "mean_closure_frequency": float(np.mean(values("closure_frequency"))),
        "mean_assessment_bias": float(np.mean(values("assessment_bias"))),
        "mean_assessment_rmse": float(np.mean(values("assessment_rmse"))),
        "mean_assessment_failures": float(np.mean(values("assessment_failures"))),
        "mean_false_healthy_rate": float(np.mean(values("false_healthy_rate"))),
        "mean_false_overfished_rate": float(np.mean(values("false_overfished_rate"))),
        "mean_cumulative_economic_value": float(np.mean(values("cumulative_economic_value"))),
    }


def _cell_utility(row: Mapping[str, Any], settings: AdvancedMSESettings) -> float:
    safety = max(1.0 - float(row["prob_ever_below_limit"]), 0.0)
    yield_value = max(float(row["median_annual_catch"]), 0.0)
    stability = 1.0 / (1.0 + max(float(row["median_catch_cv"]), 0.0))
    assessment_quality = 1.0 / (1.0 + 4.0 * max(float(row["mean_assessment_rmse"]), 0.0))
    return float(yield_value * safety ** settings.risk_aversion * stability * assessment_quality)


def _attach_scenario_regret(
    rows: list[dict[str, Any]],
    scenario_weights: Mapping[str, float],
    settings: AdvancedMSESettings,
) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["scenario_utility"] = _cell_utility(row, settings)
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    perfect_information = 0.0
    scenario_best: list[dict[str, Any]] = []
    for scenario, values in by_scenario.items():
        best = max(values, key=lambda item: float(item["scenario_utility"]))
        best_utility = float(best["scenario_utility"])
        weight = max(float(scenario_weights.get(scenario, 0.0)), 0.0)
        perfect_information += weight * best_utility
        scenario_best.append(
            {
                "scenario": scenario,
                "weight": weight,
                "best_procedure": best["procedure"],
                "best_utility": best_utility,
            }
        )
        for row in values:
            regret = max(best_utility - float(row["scenario_utility"]), 0.0)
            row["regret"] = float(regret)
            row["relative_regret"] = float(regret / max(abs(best_utility), _EPS))
            row["scenario_winner"] = bool(row is best)
    return {
        "scenario_best_procedures": scenario_best,
        "perfect_information_expected_utility": float(perfect_information),
    }


def _aggregate_procedure(
    procedure: MSEManagementProcedure,
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_weights: Mapping[str, float],
    settings: AdvancedMSESettings,
) -> dict[str, Any]:
    weights = np.asarray([max(float(scenario_weights.get(str(row["scenario"]), 0.0)), 0.0) for row in scenario_rows], dtype=float)
    if float(weights.sum()) <= 0:
        weights[:] = 1.0
    weights /= weights.sum()

    def weighted(key: str) -> float:
        return float(np.sum(weights * np.asarray([float(row[key]) for row in scenario_rows], dtype=float)))

    safety = 1.0 - weighted("prob_ever_below_limit")
    yield_value = weighted("median_annual_catch")
    stability = 1.0 / (1.0 + weighted("median_catch_cv"))
    assessment_quality = 1.0 / (1.0 + 4.0 * weighted("mean_assessment_rmse"))
    utility = yield_value * max(safety, 0.0) ** settings.risk_aversion * stability * assessment_quality
    return {
        "procedure": procedure.name,
        "weighted_probability_terminal_above_target": weighted("prob_terminal_above_target"),
        "weighted_probability_ever_below_limit": weighted("prob_ever_below_limit"),
        "weighted_median_terminal_depletion": weighted("median_terminal_depletion"),
        "weighted_median_annual_catch": yield_value,
        "weighted_catch_cv": weighted("median_catch_cv"),
        "weighted_closure_frequency": weighted("mean_closure_frequency"),
        "weighted_assessment_rmse": weighted("mean_assessment_rmse"),
        "weighted_false_healthy_rate": weighted("mean_false_healthy_rate"),
        "weighted_false_overfished_rate": weighted("mean_false_overfished_rate"),
        "weighted_economic_value": weighted("mean_cumulative_economic_value"),
        "safety_score": safety,
        "stability_score": stability,
        "assessment_quality_score": assessment_quality,
        "risk_adjusted_utility": float(utility),
        "meets_safety_constraint": bool(safety >= settings.minimum_probability_above_limit),
        "worst_case_probability_ever_below_limit": float(max(float(row["prob_ever_below_limit"]) for row in scenario_rows)),
        "worst_case_terminal_depletion_p10": float(min(float(row["terminal_depletion_p10"]) for row in scenario_rows)),
        "weighted_regret": weighted("regret") if all("regret" in row for row in scenario_rows) else float("nan"),
        "maximum_regret": float(max(float(row.get("regret", float("nan"))) for row in scenario_rows)),
        "maximum_relative_regret": float(max(float(row.get("relative_regret", float("nan"))) for row in scenario_rows)),
        "scenario_wins": int(sum(bool(row.get("scenario_winner", False)) for row in scenario_rows)),
        "scenario_win_fraction": float(np.mean([bool(row.get("scenario_winner", False)) for row in scenario_rows])),
        "minimum_scenario_utility": float(min(float(row.get("scenario_utility", 0.0)) for row in scenario_rows)),
        "lower_tail_scenario_utility": float(
            np.mean(
                np.sort(np.asarray([float(row.get("scenario_utility", 0.0)) for row in scenario_rows], dtype=float))[
                    : max(1, int(np.ceil(0.20 * len(scenario_rows))))
                ]
            )
        ),
    }


def _pareto(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            at_least = (
                float(other["weighted_median_annual_catch"]) >= float(candidate["weighted_median_annual_catch"])
                and float(other["weighted_probability_ever_below_limit"]) <= float(candidate["weighted_probability_ever_below_limit"])
                and float(other["weighted_catch_cv"]) <= float(candidate["weighted_catch_cv"])
            )
            strict = (
                float(other["weighted_median_annual_catch"]) > float(candidate["weighted_median_annual_catch"])
                or float(other["weighted_probability_ever_below_limit"]) < float(candidate["weighted_probability_ever_below_limit"])
                or float(other["weighted_catch_cv"]) < float(candidate["weighted_catch_cv"])
            )
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            result.append(dict(candidate))
    return sorted(result, key=lambda row: (-float(row["risk_adjusted_utility"]), float(row["weighted_probability_ever_below_limit"])))


def _readiness(
    scenarios: Sequence[MSEOperatingScenario],
    procedures: Sequence[MSEManagementProcedure],
    observation: MSEObservationSettings,
    assessment: MSEAssessmentSettings,
    settings: AdvancedMSESettings,
) -> dict[str, Any]:
    checks = [
        ("age_structured_operating_model", True, "The operating truth tracks ages, growth, maturity, selectivity, retention, discards and recruitment."),
        ("separate_operating_and_estimation_models", assessment.mode in {"fast_filter", "biomass_ensemble", "full_age_structured"}, "The assessment is separate from the operating truth and may be misspecified."),
        ("multiple_structural_truths", len(scenarios) >= 5, "Formal robustness requires several plausible biological and observation truths."),
        ("observation_error", observation.index_cv > 0 or observation.catch_reporting_cv > 0, "Monitoring data contain explicit error."),
        ("implementation_error", any(procedure.implementation_cv > 0 for procedure in procedures), "Management actions differ from realised fishing."),
        ("full_reassessment", assessment.mode == "full_age_structured", "Highest standard reruns the full age-structured assessment in the loop."),
        ("formal_replications", settings.simulations_per_scenario >= 500, "Formal decisions should use at least 500 simulations per operating scenario."),
        ("management_alternatives", len(procedures) >= 5, "Several management procedures are compared."),
        ("data_lag", assessment.data_lag_years >= 1, "Decision makers act on realistically delayed data."),
        ("sector_controls", all(len(procedure.sector_allocations) >= 2 for procedure in procedures), "Commercial, charter and recreational allocations are represented."),
    ]
    score = sum(1 for _name, passed, _detail in checks if passed)
    if score == len(checks):
        grade, label = "10/10", "formal high-standard configuration"
    elif score >= 8:
        grade, label = "8/10", "strong research configuration"
    elif score >= 6:
        grade, label = "6/10", "useful development configuration"
    else:
        grade, label = "4/10", "exploratory configuration only"
    return {
        "grade": grade,
        "label": label,
        "checks_passed": score,
        "checks_total": len(checks),
        "checks": [{"check": name, "passed": bool(passed), "explanation": detail} for name, passed, detail in checks],
        "boundary": "A 10/10 configuration is not the same as independent scientific certification. Real management use still requires stock-specific conditioning, verification and external review.",
    }


def run_advanced_mse(
    base_result: AgeStructuredResult,
    procedures: Sequence[MSEManagementProcedure],
    scenarios: Sequence[MSEOperatingScenario] | None = None,
    observation: MSEObservationSettings | None = None,
    assessment: MSEAssessmentSettings | None = None,
    settings: AdvancedMSESettings | None = None,
) -> dict[str, Any]:
    if not procedures:
        raise ValueError("At least one management procedure is required.")
    scenario_values = tuple(scenarios or default_operating_scenarios())
    observation_config = observation or MSEObservationSettings()
    assessment_config = assessment or MSEAssessmentSettings()
    run_config = settings or AdvancedMSESettings()
    scenario_weight_values = np.asarray([max(float(item.weight), 0.0) for item in scenario_values], dtype=float)
    if float(scenario_weight_values.sum()) <= 0:
        scenario_weight_values[:] = 1.0
    scenario_weight_values /= scenario_weight_values.sum()
    scenario_weights = {scenario.name: float(weight) for scenario, weight in zip(scenario_values, scenario_weight_values)}

    tasks: list[tuple[MSEManagementProcedure, MSEOperatingScenario, int]] = []
    for procedure in procedures:
        for scenario in scenario_values:
            for simulation_index in range(max(int(run_config.simulations_per_scenario), 1)):
                tasks.append((procedure, scenario, simulation_index))

    raw_rows: list[dict[str, Any]] = []
    workers = max(int(run_config.workers), 1)
    if workers == 1:
        for procedure, scenario, simulation_index in tasks:
            raw_rows.append(
                _single_simulation(base_result, procedure, scenario, observation_config, assessment_config, run_config, simulation_index)
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _single_simulation,
                    base_result,
                    procedure,
                    scenario,
                    observation_config,
                    assessment_config,
                    run_config,
                    simulation_index,
                ): (procedure.name, scenario.name, simulation_index)
                for procedure, scenario, simulation_index in tasks
            }
            for future in as_completed(futures):
                raw_rows.append(future.result())

    cell_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for procedure in procedures:
        for scenario in scenario_values:
            values = [row for row in raw_rows if row["procedure"] == procedure.name and row["scenario"] == scenario.name]
            cell_rows.append(_aggregate_cell(values))
            for row in values:
                for trajectory in row["trajectories"]:
                    trajectories.append({"procedure": procedure.name, "scenario": scenario.name, "simulation": row["simulation"], **trajectory})

    decision_analysis = _attach_scenario_regret(cell_rows, scenario_weights, run_config)

    aggregate_rows: list[dict[str, Any]] = []
    for procedure in procedures:
        values = [row for row in cell_rows if row["procedure"] == procedure.name]
        aggregate_rows.append(_aggregate_procedure(procedure, values, scenario_weights, run_config))
    aggregate_rows.sort(key=lambda row: (-bool(row["meets_safety_constraint"]), -float(row["risk_adjusted_utility"])))
    pareto = _pareto(aggregate_rows)
    best_safe = next((row for row in aggregate_rows if row["meets_safety_constraint"]), aggregate_rows[0])
    minimax_regret = min(aggregate_rows, key=lambda row: float(row.get("maximum_regret", float("inf"))))
    best_fixed_utility = max(float(row["risk_adjusted_utility"]) for row in aggregate_rows)
    decision_analysis.update(
        {
            "minimax_regret_procedure": minimax_regret["procedure"],
            "minimax_maximum_regret": minimax_regret["maximum_regret"],
            "best_fixed_expected_utility": best_fixed_utility,
            "expected_value_of_perfect_information": max(
                float(decision_analysis["perfect_information_expected_utility"]) - best_fixed_utility,
                0.0,
            ),
            "interpretation": "Regret compares each management procedure with the best procedure under each simulated operating scenario. Expected value of perfect information measures the upper-bound value of knowing the true scenario before choosing a procedure.",
        }
    )
    readiness = _readiness(scenario_values, procedures, observation_config, assessment_config, run_config)
    return {
        "summary": {
            "status": "COMPLETE",
            "procedures": len(procedures),
            "operating_scenarios": len(scenario_values),
            "simulations_per_scenario": run_config.simulations_per_scenario,
            "total_closed_loop_simulations": len(raw_rows),
            "recommended_procedure": best_safe["procedure"],
            "recommended_risk_adjusted_utility": best_safe["risk_adjusted_utility"],
            "recommended_meets_safety_constraint": best_safe["meets_safety_constraint"],
            "readiness_grade": readiness["grade"],
        },
        "settings": asdict(run_config),
        "observation_model": asdict(observation_config),
        "assessment_model": asdict(assessment_config),
        "operating_scenarios": [asdict(item) | {"normalised_weight": scenario_weights[item.name]} for item in scenario_values],
        "management_procedures": [asdict(item) for item in procedures],
        "scenario_results": cell_rows,
        "procedure_results": aggregate_rows,
        "pareto_front": pareto,
        "decision_analysis": decision_analysis,
        "sample_trajectories": trajectories,
        "scientific_readiness": readiness,
        "interpretation": "This MSE uses a separate age-structured operating truth, simulated monitoring, an imperfect assessment, management decisions and implementation error. The recommended procedure is conditional on the supplied operating scenarios and performance constraints.",
    }


__all__ = [
    "MSEOperatingScenario",
    "MSEObservationSettings",
    "MSEAssessmentSettings",
    "MSEManagementProcedure",
    "AdvancedMSESettings",
    "default_operating_scenarios",
    "generate_management_grid",
    "run_advanced_mse",
]
