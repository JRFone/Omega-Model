from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class OperatingModelSettings:
    k: float = 10_000.0
    r: float = 0.18
    initial_depletion: float = 0.40
    process_cv: float = 0.12
    observation_cv: float = 0.20
    catch_bias: float = 0.0
    implementation_cv: float = 0.10
    recruitment_regime_probability: float = 0.0
    poor_regime_productivity_multiplier: float = 0.6


@dataclass(frozen=True)
class ManagementProcedure:
    name: str
    assessment_interval: int = 1
    target_depletion: float = 0.40
    limit_depletion: float = 0.10
    target_f_fraction: float = 1.0
    maximum_catch_change: float = 0.20
    minimum_catch: float = 0.0
    maximum_catch: float = float("inf")
    closure_below_limit: bool = True
    pstar: float = 0.45


@dataclass(frozen=True)
class MSESettings:
    years: int = 30
    simulations: int = 500
    seed: int = 60013
    initial_catch: float = 300.0
    economic_price_per_tonne: float = 1.0
    economic_cost_per_effort: float = 0.0


def production(biomass: float, k: float, r: float) -> float:
    b = max(float(biomass), 0.0)
    return max(float(r) * b * (1.0 - b / max(float(k), _EPS)), 0.0)


def noisy_assessment(true_biomass: float, k: float, rng: np.random.Generator, observation_cv: float) -> dict[str, float]:
    sigma = np.sqrt(np.log1p(max(float(observation_cv), 0.0) ** 2))
    estimated_biomass = true_biomass * rng.lognormal(-0.5 * sigma**2, sigma)
    return {
        "estimated_biomass": float(estimated_biomass),
        "estimated_depletion": float(estimated_biomass / max(k, _EPS)),
        "estimated_k": float(k),
    }


def hcr_catch(assessment: Mapping[str, float], previous_catch: float, procedure: ManagementProcedure, r: float) -> float:
    depletion = float(assessment["estimated_depletion"])
    k = float(assessment["estimated_k"])
    if procedure.closure_below_limit and depletion <= procedure.limit_depletion:
        raw = 0.0
    else:
        ramp = np.clip(
            (depletion - procedure.limit_depletion) / max(procedure.target_depletion - procedure.limit_depletion, _EPS),
            0.0,
            1.0,
        )
        msy = r * k / 4.0
        raw = msy * procedure.target_f_fraction * ramp * procedure.pstar / 0.5
    low = previous_catch * (1.0 - max(procedure.maximum_catch_change, 0.0))
    high = previous_catch * (1.0 + max(procedure.maximum_catch_change, 0.0))
    controlled = np.clip(raw, low, high) if previous_catch > 0 else raw
    return float(np.clip(controlled, procedure.minimum_catch, procedure.maximum_catch))


def run_closed_loop_mse(
    operating: OperatingModelSettings,
    procedures: Sequence[ManagementProcedure],
    settings: MSESettings | None = None,
    assessment_function: Callable[[float, float, np.random.Generator, float], Mapping[str, float]] = noisy_assessment,
) -> dict[str, Any]:
    settings = settings or MSESettings()
    rng_master = np.random.default_rng(settings.seed)
    strategy_rows = []
    trajectories = []
    for procedure_index, procedure in enumerate(procedures):
        terminal_depletion = []
        mean_catches = []
        catch_cv = []
        years_below_limit = []
        closure_frequency = []
        rebuild_years = []
        cumulative_economic = []
        for simulation in range(settings.simulations):
            rng = np.random.default_rng(int(rng_master.integers(1, 2_000_000_000)))
            biomass = operating.k * operating.initial_depletion
            catch = settings.initial_catch
            assessed = assessment_function(biomass, operating.k, rng, operating.observation_cv)
            catches = []
            depletions = []
            closures = 0
            rebuild = None
            poor_regime = False
            economic = 0.0
            for year in range(settings.years):
                if year % max(procedure.assessment_interval, 1) == 0:
                    assessed = assessment_function(biomass, operating.k, rng, operating.observation_cv)
                    catch = hcr_catch(assessed, catch, procedure, operating.r)
                implementation_sigma = np.sqrt(np.log1p(max(operating.implementation_cv, 0.0) ** 2))
                implemented = catch * rng.lognormal(-0.5 * implementation_sigma**2, implementation_sigma)
                implemented *= 1.0 + operating.catch_bias
                implemented = min(max(implemented, 0.0), biomass * 0.95)
                if implemented <= 1e-9:
                    closures += 1
                if operating.recruitment_regime_probability > 0 and rng.uniform() < operating.recruitment_regime_probability:
                    poor_regime = not poor_regime
                productivity = operating.r * (operating.poor_regime_productivity_multiplier if poor_regime else 1.0)
                process_sigma = np.sqrt(np.log1p(max(operating.process_cv, 0.0) ** 2))
                biomass = max(
                    operating.k * 1e-6,
                    (biomass + production(biomass, operating.k, productivity) - implemented)
                    * rng.lognormal(-0.5 * process_sigma**2, process_sigma),
                )
                depletion = biomass / operating.k
                catches.append(float(implemented))
                depletions.append(float(depletion))
                effort_proxy = implemented / max(biomass, _EPS)
                economic += implemented * settings.economic_price_per_tonne - effort_proxy * settings.economic_cost_per_effort
                if rebuild is None and depletion >= procedure.target_depletion:
                    rebuild = year + 1
                if simulation < 3:
                    trajectories.append({
                        "procedure": procedure.name,
                        "simulation": simulation,
                        "year": year + 1,
                        "true_biomass": float(biomass),
                        "true_depletion": float(depletion),
                        "estimated_depletion": float(assessed["estimated_depletion"]),
                        "implemented_catch": float(implemented),
                    })
            terminal_depletion.append(depletions[-1])
            mean_catches.append(float(np.mean(catches)))
            catch_cv.append(float(np.std(catches) / max(np.mean(catches), _EPS)))
            years_below_limit.append(float(np.mean(np.asarray(depletions) < procedure.limit_depletion)))
            closure_frequency.append(closures / settings.years)
            rebuild_years.append(float(rebuild if rebuild is not None else settings.years + 1))
            cumulative_economic.append(economic)
        td = np.asarray(terminal_depletion)
        mc = np.asarray(mean_catches)
        strategy_rows.append({
            "procedure": procedure.name,
            "prob_terminal_above_target": float(np.mean(td >= procedure.target_depletion)),
            "prob_terminal_above_limit": float(np.mean(td >= procedure.limit_depletion)),
            "median_terminal_depletion": float(np.median(td)),
            "depletion_p10": float(np.quantile(td, 0.10)),
            "median_annual_catch": float(np.median(mc)),
            "catch_p10": float(np.quantile(mc, 0.10)),
            "catch_p90": float(np.quantile(mc, 0.90)),
            "median_catch_cv": float(np.median(catch_cv)),
            "mean_fraction_years_below_limit": float(np.mean(years_below_limit)),
            "mean_closure_frequency": float(np.mean(closure_frequency)),
            "median_rebuild_year": float(np.median(rebuild_years)),
            "mean_cumulative_economic_value": float(np.mean(cumulative_economic)),
        })
    pareto = pareto_front(strategy_rows)
    return {
        "operating_model": asdict(operating),
        "settings": asdict(settings),
        "procedures": [asdict(procedure) for procedure in procedures],
        "summary": strategy_rows,
        "pareto_front": pareto,
        "sample_trajectories": trajectories,
        "interpretation": "Closed-loop MSE separates operating truth, observation error, assessment, management decision and implementation error.",
    }


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            better_or_equal = (
                float(other["prob_terminal_above_limit"]) >= float(candidate["prob_terminal_above_limit"])
                and float(other["median_annual_catch"]) >= float(candidate["median_annual_catch"])
                and float(other["median_catch_cv"]) <= float(candidate["median_catch_cv"])
            )
            strictly_better = (
                float(other["prob_terminal_above_limit"]) > float(candidate["prob_terminal_above_limit"])
                or float(other["median_annual_catch"]) > float(candidate["median_annual_catch"])
                or float(other["median_catch_cv"]) < float(candidate["median_catch_cv"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(dict(candidate))
    return result


def generate_hcr_grid(
    target_values: Sequence[float] = (0.35, 0.40, 0.45, 0.50),
    limit_values: Sequence[float] = (0.10, 0.15, 0.20),
    change_values: Sequence[float] = (0.10, 0.20, 0.30),
) -> list[ManagementProcedure]:
    procedures = []
    for target in target_values:
        for limit in limit_values:
            if limit >= target:
                continue
            for change in change_values:
                procedures.append(ManagementProcedure(
                    name=f"HCR T{target:.2f} L{limit:.2f} C{change:.2f}",
                    target_depletion=float(target),
                    limit_depletion=float(limit),
                    maximum_catch_change=float(change),
                ))
    return procedures
