from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .advanced_structures import default_wa_demersal_settings, simulate_spatial_seasonal
from .benchmark_suite import run_benchmarks
from .closed_loop_mse import MSESettings, ManagementProcedure, OperatingModelSettings, run_closed_loop_mse
from .cpue_standardization import catchability_diagnostics, standardize_cpue
from .diagnostics_suite import data_conflict_matrix, reliability_grade
from .inference_engine import ParameterSpec, fit_parameters, profile_parameter, random_walk_mcmc
from .observation_models import ageing_error_matrix, apply_ageing_error, combine_likelihoods, dirichlet_multinomial_nll, lognormal_nll
from .ss3_interop import export_minimal_ss3
from .tagging import TagObservation, TagRelease, TaggingSettings, predict_tag_recaptures


def synthetic_cpue_records(years: Sequence[int], biomass: Sequence[float], seed: int = 90210) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    vessels = ["V1", "V2", "V3", "V4"]
    areas = ["North", "Central", "South"]
    for year, bio in zip(years, biomass):
        for vessel in vessels:
            for area in areas:
                effort = rng.uniform(20, 80)
                vessel_effect = {"V1": 0.9, "V2": 1.0, "V3": 1.1, "V4": 1.2}[vessel]
                area_effect = {"North": 0.9, "Central": 1.0, "South": 1.05}[area]
                depth = rng.uniform(25, 110)
                cpue = max(bio, 1.0) / 10000.0 * vessel_effect * area_effect * np.exp(-0.002 * depth) * rng.lognormal(0, 0.15)
                rows.append({"year": year, "catch": cpue * effort, "effort": effort, "vessel": vessel, "area": area, "month": int(rng.integers(1, 13)), "depth": depth})
    return pd.DataFrame(rows)


def run_complete_demo(years: int = 12, seed: int = 45120, output_dir: str | Path | None = None) -> dict[str, Any]:
    settings = default_wa_demersal_settings(start_year=2000, years=years, max_age=20)
    catch = {
        "commercial": [250.0 + 10 * np.sin(i / 2) for i in range(years)],
        "charter": [70.0 + 4 * np.sin(i / 3) for i in range(years)],
        "recreational": [150.0 + 8 * np.cos(i / 3) for i in range(years)],
    }
    spatial = simulate_spatial_seasonal(settings, catch, recruitment_deviations=[0.0] * years, environmental_effect=[0.0] * years, seed=seed)
    year_values = [int(row["year"]) for row in spatial.history]
    biomass = [float(row["total_biomass"]) for row in spatial.history]
    depletion = [float(row["depletion"]) for row in spatial.history]
    cpue_records = synthetic_cpue_records(year_values, biomass, seed + 1)
    cpue = standardize_cpue(cpue_records, categorical=("vessel", "area", "month"), continuous=("depth",))
    annual_cpue = [float(row["standardized_index"]) for row in cpue.annual_index]
    catchability = catchability_diagnostics(annual_cpue, biomass, year_values)

    # Observation likelihood demonstration.
    scaled_biomass = np.asarray(biomass) / max(np.mean(biomass), 1e-12)
    cpue_like = lognormal_nll(annual_cpue, scaled_biomass, 0.2, "CPUE")
    true_age = np.exp(-0.25 * np.arange(21)); true_age /= true_age.sum()
    age_error = ageing_error_matrix(20, 0.6)
    observed_age = apply_ageing_error(true_age, age_error)
    counts = np.round(observed_age * 300)
    age_like = dirichlet_multinomial_nll(counts, observed_age, 100.0, "Age composition")
    likelihood = combine_likelihoods([cpue_like, age_like], [1.0, 1.0])

    # Generic parameter engine demonstration: estimate a scale and trend against annual CPUE.
    x = np.arange(len(annual_cpue), dtype=float)
    observed = np.asarray(annual_cpue)
    specs = [
        ParameterSpec("scale", 1.0, 0.1, 3.0, prior_mean=1.0, prior_sd=0.5),
        ParameterSpec("trend", 0.0, -0.2, 0.2, prior_mean=0.0, prior_sd=0.1),
    ]
    def objective(parameters: Mapping[str, float]) -> float:
        predicted = parameters["scale"] * np.exp(parameters["trend"] * x)
        return lognormal_nll(observed, predicted, 0.2).nll
    inference = fit_parameters(objective, specs, starts=4, seed=seed + 2, rounds=250)
    profile = profile_parameter(objective, specs, "trend", np.linspace(-0.08, 0.08, 9), starts=1, seed=seed + 3)
    mcmc = random_walk_mcmc(lambda p: -objective(p), specs, start=inference.parameters, iterations=1200, burn=200, thin=10, seed=seed + 4)

    # Tagging demonstration.
    tag = predict_tag_recaptures(
        [TagRelease(2000, 1, 5, 1000)],
        [TagObservation(0, 2001, 0, 1, 25), TagObservation(0, 2002, 0, 2, 8)],
        np.asarray([[0.90, 0.10, 0.00], [0.05, 0.90, 0.05], [0.00, 0.10, 0.90]]),
        TaggingSettings(reporting_rates=(0.8,), fleet_capture_rates=(0.05,)),
    )

    conflict = data_conflict_matrix({"biomass": biomass, "cpue": annual_cpue, "catch": [sum(catch[fleet][i] for fleet in catch) for i in range(years)]})
    reliability_inputs = {
        **inference.diagnostics,
        **conflict,
        "mohn_rho": 0.08,
        "holdout_relative_error": 0.18,
        "optimizer_terminal_depletion_spread": 0.02,
    }
    reliability = reliability_grade(reliability_inputs)

    mse = run_closed_loop_mse(
        OperatingModelSettings(k=10000, r=0.18, initial_depletion=max(depletion[-1], 0.1)),
        [
            ManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.15, target_f_fraction=0.75, maximum_catch_change=0.15),
            ManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.10, target_f_fraction=1.0, maximum_catch_change=0.20),
            ManagementProcedure("Yield focused", target_depletion=0.35, limit_depletion=0.10, target_f_fraction=1.15, maximum_catch_change=0.30),
        ],
        MSESettings(years=20, simulations=120, seed=seed + 5, initial_catch=400.0),
    )
    benchmarks = run_benchmarks()

    payload = {
        "scope": "Omega FISH cumulative functional demonstration for Releases 4-11",
        "spatial_sex_seasonal": {"diagnostics": spatial.diagnostics, "history": spatial.history, "fleet_history": spatial.fleet_history[:100]},
        "cpue_standardization": {"annual_index": cpue.annual_index, "coefficients": cpue.coefficients, "diagnostics": cpue.diagnostics, "catchability": catchability},
        "observation_models": likelihood,
        "inference": {"parameters": inference.parameters, "objective": inference.objective, "standard_errors": inference.standard_errors, "diagnostics": inference.diagnostics, "profile": profile, "mcmc_summary": mcmc["summary"], "mcmc_acceptance_rate": mcmc["acceptance_rate"]},
        "tagging": {"predictions": tag.predictions, "diagnostics": tag.diagnostics},
        "data_conflict": conflict,
        "reliability": reliability,
        "mse": mse,
        "benchmarks": benchmarks,
        "limitations": [
            "This build provides functional foundations, not independent peer review.",
            "The generated SS3 files are starting templates and require stock-specific control configuration.",
            "Random-walk MCMC is a baseline sampler and not HMC/NUTS.",
            "Real-stock accuracy remains dependent on data quality, observation models and verified model configuration.",
        ],
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "complete_demo.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_csv(output / "spatial_history.csv", spatial.history)
        _write_csv(output / "fleet_history.csv", spatial.fleet_history)
        _write_csv(output / "cpue_standardized.csv", cpue.annual_index)
        _write_csv(output / "mse_summary.csv", mse["summary"])
        _write_csv(output / "reliability.csv", reliability["items"])
        ss3_paths = export_minimal_ss3(output / "ss3_export", year_values, [sum(catch[fleet][i] for fleet in catch) for i in range(years)], annual_cpue, max_age=20, sexes=2, areas=3, seasons=4)
        payload["ss3_export"] = ss3_paths
        (output / "complete_demo.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (output / "REPORT.html").write_text(_html_report(payload), encoding="utf-8")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _html_report(payload: Mapping[str, Any]) -> str:
    reliability = payload["reliability"]
    mse_rows = payload["mse"]["summary"]
    items = "".join(f"<tr><td>{row['diagnostic']}</td><td>{row['status']}</td><td>{row['value']}</td><td>{row['explanation']}</td></tr>" for row in reliability["items"])
    mse = "".join(f"<tr><td>{row['procedure']}</td><td>{row['prob_terminal_above_limit']:.3f}</td><td>{row['median_annual_catch']:.2f}</td><td>{row['median_catch_cv']:.3f}</td></tr>" for row in mse_rows)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Omega FISH Complete Assessment</title><style>body{{font-family:Arial;max-width:1200px;margin:auto;padding:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}.grade{{font-size:36px;font-weight:bold}}</style></head><body><h1>Omega FISH Complete Functional Demonstration</h1><p class='grade'>Reliability grade: {reliability['grade']} — {reliability['label']}</p><h2>Reliability diagnostics</h2><table><tr><th>Diagnostic</th><th>Status</th><th>Value</th><th>Meaning</th></tr>{items}</table><h2>Closed-loop MSE</h2><table><tr><th>Procedure</th><th>P above limit</th><th>Median catch</th><th>Catch CV</th></tr>{mse}</table><h2>Scope</h2><p>{payload['scope']}</p><h2>Limitations</h2><ul>{''.join(f'<li>{v}</li>' for v in payload['limitations'])}</ul></body></html>"""
