from __future__ import annotations

"""Run reproducible Omega stress tests on the DPIRD public reconstruction.

These runs test implementation and assumption sensitivity. They are not a
replacement for DPIRD's unreleased raw inputs or accepted assessment runs.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_model.age_structured import (
    AgeFitSettings,
    AgeStructuredSettings,
    SectorSettings,
    fit_age_structured,
    read_age_structured_file,
    read_composition_file,
)
from stock_model.data_io import StockDataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish"
READY = DATASET_ROOT / "Omega_Ready"
RESULTS = DATASET_ROOT / "Results"
PUBLISHED_TERMINAL_DEPLETION = 0.15


def dataset_with_frame(base: StockDataset, frame: pd.DataFrame, name: str) -> StockDataset:
    return StockDataset(
        name=name,
        frame=frame.reset_index(drop=True),
        provenance=base.provenance,
        transformations=[*base.transformations, {"operation": "DPIRD_public_stress_test", "details": {"variant": name}}],
        warnings=base.warnings,
        raw_columns=base.raw_columns,
        index_columns=base.index_columns,
    )


def base_settings() -> AgeStructuredSettings:
    female_maturity_slope = (7.01 - 3.83) / np.log(19.0)
    return AgeStructuredSettings(
        max_age=30,
        natural_mortality=0.11,
        r0=1_000.0,
        steepness=0.75,
        recruitment_sigma=0.60,
        initial_depletion=0.50,
        linf_mm=983.0,
        growth_k=0.12,
        growth_t0=0.0,
        weight_a=1.97e-8,
        weight_b=2.980,
        maturity_a50=3.83,
        maturity_slope=float(female_maturity_slope),
        m_prior_median=0.11,
        h_prior_mean=0.75,
        initial_depletion_prior=0.50,
        initial_depletion_prior_sd=0.50,
        age_comp_weight=0.05,
        sectors=(
            SectorSettings("commercial", "catch_commercial", 0.40, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
            SectorSettings("charter", "catch_charter", 0.10, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
            SectorSettings("recreational", "catch_recreational", 0.50, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
        ),
    )


def fit_configuration(seed: int, estimate_survey_selectivity: bool) -> AgeFitSettings:
    return AgeFitSettings(
        population=24 if estimate_survey_selectivity else 12,
        generations=8 if estimate_survey_selectivity else 6,
        seed=seed,
        local_rounds=2,
        estimate_natural_mortality=False,
        estimate_steepness=False,
        estimate_initial_depletion=True,
        estimate_survey_selectivity=estimate_survey_selectivity,
        estimate_recruitment_sigma=False,
    )


def geometric_index(frame: pd.DataFrame, columns: list[str], base_year: int = 2008) -> pd.Series:
    values = frame[columns]
    combined = pd.Series(np.where(values.notna().all(axis=1), np.exp(np.log(values).mean(axis=1)), np.nan), index=frame.index)
    base = float(combined.loc[frame["year"] == base_year].iloc[0])
    return combined / base


def sectors_with(settings: AgeStructuredSettings, **updates: float) -> tuple[SectorSettings, ...]:
    result: list[SectorSettings] = []
    for sector in settings.sectors:
        sector_updates: dict[str, float] = {}
        if "discard_mortality" in updates:
            sector_updates["discard_mortality"] = updates["discard_mortality"]
        if "selectivity_a50" in updates:
            sector_updates["selectivity_a50"] = updates["selectivity_a50"]
        result.append(replace(sector, **sector_updates))
    return tuple(result)


def boundary_flags(result: Any) -> list[str]:
    flags: list[str] = []
    values = result.settings
    for bound in result.diagnostics.get("parameter_bounds", []):
        name = str(bound["parameter"])
        lookup = "r0" if name == "log_r0" else name
        if lookup not in values:
            continue
        value = float(values[lookup])
        if name == "log_r0":
            value = float(np.log(max(value, 1e-12)))
        low, high = float(bound["low"]), float(bound["high"])
        width = max(high - low, 1e-12)
        if value <= low + 0.01 * width:
            flags.append(f"{lookup}_near_lower_bound")
        if value >= high - 0.01 * width:
            flags.append(f"{lookup}_near_upper_bound")
    return flags


def run() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = read_age_structured_file(READY / "dpird_wa_dhufish_public_reconstruction.csv")
    age = read_composition_file(READY / "age_composition.csv")
    settings = base_settings()

    # Estimate survey selectivity once. Public fitted survey-selectivity values
    # were not available, so subsequent sensitivity runs hold this working value.
    baseline_result = fit_age_structured(
        dataset,
        settings,
        fit_configuration(seed=8301, estimate_survey_selectivity=True),
        age_composition=age,
    )
    fitted_base = replace(
        settings,
        r0=float(baseline_result.best["r0"]),
        survey_selectivity_a50=float(baseline_result.settings["survey_selectivity_a50"]),
        survey_selectivity_slope=float(baseline_result.settings["survey_selectivity_slope"]),
    )

    base_frame = dataset.frame.copy()
    north_frame = base_frame.copy()
    north_frame["index"] = geometric_index(
        north_frame,
        ["index_index_north_dropline_daily", "index_index_north_handline_daily"],
    )
    south_frame = base_frame.copy()
    south_frame["index"] = south_frame["index_index_south_handline_daily"]
    south_base = float(south_frame.loc[south_frame["year"] == 2008, "index"].iloc[0])
    south_frame["index"] = south_frame["index"] / south_base
    no_index_frame = base_frame.copy()
    no_index_frame["index"] = np.nan
    post_2023_frame = base_frame.copy()
    for sector in ("commercial", "charter", "recreational"):
        column = f"retention_length50_{sector}"
        post_2023_frame[column] = np.nan
        post_2023_frame.loc[post_2023_frame["year"] >= 2023, column] = 300.0

    variants: list[dict[str, Any]] = [
        {"name": "base_public_working", "dataset": dataset, "settings": fitted_base, "age": age, "note": "Published biology; public composite CPUE; pooled age data; working 0.05 composition weight"},
        {"name": "optimizer_seed_8302", "dataset": dataset, "settings": fitted_base, "age": age, "seed": 8302, "note": "Repeat fit from a different optimizer seed"},
        {"name": "natural_mortality_0.09", "dataset": dataset, "settings": replace(fitted_base, natural_mortality=0.09), "age": age, "note": "Published review sensitivity"},
        {"name": "natural_mortality_0.13", "dataset": dataset, "settings": replace(fitted_base, natural_mortality=0.13), "age": age, "note": "Published review sensitivity"},
        {"name": "age_weight_0.01", "dataset": dataset, "settings": replace(fitted_base, age_comp_weight=0.01), "age": age, "note": "Lower effective composition weight"},
        {"name": "age_weight_0.20", "dataset": dataset, "settings": replace(fitted_base, age_comp_weight=0.20), "age": age, "note": "Higher effective composition weight"},
        {"name": "no_age_compositions", "dataset": dataset, "settings": fitted_base, "age": None, "note": "Catch and CPUE only"},
        {"name": "no_cpue", "dataset": dataset_with_frame(dataset, no_index_frame, "no-cpue"), "settings": fitted_base, "age": age, "note": "Catch and age compositions only"},
        {"name": "north_cpue_only", "dataset": dataset_with_frame(dataset, north_frame, "north-cpue"), "settings": fitted_base, "age": age, "note": "North dropline/handline daily CPUE only"},
        {"name": "south_cpue_only", "dataset": dataset_with_frame(dataset, south_frame, "south-cpue"), "settings": fitted_base, "age": age, "note": "South handline daily CPUE only"},
        {"name": "discard_mortality_0", "dataset": dataset, "settings": replace(fitted_base, sectors=sectors_with(fitted_base, discard_mortality=0.0)), "age": age, "note": "Zero release mortality sensitivity"},
        {"name": "discard_mortality_1", "dataset": dataset, "settings": replace(fitted_base, sectors=sectors_with(fitted_base, discard_mortality=1.0)), "age": age, "note": "100% release mortality sensitivity"},
        {"name": "gear_selectivity_A50_4", "dataset": dataset, "settings": replace(fitted_base, sectors=sectors_with(fitted_base, selectivity_a50=4.0)), "age": age, "note": "Earlier gear selectivity; fitted public values unavailable"},
        {"name": "gear_selectivity_A50_6", "dataset": dataset, "settings": replace(fitted_base, sectors=sectors_with(fitted_base, selectivity_a50=6.0)), "age": age, "note": "Later gear selectivity; fitted public values unavailable"},
        {"name": "post_2023_retention_L50_300", "dataset": dataset_with_frame(dataset, post_2023_frame, "post-2023-retention"), "settings": fitted_base, "age": age, "note": "Illustrative post-MLL-removal retention sensitivity, not a published DPIRD parameter"},
    ]

    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        result = fit_age_structured(
            variant["dataset"],
            variant["settings"],
            fit_configuration(seed=int(variant.get("seed", 8400 + index)), estimate_survey_selectivity=False),
            age_composition=variant["age"],
        )
        flags = boundary_flags(result)
        summaries.append(
            {
                "run": variant["name"],
                "note": variant["note"],
                "objective": result.best["objective"],
                "r0": result.best["r0"],
                "natural_mortality": result.best["natural_mortality"],
                "steepness": result.best["steepness"],
                "age_comp_weight": result.settings["age_comp_weight"],
                "initial_depletion": result.best["initial_depletion"],
                "terminal_depletion": result.best["terminal_depletion"],
                "published_terminal_depletion": PUBLISHED_TERMINAL_DEPLETION,
                "difference_from_published": result.best["terminal_depletion"] - PUBLISHED_TERMINAL_DEPLETION,
                "terminal_spawning_biomass_t": result.best["terminal_spawning_biomass"],
                "terminal_f": result.best["terminal_f"],
                "terminal_f_fmsy": result.best["terminal_f_fmsy"],
                "survey_selectivity_a50": result.settings["survey_selectivity_a50"],
                "survey_selectivity_slope": result.settings["survey_selectivity_slope"],
                "boundary_flags": ";".join(flags),
                "software_fit_status": "warning" if flags else "completed",
            }
        )
        for component, value in result.diagnostics["objective_components"].items():
            objective_rows.append({"run": variant["name"], "component": component, "value": value})
        for history in result.history:
            trajectories.append(
                {
                    "run": variant["name"],
                    "year": history["year"],
                    "depletion": history["depletion"],
                    "spawning_biomass_t": history["spawning_biomass"],
                    "f_scalar": history["f_scalar"],
                    "observed_catch_t": history["observed_catch"],
                }
            )

    summary = pd.DataFrame(summaries)
    trajectory = pd.DataFrame(trajectories)
    objectives = pd.DataFrame(objective_rows)
    summary.to_csv(RESULTS / "stress_test_summary.csv", index=False)
    trajectory.to_csv(RESULTS / "stress_test_trajectories.csv", index=False)
    objectives.to_csv(RESULTS / "stress_test_objective_components.csv", index=False)

    lookup = summary.set_index("run")
    terminal_range = [float(summary["terminal_depletion"].min()), float(summary["terminal_depletion"].max())]
    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "software_test_completed": True,
        "scientific_validation_completed": False,
        "baseline_terminal_depletion": float(lookup.loc["base_public_working", "terminal_depletion"]),
        "published_DPIRD_terminal_depletion": PUBLISHED_TERMINAL_DEPLETION,
        "stress_test_terminal_depletion_range": terminal_range,
        "north_vs_south_cpue_terminal_difference": float(
            lookup.loc["north_cpue_only", "terminal_depletion"] - lookup.loc["south_cpue_only", "terminal_depletion"]
        ),
        "age_information_terminal_effect": float(
            lookup.loc["base_public_working", "terminal_depletion"] - lookup.loc["no_age_compositions", "terminal_depletion"]
        ),
        "natural_mortality_terminal_range": [
            float(lookup.loc["natural_mortality_0.09", "terminal_depletion"]),
            float(lookup.loc["natural_mortality_0.13", "terminal_depletion"]),
        ],
        "main_result": "The public reconstruction is usable for stress testing, but it is not an exact reproduction of DPIRD's accepted assessment.",
        "unresolved_model_gaps": [
            "DPIRD raw input tables and accepted bespoke/SS run files are not public in the collected sources.",
            "Omega currently fits one stock, while the updated DPIRD assessment estimates North and South recruitment separately.",
            "Annual recruitment deviations are not simultaneously estimated in the current Omega integrated fit.",
            "Published fitted selectivity, retention, growth-deviation and composition-weight parameters are incomplete.",
            "The public catch, CPUE and composition series are digitised from figures and therefore approximate.",
        ],
    }
    (RESULTS / "stress_test_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return {"runs": len(summary), **findings}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
