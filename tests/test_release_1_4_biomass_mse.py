from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from stock_model.advanced_mse import (
    AdvancedMSESettings,
    MSEAssessmentSettings,
    MSEManagementProcedure,
    MSEObservationSettings,
    MSEOperatingScenario,
    _readiness,
    run_advanced_mse,
)
from stock_model.age_structured import AgeFitSettings, AgeStructuredSettings, fit_age_structured, synthetic_age_structured_dataset
from stock_model.biomass_truth_engine import BiomassTruthSettings, estimate_best_supported_biomass
from stock_model.core import ModelSettings, _simulate, fit
from stock_model.data_io import read_stock_csv
from stock_model.experimental_diagnostics import ExperimentalDiagnosticSettings, run_experimental_diagnostics
from stock_model.state_space_biomass import StateSpaceBiomassSettings, fit_state_space_biomass
from stock_model.truth_mse_charts import write_advanced_mse_dashboard, write_biomass_truth_dashboard, write_experimental_diagnostics_dashboard


def _known_biomass_dataset(with_biomass: bool = True):
    years = np.arange(1990, 2012)
    catch = np.linspace(80.0, 240.0, len(years))
    truth = _simulate(years, catch, 9000.0, 0.20, 0.82, "schaefer")
    rng = np.random.default_rng(1401)
    index = truth / 9000.0 * rng.lognormal(-0.5 * 0.06**2, 0.06, len(years))
    biomass = np.where(np.arange(len(years)) % 4 == 0, truth * rng.lognormal(-0.5 * 0.06**2, 0.06, len(years)), np.nan) if with_biomass else np.nan
    frame = pd.DataFrame({"year": years, "catch": catch, "index": index, "biomass": biomass})
    return read_stock_csv(frame.to_csv(index=False), "known-biomass"), truth


def test_biomass_evidence_recovers_known_simulation_and_weights_sum() -> None:
    dataset, truth = _known_biomass_dataset(True)
    result = estimate_best_supported_biomass(
        dataset,
        BiomassTruthSettings(search_draws=120, samples=180, holdout_years=3, seed=5001),
    )
    terminal = result.trajectory[-1]
    assert terminal["biomass_p05"] <= truth[-1] <= terminal["biomass_p95"]
    assert abs(sum(row["weight"] for row in result.candidates) - 1.0) < 1e-9
    assert result.summary["identifiability_grade"] in {"A", "B", "C"}
    assert "not a directly observed" in result.summary["statement"]


def test_biomass_evidence_warns_when_absolute_scale_is_not_observed() -> None:
    dataset, _truth = _known_biomass_dataset(False)
    result = estimate_best_supported_biomass(
        dataset,
        BiomassTruthSettings(models=("schaefer",), search_draws=120, samples=80, holdout_years=2, seed=5002),
    )
    assert result.diagnostics["identifiability"]["absolute_scale_warning"] is True



def test_state_space_biomass_filter_returns_latent_intervals() -> None:
    dataset, truth = _known_biomass_dataset(True)
    result = fit_state_space_biomass(
        dataset,
        StateSpaceBiomassSettings(particles=80, candidates=8, seed=5003),
    )
    assert result.diagnostics["state_space"] is True
    assert result.diagnostics["filtered_interval_available"] is True
    intervals = result.diagnostics["history_intervals"]
    assert len(intervals) == len(dataset.frame)
    assert all(row["biomass_p10"] <= row["biomass"] <= row["biomass_p90"] for row in intervals)
    assert np.isfinite(result.best["terminal_biomass"])
    assert 0.20 * truth[-1] <= result.best["terminal_biomass"] <= 2.5 * truth[-1]


def test_biomass_evidence_can_weight_state_space_structure() -> None:
    dataset, _truth = _known_biomass_dataset(True)
    result = estimate_best_supported_biomass(
        dataset,
        BiomassTruthSettings(
            models=("schaefer", "state_space_schaefer"),
            search_draws=120,
            samples=80,
            holdout_years=2,
            state_space_particles=72,
            state_space_candidates=8,
            seed=5004,
        ),
    )
    state_rows = [row for row in result.candidates if row["state_space"]]
    assert state_rows
    assert all(row["backend"] == "bootstrap_particle_filter" for row in state_rows)
    assert abs(sum(row["weight"] for row in result.candidates) - 1.0) < 1e-9

def _small_age_fit():
    settings = AgeStructuredSettings(max_age=8, r0=90000.0, natural_mortality=0.16, linf_mm=650.0, growth_k=0.20)
    dataset, age_comp = synthetic_age_structured_dataset(years=9, settings=settings, seed=611)
    result = fit_age_structured(
        dataset,
        settings,
        AgeFitSettings(
            population=12,
            generations=1,
            local_rounds=0,
            seed=612,
            estimate_survey_selectivity=False,
            estimate_recruitment_sigma=False,
        ),
        age_composition=age_comp,
    )
    return result


def test_advanced_mse_separates_truth_assessment_and_management() -> None:
    base = _small_age_fit()
    procedures = [
        MSEManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.15, fishing_fraction_of_fmsy=0.60),
        MSEManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.10, fishing_fraction_of_fmsy=0.80),
    ]
    scenarios = [
        MSEOperatingScenario("base", 0.5),
        MSEOperatingScenario("low recruitment", 0.5, recruitment_mean_multiplier=0.65),
    ]
    result = run_advanced_mse(
        base,
        procedures,
        scenarios=scenarios,
        observation=MSEObservationSettings(age_composition_interval=2, age_sample_size=30),
        assessment=MSEAssessmentSettings(mode="fast_filter", assessment_interval=2, minimum_years=5, data_lag_years=1),
        settings=AdvancedMSESettings(years=3, simulations_per_scenario=2, workers=1, sample_trajectories_per_cell=1, seed=613),
    )
    assert result["summary"]["total_closed_loop_simulations"] == 8
    assert len(result["scenario_results"]) == 4
    assert len(result["procedure_results"]) == 2
    assert result["sample_trajectories"]
    assert all("mean_assessment_rmse" in row for row in result["scenario_results"])
    assert all("regret" in row and "scenario_utility" in row for row in result["scenario_results"])
    assert result["decision_analysis"]["minimax_regret_procedure"] in {"Conservative", "Balanced"}
    assert result["decision_analysis"]["expected_value_of_perfect_information"] >= 0.0
    assert result["summary"]["recommended_procedure"] in {"Conservative", "Balanced"}


def test_formal_mse_readiness_requires_full_configuration() -> None:
    scenarios = [MSEOperatingScenario(f"scenario-{i}") for i in range(5)]
    procedures = [MSEManagementProcedure(f"procedure-{i}") for i in range(5)]
    result = _readiness(
        scenarios,
        procedures,
        MSEObservationSettings(),
        MSEAssessmentSettings(mode="full_age_structured", data_lag_years=1),
        AdvancedMSESettings(simulations_per_scenario=500),
    )
    assert result["grade"] == "10/10"
    assert "not the same as independent scientific certification" in result["boundary"]


def test_experimental_diagnostics_are_complete_and_explicitly_experimental() -> None:
    dataset, _truth = _known_biomass_dataset(True)
    fitted = fit(dataset, ModelSettings(search_draws=120, seed=7001))
    result = run_experimental_diagnostics(
        dataset,
        fitted,
        ExperimentalDiagnosticSettings(
            search_draws=120,
            posterior_predictive_replicates=60,
            mutual_information_permutations=20,
            data_clone_factors=(1, 2),
            seed=7002,
        ),
    )
    assert result["summary"]["status"] == "COMPLETE"
    assert result["summary"]["diagnostics_tested"] >= 6
    assert "hypothesis generators" in result["summary"]["boundary"]
    assert "parameter_sloppiness" in result["diagnostics"]
    assert "adversarial_stress" in result["diagnostics"]


def test_release_1_4_interactive_dashboards_write_offline_html() -> None:
    dataset, _truth = _known_biomass_dataset(True)
    biomass = estimate_best_supported_biomass(dataset, BiomassTruthSettings(models=("schaefer",), search_draws=120, samples=60, holdout_years=2, seed=8001))
    experimental = run_experimental_diagnostics(
        dataset,
        settings=ExperimentalDiagnosticSettings(search_draws=120, posterior_predictive_replicates=50, mutual_information_permutations=20, data_clone_factors=(1, 2), seed=8002),
    )
    base = _small_age_fit()
    mse = run_advanced_mse(
        base,
        [MSEManagementProcedure("Balanced")],
        scenarios=[MSEOperatingScenario("base")],
        settings=AdvancedMSESettings(years=2, simulations_per_scenario=1, sample_trajectories_per_cell=1, seed=8003),
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [
            write_biomass_truth_dashboard(biomass, root / "biomass.html"),
            write_experimental_diagnostics_dashboard(experimental, root / "diagnostics.html"),
            write_advanced_mse_dashboard(mse, root / "mse.html"),
        ]
        assert all(path.exists() and path.stat().st_size > 1000 for path in paths)
