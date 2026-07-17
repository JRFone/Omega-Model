from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_model.advanced_structures import (
    default_wa_demersal_settings,
    effective_discard_mortality,
    simulate_spatial_seasonal,
    validate_movement,
)
from stock_model.benchmark_suite import run_benchmarks
from stock_model.closed_loop_mse import MSESettings, ManagementProcedure, OperatingModelSettings, pareto_front, run_closed_loop_mse
from stock_model.cpue_standardization import catchability_diagnostics, standardize_cpue
from stock_model.diagnostics_suite import data_conflict_matrix, reliability_grade, retrospective_metrics
from stock_model.inference_engine import ParameterSpec, fit_parameters, profile_parameter, random_walk_mcmc
from stock_model.observation_models import (
    ageing_error_matrix,
    apply_ageing_error,
    combine_likelihoods,
    dirichlet_multinomial_nll,
    francis_weight,
    lognormal_nll,
    multinomial_nll,
)
from stock_model.ss3_interop import compare_time_series, export_minimal_ss3, parse_report_sso
from stock_model.tagging import TagObservation, TagRelease, TaggingSettings, estimate_reporting_rate, predict_tag_recaptures


class ReleaseFourToElevenTests(unittest.TestCase):
    def test_spatial_sex_seasonal_model(self):
        settings = default_wa_demersal_settings(start_year=2000, years=4, max_age=8)
        movement = validate_movement(settings)
        self.assertTrue(np.allclose(movement.sum(axis=-1), 1.0))
        catch = {fleet.name: [20.0, 25.0, 22.0, 18.0] for fleet in settings.fleets}
        result = simulate_spatial_seasonal(
            settings,
            catch,
            recruitment_deviations=[0.0] * 4,
            environmental_effect=[0.0] * 4,
            seed=1,
        )
        self.assertEqual(len(result.history), 4)
        self.assertEqual(result.terminal_numbers.shape, (2, 3, 9))
        self.assertTrue(np.isfinite(result.diagnostics["terminal_depletion"]))
        self.assertGreater(result.diagnostics["total_dead_discards"], 0.0)
        self.assertGreater(effective_discard_mortality(settings.fleets[0]), effective_discard_mortality(settings.fleets[2]))

    def test_observation_likelihoods_and_ageing_error(self):
        matrix = ageing_error_matrix(10, 0.6)
        self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0))
        true = np.zeros(11); true[5] = 1.0
        observed = apply_ageing_error(true, matrix)
        self.assertAlmostEqual(float(observed.sum()), 1.0)
        counts = np.round(observed * 200)
        multi = multinomial_nll(counts, observed)
        dm = dirichlet_multinomial_nll(counts, observed, 50.0)
        combined = combine_likelihoods([multi, dm], [1.0, 0.5])
        self.assertAlmostEqual(combined["total_objective"], multi.nll + 0.5 * dm.nll)
        obs = np.vstack([observed, np.roll(observed, 1)])
        pred = np.vstack([observed, observed])
        weights = francis_weight(obs, pred, [200, 200])
        self.assertGreater(weights["francis_multiplier"], 0.0)

    def test_cpue_standardisation_and_hyperstability(self):
        rng = np.random.default_rng(44)
        rows = []
        for year in range(2000, 2008):
            biomass = 1000 - 50 * (year - 2000)
            for vessel in ["A", "B", "C"]:
                for area in ["N", "S"]:
                    effort = rng.uniform(10, 50)
                    cpue = biomass / 1000 * ({"A": 0.9, "B": 1.0, "C": 1.1}[vessel]) * rng.lognormal(0, 0.05)
                    rows.append({"year": year, "catch": cpue * effort, "effort": effort, "vessel": vessel, "area": area, "depth": rng.uniform(20, 100)})
        result = standardize_cpue(pd.DataFrame(rows), categorical=("vessel", "area"), continuous=("depth",))
        self.assertEqual(len(result.annual_index), 8)
        self.assertAlmostEqual(np.mean([row["standardized_index"] for row in result.annual_index]), 1.0, places=8)
        diagnostic = catchability_diagnostics([1.0, 0.95, 0.90, 0.86], [1.0, 0.8, 0.6, 0.4])
        self.assertEqual(diagnostic["classification"], "hyperstable")

    def test_inference_profiles_and_mcmc(self):
        specs = [
            ParameterSpec("a", 1.5, -5.0, 5.0, prior_mean=2.0, prior_sd=2.0),
            ParameterSpec("b", -1.0, -5.0, 5.0),
        ]
        def objective(p):
            return (p["a"] - 2.0) ** 2 + 2.0 * (p["b"] + 0.5) ** 2
        result = fit_parameters(objective, specs, starts=3, rounds=300, seed=5)
        self.assertAlmostEqual(result.parameters["a"], 2.0, delta=0.08)
        self.assertAlmostEqual(result.parameters["b"], -0.5, delta=0.08)
        self.assertLess(result.maximum_gradient, 0.02)
        profile = profile_parameter(objective, specs, "a", [1.5, 2.0, 2.5], starts=1)
        self.assertEqual(min(profile, key=lambda row: row["objective"])["value"], 2.0)
        chain = random_walk_mcmc(lambda p: -objective(p), specs, start=result.parameters, iterations=400, burn=100, thin=10, seed=7)
        self.assertGreater(len(chain["samples"]), 10)
        self.assertGreater(chain["acceptance_rate"], 0.0)

    def test_tagging_and_reporting_profile(self):
        releases = [TagRelease(2000, 0, 3, 1000)]
        observations = [TagObservation(0, 2000, 0, 0, 40)]
        settings = TaggingSettings(natural_mortality=0.0, tag_loss=0.0, tag_induced_mortality=0.0, initial_mixing_survival=1.0, reporting_rates=(0.8,), fleet_capture_rates=(0.05,))
        result = predict_tag_recaptures(releases, observations, np.eye(1), settings)
        self.assertAlmostEqual(result.predictions[0]["predicted_recaptures"], 40.0, places=8)
        profile = estimate_reporting_rate(releases, observations, np.eye(1), settings, grid=[0.4, 0.8, 1.0])
        self.assertEqual(profile["best_reporting_rate"], 0.8)

    def test_diagnostics_and_reliability(self):
        conflict = data_conflict_matrix({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [1, 2, 3, 4, 5]})
        self.assertGreater(conflict["conflict_score_0_100"], 0.0)
        retro = retrospective_metrics({2000: 1.0, 2001: 0.8, 2002: 0.6}, [{2000: 1.0, 2001: 0.9}, {2000: 1.1}])
        self.assertTrue(np.isfinite(retro["mohn_rho"]))
        grade = reliability_grade({
            "maximum_gradient": 0.0001,
            "hessian_positive_definite": True,
            "mohn_rho": 0.05,
            "holdout_relative_error": 0.10,
            "conflict_score_0_100": 5.0,
            "hessian_condition_number": 100.0,
            "near_boundary": [],
            "optimizer_terminal_depletion_spread": 0.01,
        })
        self.assertEqual(grade["grade"], "A")

    def test_closed_loop_mse_and_pareto(self):
        procedures = [
            ManagementProcedure("A", target_depletion=0.45, limit_depletion=0.15, target_f_fraction=0.8),
            ManagementProcedure("B", target_depletion=0.40, limit_depletion=0.10, target_f_fraction=1.0),
        ]
        result = run_closed_loop_mse(OperatingModelSettings(), procedures, MSESettings(years=8, simulations=30, seed=8))
        self.assertEqual(len(result["summary"]), 2)
        self.assertGreaterEqual(len(result["pareto_front"]), 1)
        front = pareto_front([
            {"procedure": "x", "prob_terminal_above_limit": 0.9, "median_annual_catch": 100, "median_catch_cv": 0.2},
            {"procedure": "y", "prob_terminal_above_limit": 0.8, "median_annual_catch": 90, "median_catch_cv": 0.3},
        ])
        self.assertEqual([row["procedure"] for row in front], ["x"])

    def test_ss3_interoperability(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = export_minimal_ss3(folder, [2000, 2001, 2002], [100, 90, 80], [1.0, 0.9, 0.8], max_age=20, sexes=2, areas=3, seasons=4)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
        report = """HEADER\nANNUAL_TIME_SERIES\nYr SpawnBio Bio_all\n2000 1000 2000\n2001 900 1800\nPARAMETERS\n"""
        parsed = parse_report_sso(report)
        self.assertEqual(len(parsed["time_series"]), 2)
        compared = compare_time_series(
            [{"year": 2000, "spawning_biomass": 1000, "total_biomass": 2000}],
            parsed["time_series"],
        )
        self.assertAlmostEqual(compared["summary"]["spawning_biomass"]["mean_absolute_relative_difference"], 0.0)

    def test_benchmark_suite(self):
        result = run_benchmarks()
        self.assertEqual(result["summary"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
