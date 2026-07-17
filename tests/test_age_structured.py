from __future__ import annotations

import unittest

import numpy as np

from stock_model.age_structured import (
    AgeFitSettings,
    AgeProjectionSettings,
    AgeStructuredSettings,
    SectorSettings,
    fit_age_structured,
    life_history_arrays,
    normalise_composition_frame,
    project_age_structured,
    read_age_structured_csv,
    run_management_strategy_evaluation,
    sector_curves,
    simulate_age_structured,
    synthetic_age_structured_dataset,
)
from stock_model.data_io import StockDataset


CSV = """year,catch,index,biomass,catch_commercial,catch_charter,catch_recreational,recruitment_multiplier
2000,80,1.00,2000,40,12,28,1.00
2001,85,0.98,1960,42,13,30,0.95
2002,90,0.95,1900,45,14,31,1.10
2003,95,0.92,1840,47,14,34,0.90
2004,100,0.88,1770,50,15,35,1.20
2005,105,0.84,1700,52,16,37,1.00
2006,110,0.80,1630,55,17,38,0.85
2007,105,0.78,1580,52,16,37,1.05
2008,100,0.77,1550,50,15,35,1.10
2009,95,0.79,1570,47,14,34,1.00
"""


class AgeStructuredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = AgeStructuredSettings(
            max_age=10,
            r0=220_000.0,
            natural_mortality=0.14,
            linf_mm=720.0,
            growth_k=0.18,
            maturity_a50=4.0,
            survey_selectivity_a50=3.0,
            sectors=(
                SectorSettings("commercial", "catch_commercial", 0.50, 4.0, 1.0, 420.0, 30.0, 0.50, 1.0),
                SectorSettings("charter", "catch_charter", 0.15, 3.5, 1.0, 420.0, 30.0, 0.50, 1.0),
                SectorSettings("recreational", "catch_recreational", 0.35, 3.0, 1.0, 420.0, 30.0, 0.50, 1.0),
            ),
        )
        cls.dataset = read_age_structured_csv(CSV, "age-test")

    def test_age_model_columns_are_preserved(self):
        self.assertIn("catch_commercial", self.dataset.frame.columns)
        self.assertIn("recruitment_multiplier", self.dataset.frame.columns)
        self.assertAlmostEqual(float(self.dataset.frame.loc[0, "catch_charter"]), 12.0)

    def test_life_history_and_sector_curves(self):
        life = life_history_arrays(self.settings)
        self.assertEqual(len(life["age"]), 11)
        self.assertTrue(np.all(np.diff(life["length_mm"]) > 0))
        self.assertTrue(np.all(np.diff(life["maturity"]) >= 0))
        curves = sector_curves(self.settings, life)
        self.assertTrue(np.all((curves["commercial"]["retention"] >= 0) & (curves["commercial"]["retention"] <= 1)))
        self.assertGreater(curves["commercial"]["retention"][-1], curves["commercial"]["retention"][0])

    def test_catch_reconstruction_and_discards(self):
        output = simulate_age_structured(self.dataset, self.settings)
        self.assertEqual(len(output["history"]), 10)
        self.assertEqual(len(output["sector_history"]), 30)
        total_observed = sum(row["observed_catch"] for row in output["history"])
        total_mismatch = sum(abs(row["catch_mismatch"]) for row in output["history"])
        self.assertLess(total_mismatch, total_observed * 1e-5)
        self.assertGreater(sum(row["dead_discard_biomass"] for row in output["history"]), 0.0)
        self.assertEqual(len(output["age_structure"]), 10 * 11)

    def test_year_specific_retention_override_changes_only_configured_year(self):
        base = simulate_age_structured(self.dataset, self.settings)
        frame = self.dataset.frame.copy()
        frame["retention_length50_recreational"] = np.nan
        frame.loc[frame["year"] == frame["year"].max(), "retention_length50_recreational"] = 250.0
        changed_dataset = StockDataset(
            name="age-test-time-varying-retention",
            frame=frame,
            provenance=self.dataset.provenance,
            transformations=self.dataset.transformations,
            warnings=self.dataset.warnings,
            raw_columns=self.dataset.raw_columns,
            index_columns=self.dataset.index_columns,
        )
        changed = simulate_age_structured(changed_dataset, self.settings)
        self.assertAlmostEqual(base["history"][-2]["f_scalar"], changed["history"][-2]["f_scalar"], places=10)
        self.assertNotAlmostEqual(base["history"][-1]["f_scalar"], changed["history"][-1]["f_scalar"], places=6)

    def test_absolute_recruitment_multiplier_is_not_median_normalised(self):
        frame = self.dataset.frame.copy()
        frame["recruitment_multiplier_absolute"] = 1.0
        frame.loc[frame.index[0], "recruitment_multiplier_absolute"] = 2.0
        exact_dataset = StockDataset(
            name="absolute-recruitment-test",
            frame=frame,
            provenance=self.dataset.provenance,
            transformations=self.dataset.transformations,
            warnings=self.dataset.warnings,
            raw_columns=self.dataset.raw_columns,
            index_columns=self.dataset.index_columns,
        )
        output = simulate_age_structured(exact_dataset, self.settings)
        self.assertAlmostEqual(output["history"][1]["recruitment_deviation"], np.log(2.0), places=10)

    def test_progress_reporting_and_safe_cancellation(self):
        progress: list[tuple[float, str]] = []
        simulate_age_structured(
            self.dataset,
            self.settings,
            progress_callback=lambda value, phase: progress.append((value, phase)),
        )
        self.assertTrue(progress)
        self.assertAlmostEqual(progress[-1][0], 1.0)
        with self.assertRaises(InterruptedError):
            simulate_age_structured(self.dataset, self.settings, cancel_check=lambda: True)

    def test_composition_normalisation(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "year": [2000, 2000, 2001, 2001],
                "age": [1, 2, 1, 2],
                "count": [30, 70, 2, 8],
                "sector": ["all", "all", "all", "all"],
            }
        )
        output = normalise_composition_frame(frame)
        grouped = output.groupby(["year", "sector"])["proportion"].sum()
        self.assertTrue(np.allclose(grouped.to_numpy(), 1.0))

    def test_fit_projection_and_strategy_evaluation(self):
        synthetic, age_comp = synthetic_age_structured_dataset(
            years=12,
            settings=AgeStructuredSettings(max_age=8, r0=180_000.0, natural_mortality=0.16, linf_mm=650.0, growth_k=0.20),
            seed=77,
        )
        result = fit_age_structured(
            synthetic,
            AgeStructuredSettings(max_age=8, r0=150_000.0, natural_mortality=0.14, linf_mm=650.0, growth_k=0.20),
            AgeFitSettings(
                population=12,
                generations=1,
                local_rounds=1,
                seed=88,
                estimate_survey_selectivity=False,
                estimate_recruitment_sigma=False,
            ),
            age_composition=age_comp,
        )
        self.assertGreater(result.best["r0"], 0.0)
        self.assertGreater(result.best["msy"], 0.0)
        self.assertIn("age_composition_deviance", result.diagnostics["objective_components"])
        projection = project_age_structured(
            result,
            AgeProjectionSettings(years=3, iterations=30, seed=99, implementation_cv=0.0, recruitment_sigma=0.1),
        )
        self.assertEqual(len(projection["projection"]), 3)
        self.assertIn("prob_ever_below_limit", projection["risk_summary"])
        mse = run_management_strategy_evaluation(result, years=2, iterations=20, seed=111)
        self.assertEqual(len(mse["strategies"]), 9)
        self.assertGreaterEqual(len(mse["pareto_front"]), 1)


if __name__ == "__main__":
    unittest.main()
