from __future__ import annotations

import unittest

from stock_model.core import ModelSettings, fit
from stock_model.data_io import read_stock_csv
from stock_model.quant_lab import (
    QuantOptimizerSettings,
    _dominates,
    detect_index_regime_shift,
    run_global_optimizer,
    run_hcr_genetic_optimization,
    run_stress_tests,
    sobol_projection_screen,
)


CSV = """year,catch,index,biomass
2000,100,1.00,1200
2001,105,0.97,1160
2002,110,0.94,1110
2003,120,0.90,1050
2004,125,0.86,990
2005,130,0.82,930
2006,135,0.78,880
2007,130,0.76,850
2008,125,0.75,840
2009,120,0.77,850
"""


class QuantLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_stock_csv(CSV, "quant-test")

    def test_global_optimizer_produces_eight_dimensions(self):
        output = run_global_optimizer(
            self.dataset,
            ModelSettings(search_draws=120, seed=2),
            QuantOptimizerSettings(population=12, generations=2, seed=3, local_refinement_rounds=1),
        )
        self.assertEqual(output["summary"]["dimensions"], 8)
        self.assertEqual(len(output["parameter_bounds"]), 8)
        self.assertGreaterEqual(len(output["candidates"]), 12)
        self.assertEqual(output["candidates"][0]["rank"], 1)
        self.assertIn("parallel_coordinates", output["diagnostics"])

    def test_pareto_rows_do_not_dominate_each_other(self):
        result = fit(self.dataset, ModelSettings(search_draws=120, seed=4))
        output = run_hcr_genetic_optimization(result, years=3, iterations=40, population=12, generations=2, seed=5)
        pareto = output["pareto"]
        self.assertGreaterEqual(len(pareto), 1)
        for i, left in enumerate(pareto):
            for j, right in enumerate(pareto):
                if i != j:
                    self.assertFalse(_dominates(left["objectives"], right["objectives"]))

    def test_stress_tests_and_regime_screen(self):
        output = run_stress_tests(self.dataset, ModelSettings(search_draws=120, seed=6), search_draws=120, seed=7)
        self.assertGreaterEqual(output["summary"]["scenarios"], 8)
        self.assertTrue(any(row["scenario"] == "index_hyperstable" for row in output["stress_tests"]))
        regime = detect_index_regime_shift(self.dataset)
        self.assertEqual(regime["status"], "candidate_change_point")
        self.assertIn("best", regime)

    def test_sobol_screen(self):
        result = fit(self.dataset, ModelSettings(search_draws=120, seed=8))
        output = sobol_projection_screen(result, years=3, samples=32, seed=9)
        self.assertEqual(len(output["sensitivity"]), 5)
        self.assertEqual(output["summary"]["model_evaluations"], 32 * 7)


if __name__ == "__main__":
    unittest.main()
