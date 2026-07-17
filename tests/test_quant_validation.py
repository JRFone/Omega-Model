from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stock_model.core import ModelSettings
from stock_model.data_io import read_stock_csv
from stock_model.quant_lab import QuantOptimizerSettings, run_global_optimizer
from stock_model.quant_report import generate_quant_report
from stock_model.quant_validation import (
    EnsembleSettings,
    OptimizerAgreementSettings,
    WalkForwardSettings,
    projection_risk_metrics,
    run_model_ensemble,
    run_optimizer_agreement,
    run_walk_forward_validation,
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
2010,118,0.79,865
2011,115,0.82,890
"""


class QuantValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_stock_csv(CSV, "quant-validation")
        cls.base = ModelSettings(search_draws=120, seed=12)

    def test_covariance_and_simplex_optimizers(self):
        for index, algorithm in enumerate(("cma_es", "nelder_mead")):
            output = run_global_optimizer(
                self.dataset,
                self.base,
                QuantOptimizerSettings(
                    algorithm=algorithm,
                    population=12,
                    generations=1,
                    seed=20 + index,
                    local_refinement_rounds=0,
                ),
            )
            self.assertEqual(output["summary"]["algorithm"], algorithm)
            self.assertGreaterEqual(len(output["candidates"]), 12)
            ident = output["diagnostics"]["local_identifiability"]
            self.assertEqual(ident["dimensions"], 8)
            self.assertEqual(len(ident["profiles"]), 8 * 9)
            self.assertEqual(len(ident["hessian"]), 8)

    def test_walk_forward_validation(self):
        output = run_walk_forward_validation(
            self.dataset,
            self.base,
            WalkForwardSettings(minimum_training_years=7, holdout_years=1, search_draws=120, seed=30),
        )
        self.assertEqual(output["summary"]["status"], "completed")
        self.assertGreaterEqual(output["summary"]["folds"], 4)
        self.assertEqual(len(output["predictions"]), output["summary"]["predictions"])
        self.assertIn("index_log_rmse", output["summary"])

    def test_optimizer_agreement(self):
        output = run_optimizer_agreement(
            self.dataset,
            self.base,
            OptimizerAgreementSettings(
                algorithms=("differential_evolution", "cma_es", "nelder_mead"),
                population=12,
                generations=1,
                seed=40,
            ),
        )
        self.assertEqual(output["summary"]["algorithms"], 3)
        self.assertEqual(len(output["runs"]), 3)
        self.assertTrue(any(row["quantity"] == "terminal_depletion" for row in output["agreement"]))

    def test_model_ensemble_and_risk_metrics(self):
        output = run_model_ensemble(
            self.dataset,
            self.base,
            EnsembleSettings(search_draws=120, projection_years=3, projection_iterations=50, seed=50),
        )
        self.assertEqual(output["summary"]["models"], 3)
        self.assertEqual(len(output["models"]), 3)
        self.assertEqual(len(output["combined_projection"]), 3)
        self.assertAlmostEqual(sum(row["candidate_weight"] for row in output["models"]), 1.0, places=8)
        schaefer = output["model_projections"]["schaefer"]
        risk = projection_risk_metrics(schaefer)
        self.assertEqual(risk["summary"]["status"], "completed")
        self.assertIn("risk_adjusted_yield_index", risk["summary"])

    def test_report_package(self):
        payload = {
            "summary": {"dataset": "test", "rows": 12},
            "optimizer": {"summary": {"best_objective": 1.0}, "candidates": [{"rank": 1, "objective": 1.0}]},
            "walk_forward": {"summary": {"folds": 2}, "folds": [{"fold": 1}, {"fold": 2}]},
        }
        with tempfile.TemporaryDirectory() as folder:
            result = generate_quant_report(payload, folder)
            self.assertTrue(Path(result["html"]).exists())
            self.assertTrue(Path(result["json"]).exists())
            loaded = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"]["rows"], 12)
            self.assertTrue(Path(result["csv"]["optimizer_candidates"]).exists())


if __name__ == "__main__":
    unittest.main()
