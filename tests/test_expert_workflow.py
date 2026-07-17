from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_model.core import FitResult, ModelSettings, fit
from stock_model.data_io import StockDataset
from stock_model.expert_workflow import (
    ExpertWorkflowSettings,
    finite_difference_gradient_diagnostic,
    mean_absolute_scaled_error,
    parameter_boundary_diagnostic,
    residual_diagnostics,
    run_expert_workflow,
)


class ExpertWorkflowTests(unittest.TestCase):
    @staticmethod
    def dataset() -> StockDataset:
        years = np.arange(2000, 2012)
        catch = np.array([90, 100, 110, 120, 125, 130, 125, 118, 112, 105, 100, 95], dtype=float)
        index = np.array([1000, 970, 930, 890, 850, 810, 790, 780, 775, 790, 810, 830], dtype=float)
        frame = pd.DataFrame({"year": years, "catch": catch, "index": index, "biomass": np.nan})
        return StockDataset(name="workflow test", frame=frame, index_columns=["index"])

    def test_mase(self) -> None:
        observed = [10.0, 12.0, 13.0, 15.0]
        predicted = [10.5, 11.5, 13.5, 14.5]
        value = mean_absolute_scaled_error(observed, predicted)
        self.assertTrue(np.isfinite(value))
        self.assertLess(value, 1.0)

    def test_boundary_and_residual_diagnostics(self) -> None:
        dataset = self.dataset()
        fitted = fit(dataset, ModelSettings(search_draws=120, seed=3))
        boundaries = parameter_boundary_diagnostic(dataset, fitted)
        residuals = residual_diagnostics(dataset, fitted)
        self.assertEqual(len(boundaries["parameters"]), 4)
        self.assertEqual(len(residuals["rows"]), len(dataset.frame))
        self.assertEqual(len(residuals["heatmap"]["matrix"]), 2)

    def test_gradient_diagnostic_does_not_claim_central_parity_at_sigma_clamp(self) -> None:
        dataset = self.dataset()
        fitted = FitResult(
            name="active-bound fit",
            settings={},
            best={"k_b0": 3000.0, "r": 0.2, "initial_depletion": 0.8, "sigma": 0.03},
            diagnostics={},
            history=[],
            ensemble=[],
        )
        result = finite_difference_gradient_diagnostic(dataset, ModelSettings(), fitted)
        sigma = next(row for row in result["gradients"] if row["parameter"] == "log_sigma")
        self.assertEqual(sigma["active_bound"], "lower")
        self.assertFalse(sigma["parity_tested"])
        self.assertIsNone(sigma["parity_error"])
        self.assertNotEqual(result["summary"]["ad_parity_status"], "FAIL")

    def test_exploration_workflow_records_skips(self) -> None:
        dataset = self.dataset()
        heavy = (
            "Jitter and multi-start distribution",
            "Multi-optimizer agreement",
            "Likelihood profiles and local identifiability",
            "Retrospective analysis and Mohn's rho",
            "Hindcast prediction and MASE",
            "ASPM-style catch and index diagnostic",
            "Data-removal influence analysis",
            "Data-weighting comparison",
            "Structural model ensemble",
            "Simulation-recovery and interval-coverage testing",
            "Closed-loop management strategy evaluation",
        )
        result = run_expert_workflow(
            dataset,
            ModelSettings(search_draws=120, seed=9),
            ExpertWorkflowSettings(mode="exploration", speed="quick", skipped_steps=heavy, continue_after_failure=True),
        )
        skipped = [row for row in result["steps"] if row["status"] == "SKIPPED"]
        self.assertEqual(len(skipped), len(heavy))
        self.assertGreaterEqual(result["summary"]["steps"], 10)
        self.assertIn("exploration_policy", result["summary"])


if __name__ == "__main__":
    unittest.main()
