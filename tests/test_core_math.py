from __future__ import annotations

import math
import unittest

import numpy as np

from stock_model.core import (
    FitResult,
    ModelSettings,
    ProjectionSettings,
    _objective_breakdown,
    _production,
    _reference_points,
    _simulate,
    project,
)


class CoreMathTests(unittest.TestCase):
    def test_schaefer_reference_points(self):
        points = _reference_points(1000.0, 0.2, "schaefer")
        self.assertAlmostEqual(points["bmsy"], 500.0)
        self.assertAlmostEqual(points["msy"], 50.0)
        self.assertAlmostEqual(points["fmsy"], 0.1)
        self.assertAlmostEqual(_production(500.0, 1000.0, 0.2, "schaefer"), 50.0)

    def test_fox_reference_points(self):
        points = _reference_points(1000.0, 0.2, "fox")
        self.assertAlmostEqual(points["bmsy"], 1000.0 / math.e)
        self.assertAlmostEqual(points["msy"], 200.0 / math.e)
        self.assertAlmostEqual(points["fmsy"], 0.2)
        self.assertAlmostEqual(_production(points["bmsy"], 1000.0, 0.2, "fox"), points["msy"])

    def test_pella_reference_points(self):
        shape = 1.35
        points = _reference_points(1000.0, 0.2, "pella", shape)
        expected_bmsy = 1000.0 * (1.0 / (1.0 + shape)) ** (1.0 / shape)
        expected_msy = 0.2 * expected_bmsy / (1.0 + shape)
        self.assertAlmostEqual(points["bmsy"], expected_bmsy)
        self.assertAlmostEqual(points["msy"], expected_msy)
        self.assertAlmostEqual(points["fmsy"], 0.2 / (1.0 + shape))
        self.assertAlmostEqual(_production(points["bmsy"], 1000.0, 0.2, "pella", shape), points["msy"])

    def test_initial_biomass_is_not_replaced_by_catch(self):
        years = np.array([2000, 2001])
        catches = np.array([400.0, 0.0])
        biomass = _simulate(years, catches, 1000.0, 0.2, 0.1, "schaefer")
        self.assertAlmostEqual(biomass[0], 100.0)

    def test_objective_components_sum_to_total(self):
        years = np.arange(2000, 2005)
        catches = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        index = np.array([1.0, 0.95, 0.9, 0.85, 0.8])
        biomass = np.array([500.0, 480.0, 460.0, 440.0, 420.0])
        theta = np.array([math.log(1000.0), math.log(0.2), 0.0, math.log(0.2)])
        total, _pred, _sigma, components = _objective_breakdown(theta, years, catches, index, biomass, ModelSettings())
        self.assertAlmostEqual(total, sum(components.values()), places=10)
        required = {
            "index_likelihood",
            "biomass_likelihood",
            "terminal_depletion_constraint",
            "observation_error_prior",
            "productivity_prior",
            "initial_depletion_prior",
            "catch_to_capacity_penalty",
        }
        self.assertTrue(required <= set(components))

    def test_projection_depletion_uses_each_members_k(self):
        fit_result = FitResult(
            name="member-k-test",
            settings={"model": "schaefer", "pella_shape": 1.35, "process_cv": 0.0},
            best={
                "k_b0": 100.0,
                "r": 1e-9,
                "sigma": 0.1,
                "terminal_biomass": 50.0,
                "terminal_depletion": 0.5,
            },
            diagnostics={},
            history=[{"year": 2020, "catch": 0.0, "biomass": 50.0, "depletion": 0.5}],
            ensemble=[
                {"weight": 0.5, "k": 100.0, "r": 1e-9, "sigma": 0.1, "terminal_biomass": 50.0, "terminal_depletion": 0.5},
                {"weight": 0.5, "k": 1000.0, "r": 1e-9, "sigma": 0.1, "terminal_biomass": 500.0, "terminal_depletion": 0.5},
            ],
        )
        output = project(
            fit_result,
            ProjectionSettings(years=1, iterations=4000, strategy="fixed_f", fixed_f=0.0, process_cv=0.0, seed=5),
        )
        median = output["projection"][0]["depletion_median"]
        self.assertGreater(median, 0.45)
        self.assertLess(median, 0.55)
        self.assertEqual(output["depletion_denominator"], "each simulation member's own K")

    def test_hcr_uses_model_specific_msy(self):
        catches = {}
        for model in ["schaefer", "fox", "pella"]:
            reference = _reference_points(1000.0, 0.2, model)
            fit_result = FitResult(
                name=model,
                settings={"model": model, "pella_shape": 1.35, "process_cv": 0.0},
                best={
                    "k_b0": 1000.0,
                    "r": 0.2,
                    "sigma": 0.1,
                    "terminal_biomass": 600.0,
                    "terminal_depletion": 0.6,
                },
                diagnostics={},
                history=[{"year": 2020, "catch": 0.0, "biomass": 600.0, "depletion": 0.6}],
                ensemble=[
                    {
                        "weight": 1.0,
                        "k": 1000.0,
                        "r": 0.2,
                        "sigma": 0.1,
                        "terminal_biomass": 600.0,
                        "terminal_depletion": 0.6,
                    }
                ],
            )
            output = project(
                fit_result,
                ProjectionSettings(years=1, iterations=2000, strategy="hcr_40_10", pstar=0.5, process_cv=0.0, seed=9),
            )
            catches[model] = output["projection"][0]["catch_median"]
            self.assertAlmostEqual(catches[model], reference["msy"], delta=reference["msy"] * 0.03)
        self.assertGreater(catches["fox"], catches["schaefer"])
        self.assertLess(catches["pella"], catches["schaefer"])


if __name__ == "__main__":
    unittest.main()
