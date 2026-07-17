from __future__ import annotations

import numpy as np
import pandas as pd

from stock_model.age_structured import AgeFitSettings, AgeStructuredSettings, fit_age_structured
from stock_model.aspm_diagnostic import ASPMSettings, run_age_structured_aspm
from stock_model.core import ModelSettings, fit
from stock_model.data_io import StockDataset
from stock_model.interval_coverage import CoverageSettings, run_interval_coverage
from stock_model.likelihood_profiles import ProfileSettings, profile_likelihood


def _production_dataset() -> StockDataset:
    years = np.arange(2000, 2014)
    catch = np.asarray([20, 24, 28, 32, 36, 34, 31, 29, 27, 25, 23, 22, 21, 20], dtype=float)
    index = np.asarray([900, 880, 850, 820, 790, 770, 755, 740, 730, 720, 715, 710, 708, 705], dtype=float)
    return StockDataset(
        "priority-production",
        pd.DataFrame({"year": years, "catch": catch, "index": index, "biomass": np.nan}),
        index_columns=["index"],
    )


def _age_dataset() -> StockDataset:
    years = np.arange(2000, 2008)
    return StockDataset(
        "priority-age",
        pd.DataFrame(
            {
                "year": years,
                "catch": [5, 7, 9, 10, 9, 8, 7, 6],
                "index": [120, 116, 111, 106, 102, 99, 98, 97],
                "biomass": [np.nan] * len(years),
            }
        ),
        index_columns=["index"],
    )


def test_profile_refits_other_parameters_and_reports_intervals() -> None:
    dataset = _production_dataset()
    settings = ModelSettings(search_draws=120, seed=61)
    fitted = fit(dataset, settings)
    output = profile_likelihood(
        dataset,
        settings,
        fitted,
        "initial_depletion",
        ProfileSettings(points=7, multistarts=2, workers=1, confidence_levels=(0.8, 0.95), use_cache=False, max_iterations=150),
    )
    assert output["summary"]["all_other_active_parameters_refitted"] is True
    assert len(output["profile"]) == 7
    assert len(output["intervals"]) == 2
    assert all("components" in row for row in output["profile"])
    assert all("parameters" in row for row in output["profile"] if row.get("success"))


def test_aspm_is_age_structured_and_removes_compositions() -> None:
    dataset = _age_dataset()
    base = AgeStructuredSettings(max_age=8, r0=5000.0, natural_mortality=0.18, initial_depletion=0.9)
    full = fit_age_structured(
        dataset,
        base,
        AgeFitSettings(
            population=12,
            generations=1,
            local_rounds=0,
            seed=13,
            estimate_natural_mortality=False,
            estimate_steepness=False,
            estimate_survey_selectivity=False,
            estimate_recruitment_sigma=False,
        ),
    )
    output = run_age_structured_aspm(
        dataset,
        full_result=full,
        settings=ASPMSettings(
            multistarts=1,
            max_iterations=60,
            estimate_recruitment_deviations=False,
            run_no_index=True,
            run_index_influence=False,
        ),
    )
    assert output["summary"]["age_structured"] is True
    assert output["summary"]["biology_and_selectivity_fixed_from_full_model"] is True
    aspm = next(row for row in output["variants"] if row["name"] == "ASPM")
    assert aspm["composition_likelihoods_removed"] is True
    assert aspm["biology_fixed_from_full_model"] is True
    assert len(aspm["history"]) == len(dataset.frame)


def test_formal_coverage_counts_attempted_replicates() -> None:
    dataset = _production_dataset()
    settings = ModelSettings(search_draws=120, seed=91)
    truth_fit = fit(dataset, settings)
    output = run_interval_coverage(
        dataset,
        settings,
        truth_fit,
        CoverageSettings(
            replicates=2,
            confidence_levels=(0.8, 0.95),
            methods=("hessian",),
            workers=1,
            search_draws=120,
            include_time_series=True,
        ),
    )
    assert output["summary"]["formal_known_truth_testing"] is True
    assert output["summary"]["attempted_replicates"] == 2
    assert output["coverage"]
    assert all(row["attempted_replicates"] == 2 for row in output["coverage"])
    assert output["time_series_coverage"]
