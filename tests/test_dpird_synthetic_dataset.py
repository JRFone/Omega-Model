from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish" / "Synthetic_DPIRD_Like"


def test_dpird_like_synthetic_dataset_is_complete_and_unambiguous():
    required = {
        "model_ready_timeseries_conditioned.csv",
        "model_ready_timeseries_blind.csv",
        "age_composition.csv",
        "length_composition.csv",
        "synthetic_truth.csv",
        "parameter_register.csv",
        "omega_dataset.json",
        "source_manifest.json",
        "README.md",
    }
    assert not [name for name in required if not (DATASET / name).exists()]
    metadata = json.loads((DATASET / "omega_dataset.json").read_text(encoding="utf-8"))
    assert metadata["synthetic"] is True
    assert metadata["raw_dpird_dataset"] is False
    assert metadata["original_dpird_inputs_available"] is False


def test_conditioned_and_blind_inputs_are_separated():
    conditioned = pd.read_csv(DATASET / "model_ready_timeseries_conditioned.csv")
    blind = pd.read_csv(DATASET / "model_ready_timeseries_blind.csv")
    assert "recruitment_multiplier_absolute" in conditioned
    assert "recruitment_multiplier_absolute" not in blind
    assert np.allclose(
        conditioned[["catch", "catch_commercial", "catch_charter", "catch_recreational"]],
        blind[["catch", "catch_commercial", "catch_charter", "catch_recreational"]],
    )


def test_synthetic_truth_is_close_to_digitised_dpird_trajectory():
    truth = pd.read_csv(DATASET / "synthetic_truth.csv")
    difference = truth["synthetic_operating_truth_depletion"] - truth["digitised_dpird_relative_spawning_biomass"]
    assert float(np.sqrt(np.mean(difference**2))) < 0.04
    assert abs(float(difference.iloc[-1])) < 0.01


def test_synthetic_compositions_sum_to_one():
    for filename in ("age_composition.csv", "length_composition.csv"):
        frame = pd.read_csv(DATASET / filename)
        sums = frame.groupby(["year", "sector"])["proportion"].sum()
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_all_synthetic_charts_have_explicit_axes():
    expected = {
        "01_depletion_truth_vs_dpird.html": ("Year", "Relative spawning biomass (B/B0)"),
        "02_cpue_comparison.html": ("Year", "Relative CPUE index (2008 = 1)"),
        "03_catch_history.html": ("Year", "Retained catch (tonnes)"),
        "04_recruitment_deviations.html": ("Year", "Log recruitment deviation"),
        "05_age_composition_heatmap.html": ("Year", "Age (years; 30 is plus group)"),
    }
    for filename, labels in expected.items():
        text = (DATASET / "Charts" / filename).read_text(encoding="utf-8").replace("\\u002f", "/")
        assert all(label in text for label in labels)


def test_integrated_ui_exposes_progress_time_and_stop_controls():
    source = (ROOT / "integrated_assessment_app.py").read_text(encoding="utf-8")
    assert "{prefix} {percent:3.0f}%" in source
    assert "remaining {remaining_text}" in source
    assert 'text="Stop"' in source
    assert "_request_stop" in source
    for label in ("Year", "Age (years)", "Relative spawning biomass (B/B0)"):
        assert label in source
