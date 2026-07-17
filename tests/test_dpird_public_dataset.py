from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_model.age_structured import read_age_structured_file, read_composition_file


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish"
READY = DATASET / "Omega_Ready"


def test_dpird_public_reconstruction_is_model_ready_and_labelled() -> None:
    annual = pd.read_csv(READY / "dpird_wa_dhufish_public_reconstruction.csv")
    assert annual["year"].tolist() == list(range(1975, 2025))
    assert abs(float(annual.loc[annual["year"] == 2024, "catch"].iloc[0]) - 137.0) < 5.0
    assert set(annual["dataset_status"]) == {"public_evidence_reconstruction_not_raw_dpird_data"}
    assert set(annual["catch_evidence_class"]).issubset(
        {
            "digitised_from_published_figure",
            "digitised_sector_sum_reconciled_where_total_curve_obscured",
        }
    )
    assert np.allclose(
        annual["catch"].to_numpy(),
        annual[["catch_commercial", "catch_charter", "catch_recreational"]].sum(axis=1).to_numpy(),
    )
    assert {"catch_total_curve_digitised", "catch_total_reconciliation_t"}.issubset(annual.columns)
    assert np.isclose(float(annual.loc[annual["year"] == 2008, "index"].iloc[0]), 1.0)
    loaded = read_age_structured_file(READY / "dpird_wa_dhufish_public_reconstruction.csv")
    assert len(loaded.frame) == 50
    assert {"catch_commercial", "catch_charter", "catch_recreational"}.issubset(loaded.frame.columns)
    by_area = pd.read_csv(READY / "catch_by_area_sector.csv")
    totals = by_area.groupby(["year", "sector"])["retained_catch_t"].sum()
    for sector, column in {
        "commercial": "catch_commercial",
        "charter": "catch_charter",
        "recreational": "catch_recreational",
    }.items():
        expected = annual.set_index("year")[column]
        actual = totals.xs(sector, level="sector")
        assert np.allclose(actual.loc[expected.index].to_numpy(), expected.to_numpy())


def test_dpird_age_compositions_are_valid() -> None:
    pooled = read_composition_file(READY / "age_composition.csv")
    assert set(pooled["age"].astype(int)) == set(range(31))
    assert np.allclose(pooled.groupby("year")["proportion"].sum().to_numpy(), 1.0)
    by_area = pd.read_csv(READY / "age_composition_by_area.csv")
    assert set(by_area["area"]) == {"north", "south"}
    assert np.allclose(by_area.groupby(["year", "area"])["proportion"].sum().to_numpy(), 1.0)


def test_dpird_parameter_and_source_provenance_is_explicit() -> None:
    parameters = pd.read_csv(READY / "parameter_register.csv")
    m = parameters.loc[parameters["parameter"] == "natural_mortality_M", "value"].astype(float).iloc[0]
    h = parameters.loc[parameters["parameter"] == "steepness_h", "value"].astype(float).iloc[0]
    assert np.isclose(m, 0.11)
    assert np.isclose(h, 0.75)
    missing = parameters[parameters["evidence_class"] == "not_publicly_available"]
    assert {"selectivity_A50", "unfished_recruitment_R0", "initial_depletion"}.issubset(set(missing["parameter"]))
    manifest = json.loads((READY / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_dpird_model_files_available"] is False
    assert len(manifest["sources"]) == 4
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"])
    published = pd.read_csv(READY / "published_assessment_outputs_digitised.csv")
    terminal = float(
        published.loc[
            published["year"] == 2024,
            "published_wcb_relative_female_spawning_biomass",
        ].iloc[0]
    )
    assert abs(terminal - 0.15) < 0.01
