from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stock_model.core import ModelSettings, _objective_breakdown, fit
from stock_model.data_io import StockDataset
from stock_model.native_backend import get_native_engine, native_status


def _dataset() -> StockDataset:
    years = np.arange(2000, 2012)
    return StockDataset(
        "native-test",
        pd.DataFrame(
            {
                "year": years,
                "catch": [20, 24, 28, 32, 35, 33, 30, 27, 25, 24, 22, 20],
                "index": [800, 790, 770, 745, 720, 705, 695, 690, 685, 680, 678, 675],
                "biomass": [np.nan] * len(years),
            }
        ),
        index_columns=["index"],
    )


def test_native_objective_and_gradient_match_python() -> None:
    status = native_status()
    assert status["available"] is True
    assert status["abi_version"] == 3
    dataset = _dataset()
    settings = ModelSettings(search_draws=120)
    frame = dataset.frame
    theta = np.asarray([math.log(1000.0), math.log(0.2), math.log(0.8 / 0.2), math.log(0.22)])
    years = frame["year"].to_numpy(dtype=int)
    catches = frame["catch"].to_numpy(dtype=float)
    index = frame["index"].to_numpy(dtype=float)
    biomass = frame["biomass"].to_numpy(dtype=float)
    native = get_native_engine().objective_gradient(theta, years, catches, index, biomass, settings)
    python = _objective_breakdown(theta, years, catches, index, biomass, settings)
    assert abs(native.objective - python[0]) < 1e-9
    assert np.allclose(native.biomass, python[1], rtol=1e-12, atol=1e-9)
    for position in range(4):
        step = 1e-6
        plus = theta.copy(); plus[position] += step
        minus = theta.copy(); minus[position] -= step
        finite = (
            _objective_breakdown(plus, years, catches, index, biomass, settings)[0]
            - _objective_breakdown(minus, years, catches, index, biomass, settings)[0]
        ) / (2.0 * step)
        assert abs(native.gradient[position] - finite) < 1e-4


def test_fit_uses_native_batch_and_ad_refinement() -> None:
    result = fit(_dataset(), ModelSettings(search_draws=120, seed=101))
    assert result.diagnostics["native_backend_available"] is True
    assert str(result.diagnostics["scoring_backend"]).startswith("cpp")
    assert "native-AD" in str(result.diagnostics["refinement_backend"])
    assert np.isfinite(result.best["objective"])
