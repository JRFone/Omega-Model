from __future__ import annotations

"""Run conditioned and blind recovery tests on the DPIRD-like synthetic data."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from stock_model.age_structured import (
    AgeFitSettings,
    fit_age_structured,
    read_age_structured_file,
    read_composition_file,
    simulate_age_structured,
)
from build_dpird_dhufish_synthetic_dataset import base_settings


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish" / "Synthetic_DPIRD_Like"
RESULTS = DATASET / "Results"


def trajectory_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = actual - target
    return {
        "trajectory_rmse": float(np.sqrt(np.mean(difference**2))),
        "trajectory_mae": float(np.mean(np.abs(difference))),
        "trajectory_max_absolute_error": float(np.max(np.abs(difference))),
        "terminal_depletion": float(actual[-1]),
        "truth_terminal_depletion": float(target[-1]),
        "terminal_absolute_error": float(abs(difference[-1])),
    }


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((DATASET / "omega_dataset.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(DATASET / "synthetic_truth.csv")
    target = truth["synthetic_operating_truth_depletion"].to_numpy(dtype=float)
    initial_depletion = float(target[0])
    settings = base_settings(initial_depletion, float(metadata["synthetic_calibrated_r0"]))
    age = read_composition_file(DATASET / "age_composition.csv")
    length = read_composition_file(DATASET / "length_composition.csv")
    conditioned = read_age_structured_file(DATASET / "model_ready_timeseries_conditioned.csv")
    blind = read_age_structured_file(DATASET / "model_ready_timeseries_blind.csv")

    exact = simulate_age_structured(conditioned, settings)
    exact_depletion = np.asarray([row["depletion"] for row in exact["history"]], dtype=float)

    conditioned_fit_config = AgeFitSettings(
        population=12,
        generations=6,
        seed=20250718,
        local_rounds=3,
        estimate_natural_mortality=False,
        estimate_steepness=False,
        estimate_initial_depletion=False,
        estimate_survey_selectivity=False,
        estimate_recruitment_sigma=False,
    )
    blind_fit_config = AgeFitSettings(
        population=24,
        generations=10,
        seed=20250719,
        local_rounds=3,
        estimate_natural_mortality=False,
        estimate_steepness=False,
        estimate_initial_depletion=True,
        estimate_survey_selectivity=True,
        estimate_recruitment_sigma=False,
    )
    conditioned_fit = fit_age_structured(conditioned, settings, conditioned_fit_config, age, length)
    blind_fit = fit_age_structured(blind, settings, blind_fit_config, age, length)

    runs = {
        "conditioned_forward_truth": exact_depletion,
        "conditioned_estimation": np.asarray([row["depletion"] for row in conditioned_fit.history], dtype=float),
        "blind_estimation": np.asarray([row["depletion"] for row in blind_fit.history], dtype=float),
    }
    summaries = []
    for name, values in runs.items():
        metrics = trajectory_metrics(values, target)
        if name == "conditioned_forward_truth":
            objective = np.nan
            r0 = settings.r0
            fitted_initial = settings.initial_depletion
        else:
            result = conditioned_fit if name == "conditioned_estimation" else blind_fit
            objective = result.best["objective"]
            r0 = result.best["r0"]
            fitted_initial = result.best["initial_depletion"]
        summaries.append(
            {
                "run": name,
                **metrics,
                "objective": objective,
                "r0": r0,
                "initial_depletion": fitted_initial,
                "truth_supplied_to_model": name != "blind_estimation",
                "initial_depletion_fixed_to_truth": name in {"conditioned_forward_truth", "conditioned_estimation"},
                "annual_recruitment_fixed_to_truth": name in {"conditioned_forward_truth", "conditioned_estimation"},
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "recovery_test_summary.csv", index=False)

    trajectory_rows = []
    for name, values in runs.items():
        for year, value, truth_value in zip(truth["year"], values, target):
            trajectory_rows.append(
                {
                    "run": name,
                    "year": int(year),
                    "estimated_depletion": float(value),
                    "truth_depletion": float(truth_value),
                    "difference": float(value - truth_value),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(RESULTS / "recovery_test_trajectories.csv", index=False)

    lookup = summary.set_index("run")
    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "conditioned_forward_rmse": float(lookup.loc["conditioned_forward_truth", "trajectory_rmse"]),
        "conditioned_fit_rmse": float(lookup.loc["conditioned_estimation", "trajectory_rmse"]),
        "blind_fit_rmse": float(lookup.loc["blind_estimation", "trajectory_rmse"]),
        "conditioned_fit_terminal_error": float(lookup.loc["conditioned_estimation", "terminal_absolute_error"]),
        "blind_fit_terminal_error": float(lookup.loc["blind_estimation", "terminal_absolute_error"]),
        "interpretation": (
            "The exact conditioned forward run is an implementation check. The conditioned estimation checks parameter recovery. "
            "The blind estimation intentionally removes the annual recruitment truth and measures the current model-structure gap."
        ),
        "scientific_status": "controlled_synthetic_recovery_test_not_validation_of_dpird_raw_assessment",
    }
    (RESULTS / "recovery_test_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    figure = go.Figure()
    figure.add_scatter(x=truth["year"], y=target, mode="lines", name="Known synthetic truth", line=dict(width=3))
    figure.add_scatter(x=truth["year"], y=runs["conditioned_estimation"], mode="lines", name="Conditioned fit")
    figure.add_scatter(x=truth["year"], y=runs["blind_estimation"], mode="lines", name="Blind fit")
    figure.update_layout(
        title="Omega recovery of the DPIRD-like synthetic biomass trajectory",
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title="Relative spawning biomass (B/B0)",
        hovermode="x unified",
        margin=dict(l=80, r=35, t=75, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)")
    figure.write_html(RESULTS / "recovery_test_chart.html", include_plotlyjs=True, full_html=True)
    return {"runs": len(summary), **findings}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
