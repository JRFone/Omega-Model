from __future__ import annotations

"""Build a controlled DPIRD-like dhufish dataset from captured public evidence.

The output is deliberately labelled synthetic. It combines the public catch
history and sampling schedule with an Omega operating truth calibrated to the
digitised DPIRD relative spawning-biomass trajectory. It is a recovery and
model-structure test, not a substitute for DPIRD's unreleased raw inputs.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import least_squares
from scipy.stats import norm

from stock_model.age_structured import (
    AgeStructuredSettings,
    SectorSettings,
    life_history_arrays,
    read_age_structured_file,
    simulate_age_structured,
)


ROOT = Path(__file__).resolve().parents[1]
DPIRD_ROOT = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish"
CAPTURED = DPIRD_ROOT / "Omega_Ready"
OUTPUT = DPIRD_ROOT / "Synthetic_DPIRD_Like"
CHARTS = OUTPUT / "Charts"
SEED = 20250717


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_settings(initial_depletion: float, r0: float) -> AgeStructuredSettings:
    maturity_slope = (7.01 - 3.83) / np.log(19.0)
    return AgeStructuredSettings(
        max_age=30,
        natural_mortality=0.11,
        r0=float(r0),
        steepness=0.75,
        recruitment_sigma=0.60,
        initial_depletion=float(initial_depletion),
        linf_mm=983.0,
        growth_k=0.12,
        growth_t0=0.0,
        length_cv=0.10,
        weight_a=1.97e-8,
        weight_b=2.980,
        maturity_a50=3.83,
        maturity_slope=float(maturity_slope),
        m_prior_median=0.11,
        h_prior_mean=0.75,
        initial_depletion_prior=float(initial_depletion),
        initial_depletion_prior_sd=0.50,
        index_cv=0.15,
        age_comp_weight=0.05,
        length_comp_weight=0.05,
        sectors=(
            SectorSettings("commercial", "catch_commercial", 0.40, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
            SectorSettings("charter", "catch_charter", 0.10, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
            SectorSettings("recreational", "catch_recreational", 0.50, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
        ),
    )


def captured_recruitment_prior(published: pd.DataFrame) -> np.ndarray:
    values = published[
        ["published_north_log_recruitment_deviation", "published_south_log_recruitment_deviation"]
    ].to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if len(finite):
            result[index] = float(np.mean(finite))
    return result


def calibrate_operating_truth(dataset, published: pd.DataFrame) -> tuple[AgeStructuredSettings, np.ndarray, dict]:
    years = published["year"].to_numpy(dtype=int)
    target = published["published_wcb_relative_female_spawning_biomass"].to_numpy(dtype=float)
    known = captured_recruitment_prior(published)
    known_mask = np.isfinite(known)
    knots = np.arange(int(years.min()), 2021, 3)
    initial = base_settings(float(target[0]), 2_500.0)
    evaluations = 0

    def evaluate(values: np.ndarray):
        nonlocal evaluations
        evaluations += 1
        settings = replace(initial, r0=float(np.exp(values[0])))
        deviations = np.interp(years, knots, values[1:], left=values[1], right=0.0)
        simulation = simulate_age_structured(dataset, settings, np.exp(deviations))
        depletion = np.asarray([row["depletion"] for row in simulation["history"]], dtype=float)
        return settings, deviations, simulation, depletion

    def residuals(values: np.ndarray) -> np.ndarray:
        _settings, deviations, _simulation, depletion = evaluate(values)
        return np.r_[
            3.0 * np.log(np.clip(depletion, 1e-8, None) / target),
            0.25 * (deviations[known_mask] - known[known_mask]) / 0.60,
            0.035 * deviations / 0.60,
        ]

    start = np.r_[np.log(2_500.0), np.zeros(len(knots), dtype=float)]
    lower = np.r_[np.log(50.0), np.full(len(knots), -2.5)]
    upper = np.r_[np.log(200_000.0), np.full(len(knots), 2.5)]
    fit = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        max_nfev=140,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )
    settings, deviations, simulation, depletion = evaluate(fit.x)
    metrics = {
        "optimizer_success": bool(fit.success),
        "optimizer_status": int(fit.status),
        "optimizer_message": str(fit.message),
        "objective_evaluations": int(evaluations),
        "calibrated_synthetic_r0": float(settings.r0),
        "target_trajectory_rmse": float(np.sqrt(np.mean((depletion - target) ** 2))),
        "target_trajectory_max_absolute_error": float(np.max(np.abs(depletion - target))),
        "terminal_operating_truth_depletion": float(depletion[-1]),
        "terminal_digitised_dpird_depletion": float(target[-1]),
        "terminal_absolute_error": float(abs(depletion[-1] - target[-1])),
        "log_recruitment_deviation_min": float(deviations.min()),
        "log_recruitment_deviation_max": float(deviations.max()),
        "scientific_status": "synthetic_conditioned_operating_truth_not_raw_dpird_assessment",
    }
    return settings, deviations, {"simulation": simulation, "depletion": depletion, "metrics": metrics}


def synthetic_index(history: list[dict], captured_frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    survey = np.asarray([row["survey_biomass"] for row in history], dtype=float)
    years = captured_frame["year"].to_numpy(dtype=int)
    observed_mask = captured_frame["index"].notna().to_numpy()
    errors = rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15, size=len(years))
    result = survey * errors
    base_index = int(np.flatnonzero(years == 2008)[0])
    result = result / max(float(result[base_index]), 1e-12)
    result[~observed_mask] = np.nan
    return result


def synthetic_age_composition(
    simulation: dict,
    captured_age: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    lookup = {
        (int(row["year"]), int(row["age"])): float(row["proportion"])
        for row in simulation["predicted_age_composition"]
        if row["sector"] == "all"
    }
    rows: list[dict] = []
    for year, group in captured_age.groupby("year"):
        ages = np.arange(31, dtype=int)
        probabilities = np.asarray([lookup.get((int(year), int(age)), 0.0) for age in ages], dtype=float)
        probabilities /= max(float(probabilities.sum()), 1e-12)
        sample_size = int(round(float(group["sample_size"].median())))
        counts = rng.multinomial(max(sample_size, 1), probabilities)
        for age, count in zip(ages, counts):
            rows.append(
                {
                    "year": int(year),
                    "sector": "all",
                    "age": int(age),
                    "proportion": float(count / max(sample_size, 1)),
                    "sample_size": int(sample_size),
                    "evidence_class": "synthetic_multinomial_from_conditioned_operating_truth",
                }
            )
    return pd.DataFrame(rows)


def length_bin_probabilities(settings: AgeStructuredSettings, bins: np.ndarray) -> np.ndarray:
    life = life_history_arrays(settings)
    means = life["length_mm"]
    standard_deviations = np.maximum(means * settings.length_cv, 1.0)
    edges = np.r_[bins[0] - 25.0, (bins[:-1] + bins[1:]) / 2.0, bins[-1] + 25.0]
    probabilities = np.empty((len(means), len(bins)), dtype=float)
    for age_index, (mean, sd) in enumerate(zip(means, standard_deviations)):
        probabilities[age_index] = np.diff(norm.cdf(edges, loc=mean, scale=sd))
        probabilities[age_index] /= max(float(probabilities[age_index].sum()), 1e-12)
    return probabilities


def synthetic_length_composition(
    simulation: dict,
    settings: AgeStructuredSettings,
    captured_age: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    bins = np.arange(250.0, 1_051.0, 50.0)
    age_to_length = length_bin_probabilities(settings, bins)
    rows: list[dict] = []
    for year, group in captured_age.groupby("year"):
        numbers = np.asarray(simulation["catch_numbers_by_year"].get(int(year)), dtype=float)
        probabilities = numbers @ age_to_length
        probabilities /= max(float(probabilities.sum()), 1e-12)
        sample_size = int(round(float(group["sample_size"].median())))
        counts = rng.multinomial(max(sample_size, 1), probabilities)
        for length, count in zip(bins, counts):
            rows.append(
                {
                    "year": int(year),
                    "sector": "all",
                    "length_mm": float(length),
                    "proportion": float(count / max(sample_size, 1)),
                    "sample_size": int(sample_size),
                    "evidence_class": "synthetic_multinomial_from_published_growth_and_operating_truth",
                }
            )
    return pd.DataFrame(rows)


def save_charts(
    timeseries: pd.DataFrame,
    truth: pd.DataFrame,
    age: pd.DataFrame,
) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)

    def finish(figure: go.Figure, filename: str, title: str, x_title: str, y_title: str) -> None:
        figure.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title=x_title,
            yaxis_title=y_title,
            hovermode="x unified",
            margin=dict(l=80, r=35, t=75, b=70),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        figure.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False)
        figure.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.12)", zeroline=False)
        figure.write_html(CHARTS / filename, include_plotlyjs=True, full_html=True)

    figure = go.Figure()
    figure.add_scatter(x=truth["year"], y=truth["digitised_dpird_relative_spawning_biomass"], mode="lines", name="Digitised DPIRD trajectory", line=dict(width=3))
    figure.add_scatter(x=truth["year"], y=truth["synthetic_operating_truth_depletion"], mode="lines", name="Omega synthetic operating truth", line=dict(width=2))
    finish(figure, "01_depletion_truth_vs_dpird.html", "DPIRD-like synthetic truth calibration", "Year", "Relative spawning biomass (B/B0)")

    figure = go.Figure()
    figure.add_scatter(x=timeseries["year"], y=timeseries["index"], mode="lines+markers", name="Synthetic CPUE")
    figure.add_scatter(x=timeseries["year"], y=timeseries["captured_public_cpue_index"], mode="lines", name="Captured public CPUE", opacity=0.75)
    finish(figure, "02_cpue_comparison.html", "Synthetic and captured CPUE series", "Year", "Relative CPUE index (2008 = 1)")

    figure = go.Figure(go.Scatter(x=timeseries["year"], y=timeseries["catch"], mode="lines", line=dict(color="#60a5fa", width=2)))
    finish(figure, "03_catch_history.html", "Captured public retained-catch reconstruction", "Year", "Retained catch (tonnes)")

    figure = go.Figure()
    figure.add_scatter(x=truth["year"], y=truth["synthetic_log_recruitment_deviation"], mode="lines", name="Synthetic conditioned deviation")
    figure.add_scatter(x=truth["year"], y=truth["captured_north_log_recruitment_deviation"], mode="markers", name="Captured North points")
    figure.add_scatter(x=truth["year"], y=truth["captured_south_log_recruitment_deviation"], mode="markers", name="Captured South points")
    figure.add_hline(y=0.0, line_width=1, line_color="rgba(255,255,255,0.5)")
    finish(figure, "04_recruitment_deviations.html", "Recruitment information used in the synthetic operating truth", "Year", "Log recruitment deviation")

    pivot = age.pivot(index="age", columns="year", values="proportion").sort_index()
    figure = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.to_numpy(), colorscale="Magma", colorbar=dict(title="Proportion")))
    finish(figure, "05_age_composition_heatmap.html", "Synthetic catch age composition", "Year", "Age (years; 30 is plus group)")


def build() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    captured_raw = pd.read_csv(CAPTURED / "dpird_wa_dhufish_public_reconstruction.csv")
    dataset = read_age_structured_file(CAPTURED / "dpird_wa_dhufish_public_reconstruction.csv")
    published = pd.read_csv(CAPTURED / "published_assessment_outputs_digitised.csv")
    captured_age = pd.read_csv(CAPTURED / "age_composition.csv")
    settings, deviations, calibration = calibrate_operating_truth(dataset, published)
    simulation = calibration["simulation"]
    depletion = calibration["depletion"]
    metrics = calibration["metrics"]
    rng = np.random.default_rng(SEED)

    frame = captured_raw[["year", "catch", "catch_commercial", "catch_charter", "catch_recreational"]].copy()
    frame["index"] = synthetic_index(simulation["history"], captured_raw, rng)
    frame["biomass"] = np.nan
    frame["recruitment_multiplier_absolute"] = np.exp(deviations)
    frame["captured_public_cpue_index"] = captured_raw["index"]
    for column in captured_raw.columns:
        if column.startswith("index_") and column != "index_evidence_class":
            frame[f"captured_{column}"] = captured_raw[column]
    frame["dataset_status"] = "controlled_dpird_like_synthetic_conditioned_recovery_test"
    conditioned_path = OUTPUT / "model_ready_timeseries_conditioned.csv"
    frame.to_csv(conditioned_path, index=False)
    blind = frame.drop(columns=["recruitment_multiplier_absolute"]).copy()
    blind["dataset_status"] = "controlled_dpird_like_synthetic_blind_structure_test"
    blind.to_csv(OUTPUT / "model_ready_timeseries_blind.csv", index=False)

    synthetic_age = synthetic_age_composition(simulation, captured_age, rng)
    synthetic_length = synthetic_length_composition(simulation, settings, captured_age, rng)
    synthetic_age.to_csv(OUTPUT / "age_composition.csv", index=False)
    synthetic_length.to_csv(OUTPUT / "length_composition.csv", index=False)

    history = pd.DataFrame(simulation["history"])
    truth = pd.DataFrame(
        {
            "year": published["year"].astype(int),
            "digitised_dpird_relative_spawning_biomass": published["published_wcb_relative_female_spawning_biomass"],
            "synthetic_operating_truth_depletion": depletion,
            "synthetic_spawning_biomass_t": history["spawning_biomass"],
            "synthetic_total_biomass_t": history["total_biomass"],
            "synthetic_survey_biomass_t": history["survey_biomass"],
            "synthetic_fishing_mortality": history["f_scalar"],
            "synthetic_log_recruitment_deviation": deviations,
            "captured_north_relative_spawning_biomass": published["published_north_relative_female_spawning_biomass"],
            "captured_south_relative_spawning_biomass": published["published_south_relative_female_spawning_biomass"],
            "captured_north_log_recruitment_deviation": published["published_north_log_recruitment_deviation"],
            "captured_south_log_recruitment_deviation": published["published_south_log_recruitment_deviation"],
            "truth_status": "synthetic_operating_truth_conditioned_to_digitised_dpird_central_trajectory",
        }
    )
    truth.to_csv(OUTPUT / "synthetic_truth.csv", index=False)

    for source, destination in (
        (CAPTURED / "dpird_wa_dhufish_public_reconstruction.csv", OUTPUT / "captured_public_timeseries.csv"),
        (CAPTURED / "age_composition.csv", OUTPUT / "captured_age_composition_digitised.csv"),
        (CAPTURED / "age_composition_by_area.csv", OUTPUT / "captured_age_composition_by_area_digitised.csv"),
        (CAPTURED / "published_assessment_outputs_digitised.csv", OUTPUT / "captured_published_assessment_outputs_digitised.csv"),
    ):
        shutil.copy2(source, destination)

    parameter_rows = [
        ("maximum age / plus group", 30, "years", "published", "RAR2 Table 3.7"),
        ("natural mortality M", 0.11, "per year", "published", "RAR2 Table 3.7"),
        ("steepness h", 0.75, "proportion", "published fixed", "RAR2 Table 3.7"),
        ("recruitment sigma", 0.60, "log scale", "published fixed", "RAR2 Table 3.7"),
        ("female Linf", 983.0, "mm", "published", "RAR2 Table 3.7"),
        ("female growth k", 0.12, "per year", "published", "RAR2 Table 3.7"),
        ("female maturity A50", 3.83, "years", "published", "RAR2 Table 3.7"),
        ("female maturity A95", 7.01, "years", "published", "RAR2 Table 3.7"),
        ("weight-length a", 1.97e-8, "kg and mm", "published converted from grams", "RAR2 Table 3.7"),
        ("weight-length b", 2.980, "dimensionless", "published", "RAR2 Table 3.7"),
        ("post-release mortality", 0.50, "proportion", "published working value", "RAR2/FOP151"),
        ("historical retention L50", 500.0, "mm", "published historical legal-size proxy", "RAR2/FOP151"),
        ("initial depletion", float(settings.initial_depletion), "B/B0", "synthetic target from digitised 1975 WCB trajectory", "RAR2 Figure 3.21"),
        ("R0", float(settings.r0), "recruits", "synthetic calibrated; not a DPIRD estimate", "Omega calibration"),
        ("fleet selectivity A50", 5.0, "years", "synthetic working assumption; fitted DPIRD values unavailable", "Omega calibration"),
        ("fleet selectivity slope", 1.2, "years", "synthetic working assumption; fitted DPIRD values unavailable", "Omega calibration"),
    ]
    pd.DataFrame(parameter_rows, columns=["parameter", "value", "units", "evidence_class", "source"]).to_csv(
        OUTPUT / "parameter_register.csv", index=False
    )

    save_charts(frame, truth, synthetic_age)
    (OUTPUT / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata = {
        "id": "dpird-wa-dhufish-synthetic-public-evidence-2025",
        "display_name": "WA Dhufish — DPIRD-like controlled synthetic recovery test",
        "description": "Known synthetic truth conditioned to the digitised public DPIRD depletion trajectory, with captured public catch and sampling schedules.",
        "source": "Omega controlled simulation using captured DPIRD public evidence",
        "source_url": "https://library.dpird.wa.gov.au/fish_rar/2/",
        "difficulty": "Intermediate",
        "model_type": "Age structured",
        "primary_file": "model_ready_timeseries_conditioned.csv",
        "blind_test_file": "model_ready_timeseries_blind.csv",
        "age_composition": "age_composition.csv",
        "length_composition": "length_composition.csv",
        "truth_file": "synthetic_truth.csv",
        "synthetic_calibrated_r0": float(settings.r0),
        "data_types": ["catch", "CPUE/index", "age", "length", "sector catch", "known truth", "recruitment deviations"],
        "recommended_tools": ["Integrated Assessment", "Diagnostics", "Retrospective", "ASPM", "Likelihood Profiles"],
        "expected_behavior": "Conditioned run should reproduce the known synthetic trajectory closely; blind run exposes the current recruitment-structure limitation.",
        "raw_dpird_dataset": False,
        "original_dpird_inputs_available": False,
        "synthetic": True,
        "random_seed": SEED,
        "scientific_warning": "This must never be described as DPIRD raw data, an accepted assessment, or proof that DPIRD biomass is correct or incorrect.",
    }
    (OUTPUT / "omega_dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""# WA dhufish DPIRD-like controlled synthetic dataset

This is the closest internally coherent Omega test dataset that can be built from the public material captured so far. It is **not DPIRD's raw assessment dataset**.

## What is public evidence

- Retained catch and sector histories digitised from RAR2 Figure 3.15.
- Commercial CPUE digitised from RAR2 Figure 3.17.
- North/South age-composition sampling schedule and sample sizes digitised from RAR2 Figure 3.19.
- Relative spawning biomass, fishing mortality and available recruitment-deviation points digitised from RAR2 Figure 3.21.
- Published biological parameters recorded in `parameter_register.csv`.

## What is synthetic

- A one-stock Omega operating truth calibrated to the digitised WCB relative-spawning-biomass trajectory.
- Synthetic CPUE observations generated from the operating truth at the captured CPUE sampling years.
- Synthetic age and length samples generated from the known operating truth using the captured age-sampling years and sample sizes.
- Calibrated R0 and working selectivity values. These are not DPIRD estimates.

## Which file to use

- `model_ready_timeseries_conditioned.csv`: exact annual synthetic recruitment multipliers are included. Use this first to test whether Omega can recover a known truth.
- `model_ready_timeseries_blind.csv`: recruitment truth is hidden. Use this to expose model-structure and identifiability limitations.
- Load `age_composition.csv` and `length_composition.csv` with either time series.
- `synthetic_truth.csv` is the answer key and must not be treated as an observation.

## Calibration result

- Target-trajectory RMSE: {metrics['target_trajectory_rmse']:.4f} B/B0.
- Synthetic terminal depletion: {metrics['terminal_operating_truth_depletion']:.4f}.
- Digitised DPIRD terminal depletion: {metrics['terminal_digitised_dpird_depletion']:.4f}.

Every chart in `Charts` includes labelled X and Y axes. The original captured files are copied alongside the synthetic observations so each transformation can be audited.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    tracked = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "source_manifest.json")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "random_seed": SEED,
        "official_source": "https://library.dpird.wa.gov.au/fish_rar/2/",
        "files": [
            {"path": str(path.relative_to(OUTPUT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in tracked
        ],
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(OUTPUT), "files": len(tracked) + 1, **metrics}


def check() -> dict:
    required = [
        "model_ready_timeseries_conditioned.csv",
        "model_ready_timeseries_blind.csv",
        "age_composition.csv",
        "length_composition.csv",
        "synthetic_truth.csv",
        "parameter_register.csv",
        "omega_dataset.json",
        "calibration_metrics.json",
        "source_manifest.json",
        "README.md",
    ]
    missing = [name for name in required if not (OUTPUT / name).exists()]
    if missing:
        raise SystemExit("Missing synthetic dataset files: " + ", ".join(missing))
    age = pd.read_csv(OUTPUT / "age_composition.csv")
    length = pd.read_csv(OUTPUT / "length_composition.csv")
    truth = pd.read_csv(OUTPUT / "synthetic_truth.csv")
    metrics = json.loads((OUTPUT / "calibration_metrics.json").read_text(encoding="utf-8"))
    age_error = float((age.groupby(["year", "sector"])["proportion"].sum() - 1.0).abs().max())
    length_error = float((length.groupby(["year", "sector"])["proportion"].sum() - 1.0).abs().max())
    if age_error > 1e-9 or length_error > 1e-9:
        raise SystemExit(f"Composition sum check failed: age={age_error}, length={length_error}")
    if abs(float(truth.iloc[-1]["synthetic_operating_truth_depletion"]) - float(truth.iloc[-1]["digitised_dpird_relative_spawning_biomass"])) > 0.01:
        raise SystemExit("Terminal synthetic truth is not sufficiently close to the digitised DPIRD target.")
    if metrics["target_trajectory_rmse"] > 0.04:
        raise SystemExit("Synthetic truth trajectory RMSE exceeds the 0.04 B/B0 build tolerance.")
    return {"status": "passed", "age_sum_max_error": age_error, "length_sum_max_error": length_error, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = check() if args.check_only else build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
