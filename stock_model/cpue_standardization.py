from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass
class CPUEStandardizationResult:
    annual_index: list[dict[str, float | int]]
    coefficients: list[dict[str, float | str]]
    diagnostics: dict[str, Any]
    design_columns: list[str]
    fitted_rows: list[dict[str, float | int]]


def _one_hot(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, list[str], dict[str, str]]:
    pieces = [np.ones((len(frame), 1), dtype=float)]
    names = ["intercept"]
    references: dict[str, str] = {}
    for column in columns:
        values = frame[column].astype(str).fillna("missing")
        levels = sorted(values.unique().tolist())
        if not levels:
            continue
        references[column] = levels[0]
        for level in levels[1:]:
            pieces.append((values == level).to_numpy(dtype=float)[:, None])
            names.append(f"{column}={level}")
    return np.hstack(pieces), names, references


def _continuous(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, list[str], dict[str, tuple[float, float]]]:
    pieces = []
    names = []
    scaling: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        sd = float(np.nanstd(values))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        values = np.where(np.isfinite(values), values, mean)
        pieces.append(((values - mean) / sd)[:, None])
        names.append(column)
        scaling[column] = (mean, sd)
    return (np.hstack(pieces) if pieces else np.zeros((len(frame), 0))), names, scaling


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    penalty = np.eye(x.shape[1]) * max(float(ridge), 0.0)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    fitted = x @ beta
    residual = y - fitted
    sigma = float(np.sqrt(np.sum(residual**2) / max(len(y) - x.shape[1], 1)))
    covariance = np.linalg.pinv(x.T @ x + penalty) * sigma**2
    return beta, covariance, sigma


def standardize_cpue(
    records: pd.DataFrame,
    year_column: str = "year",
    catch_column: str = "catch",
    effort_column: str = "effort",
    categorical: Sequence[str] = ("vessel", "area", "month"),
    continuous: Sequence[str] = ("depth",),
    ridge: float = 1e-6,
    technology_year_column: str | None = None,
) -> CPUEStandardizationResult:
    required = {year_column, catch_column, effort_column}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing CPUE columns: {sorted(missing)}")
    frame = records.copy()
    frame[catch_column] = pd.to_numeric(frame[catch_column], errors="coerce")
    frame[effort_column] = pd.to_numeric(frame[effort_column], errors="coerce")
    frame[year_column] = pd.to_numeric(frame[year_column], errors="coerce")
    frame = frame.loc[(frame[catch_column] > 0) & (frame[effort_column] > 0) & frame[year_column].notna()].copy()
    if len(frame) < 8:
        raise ValueError("At least eight positive catch-effort records are required.")
    frame[year_column] = frame[year_column].astype(int)
    frame["raw_cpue"] = frame[catch_column] / frame[effort_column]
    frame["log_cpue"] = np.log(frame["raw_cpue"])

    cat = [column for column in categorical if column in frame.columns and column != year_column]
    cont = [column for column in continuous if column in frame.columns and column != year_column]
    year_values = frame[year_column].astype(str)
    year_levels = sorted(year_values.unique().tolist(), key=int)
    year_reference = year_levels[0]
    x_parts = [np.ones((len(frame), 1), dtype=float)]
    names = ["intercept"]
    for level in year_levels[1:]:
        x_parts.append((year_values == level).to_numpy(dtype=float)[:, None])
        names.append(f"year={level}")
    x_cat, names_cat, references = _one_hot(frame, cat)
    if x_cat.shape[1] > 1:
        x_parts.append(x_cat[:, 1:])
        names.extend(names_cat[1:])
    x_cont, names_cont, scaling = _continuous(frame, cont)
    if x_cont.shape[1]:
        x_parts.append(x_cont)
        names.extend(names_cont)
    if technology_year_column and technology_year_column in frame.columns:
        tech = pd.to_numeric(frame[technology_year_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        tech = (tech - tech.mean()) / max(float(tech.std()), 1.0)
        x_parts.append(tech[:, None])
        names.append("technology_trend")
    x = np.hstack(x_parts)
    y = frame["log_cpue"].to_numpy(dtype=float)
    beta, covariance, sigma = _ridge_fit(x, y, ridge)
    fitted = x @ beta
    frame["fitted_log_cpue"] = fitted
    frame["pearson_residual"] = (y - fitted) / max(sigma, 1e-9)

    annual = []
    for level in year_levels:
        row = np.zeros(x.shape[1], dtype=float)
        row[0] = 1.0
        coefficient_name = f"year={level}"
        if coefficient_name in names:
            row[names.index(coefficient_name)] = 1.0
        eta = float(row @ beta)
        variance = float(row @ covariance @ row)
        index = float(np.exp(eta - beta[0]))
        se_log = float(np.sqrt(max(variance, 0.0)))
        annual.append({
            "year": int(level),
            "standardized_index": index,
            "log_index": eta - float(beta[0]),
            "se_log": se_log,
            "lower_95": float(index * np.exp(-1.96 * se_log)),
            "upper_95": float(index * np.exp(1.96 * se_log)),
            "raw_mean_cpue": float(frame.loc[frame[year_column] == int(level), "raw_cpue"].mean()),
            "records": int((frame[year_column] == int(level)).sum()),
        })
    mean_index = np.mean([row["standardized_index"] for row in annual])
    for row in annual:
        row["standardized_index"] = float(row["standardized_index"] / max(mean_index, _EPS))
        row["lower_95"] = float(row["lower_95"] / max(mean_index, _EPS))
        row["upper_95"] = float(row["upper_95"] / max(mean_index, _EPS))

    coefficient_rows = []
    for i, name in enumerate(names):
        se = float(np.sqrt(max(covariance[i, i], 0.0)))
        coefficient_rows.append({"term": name, "estimate": float(beta[i]), "se": se, "z": float(beta[i] / max(se, 1e-12))})

    residual = y - fitted
    diagnostics = {
        "records_used": int(len(frame)),
        "years": len(year_levels),
        "first_year_reference": int(year_reference),
        "categorical_references": references,
        "continuous_scaling": {key: {"mean": value[0], "sd": value[1]} for key, value in scaling.items()},
        "rmse_log": float(np.sqrt(np.mean(residual**2))),
        "mean_residual": float(np.mean(residual)),
        "lag1_residual_correlation": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if len(residual) > 2 else 0.0,
        "condition_number": float(np.linalg.cond(x.T @ x + np.eye(x.shape[1]) * ridge)),
        "ridge": float(ridge),
        "formula": "log(catch/effort) ~ year + categorical effects + standardized continuous covariates",
        "privacy_note": "Outputs contain coefficients and annual indices only; vessel/skipper identifiers should remain in protected source data.",
    }
    fitted_rows = [
        {
            "year": int(row[year_column]),
            "raw_cpue": float(row["raw_cpue"]),
            "fitted_cpue": float(np.exp(row["fitted_log_cpue"])),
            "pearson_residual": float(row["pearson_residual"]),
        }
        for _, row in frame.iterrows()
    ]
    return CPUEStandardizationResult(annual, coefficient_rows, diagnostics, names, fitted_rows)


def catchability_diagnostics(
    cpue: Sequence[float],
    biomass: Sequence[float],
    years: Sequence[int] | None = None,
) -> dict[str, float | str]:
    cpue_arr = np.asarray(cpue, dtype=float)
    bio_arr = np.asarray(biomass, dtype=float)
    mask = np.isfinite(cpue_arr) & np.isfinite(bio_arr) & (cpue_arr > 0) & (bio_arr > 0)
    if mask.sum() < 4:
        raise ValueError("At least four positive CPUE and biomass pairs are required.")
    x = np.column_stack([np.ones(mask.sum()), np.log(bio_arr[mask])])
    beta = np.linalg.lstsq(x, np.log(cpue_arr[mask]), rcond=None)[0]
    elasticity = float(beta[1])
    classification = "proportional"
    if elasticity < 0.8:
        classification = "hyperstable"
    elif elasticity > 1.2:
        classification = "hyperdepleted"
    q = cpue_arr[mask] / bio_arr[mask]
    trend = 0.0
    if years is not None:
        year_arr = np.asarray(years, dtype=float)[mask]
        trend = float(np.polyfit(year_arr - year_arr.mean(), np.log(q), 1)[0])
    return {
        "cpue_biomass_elasticity": elasticity,
        "classification": classification,
        "annual_log_catchability_trend": trend,
        "catchability_change_per_year": float(np.exp(trend) - 1.0),
        "pairs": int(mask.sum()),
    }
