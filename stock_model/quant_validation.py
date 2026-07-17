from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, log
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .core import FitResult, ModelSettings, ProjectionSettings, _production, _reference_points, fit, project
from .data_io import StockDataset
from .quant_lab import QuantOptimizerSettings, run_global_optimizer


@dataclass(frozen=True)
class WalkForwardSettings:
    minimum_training_years: int = 6
    holdout_years: int = 1
    search_draws: int = 180
    seed: int = 19031


@dataclass(frozen=True)
class OptimizerAgreementSettings:
    algorithms: tuple[str, ...] = (
        "differential_evolution",
        "genetic",
        "cma_es",
        "nelder_mead",
        "random_multistart",
    )
    population: int = 24
    generations: int = 8
    seed: int = 30011


@dataclass(frozen=True)
class EnsembleSettings:
    models: tuple[str, ...] = ("schaefer", "fox", "pella")
    search_draws: int = 220
    projection_years: int = 20
    projection_iterations: int = 240
    seed: int = 23017


def _copy_dataset(dataset: StockDataset, frame: pd.DataFrame, suffix: str) -> StockDataset:
    return StockDataset(
        name=f"{dataset.name} {suffix}",
        frame=frame.reset_index(drop=True).copy(),
        provenance=dict(dataset.provenance),
        transformations=list(dataset.transformations),
        warnings=list(dataset.warnings),
        raw_columns=list(dataset.raw_columns),
        index_columns=list(dataset.index_columns),
    )


def _profile_q(index: np.ndarray, biomass: np.ndarray) -> float:
    mask = np.isfinite(index) & (index > 0) & np.isfinite(biomass) & (biomass > 0)
    if not mask.any():
        return float("nan")
    return float(exp(np.mean(np.log(index[mask]) - np.log(biomass[mask]))))


def _forward_biomass(
    start_biomass: float,
    catches: np.ndarray,
    k: float,
    r: float,
    model: str,
    pella_shape: float,
) -> np.ndarray:
    rows = np.empty(len(catches), dtype=float)
    biomass = max(float(start_biomass), 1e-9)
    for index, catch in enumerate(catches):
        biomass = max(
            1e-6 * k,
            biomass + _production(biomass, k, r, model, pella_shape) - max(float(catch), 0.0),
        )
        rows[index] = biomass
    return rows


def run_walk_forward_validation(
    dataset: StockDataset,
    base_settings: ModelSettings | None = None,
    settings: WalkForwardSettings | None = None,
) -> dict[str, Any]:
    base = base_settings or ModelSettings()
    config = settings or WalkForwardSettings()
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    minimum = max(5, int(config.minimum_training_years))
    holdout = max(1, int(config.holdout_years))
    if len(frame) < minimum + holdout:
        return {
            "summary": {
                "status": "insufficient_data",
                "rows": int(len(frame)),
                "minimum_required": minimum + holdout,
            },
            "folds": [],
            "predictions": [],
        }

    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    fold_number = 0
    for train_end in range(minimum, len(frame) - holdout + 1):
        fold_number += 1
        train = frame.iloc[:train_end].copy()
        test = frame.iloc[train_end : train_end + holdout].copy()
        train_dataset = _copy_dataset(dataset, train, f"walk-forward fold {fold_number}")
        fold_settings = replace(
            base,
            search_draws=max(120, int(config.search_draws)),
            seed=int(config.seed) + fold_number * 997,
        )
        fitted = fit(train_dataset, fold_settings)
        k = float(fitted.best["k_b0"])
        r = float(fitted.best["r"])
        model = str(fitted.settings.get("model", "schaefer"))
        pella_shape = float(fitted.settings.get("pella_shape", 1.35))
        train_index = train["index"].to_numpy(dtype=float)
        train_biomass = np.array([row["biomass"] for row in fitted.history], dtype=float)
        q = _profile_q(train_index, train_biomass)
        future_biomass = _forward_biomass(
            float(fitted.best["terminal_biomass"]),
            test["catch"].to_numpy(dtype=float) * float(fold_settings.catch_multiplier),
            k,
            r,
            model,
            pella_shape,
        )
        observed_index = test["index"].to_numpy(dtype=float)
        observed_biomass = test["biomass"].to_numpy(dtype=float)
        predicted_index = q * future_biomass if np.isfinite(q) else np.full(len(test), np.nan)

        fold_index_errors: list[float] = []
        fold_biomass_errors: list[float] = []
        for local_index, (_, row) in enumerate(test.iterrows()):
            idx_obs = float(observed_index[local_index])
            idx_pred = float(predicted_index[local_index])
            bio_obs = float(observed_biomass[local_index])
            bio_pred = float(future_biomass[local_index])
            index_log_error = (
                float(log(idx_pred) - log(idx_obs))
                if np.isfinite(idx_obs) and idx_obs > 0 and np.isfinite(idx_pred) and idx_pred > 0
                else float("nan")
            )
            biomass_relative_error = (
                float((bio_pred - bio_obs) / bio_obs)
                if np.isfinite(bio_obs) and bio_obs > 0
                else float("nan")
            )
            if np.isfinite(index_log_error):
                fold_index_errors.append(index_log_error)
            if np.isfinite(biomass_relative_error):
                fold_biomass_errors.append(biomass_relative_error)
            predictions.append(
                {
                    "fold": fold_number,
                    "train_end_year": int(train["year"].iloc[-1]),
                    "prediction_year": int(row["year"]),
                    "catch": float(row["catch"]),
                    "observed_index": idx_obs if np.isfinite(idx_obs) else None,
                    "predicted_index": idx_pred if np.isfinite(idx_pred) else None,
                    "index_log_error": index_log_error if np.isfinite(index_log_error) else None,
                    "observed_biomass": bio_obs if np.isfinite(bio_obs) else None,
                    "predicted_biomass": bio_pred,
                    "biomass_relative_error": biomass_relative_error if np.isfinite(biomass_relative_error) else None,
                    "predicted_depletion": float(bio_pred / k),
                }
            )
        folds.append(
            {
                "fold": fold_number,
                "train_start_year": int(train["year"].iloc[0]),
                "train_end_year": int(train["year"].iloc[-1]),
                "test_start_year": int(test["year"].iloc[0]),
                "test_end_year": int(test["year"].iloc[-1]),
                "objective": float(fitted.best["objective"]),
                "terminal_depletion_at_origin": float(fitted.best["terminal_depletion"]),
                "index_log_rmse": float(np.sqrt(np.mean(np.square(fold_index_errors)))) if fold_index_errors else None,
                "index_log_bias": float(np.mean(fold_index_errors)) if fold_index_errors else None,
                "biomass_relative_rmse": float(np.sqrt(np.mean(np.square(fold_biomass_errors)))) if fold_biomass_errors else None,
                "biomass_relative_bias": float(np.mean(fold_biomass_errors)) if fold_biomass_errors else None,
            }
        )

    index_errors = np.array(
        [row["index_log_error"] for row in predictions if row.get("index_log_error") is not None],
        dtype=float,
    )
    biomass_errors = np.array(
        [row["biomass_relative_error"] for row in predictions if row.get("biomass_relative_error") is not None],
        dtype=float,
    )
    depletion_origins = np.array([row["terminal_depletion_at_origin"] for row in folds], dtype=float)
    return {
        "summary": {
            "status": "completed",
            "folds": len(folds),
            "predictions": len(predictions),
            "minimum_training_years": minimum,
            "holdout_years": holdout,
            "index_log_rmse": float(np.sqrt(np.mean(index_errors**2))) if len(index_errors) else None,
            "index_log_bias": float(np.mean(index_errors)) if len(index_errors) else None,
            "biomass_relative_rmse": float(np.sqrt(np.mean(biomass_errors**2))) if len(biomass_errors) else None,
            "biomass_relative_bias": float(np.mean(biomass_errors)) if len(biomass_errors) else None,
            "origin_depletion_range": float(np.ptp(depletion_origins)) if len(depletion_origins) else None,
            "interpretation": (
                "Walk-forward validation fits only earlier years and predicts later observations. "
                "Large holdout errors or unstable origin depletion indicate weak predictive performance."
            ),
        },
        "settings": asdict(config),
        "folds": folds,
        "predictions": predictions,
    }


def run_optimizer_agreement(
    dataset: StockDataset,
    base_settings: ModelSettings | None = None,
    settings: OptimizerAgreementSettings | None = None,
) -> dict[str, Any]:
    base = base_settings or ModelSettings()
    config = settings or OptimizerAgreementSettings()
    results = []
    for index, algorithm in enumerate(config.algorithms):
        output = run_global_optimizer(
            dataset,
            base,
            QuantOptimizerSettings(
                algorithm=algorithm,
                population=max(12, int(config.population)),
                generations=max(1, int(config.generations)),
                seed=int(config.seed) + index * 4001,
                local_refinement_rounds=3,
            ),
        )
        best = output["candidates"][0]
        results.append(
            {
                "algorithm": algorithm,
                "objective": float(best["objective"]),
                "terminal_depletion": float(best["terminal_depletion"]),
                "k": float(best["k"]),
                "r": float(best["r"]),
                "initial_depletion": float(best["initial_depletion"]),
                "sigma": float(best["sigma"]),
                "index_weight": float(best["index_weight"]),
                "biomass_weight": float(best["biomass_weight"]),
                "catch_multiplier": float(best["catch_multiplier"]),
                "pella_shape": float(best["pella_shape"]),
                "identifiability_status": output["diagnostics"]["local_identifiability"]["status"],
            }
        )
    objectives = np.array([row["objective"] for row in results], dtype=float)
    best_objective = float(np.min(objectives))
    for row in results:
        row["objective_delta"] = float(row["objective"] - best_objective)
    parameter_names = [
        "terminal_depletion",
        "k",
        "r",
        "initial_depletion",
        "sigma",
        "index_weight",
        "biomass_weight",
        "catch_multiplier",
        "pella_shape",
    ]
    agreement = []
    for name in parameter_names:
        values = np.array([float(row[name]) for row in results], dtype=float)
        mean = float(np.mean(values))
        cv = float(np.std(values) / max(abs(mean), 1e-12))
        agreement.append(
            {
                "quantity": name,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "range": float(np.ptp(values)),
                "mean": mean,
                "coefficient_of_variation": cv,
            }
        )
    terminal_cv = next(row["coefficient_of_variation"] for row in agreement if row["quantity"] == "terminal_depletion")
    objective_spread = float(np.max(objectives) - np.min(objectives))
    status = "strong_agreement"
    if terminal_cv > 0.30 or objective_spread > 20:
        status = "weak_agreement"
    elif terminal_cv > 0.15 or objective_spread > 8:
        status = "moderate_agreement"
    return {
        "summary": {
            "status": status,
            "algorithms": len(results),
            "best_objective": best_objective,
            "objective_spread": objective_spread,
            "terminal_depletion_cv": terminal_cv,
            "interpretation": (
                "Material disagreement between independent optimisers is evidence that the surface is difficult, "
                "multi-modal, flat or insufficiently searched. Agreement is necessary but not sufficient for validity."
            ),
        },
        "settings": asdict(config),
        "runs": sorted(results, key=lambda row: row["objective"]),
        "agreement": agreement,
    }


def projection_risk_metrics(projection: dict[str, Any], target_probability: float = 0.5) -> dict[str, Any]:
    rows = projection.get("projection") or []
    if not rows:
        return {"summary": {"status": "no_projection"}, "yearly": []}
    catch = np.array([float(row.get("catch_median", 0.0)) for row in rows], dtype=float)
    depletion = np.array([float(row.get("depletion_median", np.nan)) for row in rows], dtype=float)
    depletion_p10 = np.array([float(row.get("depletion_p10", np.nan)) for row in rows], dtype=float)
    risk = np.array([float(row.get("prob_below_limit", 0.0)) for row in rows], dtype=float)
    expected_shortfall = np.array([float(row.get("expected_limit_shortfall", 0.0)) for row in rows], dtype=float)
    target_probability_rows = np.array([float(row.get("prob_above_target", 0.0)) for row in rows], dtype=float)
    changes = np.diff(catch)
    catch_volatility = float(np.std(changes) / max(np.mean(catch), 1e-12)) if len(changes) else 0.0
    running_peak = np.maximum.accumulate(depletion)
    drawdown = running_peak - depletion
    rebuilding_year = None
    qualifying = np.where(target_probability_rows >= float(target_probability))[0]
    if len(qualifying):
        rebuilding_year = int(rows[int(qualifying[0])]["year"])
    risk_adjusted_yield = float(np.mean(catch) * (1.0 - np.max(risk)) / (1.0 + catch_volatility))
    yearly = []
    for index, row in enumerate(rows):
        yearly.append(
            {
                "year": int(row["year"]),
                "catch_median": float(catch[index]),
                "depletion_median": float(depletion[index]),
                "depletion_downside_p10": float(depletion_p10[index]),
                "probability_below_limit": float(risk[index]),
                "expected_limit_shortfall": float(expected_shortfall[index]),
                "depletion_drawdown": float(drawdown[index]),
                "probability_above_target": float(target_probability_rows[index]),
            }
        )
    return {
        "summary": {
            "status": "completed",
            "cumulative_median_catch": float(np.sum(catch)),
            "mean_median_catch": float(np.mean(catch)),
            "catch_volatility": catch_volatility,
            "maximum_probability_below_limit": float(np.max(risk)),
            "terminal_probability_below_limit": float(risk[-1]),
            "mean_expected_limit_shortfall": float(np.mean(expected_shortfall)),
            "maximum_depletion_drawdown": float(np.max(drawdown)),
            "terminal_downside_depletion_p10": float(depletion_p10[-1]),
            "rebuilding_year_at_probability": rebuilding_year,
            "rebuilding_probability_threshold": float(target_probability),
            "risk_adjusted_yield_index": risk_adjusted_yield,
            "warning": (
                "The p10 depletion is a simulation quantile, not financial VaR. Expected shortfall here is the "
                "average biological-limit deficit conditional on a simulated breach."
            ),
        },
        "yearly": yearly,
    }


def run_model_ensemble(
    dataset: StockDataset,
    base_settings: ModelSettings | None = None,
    settings: EnsembleSettings | None = None,
) -> dict[str, Any]:
    base = base_settings or ModelSettings()
    config = settings or EnsembleSettings()
    model_rows = []
    projections: dict[str, dict[str, Any]] = {}
    for index, model in enumerate(config.models):
        model_settings = replace(
            base,
            model=model,
            search_draws=max(120, int(config.search_draws)),
            seed=int(config.seed) + index * 701,
        )
        fitted = fit(dataset, model_settings)
        projection_settings = ProjectionSettings(
            years=max(1, int(config.projection_years)),
            iterations=max(40, int(config.projection_iterations)),
            process_cv=float(model_settings.process_cv),
            seed=int(config.seed) + index * 1709,
        )
        projected = project(fitted, projection_settings)
        risk = projection_risk_metrics(projected)
        projections[model] = projected
        model_rows.append(
            {
                "model": model,
                "objective": float(fitted.best["objective"]),
                "terminal_depletion": float(fitted.best["terminal_depletion"]),
                "msy": float(fitted.best["msy"]),
                "k": float(fitted.best["k_b0"]),
                "r": float(fitted.best["r"]),
                "projection_terminal_depletion": float(projected["projection"][-1]["depletion_median"]),
                "projection_terminal_limit_risk": float(projected["projection"][-1]["prob_below_limit"]),
                "risk_adjusted_yield_index": float(risk["summary"]["risk_adjusted_yield_index"]),
                "fit": fitted,
                "risk": risk,
            }
        )
    objective = np.array([row["objective"] for row in model_rows], dtype=float)
    delta = objective - float(np.min(objective))
    weights = np.exp(-0.5 * np.clip(delta, 0.0, 1400.0))
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        weights = np.ones(len(weights), dtype=float)
    weights /= weights.sum()
    for row, weight, value in zip(model_rows, weights, delta):
        row["candidate_weight"] = float(weight)
        row["objective_delta"] = float(value)

    combined_rows = []
    years = [int(row["year"]) for row in next(iter(projections.values()))["projection"]]
    for position, year in enumerate(years):
        model_year_rows = [projections[row["model"]]["projection"][position] for row in model_rows]
        dep = np.array([float(row["depletion_median"]) for row in model_year_rows], dtype=float)
        catch = np.array([float(row["catch_median"]) for row in model_year_rows], dtype=float)
        risk = np.array([float(row["prob_below_limit"]) for row in model_year_rows], dtype=float)
        combined_rows.append(
            {
                "year": year,
                "candidate_weighted_depletion": float(np.sum(weights * dep)),
                "minimum_model_depletion": float(np.min(dep)),
                "maximum_model_depletion": float(np.max(dep)),
                "model_depletion_range": float(np.ptp(dep)),
                "candidate_weighted_catch": float(np.sum(weights * catch)),
                "candidate_weighted_limit_risk": float(np.sum(weights * risk)),
            }
        )

    terminal_depletion = np.array([row["projection_terminal_depletion"] for row in model_rows], dtype=float)
    terminal_risk = np.array([row["projection_terminal_limit_risk"] for row in model_rows], dtype=float)
    public_rows = []
    for row in model_rows:
        public_rows.append({key: value for key, value in row.items() if key not in {"fit", "risk"}})
    return {
        "summary": {
            "models": len(model_rows),
            "terminal_depletion_range": float(np.ptp(terminal_depletion)),
            "terminal_limit_risk_range": float(np.ptp(terminal_risk)),
            "candidate_weighted_terminal_depletion": float(np.sum(weights * terminal_depletion)),
            "candidate_weighted_terminal_limit_risk": float(np.sum(weights * terminal_risk)),
            "weighting_method": (
                "relative objective weights exp(-0.5 * objective delta); candidate comparison only, "
                "not Bayesian model probabilities"
            ),
            "interpretation": (
                "Large between-model ranges mean the conclusion depends materially on production-model structure."
            ),
        },
        "settings": asdict(config),
        "models": public_rows,
        "combined_projection": combined_rows,
        "model_projections": projections,
        "model_risk": {row["model"]: row["risk"] for row in model_rows},
    }
