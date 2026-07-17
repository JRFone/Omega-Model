from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log, pi
from typing import Any

import numpy as np

from .data_io import StockDataset


@dataclass(frozen=True)
class ModelSettings:
    model: str = "schaefer"
    search_draws: int = 900
    seed: int = 4107
    target_depletion: float | None = None
    target_depletion_cv: float = 0.25
    r_prior_median: float = 0.18
    r_prior_cv: float = 0.75
    obs_cv: float = 0.22
    process_cv: float = 0.12
    index_weight: float = 1.0
    biomass_weight: float = 1.0
    catch_multiplier: float = 1.0
    pella_shape: float = 1.35
    initial_depletion_prior_mean: float = 0.85
    initial_depletion_prior_sd: float = 0.30
    initial_depletion_prior_weight: float = 0.04
    catch_to_capacity_penalty_weight: float = 0.08
    observation_prior_log_sd: float = 0.75


@dataclass(frozen=True)
class ProjectionSettings:
    years: int = 20
    iterations: int = 800
    strategy: str = "hcr_40_10"
    fixed_catch: float = 250.0
    fixed_f: float = 0.08
    target_depletion: float = 0.40
    limit_depletion: float = 0.10
    pstar: float = 0.45
    process_cv: float = 0.12
    seed: int = 9101
    maximum_exploitation_fraction: float = 0.85


@dataclass
class FitResult:
    name: str
    settings: dict[str, Any]
    best: dict[str, Any]
    diagnostics: dict[str, Any]
    history: list[dict[str, float]]
    ensemble: list[dict[str, float]]


def _canonical_model(model: str) -> str:
    value = str(model or "schaefer").strip().lower()
    if value.startswith("state_space_"):
        value = value.removeprefix("state_space_")
    if value in {"pella_tomlinson", "pella-tomlinson"}:
        value = "pella"
    return value if value in {"schaefer", "fox", "pella"} else "schaefer"


def _production(biomass: float, k: float, r: float, model: str, pella_shape: float = 1.35) -> float:
    """Return deterministic non-negative surplus production for the selected model."""
    k = max(float(k), 1e-12)
    biomass = max(1e-12, min(float(biomass), k * 2.0))
    r = max(float(r), 0.0)
    model = _canonical_model(model)
    if model == "fox":
        return max(0.0, float(r * biomass * log(k / biomass)))
    if model == "pella":
        shape = max(float(pella_shape), 1e-6)
        return max(0.0, float(r * biomass * (1.0 - (biomass / k) ** shape) / shape))
    return max(0.0, float(r * biomass * (1.0 - biomass / k)))


def _reference_points(k: float, r: float, model: str, pella_shape: float = 1.35) -> dict[str, float]:
    """Analytic BMSY, MSY and FMSY for the implemented production functions."""
    k = max(float(k), 1e-12)
    r = max(float(r), 1e-12)
    model = _canonical_model(model)
    if model == "fox":
        bmsy = k / exp(1.0)
        msy = r * bmsy
    elif model == "pella":
        shape = max(float(pella_shape), 1e-6)
        bmsy = k * (1.0 / (1.0 + shape)) ** (1.0 / shape)
        msy = r * bmsy / (1.0 + shape)
    else:
        bmsy = k / 2.0
        msy = r * k / 4.0
    return {
        "bmsy": float(bmsy),
        "msy": float(msy),
        "fmsy": float(msy / max(bmsy, 1e-12)),
    }


def _initial_biomass(k: float, b0_frac: float) -> float:
    value = float(k) * float(b0_frac)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Initial biomass must be finite and positive.")
    return value


def _simulate(
    years: np.ndarray,
    catches: np.ndarray,
    k: float,
    r: float,
    b0_frac: float,
    model: str,
    pella_shape: float = 1.35,
) -> np.ndarray:
    biomass = np.empty(len(years), dtype=float)
    # Do not silently replace the requested initial state with first-year catch.
    biomass[0] = _initial_biomass(k, b0_frac)
    for i in range(1, len(years)):
        previous = biomass[i - 1]
        biomass[i] = max(
            1e-6 * k,
            previous + _production(previous, k, r, model, pella_shape) - catches[i - 1],
        )
    return biomass


def _lognormal_nll(obs: np.ndarray, pred: np.ndarray, sigma: float) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if not mask.any():
        return 0.0
    sigma = max(float(sigma), 1e-12)
    resid = np.log(obs[mask]) - np.log(pred[mask])
    return float(np.sum(0.5 * (resid / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2.0 * pi)))


def _objective_breakdown(
    theta: np.ndarray,
    years: np.ndarray,
    catches: np.ndarray,
    index: np.ndarray,
    biomass_obs: np.ndarray,
    settings: ModelSettings,
) -> tuple[float, np.ndarray, float, dict[str, float]]:
    log_k, log_r, logit_b0, log_sigma = theta
    k = exp(log_k)
    r = exp(log_r)
    b0_frac = 1.0 / (1.0 + exp(-logit_b0))
    sigma = max(0.03, min(exp(log_sigma), 1.5))
    components = {
        "index_likelihood": 0.0,
        "biomass_likelihood": 0.0,
        "terminal_depletion_constraint": 0.0,
        "observation_error_prior": 0.0,
        "productivity_prior": 0.0,
        "initial_depletion_prior": 0.0,
        "catch_to_capacity_penalty": 0.0,
    }
    if k <= max(catches) * 1.05 or r <= 0.005 or r > 1.2:
        components["invalid_parameter_penalty"] = 1e18
        return 1e18, np.zeros_like(catches), 1.0, components
    try:
        pred_b = _simulate(years, catches, k, r, b0_frac, settings.model, settings.pella_shape)
    except ValueError:
        components["invalid_parameter_penalty"] = 1e18
        return 1e18, np.zeros_like(catches), sigma, components
    if not np.all(np.isfinite(pred_b)):
        components["invalid_parameter_penalty"] = 1e18
        return 1e18, pred_b, sigma, components

    idx_mask = np.isfinite(index) & (index > 0)
    if idx_mask.any():
        q = exp(float(np.mean(np.log(index[idx_mask]) - np.log(pred_b[idx_mask]))))
        components["index_likelihood"] = max(settings.index_weight, 0.0) * _lognormal_nll(index, q * pred_b, sigma)
    bio_mask = np.isfinite(biomass_obs) & (biomass_obs > 0)
    if bio_mask.any():
        components["biomass_likelihood"] = max(settings.biomass_weight, 0.0) * _lognormal_nll(
            biomass_obs,
            pred_b,
            max(sigma, 0.10),
        )
    if settings.target_depletion is not None:
        pred_dep = max(pred_b[-1] / k, 1e-6)
        target = max(float(settings.target_depletion), 1e-6)
        sd = max(settings.target_depletion_cv, 0.05)
        components["terminal_depletion_constraint"] = 0.5 * ((log(pred_dep) - log(target)) / sd) ** 2

    components["observation_error_prior"] = 0.5 * (
        (log(sigma) - log(max(settings.obs_cv, 0.03))) / max(settings.observation_prior_log_sd, 1e-6)
    ) ** 2
    r_sd = max((log(1 + settings.r_prior_cv**2)) ** 0.5, 1e-6)
    components["productivity_prior"] = 0.5 * ((log(r) - log(max(settings.r_prior_median, 1e-6))) / r_sd) ** 2
    components["catch_to_capacity_penalty"] = max(settings.catch_to_capacity_penalty_weight, 0.0) * (
        max(catches) / max(k, 1e-9)
    ) ** 2
    components["initial_depletion_prior"] = max(settings.initial_depletion_prior_weight, 0.0) * (
        (b0_frac - settings.initial_depletion_prior_mean) / max(settings.initial_depletion_prior_sd, 1e-6)
    ) ** 2
    nll = float(sum(components.values()))
    return nll, pred_b, sigma, {key: float(value) for key, value in components.items()}


def _objective(
    theta: np.ndarray,
    years: np.ndarray,
    catches: np.ndarray,
    index: np.ndarray,
    biomass_obs: np.ndarray,
    settings: ModelSettings,
) -> tuple[float, np.ndarray, float]:
    nll, pred_b, sigma, _components = _objective_breakdown(theta, years, catches, index, biomass_obs, settings)
    return nll, pred_b, sigma


def fit(dataset: StockDataset, settings: ModelSettings | None = None) -> FitResult:
    settings = settings or ModelSettings()
    df = dataset.frame
    years = df["year"].to_numpy(dtype=int)
    catches = df["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)
    index = df["index"].to_numpy(dtype=float)
    biomass_obs = df["biomass"].to_numpy(dtype=float)
    rng = np.random.default_rng(settings.seed)

    max_catch = max(float(np.nanmax(catches)), 1.0)
    total_catch = max(float(np.nansum(catches)), max_catch)
    k_low = max(max_catch * 3.0, total_catch * 0.25)
    k_high = max(k_low * 5.0, total_catch * 12.0)
    draws = int(max(120, settings.search_draws))
    candidates = np.column_stack(
        [
            rng.uniform(log(k_low), log(k_high), draws),
            rng.uniform(log(0.025), log(0.65), draws),
            rng.uniform(-2.1, 3.0, draws),
            rng.uniform(log(0.08), log(0.75), draws),
        ]
    )
    # Score the broad search as one native batch when the compiled backend is
    # available. The Python implementation remains the parity-tested fallback.
    scoring_backend = "python"
    try:
        from .native_backend import get_native_engine

        native_engine = get_native_engine()
        objective_values, _batch_gradients, scoring_backend = native_engine.batch_objective(
            candidates,
            years,
            catches,
            index,
            biomass_obs,
            settings,
            gradients=False,
        )
    except Exception:
        native_engine = None
        objective_values = np.asarray(
            [_objective(theta, years, catches, index, biomass_obs, settings)[0] for theta in candidates],
            dtype=float,
        )
        scoring_backend = "python-fallback"

    scored: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for nll, theta in zip(objective_values, candidates):
        _check_nll, pred_b, sigma = _objective(theta, years, catches, index, biomass_obs, settings)
        scored.append((float(nll), theta, pred_b, sigma))
    scored.sort(key=lambda row: row[0])

    refined: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    refinement_backend = "coordinate-search"
    bounds = [
        (log(k_low), log(k_high)),
        (log(0.0051), log(1.199)),
        (-6.0, 6.0),
        (log(0.03), log(1.5)),
    ]
    try:
        from scipy.optimize import minimize

        for _, start, _, _ in scored[:18]:
            def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
                if native_engine is not None and native_engine.available:
                    result = native_engine.objective_gradient(theta, years, catches, index, biomass_obs, settings)
                    return result.objective, result.gradient
                value = _objective(theta, years, catches, index, biomass_obs, settings)[0]
                gradient = np.empty(4, dtype=float)
                for position in range(4):
                    step = 1e-5 * max(1.0, abs(float(theta[position])))
                    plus = np.asarray(theta, dtype=float).copy(); plus[position] += step
                    minus = np.asarray(theta, dtype=float).copy(); minus[position] -= step
                    gradient[position] = (
                        _objective(plus, years, catches, index, biomass_obs, settings)[0]
                        - _objective(minus, years, catches, index, biomass_obs, settings)[0]
                    ) / (2.0 * step)
                return float(value), gradient

            result = minimize(
                lambda value: fun(value)[0],
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                jac=lambda value: fun(value)[1],
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
            )
            theta = np.asarray(result.x, dtype=float)
            nll, pred_b, sigma = _objective(theta, years, catches, index, biomass_obs, settings)
            refined.append((float(nll), theta, pred_b, sigma))
        refinement_backend = "scipy-L-BFGS-B+native-AD" if native_engine is not None and native_engine.available else "scipy-L-BFGS-B+finite-difference"
    except Exception:
        for _, start, _, _ in scored[:18]:
            theta = start.copy()
            steps = np.array([0.35, 0.22, 0.35, 0.16])
            best_nll, best_b, best_sigma = _objective(theta, years, catches, index, biomass_obs, settings)
            for _round in range(8):
                improved = False
                for j in range(len(theta)):
                    for direction in (-1.0, 1.0):
                        trial = theta.copy()
                        trial[j] += steps[j] * direction
                        nll, pred_b, sigma = _objective(trial, years, catches, index, biomass_obs, settings)
                        if nll < best_nll:
                            theta, best_nll, best_b, best_sigma = trial, nll, pred_b, sigma
                            improved = True
                if not improved:
                    steps *= 0.58
            refined.append((best_nll, theta, best_b, best_sigma))
    refined.sort(key=lambda row: row[0])
    best_nll, best_theta, best_b, best_sigma = refined[0]
    log_k, log_r, logit_b0, _ = best_theta
    k = exp(log_k)
    r = exp(log_r)
    b0_frac = 1.0 / (1.0 + exp(-logit_b0))
    reference = _reference_points(k, r, settings.model, settings.pella_shape)
    _, _, _, objective_components = _objective_breakdown(best_theta, years, catches, index, biomass_obs, settings)

    ensemble: list[dict[str, float]] = []
    keep = refined[: min(30, len(refined))]
    base = keep[0][0]
    weights = np.exp(-0.5 * np.array([row[0] - base for row in keep]))
    weights = weights / weights.sum()
    for weight, (nll, theta, pred_b, sigma) in zip(weights, keep):
        lk, lr, lb, _ls = theta
        member_k = float(exp(lk))
        member_r = float(exp(lr))
        member_ref = _reference_points(member_k, member_r, settings.model, settings.pella_shape)
        ensemble.append(
            {
                "weight": float(weight),
                "nll": float(nll),
                "k": member_k,
                "r": member_r,
                "b0_frac": float(1 / (1 + exp(-lb))),
                "sigma": float(sigma),
                "terminal_biomass": float(pred_b[-1]),
                "terminal_depletion": float(pred_b[-1] / member_k),
                "msy": member_ref["msy"],
                "bmsy": member_ref["bmsy"],
                "fmsy": member_ref["fmsy"],
            }
        )

    history = [
        {"year": int(y), "catch": float(c), "biomass": float(b), "depletion": float(b / k)}
        for y, c, b in zip(years, catches, best_b)
    ]
    best: dict[str, Any] = {
        "k_b0": float(k),
        "r": float(r),
        "initial_depletion": float(b0_frac),
        "initial_biomass": float(best_b[0]),
        "terminal_biomass": float(best_b[-1]),
        "terminal_depletion": float(best_b[-1] / k),
        "msy": reference["msy"],
        "bmsy": reference["bmsy"],
        "fmsy": reference["fmsy"],
        "sigma": float(best_sigma),
        "objective": float(best_nll),
    }
    diagnostics: dict[str, Any] = {
        "years": float(len(years)),
        "index_points": float(np.isfinite(index).sum()),
        "biomass_points": float(np.isfinite(biomass_obs).sum()),
        "total_catch": float(np.sum(catches)),
        "max_catch": float(np.max(catches)),
        "initial_catch_fraction_of_biomass": float(catches[0] / max(best_b[0], 1e-12)),
        "initial_state_overridden": False,
        "objective_components": objective_components,
        "scoring_backend": scoring_backend,
        "refinement_backend": refinement_backend,
        "native_backend_available": bool(native_engine is not None and getattr(native_engine, "available", False)),
        "uncertainty_method": "candidate-fit spread; not a posterior, bootstrap interval, or formal confidence interval",
    }
    return FitResult(dataset.name, asdict(settings), best, diagnostics, history, ensemble)


def project(fit_result: FitResult, settings: ProjectionSettings | None = None) -> dict[str, Any]:
    settings = settings or ProjectionSettings()
    rng = np.random.default_rng(settings.seed)
    last_year = int(fit_result.history[-1]["year"])
    best_k = max(float(fit_result.best["k_b0"]), 1e-12)
    best_terminal_depletion = float(fit_result.best.get("terminal_depletion", fit_result.history[-1]["biomass"] / best_k))
    ensemble = fit_result.ensemble or [
        {
            "weight": 1.0,
            "k": fit_result.best["k_b0"],
            "r": fit_result.best["r"],
            "sigma": fit_result.best["sigma"],
            "terminal_biomass": fit_result.best["terminal_biomass"],
            "terminal_depletion": best_terminal_depletion,
            **_reference_points(
                fit_result.best["k_b0"],
                fit_result.best["r"],
                fit_result.settings.get("model", "schaefer"),
                fit_result.settings.get("pella_shape", 1.35),
            ),
        }
    ]
    weights = np.array([max(float(e.get("weight", 0.0)), 0.0) for e in ensemble], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(ensemble), dtype=float)
    weights = weights / weights.sum()
    choices = rng.choice(len(ensemble), size=max(1, int(settings.iterations)), p=weights)

    years = np.arange(last_year + 1, last_year + max(1, int(settings.years)) + 1)
    biomass = np.empty((len(choices), len(years)))
    catches = np.empty_like(biomass)
    depletion = np.empty_like(biomass)
    member_k = np.empty(len(choices), dtype=float)
    model = fit_result.settings.get("model", "schaefer")
    pella_shape = float(fit_result.settings.get("pella_shape", 1.35))
    for i, choice in enumerate(choices):
        e = ensemble[int(choice)]
        k = max(float(e["k"]), 1e-9)
        r = max(float(e["r"]), 1e-9)
        reference = _reference_points(k, r, model, pella_shape)
        member_terminal_depletion = float(e.get("terminal_depletion", best_terminal_depletion))
        terminal_biomass = float(e.get("terminal_biomass", member_terminal_depletion * k))
        b = max(1e-6 * k, terminal_biomass * rng.lognormal(0.0, 0.08))
        member_k[i] = k
        for t in range(len(years)):
            dep = b / k
            if settings.strategy == "fixed_f":
                catch = settings.fixed_f * b
            elif settings.strategy == "fixed_catch":
                catch = settings.fixed_catch
            else:
                ramp = min(
                    1.0,
                    max(
                        0.0,
                        (dep - settings.limit_depletion)
                        / max(settings.target_depletion - settings.limit_depletion, 1e-9),
                    ),
                )
                # Use the selected production model's MSY, not Schaefer rK/4.
                catch = reference["msy"] * ramp * settings.pstar / 0.5
            catch = min(max(catch, 0.0), b * max(min(settings.maximum_exploitation_fraction, 1.0), 0.0))
            catches[i, t] = catch
            process = rng.lognormal(-0.5 * settings.process_cv**2, settings.process_cv)
            b = max(1e-6 * k, (b + _production(b, k, r, model, pella_shape) - catch) * process)
            biomass[i, t] = b
            depletion[i, t] = b / k

    rows = []
    for j, year in enumerate(years):
        catch_column = catches[:, j]
        dep_column = depletion[:, j]
        biomass_column = biomass[:, j]
        below_limit = dep_column < settings.limit_depletion
        shortfall = np.maximum(settings.limit_depletion - dep_column, 0.0)
        rows.append(
            {
                "year": int(year),
                "biomass_p10": float(np.quantile(biomass_column, 0.10)),
                "biomass_median": float(np.quantile(biomass_column, 0.50)),
                "biomass_p90": float(np.quantile(biomass_column, 0.90)),
                "catch_p10": float(np.quantile(catch_column, 0.10)),
                "catch_median": float(np.quantile(catch_column, 0.50)),
                "catch_p90": float(np.quantile(catch_column, 0.90)),
                "depletion_p10": float(np.quantile(dep_column, 0.10)),
                "depletion_median": float(np.quantile(dep_column, 0.50)),
                "depletion_p90": float(np.quantile(dep_column, 0.90)),
                "prob_above_target": float(np.mean(dep_column >= settings.target_depletion)),
                "prob_above_limit": float(np.mean(dep_column >= settings.limit_depletion)),
                "prob_below_limit": float(np.mean(below_limit)),
                "expected_limit_shortfall": float(np.mean(shortfall[below_limit])) if below_limit.any() else 0.0,
            }
        )
    return {
        "settings": asdict(settings),
        "projection": rows,
        "uncertainty_label": "candidate-fit spread plus process simulation; not a posterior distribution",
        "depletion_denominator": "each simulation member's own K",
        "sampled_k_p10": float(np.quantile(member_k, 0.10)),
        "sampled_k_median": float(np.quantile(member_k, 0.50)),
        "sampled_k_p90": float(np.quantile(member_k, 0.90)),
    }
