from __future__ import annotations

from dataclasses import asdict, dataclass
from math import lgamma, log
from typing import Any, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class TagRelease:
    release_year: int
    area: int
    age: int
    number: float
    sex: int = 0


@dataclass(frozen=True)
class TagObservation:
    release_id: int
    recapture_year: int
    fleet: int
    area: int
    observed: int


@dataclass(frozen=True)
class TaggingSettings:
    natural_mortality: float = 0.12
    tag_loss: float = 0.03
    tag_induced_mortality: float = 0.02
    initial_mixing_survival: float = 0.95
    reporting_rates: tuple[float, ...] = (0.8,)
    fleet_capture_rates: tuple[float, ...] = (0.10,)
    overdispersion: float = 50.0


@dataclass
class TaggingResult:
    predictions: list[dict[str, float | int]]
    objective: float
    diagnostics: dict[str, Any]


def validate_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Tag movement matrix must be square.")
    if np.any(value < 0):
        raise ValueError("Tag movement probabilities cannot be negative.")
    sums = value.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Each tag movement origin must have positive probability.")
    return value / sums


def _binomial_nll(observed: int, total: float, probability: float) -> float:
    n = max(int(round(total)), observed)
    p = float(np.clip(probability, 1e-10, 1 - 1e-10))
    return -(lgamma(n + 1) - lgamma(observed + 1) - lgamma(n - observed + 1) + observed * log(p) + (n - observed) * log(1 - p))


def predict_tag_recaptures(
    releases: Sequence[TagRelease],
    observations: Sequence[TagObservation],
    movement: np.ndarray,
    settings: TaggingSettings | None = None,
) -> TaggingResult:
    settings = settings or TaggingSettings()
    move = validate_transition_matrix(movement)
    if len(settings.reporting_rates) != len(settings.fleet_capture_rates):
        raise ValueError("reporting_rates and fleet_capture_rates must have the same length.")
    predictions: list[dict[str, float | int]] = []
    objective = 0.0
    by_release: dict[int, list[TagObservation]] = {}
    for obs in observations:
        by_release.setdefault(obs.release_id, []).append(obs)
    for release_id, release in enumerate(releases):
        if release.area < 0 or release.area >= move.shape[0]:
            raise ValueError("Tag release area is outside the movement matrix.")
        alive = np.zeros(move.shape[0], dtype=float)
        alive[release.area] = max(release.number, 0.0) * (1.0 - np.clip(settings.tag_induced_mortality, 0, 1)) * np.clip(settings.initial_mixing_survival, 0, 1)
        release_obs = sorted(by_release.get(release_id, []), key=lambda row: row.recapture_year)
        current_year = release.release_year
        for obs in release_obs:
            while current_year < obs.recapture_year:
                alive = alive @ move
                alive *= np.exp(-max(settings.natural_mortality, 0.0) - max(settings.tag_loss, 0.0))
                current_year += 1
            if obs.fleet < 0 or obs.fleet >= len(settings.reporting_rates):
                raise ValueError("Tag observation fleet is outside reporting-rate settings.")
            available = float(alive[obs.area])
            capture_rate = float(np.clip(settings.fleet_capture_rates[obs.fleet], 0, 1))
            report_rate = float(np.clip(settings.reporting_rates[obs.fleet], 0, 1))
            predicted = available * capture_rate * report_rate
            probability = capture_rate * report_rate
            objective += _binomial_nll(obs.observed, available, probability)
            predictions.append({
                "release_id": release_id,
                "release_year": release.release_year,
                "recapture_year": obs.recapture_year,
                "fleet": obs.fleet,
                "area": obs.area,
                "available_tags": available,
                "predicted_recaptures": predicted,
                "observed_recaptures": obs.observed,
                "reporting_rate": report_rate,
                "capture_rate": capture_rate,
            })
            alive[obs.area] = max(alive[obs.area] - available * capture_rate, 0.0)
    residuals = [float(row["observed_recaptures"]) - float(row["predicted_recaptures"]) for row in predictions]
    diagnostics = {
        "releases": len(releases),
        "observations": len(observations),
        "objective": float(objective),
        "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0,
        "settings": asdict(settings),
    }
    return TaggingResult(predictions, float(objective), diagnostics)


def estimate_reporting_rate(
    releases: Sequence[TagRelease],
    observations: Sequence[TagObservation],
    movement: np.ndarray,
    base_settings: TaggingSettings | None = None,
    grid: Sequence[float] = tuple(np.linspace(0.05, 1.0, 40)),
) -> dict[str, Any]:
    base = base_settings or TaggingSettings()
    rows = []
    best = None
    for rate in grid:
        settings = TaggingSettings(**{**asdict(base), "reporting_rates": tuple(float(rate) for _ in base.reporting_rates)})
        result = predict_tag_recaptures(releases, observations, movement, settings)
        row = {"reporting_rate": float(rate), "objective": result.objective}
        rows.append(row)
        if best is None or result.objective < best[0]:
            best = (result.objective, float(rate), result)
    assert best is not None
    return {
        "best_reporting_rate": best[1],
        "objective": best[0],
        "profile": rows,
        "result": best[2],
    }
