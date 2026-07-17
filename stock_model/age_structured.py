from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .data_io import StockDataset, read_stock_csv, read_stock_file


_EPS = 1e-12


@dataclass(frozen=True)
class SectorSettings:
    name: str
    catch_column: str | None = None
    catch_share: float = 1.0
    selectivity_a50: float = 5.0
    selectivity_slope: float = 1.25
    retention_length50_mm: float = 500.0
    retention_slope_mm: float = 35.0
    discard_mortality: float = 0.50
    fishing_power: float = 1.0


@dataclass(frozen=True)
class AgeStructuredSettings:
    max_age: int = 30
    natural_mortality: float = 0.12
    r0: float = 1_000_000.0
    steepness: float = 0.75
    recruitment_sigma: float = 0.60
    recruitment_rho: float = 0.0
    initial_depletion: float = 0.85
    linf_mm: float = 850.0
    growth_k: float = 0.13
    growth_t0: float = -0.50
    length_cv: float = 0.10
    weight_a: float = 1.0e-8
    weight_b: float = 3.05
    maturity_a50: float = 5.0
    maturity_slope: float = 1.20
    maturity_model: str = "age_logistic"
    maturity_length50_mm: float = 500.0
    maturity_slope_coefficient: float = -0.05
    female_fraction: float = 0.50
    survey_selectivity_a50: float = 4.0
    survey_selectivity_slope: float = 1.25
    index_cv: float = 0.20
    biomass_cv: float = 0.20
    age_comp_weight: float = 1.0
    length_comp_weight: float = 1.0
    m_prior_median: float = 0.12
    m_prior_cv: float = 0.50
    h_prior_mean: float = 0.75
    h_prior_sd: float = 0.15
    initial_depletion_prior: float = 0.85
    initial_depletion_prior_sd: float = 0.25
    sectors: tuple[SectorSettings, ...] = (
        SectorSettings("commercial", "catch_commercial", 0.50, 5.0, 1.2, 500.0, 35.0, 0.50, 1.0),
        SectorSettings("charter", "catch_charter", 0.15, 4.5, 1.3, 500.0, 35.0, 0.50, 1.0),
        SectorSettings("recreational", "catch_recreational", 0.35, 4.0, 1.4, 500.0, 35.0, 0.50, 1.0),
    )


@dataclass(frozen=True)
class AgeFitSettings:
    population: int = 36
    generations: int = 24
    seed: int = 8301
    mutation: float = 0.75
    crossover: float = 0.80
    local_rounds: int = 4
    estimate_natural_mortality: bool = True
    estimate_steepness: bool = True
    estimate_initial_depletion: bool = True
    estimate_survey_selectivity: bool = True
    estimate_recruitment_sigma: bool = False


@dataclass(frozen=True)
class AgeProjectionSettings:
    years: int = 20
    iterations: int = 400
    strategy: str = "hcr_40_10"
    fixed_catch: float = 250.0
    fixed_f: float = 0.08
    target_depletion: float = 0.40
    limit_depletion: float = 0.10
    pstar: float = 0.45
    recruitment_sigma: float | None = None
    implementation_cv: float = 0.10
    seed: int = 9331


@dataclass
class AgeStructuredResult:
    name: str
    settings: dict[str, Any]
    fit_settings: dict[str, Any]
    best: dict[str, float]
    diagnostics: dict[str, Any]
    history: list[dict[str, float]]
    sector_history: list[dict[str, float]]
    age_structure: list[dict[str, float]]
    predicted_age_composition: list[dict[str, float]]
    predicted_length_composition: list[dict[str, float]]
    ensemble: list[dict[str, float]]
    state: dict[str, Any]


def logistic(x: np.ndarray | float, x50: float, slope: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    scale = max(float(slope), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip((values - float(x50)) / scale, -60.0, 60.0)))


def life_history_arrays(settings: AgeStructuredSettings) -> dict[str, np.ndarray]:
    ages = np.arange(max(int(settings.max_age), 1) + 1, dtype=float)
    length = settings.linf_mm * (1.0 - np.exp(-settings.growth_k * (ages - settings.growth_t0)))
    length = np.maximum(length, 1.0)
    weight = settings.weight_a * np.power(length, settings.weight_b)
    if settings.maturity_model == "length_logistic":
        maturity = 1.0 / (1.0 + np.exp(np.clip(settings.maturity_slope_coefficient * (length - settings.maturity_length50_mm), -60.0, 60.0)))
    elif settings.maturity_model == "none":
        maturity = np.zeros_like(ages, dtype=float)
    else:
        maturity = logistic(ages, settings.maturity_a50, settings.maturity_slope)
    survey_selectivity = logistic(ages, settings.survey_selectivity_a50, settings.survey_selectivity_slope)
    return {
        "age": ages,
        "length_mm": length,
        "weight_kg": weight,
        "maturity": maturity,
        "survey_selectivity": survey_selectivity,
    }


def sector_curves(settings: AgeStructuredSettings, life_history: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, Any]]:
    life = life_history or life_history_arrays(settings)
    ages = life["age"]
    length = life["length_mm"]
    curves: dict[str, dict[str, np.ndarray]] = {}
    for sector in settings.sectors:
        selectivity = logistic(ages, sector.selectivity_a50, sector.selectivity_slope)
        retention = logistic(length, sector.retention_length50_mm, sector.retention_slope_mm)
        retained_fraction = selectivity * retention
        released_fraction = selectivity * (1.0 - retention)
        mortality_fraction = retained_fraction + released_fraction * np.clip(sector.discard_mortality, 0.0, 1.0)
        curves[sector.name] = {
            "selectivity": selectivity,
            "retention": retention,
            "retained_fraction": retained_fraction,
            "released_fraction": released_fraction,
            "mortality_fraction": mortality_fraction,
            "discard_mortality": float(np.clip(sector.discard_mortality, 0.0, 1.0)),
            "fishing_power": float(max(sector.fishing_power, 0.0)),
        }
    return curves


def _annual_sector_curves(
    settings: AgeStructuredSettings,
    life_history: dict[str, np.ndarray],
    frame: pd.DataFrame,
    row_index: int,
) -> dict[str, dict[str, Any]]:
    """Build sector curves with optional year-specific values from the dataset.

    Supported columns are suffixed with the lower-case sector name, for example
    ``retention_length50_recreational`` or
    ``discard_mortality_charter``. Missing values retain the configured base
    setting. This keeps historical regulation and fleet changes explicit in the
    data rather than silently applying one curve to every year.
    """

    ages = life_history["age"]
    length = life_history["length_mm"]

    def value(column: str, default: float) -> float:
        if column not in frame.columns:
            return float(default)
        raw = frame.iloc[row_index][column]
        return float(raw) if pd.notna(raw) else float(default)

    curves: dict[str, dict[str, Any]] = {}
    for sector in settings.sectors:
        suffix = sector.name.lower()
        selectivity_a50 = value(f"selectivity_a50_{suffix}", sector.selectivity_a50)
        selectivity_slope = value(f"selectivity_slope_{suffix}", sector.selectivity_slope)
        retention_length50 = value(f"retention_length50_{suffix}", sector.retention_length50_mm)
        retention_slope = value(f"retention_slope_{suffix}", sector.retention_slope_mm)
        discard_mortality = float(np.clip(value(f"discard_mortality_{suffix}", sector.discard_mortality), 0.0, 1.0))
        fishing_power = max(value(f"fishing_power_{suffix}", sector.fishing_power), 0.0)
        selectivity = logistic(ages, selectivity_a50, selectivity_slope)
        retention = logistic(length, retention_length50, retention_slope)
        retained_fraction = selectivity * retention
        released_fraction = selectivity * (1.0 - retention)
        curves[sector.name] = {
            "selectivity": selectivity,
            "retention": retention,
            "retained_fraction": retained_fraction,
            "released_fraction": released_fraction,
            "mortality_fraction": retained_fraction + released_fraction * discard_mortality,
            "discard_mortality": discard_mortality,
            "fishing_power": float(fishing_power),
        }
    return curves


def _unfished_numbers(settings: AgeStructuredSettings) -> np.ndarray:
    ages = np.arange(settings.max_age + 1, dtype=float)
    numbers = settings.r0 * np.exp(-max(settings.natural_mortality, 1e-6) * ages)
    if len(numbers) > 1:
        numbers[-1] /= max(1.0 - exp(-max(settings.natural_mortality, 1e-6)), 1e-9)
    return numbers


def _spawning_biomass(numbers: np.ndarray, life: dict[str, np.ndarray], settings: AgeStructuredSettings) -> float:
    return float(np.sum(numbers * life["weight_kg"] * life["maturity"] * np.clip(settings.female_fraction, 0.0, 1.0)))


def _total_biomass(numbers: np.ndarray, life: dict[str, np.ndarray]) -> float:
    return float(np.sum(numbers * life["weight_kg"]))


def _survey_biomass(numbers: np.ndarray, life: dict[str, np.ndarray]) -> float:
    return float(np.sum(numbers * life["weight_kg"] * life["survey_selectivity"]))


def _beverton_holt(ssb: float, b0: float, settings: AgeStructuredSettings) -> float:
    h = min(max(settings.steepness, 0.2001), 0.999)
    numerator = 4.0 * h * settings.r0 * max(ssb, 0.0)
    denominator = max(b0 * (1.0 - h) + max(ssb, 0.0) * (5.0 * h - 1.0), _EPS)
    return max(numerator / denominator, 1e-12)


def _normal_cdf(value: np.ndarray | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))


def _length_bin_probabilities(life: dict[str, np.ndarray], bins: np.ndarray, length_cv: float) -> np.ndarray:
    means = life["length_mm"]
    sd = np.maximum(means * max(float(length_cv), 1e-4), 1.0)
    if len(bins) == 1:
        return np.ones((len(means), 1), dtype=float)
    edges = np.empty(len(bins) + 1, dtype=float)
    edges[1:-1] = 0.5 * (bins[:-1] + bins[1:])
    edges[0] = max(0.0, bins[0] - (edges[1] - bins[0]))
    edges[-1] = bins[-1] + (bins[-1] - edges[-2])
    probabilities = np.empty((len(means), len(bins)), dtype=float)
    for age_index, (mean, sigma) in enumerate(zip(means, sd)):
        z_hi = (edges[1:] - mean) / sigma
        z_lo = (edges[:-1] - mean) / sigma
        row = _normal_cdf(z_hi) - _normal_cdf(z_lo)
        row = np.maximum(row, 0.0)
        probabilities[age_index] = row / max(float(row.sum()), _EPS)
    return probabilities


def _sector_observed_catches(frame: pd.DataFrame, row_index: int, settings: AgeStructuredSettings) -> dict[str, float]:
    total = max(float(frame.iloc[row_index]["catch"]), 0.0)
    explicit: dict[str, float] = {}
    for sector in settings.sectors:
        if sector.catch_column and sector.catch_column in frame.columns:
            value = frame.iloc[row_index][sector.catch_column]
            if pd.notna(value):
                explicit[sector.name] = max(float(value), 0.0)
    if explicit:
        explicit_total = sum(explicit.values())
        if explicit_total < total and total > 0:
            missing = [sector for sector in settings.sectors if sector.name not in explicit]
            shares = np.array([max(sector.catch_share, 0.0) for sector in missing], dtype=float)
            if missing:
                shares = shares / max(float(shares.sum()), _EPS)
                remainder = total - explicit_total
                for sector, share in zip(missing, shares):
                    explicit[sector.name] = remainder * float(share)
        return {sector.name: explicit.get(sector.name, 0.0) for sector in settings.sectors}
    shares = np.array([max(sector.catch_share, 0.0) for sector in settings.sectors], dtype=float)
    if float(shares.sum()) <= 0:
        shares[:] = 1.0
    shares /= shares.sum()
    return {sector.name: total * float(share) for sector, share in zip(settings.sectors, shares)}


def _annual_catch(
    numbers: np.ndarray,
    f_scalar: float,
    sector_shares: dict[str, float],
    settings: AgeStructuredSettings,
    life: dict[str, np.ndarray],
    curves: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    sector_dead_f: dict[str, np.ndarray] = {}
    sector_land_f: dict[str, np.ndarray] = {}
    sector_discard_f: dict[str, np.ndarray] = {}
    total_target = max(sum(sector_shares.values()), _EPS)
    for sector in settings.sectors:
        share = max(float(sector_shares.get(sector.name, 0.0)) / total_target, 0.0)
        fishing_power = max(float(curves[sector.name].get("fishing_power", sector.fishing_power)), 0.0)
        discard_mortality = float(np.clip(curves[sector.name].get("discard_mortality", sector.discard_mortality), 0.0, 1.0))
        encounter_f = max(float(f_scalar), 0.0) * share * fishing_power * curves[sector.name]["selectivity"]
        land_f = encounter_f * curves[sector.name]["retention"]
        discard_f = encounter_f * (1.0 - curves[sector.name]["retention"]) * discard_mortality
        sector_land_f[sector.name] = land_f
        sector_discard_f[sector.name] = discard_f
        sector_dead_f[sector.name] = land_f + discard_f
    total_dead_f = np.sum(np.vstack(list(sector_dead_f.values())), axis=0) if sector_dead_f else np.zeros_like(numbers)
    z = max(float(settings.natural_mortality), 1e-9) + total_dead_f
    deaths_factor = (1.0 - np.exp(-z)) / np.maximum(z, _EPS)
    landed_biomass: dict[str, float] = {}
    discard_dead_biomass: dict[str, float] = {}
    landed_numbers_at_age: dict[str, np.ndarray] = {}
    discard_dead_numbers_at_age: dict[str, np.ndarray] = {}
    for sector in settings.sectors:
        landed_n = numbers * sector_land_f[sector.name] * deaths_factor
        discard_n = numbers * sector_discard_f[sector.name] * deaths_factor
        landed_numbers_at_age[sector.name] = landed_n
        discard_dead_numbers_at_age[sector.name] = discard_n
        landed_biomass[sector.name] = float(np.sum(landed_n * life["weight_kg"]))
        discard_dead_biomass[sector.name] = float(np.sum(discard_n * life["weight_kg"]))
    survivors = numbers * np.exp(-z)
    return {
        "z": z,
        "survivors": survivors,
        "landed_biomass": landed_biomass,
        "discard_dead_biomass": discard_dead_biomass,
        "landed_numbers_at_age": landed_numbers_at_age,
        "discard_dead_numbers_at_age": discard_dead_numbers_at_age,
        "total_landed_biomass": float(sum(landed_biomass.values())),
        "total_dead_discard_biomass": float(sum(discard_dead_biomass.values())),
        "f_scalar": float(f_scalar),
    }


def _solve_f_for_catch(
    numbers: np.ndarray,
    observed_sector_catch: dict[str, float],
    settings: AgeStructuredSettings,
    life: dict[str, np.ndarray],
    curves: dict[str, dict[str, np.ndarray]],
) -> tuple[float, dict[str, Any], float]:
    target = max(float(sum(observed_sector_catch.values())), 0.0)
    if target <= 0:
        outcome = _annual_catch(numbers, 0.0, observed_sector_catch, settings, life, curves)
        return 0.0, outcome, 0.0
    low, high = 0.0, 0.25
    high_outcome = _annual_catch(numbers, high, observed_sector_catch, settings, life, curves)
    while high_outcome["total_landed_biomass"] < target and high < 20.0:
        high *= 2.0
        high_outcome = _annual_catch(numbers, high, observed_sector_catch, settings, life, curves)
    for _ in range(42):
        mid = 0.5 * (low + high)
        outcome = _annual_catch(numbers, mid, observed_sector_catch, settings, life, curves)
        if outcome["total_landed_biomass"] < target:
            low = mid
        else:
            high = mid
    f_value = 0.5 * (low + high)
    outcome = _annual_catch(numbers, f_value, observed_sector_catch, settings, life, curves)
    mismatch = outcome["total_landed_biomass"] - target
    return float(f_value), outcome, float(mismatch)


def _advance_numbers(survivors: np.ndarray, recruitment: float) -> np.ndarray:
    next_numbers = np.zeros_like(survivors)
    next_numbers[0] = max(float(recruitment), 1e-12)
    if len(survivors) > 1:
        next_numbers[1:-1] = survivors[:-2]
        next_numbers[-1] = survivors[-2] + survivors[-1]
    return next_numbers


def _recruitment_multipliers(frame: pd.DataFrame) -> np.ndarray:
    # Controlled operating-model and recovery tests sometimes need the exact
    # annual multiplier that generated their known truth. Keep that explicit
    # path separate from observational recruitment indices, which remain
    # normalised by their median below.
    if "recruitment_multiplier_absolute" in frame.columns:
        raw = pd.to_numeric(frame["recruitment_multiplier_absolute"], errors="coerce").to_numpy(dtype=float)
        multipliers = np.ones(len(frame), dtype=float)
        valid = np.isfinite(raw) & (raw > 0)
        multipliers[valid] = raw[valid]
        return np.clip(multipliers, 0.05, 20.0)
    for column in ("recruitment_multiplier", "recruitment_index", "recruitment", "juvenile_index"):
        if column not in frame.columns:
            continue
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(raw) & (raw > 0)
        if not valid.any():
            continue
        median = max(float(np.nanmedian(raw[valid])), _EPS)
        multipliers = np.ones(len(frame), dtype=float)
        multipliers[valid] = raw[valid] / median
        return np.clip(multipliers, 0.05, 20.0)
    return np.ones(len(frame), dtype=float)


def simulate_age_structured(
    dataset: StockDataset,
    settings: AgeStructuredSettings | None = None,
    recruitment_multipliers: np.ndarray | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    config = settings or AgeStructuredSettings()
    frame = dataset.frame.reset_index(drop=True)
    years = frame["year"].to_numpy(dtype=int)
    life = life_history_arrays(config)
    curves = sector_curves(config, life)
    unfished = _unfished_numbers(config)
    b0 = _spawning_biomass(unfished, life, config)
    numbers = unfished * np.clip(config.initial_depletion, 0.01, 1.50)
    multipliers = recruitment_multipliers if recruitment_multipliers is not None else _recruitment_multipliers(frame)
    if len(multipliers) != len(frame):
        raise ValueError("Recruitment multipliers must have one value per model year.")

    history: list[dict[str, float]] = []
    sector_history: list[dict[str, float]] = []
    age_structure: list[dict[str, float]] = []
    predicted_age_composition: list[dict[str, float]] = []
    catch_numbers_by_year: dict[int, np.ndarray] = {}
    catch_numbers_by_year_sector: dict[tuple[int, str], np.ndarray] = {}
    mismatch_total = 0.0
    last_recruit_deviation = 0.0

    for year_index, year in enumerate(years):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Age-structured reconstruction was cancelled.")
        if progress_callback is not None:
            progress_callback(year_index / max(len(years), 1), f"Reconstructing year {int(year)}")
        curves = _annual_sector_curves(config, life, frame, year_index)
        total_biomass = _total_biomass(numbers, life)
        ssb = _spawning_biomass(numbers, life, config)
        survey_biomass = _survey_biomass(numbers, life)
        observed_sector = _sector_observed_catches(frame, year_index, config)
        f_scalar, catch_outcome, mismatch = _solve_f_for_catch(numbers, observed_sector, config, life, curves)
        mismatch_total += abs(mismatch)
        landed_at_age = np.sum(np.vstack(list(catch_outcome["landed_numbers_at_age"].values())), axis=0)
        catch_numbers_by_year[int(year)] = landed_at_age
        catch_total_numbers = max(float(landed_at_age.sum()), _EPS)
        for age, abundance, landed_n in zip(life["age"], numbers, landed_at_age):
            age_structure.append(
                {
                    "year": int(year),
                    "age": int(age),
                    "numbers": float(abundance),
                    "biomass": float(abundance * life["weight_kg"][int(age)]),
                    "catch_numbers": float(landed_n),
                }
            )
            predicted_age_composition.append(
                {
                    "year": int(year),
                    "sector": "all",
                    "age": int(age),
                    "proportion": float(landed_n / catch_total_numbers),
                }
            )
        for sector in config.sectors:
            sector_landed_n = catch_outcome["landed_numbers_at_age"][sector.name]
            catch_numbers_by_year_sector[(int(year), sector.name)] = sector_landed_n
            sector_total_n = max(float(sector_landed_n.sum()), _EPS)
            for age, value in zip(life["age"], sector_landed_n):
                predicted_age_composition.append(
                    {
                        "year": int(year),
                        "sector": sector.name,
                        "age": int(age),
                        "proportion": float(value / sector_total_n),
                    }
                )
            sector_history.append(
                {
                    "year": int(year),
                    "sector": sector.name,
                    "observed_landed_catch": float(observed_sector.get(sector.name, 0.0)),
                    "predicted_landed_catch": float(catch_outcome["landed_biomass"][sector.name]),
                    "dead_discard_biomass": float(catch_outcome["discard_dead_biomass"][sector.name]),
                    "f_scalar": float(f_scalar),
                }
            )

        history.append(
            {
                "year": int(year),
                "total_biomass": float(total_biomass),
                "spawning_biomass": float(ssb),
                "survey_biomass": float(survey_biomass),
                "depletion": float(ssb / max(b0, _EPS)),
                "f_scalar": float(f_scalar),
                "observed_catch": float(sum(observed_sector.values())),
                "predicted_landed_catch": float(catch_outcome["total_landed_biomass"]),
                "dead_discard_biomass": float(catch_outcome["total_dead_discard_biomass"]),
                "catch_mismatch": float(mismatch),
                "recruitment": float(numbers[0]),
                "recruitment_deviation": float(last_recruit_deviation),
            }
        )
        expected_recruitment = _beverton_holt(ssb, b0, config)
        multiplier = max(float(multipliers[year_index]), 0.001)
        last_recruit_deviation = log(multiplier)
        recruitment_next = expected_recruitment * multiplier
        numbers = _advance_numbers(catch_outcome["survivors"], recruitment_next)

    if progress_callback is not None:
        progress_callback(1.0, "Reconstruction complete")

    return {
        "settings": asdict(config),
        "life_history": {key: value.tolist() for key, value in life.items()},
        "sector_curves": {
            sector: {key: values.tolist() if isinstance(values, np.ndarray) else float(values) for key, values in value.items()}
            for sector, value in curves.items()
        },
        "b0": float(b0),
        "history": history,
        "sector_history": sector_history,
        "age_structure": age_structure,
        "predicted_age_composition": predicted_age_composition,
        "catch_numbers_by_year": catch_numbers_by_year,
        "catch_numbers_by_year_sector": catch_numbers_by_year_sector,
        "final_numbers": numbers,
        "catch_mismatch_total": float(mismatch_total),
    }



def _preserve_age_model_columns(base: StockDataset, raw: pd.DataFrame) -> StockDataset:
    raw_lookup = {str(column).strip().lower(): column for column in raw.columns}
    year_column = raw_lookup.get("year") or raw_lookup.get("yr") or raw_lookup.get("date")
    if year_column is None:
        return base
    extras = pd.DataFrame({"year": pd.to_numeric(raw[year_column], errors="coerce")})
    allowed_prefixes = (
        "catch_", "recruitment", "juvenile", "environment", "temperature",
        "selectivity_", "retention_", "discard_mortality_", "fishing_power_",
    )
    for column in raw.columns:
        key = str(column).strip().lower()
        if column == year_column or not key.startswith(allowed_prefixes):
            continue
        extras[key] = pd.to_numeric(raw[column], errors="coerce")
    extras = extras.dropna(subset=["year"]).copy()
    if extras.empty or len(extras.columns) == 1:
        return base
    extras["year"] = extras["year"].astype(int)
    extras = extras.sort_values("year").drop_duplicates("year", keep="last")
    merged = base.frame.merge(extras, on="year", how="left")
    return StockDataset(
        name=base.name,
        frame=merged,
        provenance=base.provenance,
        transformations=[*base.transformations, {
            "operation": "preserve_age_structured_columns",
            "details": {"columns": [column for column in merged.columns if column not in base.frame.columns]},
        }],
        warnings=base.warnings,
        raw_columns=base.raw_columns,
        index_columns=base.index_columns,
    )


def read_age_structured_file(path: str | Path) -> StockDataset:
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xlsm"} else pd.read_csv(path)
    return _preserve_age_model_columns(read_stock_file(path), raw)


def read_age_structured_csv(text: str, name: str = "Age-structured stock") -> StockDataset:
    from io import StringIO

    raw = pd.read_csv(StringIO(text))
    return _preserve_age_model_columns(read_stock_csv(text, name), raw)

def read_composition_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    return normalise_composition_frame(frame)


def read_composition_csv(text: str) -> pd.DataFrame:
    from io import StringIO

    return normalise_composition_frame(pd.read_csv(StringIO(text)))


def normalise_composition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    aliases = {
        "year": ["year", "yr"],
        "age": ["age", "age_years"],
        "length_mm": ["length_mm", "length", "length_mid_mm", "bin"],
        "proportion": ["proportion", "prop", "frequency", "count"],
        "sample_size": ["sample_size", "n", "effective_n"],
        "sector": ["sector", "fleet"],
    }
    out = pd.DataFrame()
    for target, names in aliases.items():
        for name in names:
            if name in lookup:
                out[target] = frame[lookup[name]]
                break
    if "year" not in out or "proportion" not in out or ("age" not in out and "length_mm" not in out):
        raise ValueError("Composition data needs year, proportion/count, and either age or length_mm.")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    for column in ("age", "length_mm", "proportion", "sample_size"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "sample_size" not in out:
        out["sample_size"] = 100.0
    if "sector" not in out:
        out["sector"] = "all"
    out["sector"] = out["sector"].fillna("all").astype(str).str.lower()
    out = out.dropna(subset=["year", "proportion"]).copy()
    out["year"] = out["year"].astype(int)
    value_column = "age" if "age" in out else "length_mm"
    out = out.dropna(subset=[value_column])
    group_columns = ["year", "sector"]
    group_sum = out.groupby(group_columns)["proportion"].transform("sum")
    out["proportion"] = np.where(group_sum > 0, out["proportion"] / group_sum, 0.0)
    return out.reset_index(drop=True)


def _age_composition_deviance(
    observed: pd.DataFrame | None,
    simulation: dict[str, Any],
    weight: float,
) -> tuple[float, list[dict[str, float]]]:
    if observed is None or observed.empty or "age" not in observed:
        return 0.0, []
    predicted_rows = simulation["predicted_age_composition"]
    predicted = {(int(row["year"]), str(row["sector"]).lower(), int(row["age"])): float(row["proportion"]) for row in predicted_rows}
    details: list[dict[str, float]] = []
    total = 0.0
    for (year, sector), group in observed.groupby(["year", "sector"]):
        n_eff = max(float(group["sample_size"].dropna().median()), 1.0)
        group_deviance = 0.0
        for _, row in group.iterrows():
            obs = max(float(row["proportion"]), 0.0)
            pred = max(predicted.get((int(year), str(sector).lower(), int(row["age"])), _EPS), _EPS)
            if obs > 0:
                group_deviance += 2.0 * n_eff * obs * log(obs / pred)
        group_deviance *= max(float(weight), 0.0)
        total += group_deviance
        details.append({"year": int(year), "sector": str(sector), "deviance": float(group_deviance), "sample_size": float(n_eff)})
    return float(total), details


def _predicted_length_composition(
    observed: pd.DataFrame | None,
    simulation: dict[str, Any],
    settings: AgeStructuredSettings,
) -> list[dict[str, float]]:
    if observed is None or observed.empty or "length_mm" not in observed:
        return []
    bins = np.sort(observed["length_mm"].dropna().unique().astype(float))
    life = {key: np.asarray(value, dtype=float) for key, value in simulation["life_history"].items()}
    bin_prob = _length_bin_probabilities(life, bins, settings.length_cv)
    rows: list[dict[str, float]] = []
    for (year, sector), group in observed.groupby(["year", "sector"]):
        key = (int(year), str(sector).lower())
        if key[1] == "all":
            numbers = simulation["catch_numbers_by_year"].get(key[0])
        else:
            numbers = simulation["catch_numbers_by_year_sector"].get(key)
        if numbers is None:
            continue
        length_counts = np.asarray(numbers, dtype=float) @ bin_prob
        length_counts = length_counts / max(float(length_counts.sum()), _EPS)
        for bin_value, proportion in zip(bins, length_counts):
            rows.append({"year": int(year), "sector": key[1], "length_mm": float(bin_value), "proportion": float(proportion)})
    return rows


def _length_composition_deviance(
    observed: pd.DataFrame | None,
    predicted_rows: list[dict[str, float]],
    weight: float,
) -> tuple[float, list[dict[str, float]]]:
    if observed is None or observed.empty or "length_mm" not in observed:
        return 0.0, []
    predicted = {
        (int(row["year"]), str(row["sector"]).lower(), float(row["length_mm"])): float(row["proportion"])
        for row in predicted_rows
    }
    details: list[dict[str, float]] = []
    total = 0.0
    for (year, sector), group in observed.groupby(["year", "sector"]):
        n_eff = max(float(group["sample_size"].dropna().median()), 1.0)
        group_deviance = 0.0
        for _, row in group.iterrows():
            obs = max(float(row["proportion"]), 0.0)
            pred = max(predicted.get((int(year), str(sector).lower(), float(row["length_mm"])), _EPS), _EPS)
            if obs > 0:
                group_deviance += 2.0 * n_eff * obs * log(obs / pred)
        group_deviance *= max(float(weight), 0.0)
        total += group_deviance
        details.append({"year": int(year), "sector": str(sector), "deviance": float(group_deviance), "sample_size": float(n_eff)})
    return float(total), details


def _lognormal_component(observed: np.ndarray, predicted: np.ndarray, sigma: float) -> tuple[float, float, np.ndarray]:
    mask = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0) & (predicted > 0)
    if not mask.any():
        return 0.0, 1.0, np.full_like(observed, np.nan, dtype=float)
    q = exp(float(np.mean(np.log(observed[mask]) - np.log(predicted[mask]))))
    fitted = q * predicted
    residuals = np.full_like(observed, np.nan, dtype=float)
    residuals[mask] = np.log(observed[mask]) - np.log(fitted[mask])
    nll = float(np.sum(0.5 * (residuals[mask] / max(sigma, 0.03)) ** 2 + log(max(sigma, 0.03)) + 0.5 * log(2.0 * pi)))
    return nll, q, residuals


def _objective(
    dataset: StockDataset,
    settings: AgeStructuredSettings,
    age_composition: pd.DataFrame | None,
    length_composition: pd.DataFrame | None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[float, dict[str, Any]]:
    simulation = simulate_age_structured(dataset, settings, cancel_check=cancel_check)
    history = simulation["history"]
    frame = dataset.frame.reset_index(drop=True)
    index = frame["index"].to_numpy(dtype=float)
    biomass_obs = frame["biomass"].to_numpy(dtype=float)
    survey_pred = np.array([row["survey_biomass"] for row in history], dtype=float)
    biomass_pred = np.array([row["total_biomass"] for row in history], dtype=float)
    index_nll, q_index, index_residuals = _lognormal_component(index, survey_pred, settings.index_cv)
    biomass_nll, q_biomass, biomass_residuals = _lognormal_component(biomass_obs, biomass_pred, settings.biomass_cv)
    age_deviance, age_details = _age_composition_deviance(age_composition, simulation, settings.age_comp_weight)
    predicted_length = _predicted_length_composition(length_composition, simulation, settings)
    length_deviance, length_details = _length_composition_deviance(length_composition, predicted_length, settings.length_comp_weight)
    m_sd = sqrt(log(1.0 + max(settings.m_prior_cv, 0.05) ** 2))
    m_prior = 0.5 * ((log(max(settings.natural_mortality, 1e-6)) - log(max(settings.m_prior_median, 1e-6))) / m_sd) ** 2
    h_prior = 0.5 * ((settings.steepness - settings.h_prior_mean) / max(settings.h_prior_sd, 0.02)) ** 2
    dep_prior = 0.5 * ((settings.initial_depletion - settings.initial_depletion_prior) / max(settings.initial_depletion_prior_sd, 0.02)) ** 2
    mismatch_penalty = 0.5 * (simulation["catch_mismatch_total"] / max(float(frame["catch"].sum()), 1.0) / 0.001) ** 2
    components = {
        "index_likelihood": float(index_nll),
        "biomass_likelihood": float(biomass_nll),
        "age_composition_deviance": float(age_deviance),
        "length_composition_deviance": float(length_deviance),
        "natural_mortality_prior": float(m_prior),
        "steepness_prior": float(h_prior),
        "initial_depletion_prior": float(dep_prior),
        "catch_reconstruction_penalty": float(mismatch_penalty),
    }
    objective = float(sum(components.values()))
    details = {
        "simulation": simulation,
        "objective_components": components,
        "q_index": float(q_index),
        "q_biomass": float(q_biomass),
        "index_residuals": index_residuals,
        "biomass_residuals": biomass_residuals,
        "age_composition_diagnostics": age_details,
        "length_composition_diagnostics": length_details,
        "predicted_length_composition": predicted_length,
    }
    return objective, details


def _fit_parameter_spec(settings: AgeStructuredSettings, fit_settings: AgeFitSettings, dataset: StockDataset) -> list[tuple[str, float, float]]:
    frame = dataset.frame
    max_catch = max(float(frame["catch"].max()), 1.0)
    life = life_history_arrays(settings)
    mean_weight = max(float(np.mean(life["weight_kg"][max(1, settings.max_age // 4):])), 1e-6)
    r0_center = max(max_catch / mean_weight * 10.0, settings.r0)
    specs: list[tuple[str, float, float]] = [("log_r0", log(r0_center / 20.0), log(r0_center * 20.0))]
    if fit_settings.estimate_natural_mortality:
        specs.append(("natural_mortality", 0.03, 0.45))
    if fit_settings.estimate_steepness:
        specs.append(("steepness", 0.21, 0.99))
    if fit_settings.estimate_initial_depletion:
        specs.append(("initial_depletion", 0.05, 1.20))
    if fit_settings.estimate_survey_selectivity:
        specs.extend(
            [
                ("survey_selectivity_a50", 0.0, max(settings.max_age * 0.80, 1.0)),
                ("survey_selectivity_slope", 0.15, max(settings.max_age * 0.30, 1.0)),
            ]
        )
    if fit_settings.estimate_recruitment_sigma:
        specs.append(("recruitment_sigma", 0.05, 1.50))
    return specs


def _decode_parameters(unit: np.ndarray, specs: list[tuple[str, float, float]], base: AgeStructuredSettings) -> AgeStructuredSettings:
    values = {name: low + float(value) * (high - low) for value, (name, low, high) in zip(unit, specs)}
    if "log_r0" in values:
        values["r0"] = exp(values.pop("log_r0"))
    return replace(base, **values)


def fit_age_structured(
    dataset: StockDataset,
    settings: AgeStructuredSettings | None = None,
    fit_settings: AgeFitSettings | None = None,
    age_composition: pd.DataFrame | None = None,
    length_composition: pd.DataFrame | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> AgeStructuredResult:
    base = settings or AgeStructuredSettings()
    config = fit_settings or AgeFitSettings()
    specs = _fit_parameter_spec(base, config, dataset)
    dimensions = len(specs)
    population_size = max(int(config.population), dimensions * 6, 12)
    generations = max(int(config.generations), 1)
    rng = np.random.default_rng(config.seed)
    population = rng.random((population_size, dimensions))
    scores = np.empty(population_size, dtype=float)
    details: list[dict[str, Any]] = []
    total_evaluations = population_size + generations * population_size + max(config.local_rounds, 0) * dimensions * 2
    completed_evaluations = 0

    def evaluate(unit: np.ndarray) -> tuple[float, dict[str, Any], AgeStructuredSettings]:
        nonlocal completed_evaluations
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Integrated age-structured fit was cancelled.")
        candidate = _decode_parameters(np.clip(unit, 0.0, 1.0), specs, base)
        score, detail = _objective(dataset, candidate, age_composition, length_composition, cancel_check)
        completed_evaluations += 1
        if progress_callback is not None:
            progress_callback(
                completed_evaluations / max(total_evaluations, 1),
                f"Fit evaluation {completed_evaluations} of {total_evaluations}",
            )
        return float(score), detail, candidate

    candidate_settings: list[AgeStructuredSettings] = []
    for index in range(population_size):
        score, detail, candidate = evaluate(population[index])
        scores[index] = score
        details.append(detail)
        candidate_settings.append(candidate)

    convergence_history: list[dict[str, float]] = []
    for generation in range(generations):
        for target in range(population_size):
            choices = [index for index in range(population_size) if index != target]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = population[a] + config.mutation * (population[b] - population[c])
            mutant = np.clip(mutant, 0.0, 1.0)
            mask = rng.random(dimensions) < config.crossover
            mask[int(rng.integers(0, dimensions))] = True
            trial = np.where(mask, mutant, population[target])
            score, detail, candidate = evaluate(trial)
            if score < scores[target]:
                population[target] = trial
                scores[target] = score
                details[target] = detail
                candidate_settings[target] = candidate
        convergence_history.append(
            {
                "generation": float(generation + 1),
                "best_objective": float(np.min(scores)),
                "median_objective": float(np.median(scores)),
                "objective_spread": float(np.quantile(scores, 0.90) - np.quantile(scores, 0.10)),
            }
        )

    best_index = int(np.argmin(scores))
    best_unit = population[best_index].copy()
    best_score = float(scores[best_index])
    best_detail = details[best_index]
    best_settings = candidate_settings[best_index]
    step = np.full(dimensions, 0.08, dtype=float)
    for _ in range(max(config.local_rounds, 0)):
        improved = False
        for dimension in range(dimensions):
            for direction in (-1.0, 1.0):
                trial = best_unit.copy()
                trial[dimension] = np.clip(trial[dimension] + direction * step[dimension], 0.0, 1.0)
                score, detail, candidate = evaluate(trial)
                if score < best_score:
                    best_unit, best_score, best_detail, best_settings = trial, score, detail, candidate
                    improved = True
        if not improved:
            step *= 0.5

    if progress_callback is not None:
        progress_callback(1.0, "Integrated fit complete")

    order = np.argsort(scores)
    ensemble: list[dict[str, float]] = []
    keep = order[: min(20, population_size)]
    delta = np.array([scores[index] - scores[keep[0]] for index in keep], dtype=float)
    weights = np.exp(-0.5 * np.clip(delta, 0.0, 700.0))
    weights /= max(float(weights.sum()), _EPS)
    for rank, (index, weight) in enumerate(zip(keep, weights), start=1):
        candidate = candidate_settings[int(index)]
        simulation = details[int(index)]["simulation"]
        ensemble.append(
            {
                "rank": float(rank),
                "weight": float(weight),
                "objective": float(scores[int(index)]),
                "r0": float(candidate.r0),
                "natural_mortality": float(candidate.natural_mortality),
                "steepness": float(candidate.steepness),
                "initial_depletion": float(candidate.initial_depletion),
                "survey_selectivity_a50": float(candidate.survey_selectivity_a50),
                "survey_selectivity_slope": float(candidate.survey_selectivity_slope),
                "terminal_depletion": float(simulation["history"][-1]["depletion"]),
            }
        )

    simulation = best_detail["simulation"]
    references = equilibrium_reference_points(best_settings)
    residual_rows = []
    for row, index_resid, bio_resid in zip(simulation["history"], best_detail["index_residuals"], best_detail["biomass_residuals"]):
        residual_rows.append(
            {
                "year": row["year"],
                "index_log_residual": float(index_resid) if np.isfinite(index_resid) else float("nan"),
                "biomass_log_residual": float(bio_resid) if np.isfinite(bio_resid) else float("nan"),
            }
        )
    diagnostics = {
        "objective_components": best_detail["objective_components"],
        "convergence_history": convergence_history,
        "parameter_bounds": [{"parameter": name, "low": low, "high": high} for name, low, high in specs],
        "residuals": residual_rows,
        "q_index": best_detail["q_index"],
        "q_biomass": best_detail["q_biomass"],
        "age_composition": best_detail["age_composition_diagnostics"],
        "length_composition": best_detail["length_composition_diagnostics"],
        "interpretation": "Integrated age-structured foundation fit. Catch is reconstructed with Baranov mortality, sector retention and discard mortality. This is not yet a replacement for a fully peer-reviewed Stock Synthesis configuration.",
    }
    best = {
        "objective": float(best_score),
        "r0": float(best_settings.r0),
        "natural_mortality": float(best_settings.natural_mortality),
        "steepness": float(best_settings.steepness),
        "initial_depletion": float(best_settings.initial_depletion),
        "terminal_biomass": float(simulation["history"][-1]["total_biomass"]),
        "terminal_spawning_biomass": float(simulation["history"][-1]["spawning_biomass"]),
        "terminal_depletion": float(simulation["history"][-1]["depletion"]),
        "terminal_f": float(simulation["history"][-1]["f_scalar"]),
        "b0": float(simulation["b0"]),
        "msy": float(references["msy"]),
        "bmsy": float(references["bmsy"]),
        "fmsy": float(references["fmsy"]),
        "terminal_f_fmsy": float(simulation["history"][-1]["f_scalar"] / max(references["fmsy"], _EPS)),
    }
    state = {
        "final_numbers": np.asarray(simulation["final_numbers"], dtype=float).tolist(),
        "life_history": simulation["life_history"],
        "sector_curves": simulation["sector_curves"],
        "last_year": int(simulation["history"][-1]["year"]),
    }
    return AgeStructuredResult(
        dataset.name,
        asdict(best_settings),
        asdict(config),
        best,
        diagnostics,
        simulation["history"],
        simulation["sector_history"],
        simulation["age_structure"],
        simulation["predicted_age_composition"],
        best_detail["predicted_length_composition"],
        ensemble,
        state,
    )


def _equilibrium_at_f(settings: AgeStructuredSettings, f_scalar: float, years: int = 140) -> dict[str, float]:
    life = life_history_arrays(settings)
    curves = sector_curves(settings, life)
    numbers = _unfished_numbers(settings)
    b0 = _spawning_biomass(numbers, life, settings)
    shares = {sector.name: max(sector.catch_share, 0.0) for sector in settings.sectors}
    outcome: dict[str, Any] | None = None
    for _ in range(max(years, 20)):
        outcome = _annual_catch(numbers, f_scalar, shares, settings, life, curves)
        ssb = _spawning_biomass(numbers, life, settings)
        recruitment = _beverton_holt(ssb, b0, settings)
        numbers = _advance_numbers(outcome["survivors"], recruitment)
    assert outcome is not None
    return {
        "f": float(f_scalar),
        "yield": float(outcome["total_landed_biomass"]),
        "dead_discards": float(outcome["total_dead_discard_biomass"]),
        "biomass": float(_total_biomass(numbers, life)),
        "spawning_biomass": float(_spawning_biomass(numbers, life, settings)),
        "depletion": float(_spawning_biomass(numbers, life, settings) / max(b0, _EPS)),
    }


@lru_cache(maxsize=128)
def _equilibrium_reference_points_cached(settings: AgeStructuredSettings, grid_points: int) -> dict[str, Any]:
    f_values = np.linspace(0.0, 1.50, max(int(grid_points), 20))
    rows = [_equilibrium_at_f(settings, float(f_value)) for f_value in f_values]
    best = max(rows, key=lambda row: row["yield"])
    b0 = rows[0]["spawning_biomass"]
    return {
        "msy": float(best["yield"]),
        "bmsy": float(best["spawning_biomass"]),
        "fmsy": float(best["f"]),
        "b0": float(b0),
        "bmsy_b0": float(best["spawning_biomass"] / max(b0, _EPS)),
        "grid": rows,
    }


def equilibrium_reference_points(settings: AgeStructuredSettings, grid_points: int = 50) -> dict[str, Any]:
    return _equilibrium_reference_points_cached(settings, max(int(grid_points), 20))


def _settings_from_result(result: AgeStructuredResult) -> AgeStructuredSettings:
    value = dict(result.settings)
    sectors_raw = value.pop("sectors", [])
    sectors = tuple(SectorSettings(**sector) for sector in sectors_raw) if sectors_raw else AgeStructuredSettings().sectors
    return AgeStructuredSettings(sectors=sectors, **value)


def project_age_structured(
    result: AgeStructuredResult,
    settings: AgeProjectionSettings | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    config = settings or AgeProjectionSettings()
    model_settings = _settings_from_result(result)
    life = life_history_arrays(model_settings)
    curves = sector_curves(model_settings, life)
    references = equilibrium_reference_points(model_settings)
    b0 = float(references["b0"])
    base_numbers = np.asarray(result.state["final_numbers"], dtype=float)
    years = np.arange(int(result.state["last_year"]) + 1, int(result.state["last_year"]) + max(config.years, 1) + 1)
    rng = np.random.default_rng(config.seed)
    iterations = max(int(config.iterations), 20)
    biomass = np.empty((iterations, len(years)), dtype=float)
    ssb = np.empty_like(biomass)
    depletion = np.empty_like(biomass)
    catches = np.empty_like(biomass)
    fishing_mortality = np.empty_like(biomass)
    recruitments = np.empty_like(biomass)
    dead_discards = np.empty_like(biomass)
    shares = {sector.name: max(sector.catch_share, 0.0) for sector in model_settings.sectors}
    recruitment_sigma = model_settings.recruitment_sigma if config.recruitment_sigma is None else max(config.recruitment_sigma, 0.0)

    for iteration in range(iterations):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Age-structured projection was cancelled.")
        if progress_callback is not None:
            progress_callback(iteration / max(iterations, 1), f"Projection simulation {iteration + 1} of {iterations}")
        numbers = base_numbers.copy()
        rec_dev = 0.0
        for year_index, _year in enumerate(years):
            current_ssb = _spawning_biomass(numbers, life, model_settings)
            current_depletion = current_ssb / max(b0, _EPS)
            if config.strategy == "fixed_catch":
                target_sector = {name: config.fixed_catch * share / max(sum(shares.values()), _EPS) for name, share in shares.items()}
                f_scalar, outcome, _ = _solve_f_for_catch(numbers, target_sector, model_settings, life, curves)
            else:
                if config.strategy == "fixed_f":
                    f_scalar = max(config.fixed_f, 0.0)
                else:
                    ramp = min(1.0, max(0.0, (current_depletion - config.limit_depletion) / max(config.target_depletion - config.limit_depletion, _EPS)))
                    f_scalar = references["fmsy"] * ramp * max(config.pstar, 0.0) / 0.5
                implementation = rng.lognormal(-0.5 * config.implementation_cv**2, max(config.implementation_cv, 0.0))
                f_scalar *= implementation
                outcome = _annual_catch(numbers, f_scalar, shares, model_settings, life, curves)
            expected_recruitment = _beverton_holt(current_ssb, b0, model_settings)
            innovation = rng.normal(0.0, recruitment_sigma)
            rec_dev = model_settings.recruitment_rho * rec_dev + sqrt(max(1.0 - model_settings.recruitment_rho**2, 0.0)) * innovation
            recruitment = expected_recruitment * exp(rec_dev - 0.5 * recruitment_sigma**2)
            numbers = _advance_numbers(outcome["survivors"], recruitment)
            biomass[iteration, year_index] = _total_biomass(numbers, life)
            ssb[iteration, year_index] = _spawning_biomass(numbers, life, model_settings)
            depletion[iteration, year_index] = ssb[iteration, year_index] / max(b0, _EPS)
            catches[iteration, year_index] = outcome["total_landed_biomass"]
            dead_discards[iteration, year_index] = outcome["total_dead_discard_biomass"]
            fishing_mortality[iteration, year_index] = f_scalar
            recruitments[iteration, year_index] = recruitment

    if progress_callback is not None:
        progress_callback(1.0, "Projection complete")

    rows: list[dict[str, float]] = []
    for year_index, year in enumerate(years):
        rows.append(
            {
                "year": int(year),
                "biomass_p10": float(np.quantile(biomass[:, year_index], 0.10)),
                "biomass_median": float(np.quantile(biomass[:, year_index], 0.50)),
                "biomass_p90": float(np.quantile(biomass[:, year_index], 0.90)),
                "spawning_biomass_median": float(np.quantile(ssb[:, year_index], 0.50)),
                "depletion_p10": float(np.quantile(depletion[:, year_index], 0.10)),
                "depletion_median": float(np.quantile(depletion[:, year_index], 0.50)),
                "depletion_p90": float(np.quantile(depletion[:, year_index], 0.90)),
                "catch_median": float(np.quantile(catches[:, year_index], 0.50)),
                "dead_discard_median": float(np.quantile(dead_discards[:, year_index], 0.50)),
                "f_median": float(np.quantile(fishing_mortality[:, year_index], 0.50)),
                "recruitment_median": float(np.quantile(recruitments[:, year_index], 0.50)),
                "prob_above_target": float(np.mean(depletion[:, year_index] >= config.target_depletion)),
                "prob_above_limit": float(np.mean(depletion[:, year_index] >= config.limit_depletion)),
            }
        )
    annual_catch_mean = np.mean(catches, axis=1)
    catch_cv = np.std(catches, axis=1) / np.maximum(annual_catch_mean, _EPS)
    min_depletion = np.min(depletion, axis=1)
    terminal_depletion = depletion[:, -1]
    return {
        "settings": asdict(config),
        "reference_points": references,
        "projection": rows,
        "risk_summary": {
            "median_annual_catch": float(np.median(annual_catch_mean)),
            "median_catch_cv": float(np.median(catch_cv)),
            "prob_ever_below_limit": float(np.mean(min_depletion < config.limit_depletion)),
            "prob_terminal_above_target": float(np.mean(terminal_depletion >= config.target_depletion)),
            "terminal_depletion_p10": float(np.quantile(terminal_depletion, 0.10)),
            "terminal_depletion_median": float(np.quantile(terminal_depletion, 0.50)),
            "terminal_depletion_p90": float(np.quantile(terminal_depletion, 0.90)),
            "biological_expected_shortfall": float(np.mean(np.maximum(config.limit_depletion - min_depletion, 0.0))),
        },
    }


def run_management_strategy_evaluation(
    result: AgeStructuredResult,
    years: int = 20,
    iterations: int = 250,
    seed: int = 11221,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    model_settings = _settings_from_result(result)
    references = equilibrium_reference_points(model_settings)
    strategies: list[tuple[str, AgeProjectionSettings]] = []
    for fraction in (0.25, 0.50, 0.75, 1.00, 1.25):
        strategies.append(
            (
                f"fixed_f_{fraction:.2f}_fmsy",
                AgeProjectionSettings(years=years, iterations=iterations, strategy="fixed_f", fixed_f=references["fmsy"] * fraction, seed=seed + len(strategies)),
            )
        )
    for pstar in (0.25, 0.35, 0.45, 0.50):
        strategies.append(
            (
                f"hcr_pstar_{pstar:.2f}",
                AgeProjectionSettings(years=years, iterations=iterations, strategy="hcr_40_10", pstar=pstar, seed=seed + len(strategies)),
            )
        )
    rows: list[dict[str, Any]] = []
    projections: dict[str, Any] = {}
    for strategy_index, (label, config) in enumerate(strategies):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Management strategy evaluation was cancelled.")

        def strategy_progress(value: float, message: str) -> None:
            if progress_callback is not None:
                overall = (strategy_index + min(max(value, 0.0), 1.0)) / max(len(strategies), 1)
                progress_callback(overall, f"{label}: {message}")

        projection = project_age_structured(result, config, cancel_check, strategy_progress)
        risk = projection["risk_summary"]
        row = {"strategy": label, **risk}
        row["risk_adjusted_yield"] = row["median_annual_catch"] * max(0.0, 1.0 - row["prob_ever_below_limit"])
        rows.append(row)
        projections[label] = projection
    if progress_callback is not None:
        progress_callback(1.0, "Management strategy evaluation complete")
    pareto = _pareto_front(rows)
    return {
        "summary": {
            "strategies": len(rows),
            "pareto_strategies": len(pareto),
            "highest_risk_adjusted_yield": max(rows, key=lambda row: row["risk_adjusted_yield"])["strategy"] if rows else "",
        },
        "strategies": rows,
        "pareto_front": pareto,
        "projections": projections,
        "reference_points": references,
    }


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            at_least_as_good = (
                other["median_annual_catch"] >= candidate["median_annual_catch"]
                and other["prob_ever_below_limit"] <= candidate["prob_ever_below_limit"]
                and other["median_catch_cv"] <= candidate["median_catch_cv"]
            )
            strictly_better = (
                other["median_annual_catch"] > candidate["median_annual_catch"]
                or other["prob_ever_below_limit"] < candidate["prob_ever_below_limit"]
                or other["median_catch_cv"] < candidate["median_catch_cv"]
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (-row["median_annual_catch"], row["prob_ever_below_limit"]))


def synthetic_age_structured_dataset(
    years: int = 30,
    settings: AgeStructuredSettings | None = None,
    seed: int = 1234,
) -> tuple[StockDataset, pd.DataFrame]:
    config = settings or AgeStructuredSettings(max_age=20, r0=350_000.0)
    rng = np.random.default_rng(seed)
    year_values = np.arange(1990, 1990 + max(years, 8))
    catches = np.linspace(120.0, 320.0, len(year_values)) * (1.0 + 0.12 * np.sin(np.arange(len(year_values)) / 3.0))
    frame = pd.DataFrame({"year": year_values, "catch": catches, "index": np.nan, "biomass": np.nan})
    dataset = read_stock_csv(frame.to_csv(index=False), "synthetic_age_structured")
    simulation = simulate_age_structured(dataset, config)
    survey = np.array([row["survey_biomass"] for row in simulation["history"]])
    biomass = np.array([row["total_biomass"] for row in simulation["history"]])
    frame["index"] = survey * rng.lognormal(-0.5 * 0.15**2, 0.15, len(frame))
    frame["biomass"] = biomass * rng.lognormal(-0.5 * 0.10**2, 0.10, len(frame))
    final_dataset = read_stock_csv(frame.to_csv(index=False), "synthetic_age_structured")
    comp_rows = []
    for row in simulation["predicted_age_composition"]:
        if row["sector"] != "all" or row["year"] % 3 != 0:
            continue
        comp_rows.append({**row, "sample_size": 120})
    return final_dataset, pd.DataFrame(comp_rows)


__all__ = [
    "SectorSettings",
    "AgeStructuredSettings",
    "AgeFitSettings",
    "AgeProjectionSettings",
    "AgeStructuredResult",
    "logistic",
    "life_history_arrays",
    "sector_curves",
    "simulate_age_structured",
    "fit_age_structured",
    "equilibrium_reference_points",
    "project_age_structured",
    "run_management_strategy_evaluation",
    "read_age_structured_file",
    "read_age_structured_csv",
    "read_composition_file",
    "read_composition_csv",
    "normalise_composition_frame",
    "synthetic_age_structured_dataset",
]
