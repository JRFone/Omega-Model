from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import exp
from typing import Any, Mapping, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class SexSpec:
    name: str
    fraction_at_recruitment: float = 0.5
    natural_mortality: float = 0.12
    linf_mm: float = 850.0
    growth_k: float = 0.13
    growth_t0: float = -0.5
    weight_a: float = 1.0e-8
    weight_b: float = 3.05
    maturity_a50: float = 5.0
    maturity_slope: float = 1.2
    fecundity_power: float = 1.0
    maturity_model: str = "age_logistic"
    maturity_length50_mm: float = 500.0
    maturity_slope_coefficient: float = -0.05


@dataclass(frozen=True)
class AreaSpec:
    name: str
    recruitment_share: float = 1.0
    productivity_multiplier: float = 1.0
    habitat_capacity_multiplier: float = 1.0


@dataclass(frozen=True)
class SelectivityBlock:
    start_year: int
    end_year: int
    a50: float
    slope: float
    retention_length50_mm: float = 500.0
    retention_slope_mm: float = 35.0
    fishing_power: float = 1.0


@dataclass(frozen=True)
class DepthMortalityBand:
    minimum_depth_m: float
    maximum_depth_m: float
    mortality: float
    share: float = 1.0


@dataclass(frozen=True)
class FleetSpec:
    name: str
    sector: str
    area_shares: tuple[float, ...]
    season_shares: tuple[float, ...]
    selectivity_blocks: tuple[SelectivityBlock, ...]
    discard_mortality_by_depth: tuple[DepthMortalityBand, ...] = (
        DepthMortalityBand(0.0, 39.99, 0.25, 0.35),
        DepthMortalityBand(40.0, 79.99, 0.50, 0.40),
        DepthMortalityBand(80.0, 2000.0, 0.80, 0.25),
    )
    implementation_cv: float = 0.0


@dataclass(frozen=True)
class SeasonalSpatialSettings:
    start_year: int = 2000
    years: int = 20
    max_age: int = 30
    seasons: int = 4
    spawning_season: int = 1
    recruitment_season: int = 0
    r0: float = 1_000_000.0
    steepness: float = 0.75
    recruitment_sigma: float = 0.60
    recruitment_rho: float = 0.0
    initial_depletion: float = 0.85
    sexes: tuple[SexSpec, ...] = (
        SexSpec("female", 0.5),
        SexSpec("male", 0.5, maturity_a50=999.0),
    )
    areas: tuple[AreaSpec, ...] = (
        AreaSpec("North", 0.34),
        AreaSpec("Central", 0.33),
        AreaSpec("South", 0.33),
    )
    fleets: tuple[FleetSpec, ...] = ()
    # movement[season, sex, age, origin, destination]
    movement: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass
class SpatialSimulationResult:
    settings: dict[str, Any]
    history: list[dict[str, float | int | str]]
    fleet_history: list[dict[str, float | int | str]]
    age_area_sex: list[dict[str, float | int | str]]
    terminal_numbers: np.ndarray
    diagnostics: dict[str, Any]


def logistic(x: np.ndarray | float, x50: float, slope: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    scale = max(abs(float(slope)), 1e-8)
    return 1.0 / (1.0 + np.exp(-np.clip((values - float(x50)) / scale, -60.0, 60.0)))


def life_history(settings: SeasonalSpatialSettings) -> dict[str, np.ndarray]:
    ages = np.arange(settings.max_age + 1, dtype=float)
    lengths = []
    weights = []
    maturity = []
    fecundity = []
    for sex in settings.sexes:
        length = np.maximum(sex.linf_mm * (1.0 - np.exp(-sex.growth_k * (ages - sex.growth_t0))), 1.0)
        weight = sex.weight_a * np.power(length, sex.weight_b)
        if sex.maturity_model == "length_logistic":
            coefficient = float(sex.maturity_slope_coefficient)
            mature = 1.0 / (1.0 + np.exp(np.clip(coefficient * (length - sex.maturity_length50_mm), -60.0, 60.0)))
        elif sex.maturity_model == "none":
            mature = np.zeros_like(ages, dtype=float)
        else:
            mature = logistic(ages, sex.maturity_a50, sex.maturity_slope)
        lengths.append(length)
        weights.append(weight)
        maturity.append(mature)
        fecundity.append(mature * np.power(np.maximum(weight, _EPS), sex.fecundity_power))
    return {
        "ages": ages,
        "length_mm": np.asarray(lengths),
        "weight_kg": np.asarray(weights),
        "maturity": np.asarray(maturity),
        "fecundity": np.asarray(fecundity),
    }


def default_movement(settings: SeasonalSpatialSettings) -> np.ndarray:
    shape = (settings.seasons, len(settings.sexes), settings.max_age + 1, len(settings.areas), len(settings.areas))
    movement = np.zeros(shape, dtype=float)
    identity = np.eye(len(settings.areas), dtype=float)
    movement[...] = identity
    return movement


def validate_movement(settings: SeasonalSpatialSettings) -> np.ndarray:
    movement = default_movement(settings) if settings.movement is None else np.asarray(settings.movement, dtype=float)
    expected = (settings.seasons, len(settings.sexes), settings.max_age + 1, len(settings.areas), len(settings.areas))
    if movement.shape != expected:
        raise ValueError(f"Movement matrix shape must be {expected}; received {movement.shape}.")
    if np.any(movement < -1e-12):
        raise ValueError("Movement probabilities cannot be negative.")
    row_sums = movement.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every movement origin row must have positive probability.")
    return movement / row_sums


def selectivity_for_year(fleet: FleetSpec, year: int, life: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    block = None
    for candidate in fleet.selectivity_blocks:
        if candidate.start_year <= year <= candidate.end_year:
            block = candidate
            break
    if block is None:
        if not fleet.selectivity_blocks:
            block = SelectivityBlock(-10_000, 10_000, 5.0, 1.2)
        else:
            block = min(fleet.selectivity_blocks, key=lambda value: min(abs(year - value.start_year), abs(year - value.end_year)))
    ages = life["ages"]
    selectivity_age = logistic(ages, block.a50, block.slope)
    retention = logistic(life["length_mm"], block.retention_length50_mm, block.retention_slope_mm)
    return np.broadcast_to(selectivity_age, retention.shape), retention, max(block.fishing_power, 0.0)


def effective_discard_mortality(fleet: FleetSpec) -> float:
    if not fleet.discard_mortality_by_depth:
        return 0.0
    shares = np.asarray([max(band.share, 0.0) for band in fleet.discard_mortality_by_depth], dtype=float)
    if shares.sum() <= 0:
        shares[:] = 1.0
    shares /= shares.sum()
    mortalities = np.asarray([np.clip(band.mortality, 0.0, 1.0) for band in fleet.discard_mortality_by_depth], dtype=float)
    return float(np.dot(shares, mortalities))


def _initial_numbers(settings: SeasonalSpatialSettings) -> np.ndarray:
    sex_n = len(settings.sexes)
    area_n = len(settings.areas)
    ages = np.arange(settings.max_age + 1, dtype=float)
    numbers = np.zeros((sex_n, area_n, settings.max_age + 1), dtype=float)
    area_shares = np.asarray([max(a.recruitment_share, 0.0) for a in settings.areas], dtype=float)
    area_shares = area_shares / max(area_shares.sum(), _EPS)
    sex_shares = np.asarray([max(s.fraction_at_recruitment, 0.0) for s in settings.sexes], dtype=float)
    sex_shares = sex_shares / max(sex_shares.sum(), _EPS)
    for s, sex in enumerate(settings.sexes):
        survival = np.exp(-max(sex.natural_mortality, 1e-9) * ages)
        survival[-1] /= max(1.0 - exp(-max(sex.natural_mortality, 1e-9)), _EPS)
        for a in range(area_n):
            numbers[s, a] = settings.r0 * sex_shares[s] * area_shares[a] * survival * settings.initial_depletion
    return numbers


def _spawning_output(numbers: np.ndarray, life: dict[str, np.ndarray]) -> float:
    return float(np.sum(numbers * life["fecundity"][:, None, :]))


def _beverton_holt(ssb: float, b0: float, r0: float, steepness: float) -> float:
    h = float(np.clip(steepness, 0.2001, 0.999))
    numerator = 4.0 * h * r0 * max(ssb, 0.0)
    denominator = max(b0 * (1.0 - h) + max(ssb, 0.0) * (5.0 * h - 1.0), _EPS)
    return max(numerator / denominator, _EPS)


def _fleet_catch_at_f(
    numbers: np.ndarray,
    f_scalar: float,
    fleet: FleetSpec,
    area_shares: np.ndarray,
    season_share: float,
    selectivity: np.ndarray,
    retention: np.ndarray,
    life: dict[str, np.ndarray],
    natural_mortality: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    dm = effective_discard_mortality(fleet)
    encounter = max(f_scalar, 0.0) * max(season_share, 0.0) * selectivity[:, None, :] * area_shares[None, :, None]
    retained_f = encounter * retention[:, None, :]
    dead_discard_f = encounter * (1.0 - retention[:, None, :]) * dm
    dead_f = retained_f + dead_discard_f
    z = natural_mortality[:, None, None] + dead_f
    baranov = (1.0 - np.exp(-z)) / np.maximum(z, _EPS)
    landed_numbers = numbers * retained_f * baranov
    discard_numbers = numbers * dead_discard_f * baranov
    landed = float(np.sum(landed_numbers * life["weight_kg"][:, None, :]))
    discarded_dead = float(np.sum(discard_numbers * life["weight_kg"][:, None, :]))
    return landed, discarded_dead, dead_f


def solve_f_for_catch(
    numbers: np.ndarray,
    target_catch: float,
    fleet: FleetSpec,
    year: int,
    season: int,
    settings: SeasonalSpatialSettings,
    life: dict[str, np.ndarray],
) -> tuple[float, float, float, np.ndarray]:
    target = max(float(target_catch), 0.0)
    area_shares = np.asarray(fleet.area_shares, dtype=float)
    if area_shares.size != len(settings.areas):
        raise ValueError(f"Fleet {fleet.name} area_shares must have {len(settings.areas)} entries.")
    area_shares = np.maximum(area_shares, 0.0)
    area_shares /= max(area_shares.sum(), _EPS)
    if len(fleet.season_shares) != settings.seasons:
        raise ValueError(f"Fleet {fleet.name} season_shares must have {settings.seasons} entries.")
    season_shares = np.maximum(np.asarray(fleet.season_shares, dtype=float), 0.0)
    season_shares /= max(season_shares.sum(), _EPS)
    sel, retention, power = selectivity_for_year(fleet, year, life)
    natural_mortality = np.asarray([max(sex.natural_mortality, 1e-9) / settings.seasons for sex in settings.sexes])

    def evaluate(f_value: float) -> tuple[float, float, np.ndarray]:
        return _fleet_catch_at_f(numbers, f_value * power, fleet, area_shares, season_shares[season], sel, retention, life, natural_mortality)

    if target <= 0:
        landed, discarded, dead_f = evaluate(0.0)
        return 0.0, landed, discarded, dead_f
    lo, hi = 0.0, 0.5
    landed, discarded, dead_f = evaluate(hi)
    while landed < target and hi < 20.0:
        hi *= 2.0
        landed, discarded, dead_f = evaluate(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        landed_mid, discard_mid, dead_mid = evaluate(mid)
        if landed_mid < target:
            lo = mid
        else:
            hi = mid
            landed, discarded, dead_f = landed_mid, discard_mid, dead_mid
    return hi, landed, discarded, dead_f


def simulate_spatial_seasonal(
    settings: SeasonalSpatialSettings,
    catch_by_fleet: Mapping[str, Sequence[float]] | None = None,
    recruitment_deviations: Sequence[float] | None = None,
    environmental_effect: Sequence[float] | None = None,
    seed: int = 14531,
) -> SpatialSimulationResult:
    if settings.years < 1 or settings.seasons < 1 or settings.max_age < 1:
        raise ValueError("years, seasons and max_age must be positive.")
    life = life_history(settings)
    movement = validate_movement(settings)
    numbers = _initial_numbers(settings)
    unfished_numbers = _initial_numbers(replace(settings, initial_depletion=1.0))
    b0 = _spawning_output(unfished_numbers, life)
    rng = np.random.default_rng(seed)
    stochastic_recruitment = recruitment_deviations is None
    recruitment_deviations = np.zeros(settings.years) if stochastic_recruitment else np.asarray(recruitment_deviations, dtype=float)
    environmental_effect = np.zeros(settings.years) if environmental_effect is None else np.asarray(environmental_effect, dtype=float)
    if len(recruitment_deviations) < settings.years or len(environmental_effect) < settings.years:
        raise ValueError("Recruitment and environmental effect vectors must cover every model year.")
    catches = catch_by_fleet or {}
    for fleet in settings.fleets:
        if fleet.name in catches and len(catches[fleet.name]) < settings.years:
            raise ValueError(f"Catch series for {fleet.name} must cover every model year.")

    history: list[dict[str, float | int | str]] = []
    fleet_history: list[dict[str, float | int | str]] = []
    age_rows: list[dict[str, float | int | str]] = []
    previous_dev = 0.0

    for y in range(settings.years):
        year = settings.start_year + y
        annual_spawn = _spawning_output(numbers, life)
        total_landings = 0.0
        total_dead_discards = 0.0
        annual_f = 0.0
        for season in range(settings.seasons):
            # Movement is applied at the start of each season.
            moved = np.zeros_like(numbers)
            for s in range(len(settings.sexes)):
                for age in range(settings.max_age + 1):
                    moved[s, :, age] = numbers[s, :, age] @ movement[season, s, age]
            numbers = moved

            if season == settings.recruitment_season:
                dev = settings.recruitment_rho * previous_dev + recruitment_deviations[y]
                if stochastic_recruitment:
                    dev += rng.normal(0.0, settings.recruitment_sigma)
                expected = _beverton_holt(annual_spawn, b0, settings.r0, settings.steepness)
                recruitment = expected * exp(dev - 0.5 * settings.recruitment_sigma**2 + environmental_effect[y])
                area_shares = np.asarray([max(area.recruitment_share * area.productivity_multiplier, 0.0) for area in settings.areas])
                area_shares /= max(area_shares.sum(), _EPS)
                sex_shares = np.asarray([max(sex.fraction_at_recruitment, 0.0) for sex in settings.sexes])
                sex_shares /= max(sex_shares.sum(), _EPS)
                for s in range(len(settings.sexes)):
                    numbers[s, :, 0] += recruitment * sex_shares[s] * area_shares
                previous_dev = dev

            total_dead_f = np.zeros_like(numbers)
            for fleet in settings.fleets:
                target_annual = float(catches.get(fleet.name, [0.0] * settings.years)[y])
                shares = np.maximum(np.asarray(fleet.season_shares, dtype=float), 0.0)
                shares /= max(shares.sum(), _EPS)
                target_season = target_annual * shares[season]
                f_scalar, landed, dead_discard, dead_f = solve_f_for_catch(numbers, target_season, fleet, year, season, settings, life)
                total_dead_f += dead_f
                total_landings += landed
                total_dead_discards += dead_discard
                annual_f = max(annual_f, f_scalar)
                fleet_history.append({
                    "year": year,
                    "season": season + 1,
                    "fleet": fleet.name,
                    "sector": fleet.sector,
                    "target_landed_catch": target_season,
                    "predicted_landed_catch": landed,
                    "dead_discard_biomass": dead_discard,
                    "f_scalar": f_scalar,
                    "effective_discard_mortality": effective_discard_mortality(fleet),
                })

            m = np.asarray([max(sex.natural_mortality, 1e-9) / settings.seasons for sex in settings.sexes])[:, None, None]
            numbers *= np.exp(-(m + total_dead_f))
            if season == settings.spawning_season:
                annual_spawn = _spawning_output(numbers, life)

            if season == settings.seasons - 1:
                aged = np.zeros_like(numbers)
                aged[:, :, 1:-1] = numbers[:, :, :-2]
                aged[:, :, -1] = numbers[:, :, -2] + numbers[:, :, -1]
                numbers = aged

        total_biomass = float(np.sum(numbers * life["weight_kg"][:, None, :]))
        spawning = _spawning_output(numbers, life)
        history.append({
            "year": year,
            "total_biomass": total_biomass,
            "spawning_output": spawning,
            "depletion": spawning / max(b0, _EPS),
            "landed_catch": total_landings,
            "dead_discard_biomass": total_dead_discards,
            "maximum_f_scalar": annual_f,
        })
        for s, sex in enumerate(settings.sexes):
            for a, area in enumerate(settings.areas):
                for age in range(settings.max_age + 1):
                    age_rows.append({
                        "year": year,
                        "sex": sex.name,
                        "area": area.name,
                        "age": age,
                        "numbers": float(numbers[s, a, age]),
                        "biomass": float(numbers[s, a, age] * life["weight_kg"][s, age]),
                    })

    diagnostics = {
        "sexes": len(settings.sexes),
        "areas": len(settings.areas),
        "seasons": settings.seasons,
        "fleets": len(settings.fleets),
        "movement_rows_sum_to_one": bool(np.allclose(movement.sum(axis=-1), 1.0)),
        "terminal_depletion": float(history[-1]["depletion"]),
        "total_landed_catch": float(sum(float(row["landed_catch"]) for row in history)),
        "total_dead_discards": float(sum(float(row["dead_discard_biomass"]) for row in history)),
    }
    serializable = asdict(settings)
    serializable["movement"] = None if settings.movement is None else "array"
    return SpatialSimulationResult(serializable, history, fleet_history, age_rows, numbers.copy(), diagnostics)


def default_wa_demersal_settings(start_year: int = 2000, years: int = 20, max_age: int = 30) -> SeasonalSpatialSettings:
    blocks = (
        SelectivityBlock(start_year, start_year + max(years // 2 - 1, 0), 4.5, 1.2, 500.0, 35.0, 1.0),
        SelectivityBlock(start_year + max(years // 2, 1), start_year + years - 1, 5.0, 1.0, 500.0, 30.0, 1.1),
    )
    fleets = (
        FleetSpec("commercial", "commercial", (0.30, 0.40, 0.30), (0.25, 0.25, 0.25, 0.25), blocks,
                  (DepthMortalityBand(0, 39.99, 0.25, 0.05), DepthMortalityBand(40, 79.99, 0.50, 0.25), DepthMortalityBand(80, 2000, 0.80, 0.70))),
        FleetSpec("charter", "charter", (0.25, 0.45, 0.30), (0.20, 0.30, 0.30, 0.20), blocks,
                  (DepthMortalityBand(0, 39.99, 0.25, 0.25), DepthMortalityBand(40, 79.99, 0.50, 0.55), DepthMortalityBand(80, 2000, 0.80, 0.20))),
        FleetSpec("recreational", "recreational", (0.35, 0.40, 0.25), (0.20, 0.30, 0.30, 0.20), blocks,
                  (DepthMortalityBand(0, 39.99, 0.25, 0.55), DepthMortalityBand(40, 79.99, 0.50, 0.35), DepthMortalityBand(80, 2000, 0.80, 0.10))),
    )
    settings = SeasonalSpatialSettings(start_year=start_year, years=years, max_age=max_age, fleets=fleets)
    movement = default_movement(settings)
    # Small adjacent-area movement with greater adult mobility.
    for season in range(settings.seasons):
        for sex in range(len(settings.sexes)):
            for age in range(settings.max_age + 1):
                move = min(0.02 + 0.002 * age, 0.12)
                for origin in range(len(settings.areas)):
                    row = np.zeros(len(settings.areas))
                    row[origin] = 1.0
                    if origin > 0:
                        row[origin] -= move / (2 if origin < len(settings.areas) - 1 else 1)
                        row[origin - 1] += move / (2 if origin < len(settings.areas) - 1 else 1)
                    if origin < len(settings.areas) - 1:
                        row[origin] -= move / (2 if origin > 0 else 1)
                        row[origin + 1] += move / (2 if origin > 0 else 1)
                    movement[season, sex, age, origin] = row
    return replace(settings, movement=movement)
