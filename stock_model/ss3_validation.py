from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NOAA_REPOSITORY = "nmfs-ost/ss3-test-models"
NOAA_REPOSITORY_URL = "https://github.com/nmfs-ost/ss3-test-models"
NOAA_RAW_ROOT = "https://raw.githubusercontent.com/nmfs-ost/ss3-test-models"
NOAA_API_ROOT = "https://api.github.com/repos/nmfs-ost/ss3-test-models"
SS3_SOURCE_REPOSITORY = "https://github.com/nmfs-ost/ss3-source-code"
SS3_RELEASES_API = "https://api.github.com/repos/nmfs-ost/ss3-source-code/releases/latest"
# Pin validation to a known repository snapshot so results are reproducible.
NOAA_VALIDATION_COMMIT = "3d1f9c0aad7e439a73bd807b02d0ffe4d7b3b944"


NOAA_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "Simple": {
        "features": ["two_sexes", "length_compositions", "age_compositions", "two_surveys", "estimated_growth", "beverton_holt"],
        "tier": "core",
    },
    "Simple_NoCPUE": {
        "features": ["no_abundance_index", "two_sexes", "compositions"],
        "tier": "core",
    },
    "Simple_with_Discard": {
        "features": ["discard_data", "retention", "discard_mortality", "two_sexes"],
        "tier": "core",
    },
    "Simple_with_DM_sizefreq": {
        "features": ["dirichlet_multinomial", "generalized_size_composition", "two_areas", "two_seasons"],
        "tier": "advanced",
    },
    "three_area_nomove": {
        "features": ["three_areas", "spatial_structure", "no_movement"],
        "tier": "advanced",
    },
    "two_morph_seas_areas": {
        "features": ["two_areas", "movement", "two_growth_patterns", "two_seasons", "generalized_size_composition"],
        "tier": "advanced",
    },
    "tagging_mirrored_sel": {
        "features": ["tagging", "time_varying_growth", "compound_selectivity", "environmental_data", "mirrored_selectivity"],
        "tier": "advanced",
    },
    "Hake_2018": {
        "features": ["empirical_weight_at_age", "dirichlet_multinomial", "time_varying_selectivity"],
        "tier": "advanced",
    },
    "BigSkate_2019": {
        "features": ["hybrid_f_method", "growth_cessation", "catchability_blocks", "discard_data", "conditional_age_at_length"],
        "tier": "stress",
    },
    "Sablefish2015": {
        "features": ["cubic_spline_selectivity", "male_offsets", "double_normal_selectivity", "special_recruitment_deviations"],
        "tier": "stress",
    },
}


OMEGA_CAPABILITIES: dict[str, str] = {
    "two_sexes": "implemented",
    "length_compositions": "implemented",
    "age_compositions": "implemented",
    "two_surveys": "implemented",
    "estimated_growth": "partial",
    "beverton_holt": "implemented",
    "no_abundance_index": "implemented",
    "compositions": "implemented",
    "discard_data": "implemented",
    "retention": "implemented",
    "discard_mortality": "implemented",
    "dirichlet_multinomial": "implemented",
    "generalized_size_composition": "partial",
    "two_areas": "implemented",
    "two_seasons": "implemented",
    "three_areas": "implemented",
    "spatial_structure": "implemented",
    "no_movement": "implemented",
    "movement": "implemented",
    "two_growth_patterns": "partial",
    "tagging": "implemented",
    "time_varying_growth": "partial",
    "compound_selectivity": "partial",
    "environmental_data": "implemented",
    "mirrored_selectivity": "partial",
    "empirical_weight_at_age": "partial",
    "time_varying_selectivity": "implemented",
    "hybrid_f_method": "partial",
    "growth_cessation": "not_implemented",
    "catchability_blocks": "partial",
    "conditional_age_at_length": "partial",
    "cubic_spline_selectivity": "not_implemented",
    "male_offsets": "partial",
    "double_normal_selectivity": "not_implemented",
    "special_recruitment_deviations": "partial",
}


@dataclass(frozen=True)
class SS3StarterSummary:
    version: str | None
    data_file: str
    control_file: str
    convergence_criterion: float | None
    retrospective_year: int | None
    soft_bounds: bool | None
    raw_sha256: str


@dataclass(frozen=True)
class SS3Fleet:
    number: int
    fleet_type: int
    timing: float
    area: int
    catch_units: int
    need_catch_multiplier: int
    name: str


@dataclass(frozen=True)
class SS3CatchObservation:
    year: int
    season: int
    fleet: int
    catch: float
    standard_error: float


@dataclass(frozen=True)
class SS3IndexObservation:
    year: int
    month: float
    fleet: int
    observation: float
    standard_error: float


@dataclass
class SS3DataSummary:
    start_year: int
    end_year: int
    seasons: int
    months_per_season: list[float]
    subseasons: int
    spawning_month: float
    sexes: int
    max_age: int
    areas: int
    fleets: list[SS3Fleet] = field(default_factory=list)
    catches: list[SS3CatchObservation] = field(default_factory=list)
    indices: list[SS3IndexObservation] = field(default_factory=list)
    feature_flags: dict[str, bool] = field(default_factory=dict)
    raw_sha256: str = ""


@dataclass
class SS3ControlSummary:
    natural_mortality_type: int | None
    growth_model: int | None
    growth_age_1: float | None
    growth_age_2: float | None
    maturity_option: int | None
    fecundity_option: int | None
    stock_recruit_option: int | None
    fishing_mortality_method: int | None
    parameters: dict[str, float] = field(default_factory=dict)
    phases: dict[str, int] = field(default_factory=dict)
    recruitment_deviations: list[float] = field(default_factory=list)
    fishing_mortality_by_fleet: dict[str, list[float]] = field(default_factory=dict)
    feature_flags: dict[str, bool] = field(default_factory=dict)
    raw_sha256: str = ""


@dataclass
class NOAAValidationResult:
    model_name: str
    source_repository: str
    source_commit: str
    source_mode: str
    checks: list[dict[str, Any]]
    capability_matrix: list[dict[str, Any]]
    starter: dict[str, Any]
    data: dict[str, Any]
    control: dict[str, Any]
    native_ss3: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numeric_prefix(line: str) -> list[float]:
    value = line.split("#", 1)[0].strip()
    if not value:
        return []
    numbers: list[float] = []
    for token in value.replace(",", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            break
    return numbers


def _non_comment_lines(text: str) -> list[str]:
    rows = []
    for raw in text.splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            rows.append(value)
    return rows


def _find_number_before_label(text: str, label: str, cast=float) -> Any | None:
    pattern = re.compile(rf"^\s*([-+0-9.eE]+)\s*#.*{re.escape(label)}", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    try:
        return cast(float(match.group(1)))
    except (TypeError, ValueError):
        return None


def _find_number_before_exact_comment_label(text: str, label: str, cast=float) -> Any | None:
    pattern = re.compile(
        rf"^\s*([-+0-9.eE]+)\s*#\s*{re.escape(label)}(?:\s|;|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return None
    try:
        return cast(float(match.group(1)))
    except (TypeError, ValueError):
        return None


def parse_ss3_starter(text: str) -> SS3StarterSummary:
    cleaned = _non_comment_lines(text)
    if len(cleaned) < 2:
        raise ValueError("SS3 starter file does not contain data and control file names.")
    version_match = re.search(r"#V([0-9.]+)", text)
    return SS3StarterSummary(
        version=version_match.group(1) if version_match else None,
        data_file=cleaned[0].split()[0],
        control_file=cleaned[1].split()[0],
        convergence_criterion=_find_number_before_label(text, "converge_criterion", float),
        retrospective_year=_find_number_before_label(text, "retro_yr", int),
        soft_bounds=(bool(_find_number_before_label(text, "soft_bounds", int)) if _find_number_before_label(text, "soft_bounds", int) is not None else None),
        raw_sha256=sha256_text(text),
    )


def _first_numeric_rows(text: str, count: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in text.splitlines():
        numbers = _numeric_prefix(line)
        if numbers:
            rows.append(numbers)
            if len(rows) >= count:
                break
    return rows


def _section_after_marker(text: str, marker: str, *, case_sensitive: bool = False) -> list[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    match = re.search(marker, text, flags)
    if not match:
        return []
    return text[match.end():].splitlines()


def parse_ss3_data(text: str) -> SS3DataSummary:
    header = _first_numeric_rows(text, 10)
    if len(header) < 10:
        raise ValueError("SS3 data file header is incomplete.")
    start_year = int(header[0][0])
    end_year = int(header[1][0])
    seasons = int(header[2][0])
    months = [float(value) for value in header[3]]
    subseasons = int(header[4][0])
    spawning_month = float(header[5][0])
    sexes = int(header[6][0])
    max_age = int(header[7][0])
    areas = int(header[8][0])
    fleet_count = int(header[9][0])

    fleets: list[SS3Fleet] = []
    fleet_rows = _section_after_marker(text, r"#_fleet_type\s+fishery_timing\s+area\s+catch_units")
    for raw in fleet_rows:
        if len(fleets) >= fleet_count:
            break
        numbers = _numeric_prefix(raw)
        if len(numbers) < 5:
            continue
        before_comment, _, after_comment = raw.partition("#")
        tokens = before_comment.split()
        name = tokens[5] if len(tokens) > 5 else f"Fleet_{len(fleets) + 1}"
        fleets.append(
            SS3Fleet(
                number=len(fleets) + 1,
                fleet_type=int(numbers[0]),
                timing=float(numbers[1]),
                area=int(numbers[2]),
                catch_units=int(numbers[3]),
                need_catch_multiplier=int(numbers[4]),
                name=name or after_comment.strip() or f"Fleet_{len(fleets) + 1}",
            )
        )

    catches: list[SS3CatchObservation] = []
    catch_rows = _section_after_marker(text, r"#_Catch data")
    for raw in catch_rows:
        numbers = _numeric_prefix(raw)
        if len(numbers) < 5:
            continue
        year = int(numbers[0])
        if year == -9999:
            break
        if year == -999:
            continue
        catches.append(SS3CatchObservation(year, int(numbers[1]), int(numbers[2]), float(numbers[3]), float(numbers[4])))

    indices: list[SS3IndexObservation] = []
    index_rows = _section_after_marker(text, r"#_CPUE_and_surveyabundance_and_index_observations")
    observation_started = False
    for raw in index_rows:
        if "#_year" in raw.lower() and "obs" in raw.lower():
            observation_started = True
            continue
        if not observation_started:
            continue
        numbers = _numeric_prefix(raw)
        if len(numbers) < 5:
            continue
        year = int(numbers[0])
        if year == -9999:
            break
        indices.append(SS3IndexObservation(year, float(numbers[1]), int(numbers[2]), float(numbers[3]), float(numbers[4])))

    lower = text.lower()
    flags = {
        "catch": bool(catches),
        "abundance_indices": bool(indices),
        "discard_data": bool(re.search(r"\n\s*[1-9][0-9]*\s*#_n_fleets_with_discard", lower)),
        "length_compositions": "use length composition data" in lower and not bool(re.search(r"\n\s*0\s*#\s*use length composition data", lower)),
        "age_compositions": "age composition" in lower or "agecomp" in lower,
        "mean_size_at_age": "mean size-at-age" in lower or "mean_size_at_age" in lower,
        "environmental_data": "environmental" in lower and "0 #_n_environ" not in lower,
        "tagging": "tag release" in lower or "tag_recapture" in lower or "tagging" in lower,
        "generalized_size_composition": "generalized size" in lower or "sizefreq" in lower,
        "multiple_areas": areas > 1,
        "multiple_seasons": seasons > 1,
        "multiple_sexes": abs(sexes) == 2,
    }
    return SS3DataSummary(
        start_year=start_year,
        end_year=end_year,
        seasons=seasons,
        months_per_season=months,
        subseasons=subseasons,
        spawning_month=spawning_month,
        sexes=sexes,
        max_age=max_age,
        areas=areas,
        fleets=fleets,
        catches=catches,
        indices=indices,
        feature_flags=flags,
        raw_sha256=sha256_text(text),
    )


def _parameter_rows(text: str) -> tuple[dict[str, float], dict[str, int]]:
    parameters: dict[str, float] = {}
    phases: dict[str, int] = {}
    for raw in text.splitlines():
        if "#" not in raw:
            continue
        before, _, comment = raw.partition("#")
        numbers = _numeric_prefix(before)
        if len(numbers) < 7:
            continue
        name = comment.strip().split()[0] if comment.strip() else ""
        if not name or name.startswith("_") or name.lower() in {"cond", "autogen"}:
            continue
        parameters[name] = float(numbers[2])
        phases[name] = int(numbers[6])
    return parameters, phases


def _comment_vector(text: str, label_pattern: str) -> list[float]:
    pattern = re.compile(rf"^\s*#\s*{label_pattern}\s+(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return []
    values: list[float] = []
    for token in match.group(1).split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def parse_ss3_control(text: str) -> SS3ControlSummary:
    parameters, phases = _parameter_rows(text)
    natural_mortality_type = _find_number_before_label(text, "natM_type", int)
    growth_model = _find_number_before_label(text, "GrowthModel", int)
    growth_age_1 = _find_number_before_label(text, "Age(post-settlement) for L1", float)
    growth_age_2 = _find_number_before_label(text, "Age(post-settlement) for L2", float)
    maturity_option = _find_number_before_label(text, "maturity_option", int)
    fecundity_option = _find_number_before_label(text, "fecundity_at_length option", int)
    stock_recruit_option = _find_number_before_exact_comment_label(text, "_Spawner-Recruitment", int)
    fishing_mortality_method = _find_number_before_label(text, "F_Method", int)

    recdevs: list[float] = []
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        if "all recruitment deviations" in raw.lower():
            for candidate in lines[index + 1:index + 5]:
                values = _numeric_prefix(candidate.lstrip("# "))
                if len(values) >= 4:
                    recdevs = values
                    break
            break

    f_by_fleet: dict[str, list[float]] = {}
    f_section = False
    for raw in lines:
        lower_raw = raw.lower()
        if "f rates by fleet" in lower_raw:
            f_section = True
            continue
        if f_section and "#_q_setup" in lower_raw:
            break
        if not f_section or not raw.lstrip().startswith("#"):
            continue
        content = raw.lstrip()[1:].strip()
        if not content or content.startswith("_") or content.lower().startswith(("year:", "seas:")):
            continue
        tokens = content.split()
        if len(tokens) < 4:
            continue
        name = tokens[0]
        values: list[float] = []
        for token in tokens[1:]:
            try:
                values.append(float(token))
            except ValueError:
                values = []
                break
        if len(values) >= 8 and any(value > 0 for value in values):
            f_by_fleet[name] = values

    lower = text.lower()
    flags = {
        "time_varying_parameters": "timevary" in lower or "time-vary" in lower,
        "movement": "n_movement_definitions" in lower and "#_cond 0 # n_movement_definitions" not in lower,
        "recruitment_deviations": bool(recdevs) or "do_recdev" in lower,
        "beverton_holt": stock_recruit_option == 3,
        "ricker": stock_recruit_option == 2,
        "length_based_maturity": maturity_option in {1, 6},
        "age_based_maturity": maturity_option in {2, 3, 4},
        "catchability_power": "link type: 3=power" in lower and bool(re.search(r"\n\s*\d+\s+[36]\s+", text)),
        "cubic_spline_selectivity": "cubic spline" in lower and bool(re.search(r"\n\s*(27|42)\s+", text)),
        "double_normal_selectivity": "double_normal" in lower and bool(re.search(r"\n\s*(2|22|23|24)\s+", text)),
        "growth_cessation": growth_model == 8,
        "empirical_weight_at_age": bool(re.search(r"^\s*1\s*#.*read wtatage", text, re.IGNORECASE | re.MULTILINE)),
    }
    return SS3ControlSummary(
        natural_mortality_type=natural_mortality_type,
        growth_model=growth_model,
        growth_age_1=growth_age_1,
        growth_age_2=growth_age_2,
        maturity_option=maturity_option,
        fecundity_option=fecundity_option,
        stock_recruit_option=stock_recruit_option,
        fishing_mortality_method=fishing_mortality_method,
        parameters=parameters,
        phases=phases,
        recruitment_deviations=recdevs,
        fishing_mortality_by_fleet=f_by_fleet,
        feature_flags=flags,
        raw_sha256=sha256_text(text),
    )


def ss3_l1_l2_to_linf_t0(length_at_age1: float, length_at_age2: float, age1: float, age2: float, growth_k: float) -> tuple[float, float]:
    if growth_k <= 0 or age2 <= age1:
        raise ValueError("Growth K must be positive and age2 must exceed age1.")
    decay = math.exp(-growth_k * (age2 - age1))
    denominator = 1.0 - decay
    if abs(denominator) < 1e-12:
        raise ValueError("Growth parameters do not identify Linf.")
    linf = (length_at_age2 - length_at_age1 * decay) / denominator
    if linf <= max(length_at_age1, length_at_age2):
        raise ValueError("Converted Linf is not greater than observed reference lengths.")
    t0 = age1 + math.log(1.0 - length_at_age1 / linf) / growth_k
    return float(linf), float(t0)


def von_bertalanffy_length(ages: Sequence[float], linf: float, growth_k: float, t0: float) -> np.ndarray:
    values = np.asarray(ages, dtype=float)
    return linf * (1.0 - np.exp(-growth_k * (values - t0)))


def weight_at_length(length: Sequence[float] | float, coefficient: float, exponent: float) -> np.ndarray:
    return coefficient * np.power(np.maximum(np.asarray(length, dtype=float), 0.0), exponent)


def ss3_length_logistic(length: Sequence[float] | float, length50: float, slope_coefficient: float) -> np.ndarray:
    values = np.asarray(length, dtype=float)
    exponent = np.clip(slope_coefficient * (values - length50), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(exponent))


def selectivity_from_inflection_width(length: Sequence[float] | float, inflection: float, width_95: float) -> np.ndarray:
    if width_95 <= 0:
        raise ValueError("The 5–95% selectivity width must be positive.")
    scale = width_95 / (2.0 * math.log(19.0))
    values = np.asarray(length, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip((values - inflection) / scale, -60.0, 60.0)))


def beverton_holt_recruitment(ssb: float, unfished_ssb: float, r0: float, steepness: float) -> float:
    h = float(np.clip(steepness, 0.200001, 0.999999))
    numerator = 4.0 * h * r0 * max(float(ssb), 0.0)
    denominator = unfished_ssb * (1.0 - h) + max(float(ssb), 0.0) * (5.0 * h - 1.0)
    return float(numerator / max(denominator, 1e-12))


def capability_matrix(model_name: str | None = None) -> list[dict[str, Any]]:
    names = [model_name] if model_name else list(NOAA_MODEL_CATALOG)
    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in NOAA_MODEL_CATALOG:
            continue
        for feature in NOAA_MODEL_CATALOG[name]["features"]:
            status = OMEGA_CAPABILITIES.get(feature, "not_assessed")
            rows.append(
                {
                    "model": name,
                    "tier": NOAA_MODEL_CATALOG[name]["tier"],
                    "feature": feature,
                    "omega_status": status,
                    "parity": status == "implemented",
                }
            )
    return rows


def _check(name: str, expected: Any, actual: Any, tolerance: float | None = None, *, category: str = "parser", detail: str = "") -> dict[str, Any]:
    if tolerance is None:
        passed = expected == actual
        difference = None
    else:
        difference = abs(float(actual) - float(expected))
        passed = difference <= tolerance
    return {
        "name": name,
        "category": category,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "difference": difference,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def validate_noaa_simple(starter_text: str, data_text: str, control_text: str, *, source_mode: str = "embedded_fixture") -> NOAAValidationResult:
    starter = parse_ss3_starter(starter_text)
    data = parse_ss3_data(data_text)
    control = parse_ss3_control(control_text)
    checks: list[dict[str, Any]] = []
    checks.extend(
        [
            _check("Start year", 1971, data.start_year),
            _check("End year", 2001, data.end_year),
            _check("Seasons", 1, data.seasons),
            _check("Sexes", 2, data.sexes),
            _check("Maximum age", 40, data.max_age),
            _check("Areas", 1, data.areas),
            _check("Fleets and surveys", 3, len(data.fleets)),
            _check("Historical catch rows", 31, len(data.catches)),
            _check("Abundance-index rows", 21, len(data.indices)),
            _check("Natural mortality model", 0, control.natural_mortality_type),
            _check("Growth model", 1, control.growth_model),
            _check("Stock-recruit model", 3, control.stock_recruit_option),
            _check("Fishing mortality method", 3, control.fishing_mortality_method),
        ]
    )

    required_parameters = {
        "NatM_uniform_Fem_GP_1": 0.1,
        "L_at_Amin_Fem_GP_1": 21.6591,
        "L_at_Amax_Fem_GP_1": 71.654,
        "VonBert_K_Fem_GP_1": 0.14724,
        "Wtlen_1_Fem_GP_1": 2.44e-06,
        "Wtlen_2_Fem_GP_1": 3.34694,
        "SR_LN(R0)": 8.81206,
        "SR_BH_steep": 0.573835,
        "SR_sigmaR": 0.6,
    }
    for parameter, expected in required_parameters.items():
        actual = control.parameters.get(parameter)
        checks.append(_check(f"Parameter {parameter}", expected, actual if actual is not None else float("nan"), 1e-8 if abs(expected) < 1 else 1e-5, category="parameter"))

    try:
        linf, t0 = ss3_l1_l2_to_linf_t0(
            control.parameters["L_at_Amin_Fem_GP_1"],
            control.parameters["L_at_Amax_Fem_GP_1"],
            float(control.growth_age_1 or 0.0),
            float(control.growth_age_2 or 25.0),
            control.parameters["VonBert_K_Fem_GP_1"],
        )
        reconstructed = von_bertalanffy_length(
            [float(control.growth_age_1 or 0.0), float(control.growth_age_2 or 25.0)],
            linf,
            control.parameters["VonBert_K_Fem_GP_1"],
            t0,
        )
        checks.append(_check("Female growth L1 reconstruction", 21.6591, float(reconstructed[0]), 1e-8, category="life_history"))
        checks.append(_check("Female growth L2 reconstruction", 71.654, float(reconstructed[1]), 1e-8, category="life_history"))
        checks.append(_check("Converted Linf", 72.94632337479932, linf, 1e-8, category="life_history"))
        checks.append(_check("Converted t0", -2.3925713821956145, t0, 1e-8, category="life_history"))
    except Exception as exc:
        checks.append({"name": "Female growth conversion", "category": "life_history", "status": "FAIL", "detail": str(exc), "expected": "valid", "actual": "error", "tolerance": None, "difference": None})

    female_m = control.parameters.get("NatM_uniform_Fem_GP_1", float("nan"))
    checks.append(_check("Annual survival from M", math.exp(-0.1), math.exp(-female_m), 1e-12, category="mortality"))
    weight_50 = float(weight_at_length(50.0, control.parameters.get("Wtlen_1_Fem_GP_1", float("nan")), control.parameters.get("Wtlen_2_Fem_GP_1", float("nan"))))
    checks.append(_check("Weight at length 50", 1.1850604182217688, weight_50, 1e-10, category="life_history"))

    h = control.parameters.get("SR_BH_steep", float("nan"))
    r0 = math.exp(control.parameters.get("SR_LN(R0)", float("nan")))
    recruitment_at_b0 = beverton_holt_recruitment(1000.0, 1000.0, r0, h)
    checks.append(_check("Beverton-Holt returns R0 at B0", r0, recruitment_at_b0, 1e-8 * max(r0, 1.0), category="recruitment"))

    f_values = next(iter(control.fishing_mortality_by_fleet.values()), [])
    checks.append(_check("Fishing-mortality reference vector detected", True, len(f_values) >= 31, category="reference_output"))
    checks.append(_check("Recruitment-deviation reference vector detected", True, len(control.recruitment_deviations) >= 31, category="reference_output"))

    matrix = capability_matrix("Simple")
    passed = sum(row["status"] == "PASS" for row in checks)
    failed = len(checks) - passed
    parity_count = sum(row["parity"] for row in matrix)
    summary = {
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": failed,
        "validation_status": "PASS" if failed == 0 else "FAIL",
        "capabilities_total": len(matrix),
        "capabilities_at_parity": parity_count,
        "capability_parity_fraction": parity_count / max(len(matrix), 1),
        "claim_limit": "Passing this suite verifies parsing and selected deterministic equations. It does not prove full SS3 numerical equivalence.",
    }
    return NOAAValidationResult(
        model_name="Simple",
        source_repository=NOAA_REPOSITORY_URL,
        source_commit=NOAA_VALIDATION_COMMIT,
        source_mode=source_mode,
        checks=checks,
        capability_matrix=matrix,
        starter=asdict(starter),
        data={**asdict(data), "fleets": [asdict(row) for row in data.fleets], "catches": [asdict(row) for row in data.catches], "indices": [asdict(row) for row in data.indices]},
        control=asdict(control),
        summary=summary,
    )


def _request_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Omega-FISH-Model/1.1 NOAA-validation"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_bytes(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Omega-FISH-Model/1.1 NOAA-validation"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_noaa_model(model_name: str, destination: str | Path, *, commit: str = NOAA_VALIDATION_COMMIT, timeout: float = 30.0) -> dict[str, Any]:
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError("model_name must be a single NOAA model folder name.")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    encoded_model = urllib.parse.quote(model_name)
    listing_url = f"{NOAA_API_ROOT}/contents/models/{encoded_model}?ref={urllib.parse.quote(commit)}"
    listing = _request_json(listing_url, timeout=timeout)
    if not isinstance(listing, list):
        raise ValueError(f"NOAA model listing was not returned for {model_name}.")
    files: list[dict[str, Any]] = []
    for item in listing:
        if item.get("type") != "file":
            continue
        download_url = item.get("download_url")
        if not download_url:
            continue
        content = _download_bytes(download_url, timeout=timeout)
        path = output / str(item["name"])
        path.write_bytes(content)
        files.append({"name": path.name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "source_url": download_url})
    if not files:
        raise ValueError(f"No files were downloaded for NOAA model {model_name}.")
    manifest = {
        "model": model_name,
        "repository": NOAA_REPOSITORY_URL,
        "commit": commit,
        "downloaded_files": files,
        "warning": "NOAA states that ss3-test-models are for software testing and may have altered data; they must not be treated as assessment data sources.",
    }
    (output / "omega_noaa_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_latest_ss3_executable(
    destination: str | Path,
    *,
    platform_name: str = "windows",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Download an official SS3 release asset and return a hashed local executable.

    The release asset naming has changed over time, so selection is score based.
    A ZIP is extracted into a versioned folder and the first plausible SS3
    executable is selected. No downloaded binary is executed automatically.
    """
    release = _request_json(SS3_RELEASES_API, timeout=timeout)
    assets = release.get("assets", []) if isinstance(release, dict) else []
    if not assets:
        raise ValueError("The official SS3 release did not contain downloadable assets.")

    platform_tokens = {
        "windows": ("windows", "win", "w64", "x64"),
        "linux": ("linux", "ubuntu", "x86_64"),
        "macos": ("mac", "macos", "darwin", "osx"),
    }.get(platform_name.lower(), (platform_name.lower(),))

    def score(asset: Mapping[str, Any]) -> tuple[int, int]:
        name = str(asset.get("name", "")).lower()
        value = 0
        if "ss3" in name or "stock" in name:
            value += 5
        if any(token in name for token in platform_tokens):
            value += 8
        if name.endswith(".exe"):
            value += 5 if platform_name.lower() == "windows" else 0
        if name.endswith(".zip"):
            value += 3
        if "source" in name or "code" in name:
            value -= 8
        return value, int(asset.get("size", 0) or 0)

    ranked = sorted(assets, key=score, reverse=True)
    selected = ranked[0]
    if score(selected)[0] <= 0:
        raise ValueError(f"No plausible {platform_name} SS3 release asset was found.")
    url = selected.get("browser_download_url")
    if not url:
        raise ValueError("Selected SS3 release asset has no download URL.")

    root = Path(destination)
    tag = str(release.get("tag_name") or "latest").replace("/", "_")
    output = root / tag
    output.mkdir(parents=True, exist_ok=True)
    asset_name = str(selected.get("name") or "ss3_release_asset")
    asset_path = output / asset_name
    payload = _download_bytes(str(url), timeout=timeout)
    asset_path.write_bytes(payload)

    executable_path: Path | None = None
    if asset_path.suffix.lower() == ".zip":
        extract_root = output / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(asset_path) as archive:
            archive.extractall(extract_root)
        candidates = sorted(
            [path for path in extract_root.rglob("*") if path.is_file() and path.suffix.lower() == ".exe"],
            key=lambda path: ("ss3" not in path.name.lower(), len(path.parts), path.name.lower()),
        )
        if platform_name.lower() != "windows":
            candidates = sorted(
                [path for path in extract_root.rglob("*") if path.is_file() and os.access(path, os.X_OK)],
                key=lambda path: ("ss3" not in path.name.lower(), len(path.parts), path.name.lower()),
            ) or candidates
        if candidates:
            executable_path = candidates[0]
    elif asset_path.suffix.lower() == ".exe" or os.access(asset_path, os.X_OK):
        executable_path = asset_path

    if executable_path is None:
        raise ValueError(f"The downloaded SS3 asset {asset_name} did not contain an executable.")

    manifest = {
        "repository": SS3_SOURCE_REPOSITORY,
        "release_tag": release.get("tag_name"),
        "release_name": release.get("name"),
        "asset_name": asset_name,
        "asset_url": url,
        "asset_sha256": hashlib.sha256(payload).hexdigest(),
        "asset_size": len(payload),
        "executable": str(executable_path),
        "executable_sha256": hashlib.sha256(executable_path.read_bytes()).hexdigest(),
        "download_only": True,
        "warning": "Downloaded from the official NOAA/NMFS SS3 source repository. Omega does not execute the binary until the user explicitly runs a native validation.",
    }
    (output / "omega_ss3_executable_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_native_ss3(model_directory: str | Path, executable: str | Path, *, timeout_seconds: float = 900.0) -> dict[str, Any]:
    model_path = Path(model_directory).resolve()
    executable_path = Path(executable).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not executable_path.is_file():
        raise FileNotFoundError(executable_path)
    with tempfile.TemporaryDirectory(prefix="omega_ss3_") as temporary:
        run_dir = Path(temporary) / model_path.name
        shutil.copytree(model_path, run_dir)
        copied_executable = run_dir / executable_path.name
        shutil.copy2(executable_path, copied_executable)
        command = [str(copied_executable)]
        completed = subprocess.run(command, cwd=run_dir, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        report = run_dir / "Report.sso"
        warning = run_dir / "warning.sso"
        result = {
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-12000:],
            "stderr_tail": completed.stderr[-12000:],
            "report_created": report.exists(),
            "warning_created": warning.exists(),
            "warning_text": warning.read_text(encoding="utf-8", errors="replace")[-12000:] if warning.exists() else "",
            "status": "PASS" if completed.returncode == 0 and report.exists() else "FAIL",
        }
        if report.exists():
            from .ss3_interop import parse_report_sso

            report_text = report.read_text(encoding="utf-8", errors="replace")
            result["report_summary"] = parse_report_sso(report_text)
            result["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        return result


def load_model_file_set(model_directory: str | Path) -> tuple[str, str, str, str | None]:
    directory = Path(model_directory)
    starter_path = directory / "starter.ss"
    if not starter_path.exists():
        candidates = sorted(directory.glob("*starter*.ss"))
        if not candidates:
            raise FileNotFoundError("No starter.ss file was found.")
        starter_path = candidates[0]
    starter_text = starter_path.read_text(encoding="utf-8", errors="replace")
    starter = parse_ss3_starter(starter_text)
    data_path = directory / starter.data_file
    control_path = directory / starter.control_file
    if not data_path.exists() or not control_path.exists():
        raise FileNotFoundError(f"Starter references missing files: {data_path.name}, {control_path.name}")
    forecast_path = directory / "forecast.ss"
    return (
        starter_text,
        data_path.read_text(encoding="utf-8", errors="replace"),
        control_path.read_text(encoding="utf-8", errors="replace"),
        forecast_path.read_text(encoding="utf-8", errors="replace") if forecast_path.exists() else None,
    )


def validate_model_directory(model_directory: str | Path, *, model_name: str | None = None, native_executable: str | Path | None = None) -> NOAAValidationResult:
    starter_text, data_text, control_text, _forecast = load_model_file_set(model_directory)
    model_path = Path(model_directory)
    name = model_name or model_path.name
    if name == "Simple":
        source_mode = "downloaded_noaa_model" if (model_path / "omega_noaa_source_manifest.json").exists() else "embedded_fixture"
        result = validate_noaa_simple(starter_text, data_text, control_text, source_mode=source_mode)
    else:
        starter = parse_ss3_starter(starter_text)
        data = parse_ss3_data(data_text)
        control = parse_ss3_control(control_text)
        checks = [
            _check("Data years ordered", True, data.start_year <= data.end_year),
            _check("At least one fleet", True, len(data.fleets) > 0),
            _check("Positive maximum age", True, data.max_age > 0),
            _check("Control parameters detected", True, len(control.parameters) > 0),
        ]
        matrix = capability_matrix(name)
        passed = sum(row["status"] == "PASS" for row in checks)
        result = NOAAValidationResult(
            model_name=name,
            source_repository=NOAA_REPOSITORY_URL,
            source_commit=NOAA_VALIDATION_COMMIT,
            source_mode="downloaded_noaa_model",
            checks=checks,
            capability_matrix=matrix,
            starter=asdict(starter),
            data={**asdict(data), "fleets": [asdict(row) for row in data.fleets], "catches": [asdict(row) for row in data.catches], "indices": [asdict(row) for row in data.indices]},
            control=asdict(control),
            summary={
                "checks_total": len(checks),
                "checks_passed": passed,
                "checks_failed": len(checks) - passed,
                "validation_status": "PASS" if passed == len(checks) else "FAIL",
                "capabilities_total": len(matrix),
                "capabilities_at_parity": sum(row["parity"] for row in matrix),
                "claim_limit": "This is a structural input audit unless a native SS3 executable is also run.",
            },
        )
    if native_executable is not None:
        result.native_ss3 = run_native_ss3(model_directory, native_executable)
        if result.native_ss3.get("status") != "PASS":
            result.summary["validation_status"] = "FAIL"
    return result


def write_validation_report(result: NOAAValidationResult, output_directory: str | Path) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    json_path = output / f"{result.model_name}_omega_noaa_validation.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    rows = []
    for check in result.checks:
        rows.append(
            f"<tr><td>{check['category']}</td><td>{check['name']}</td><td>{check['expected']}</td><td>{check['actual']}</td><td class='{check['status'].lower()}'>{check['status']}</td></tr>"
        )
    capability_rows = []
    for row in result.capability_matrix:
        capability_rows.append(
            f"<tr><td>{row['model']}</td><td>{row['feature']}</td><td>{row['omega_status']}</td><td>{'YES' if row['parity'] else 'NO'}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Omega FISH NOAA Validation</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#172033}}
header{{background:#0e294b;color:white;padding:28px 36px}}
main{{max-width:1250px;margin:auto;padding:28px}}
.card{{background:white;border-radius:10px;padding:20px;margin-bottom:18px;box-shadow:0 2px 10px rgba(0,0,0,.08)}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #dfe6ef;padding:9px;text-align:left}}th{{background:#eef3f8}}
.pass{{color:#087f5b;font-weight:700}}.fail{{color:#c92a2a;font-weight:700}}
.metric{{display:inline-block;padding:12px 18px;margin:4px;background:#e9f2ff;border-radius:8px}}
</style></head>
<body><header><h1>Omega FISH × NOAA Stock Synthesis Validation</h1><p>{result.model_name} · pinned commit {result.source_commit}</p></header>
<main>
<div class='card'><h2>Validation summary</h2>
<div class='metric'>Status: <b>{result.summary.get('validation_status')}</b></div>
<div class='metric'>Checks: <b>{result.summary.get('checks_passed')}/{result.summary.get('checks_total')}</b></div>
<div class='metric'>Feature parity: <b>{result.summary.get('capabilities_at_parity')}/{result.summary.get('capabilities_total')}</b></div>
<p>{result.summary.get('claim_limit','')}</p></div>
<div class='card'><h2>Deterministic checks</h2><table><tr><th>Category</th><th>Check</th><th>Expected</th><th>Actual</th><th>Status</th></tr>{''.join(rows)}</table></div>
<div class='card'><h2>Feature parity</h2><table><tr><th>NOAA model</th><th>Feature</th><th>Omega status</th><th>Parity</th></tr>{''.join(capability_rows)}</table></div>
<div class='card'><h2>Source control</h2><p>Repository: {result.source_repository}</p><p>Commit: {result.source_commit}</p><p>Mode: {result.source_mode}</p></div>
</main></body></html>"""
    html_path = output / f"{result.model_name}_omega_noaa_validation.html"
    html_path.write_text(html, encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}


def competitive_scorecard() -> list[dict[str, Any]]:
    """Return an evidence-gated roadmap for the objective of exceeding SS3.

    A row is only marked ``omega_advantage_verified`` when the capability is
    implemented in Omega and covered by the local automated suite. Numerical
    equivalence, speed and scientific acceptance remain pending until external
    benchmark evidence exists.
    """
    return [
        {
            "dimension": "Model-form breadth",
            "current_position": "ss3_leads",
            "omega_evidence": "Core age, sex, fleet, space, tagging, compositions and MSE are implemented; several specialist SS3 patterns remain partial.",
            "proof_required": "Pass the full pinned NOAA model catalog and publish per-feature equivalence results.",
        },
        {
            "dimension": "Native numerical equivalence",
            "current_position": "pending",
            "omega_evidence": "Pinned input parsing and selected deterministic equations pass.",
            "proof_required": "Run official SS3 executables and compare objective components, time series, reference points and uncertainty within declared tolerances.",
        },
        {
            "dimension": "Audit trail and provenance",
            "current_position": "omega_advantage_verified",
            "omega_evidence": "Source hashes, transformation logs, assumption registers and run manifests are generated by the application and tested.",
            "proof_required": "Maintain coverage as new import paths are added.",
        },
        {
            "dimension": "Optimizer agreement diagnostics",
            "current_position": "omega_advantage_verified",
            "omega_evidence": "Multiple optimizers, agreement summaries, curvature diagnostics and local-minimum warnings are integrated and tested.",
            "proof_required": "Benchmark recovery and runtime on the NOAA catalog.",
        },
        {
            "dimension": "Predictive and data-conflict diagnostics",
            "current_position": "omega_advantage_verified",
            "omega_evidence": "Walk-forward validation, conflict matrices, influence analysis and reliability grading are integrated and tested.",
            "proof_required": "Validate interpretation on real assessments and simulation-estimation trials.",
        },
        {
            "dimension": "Closed-loop management testing",
            "current_position": "omega_advantage_verified",
            "omega_evidence": "Operating-model, assessment-error, HCR and implementation-error loops are available in one interface.",
            "proof_required": "Cross-check against SSMSE scenarios and publish performance metrics.",
        },
        {
            "dimension": "Interface and explainability",
            "current_position": "omega_advantage_verified",
            "omega_evidence": "Dedicated desktop workspaces expose assumptions, checks, parity gaps and plain-language reliability outcomes.",
            "proof_required": "Structured analyst usability testing.",
        },
        {
            "dimension": "Runtime and scalability",
            "current_position": "unverified",
            "omega_evidence": "No controlled SS3-versus-Omega runtime benchmark has been completed.",
            "proof_required": "Same hardware, same model, repeated timing, memory and convergence comparisons.",
        },
        {
            "dimension": "Uncertainty coverage",
            "current_position": "unverified",
            "omega_evidence": "Profiles, bootstrap and baseline posterior tools exist, but nominal coverage is not yet demonstrated across the NOAA catalog.",
            "proof_required": "Large known-truth simulation study with 50%, 80%, 90% and 95% coverage reporting.",
        },
        {
            "dimension": "Independent scientific acceptance",
            "current_position": "ss3_leads",
            "omega_evidence": "Omega has not yet undergone independent peer review or repeated external assessment use.",
            "proof_required": "Independent code review, scientific review, reproducible case studies and archived releases.",
        },
    ]
