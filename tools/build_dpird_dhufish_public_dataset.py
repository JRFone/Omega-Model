from __future__ import annotations

"""Build the public-evidence WA Dhufish dataset used by Omega.

The script never describes digitised chart values as raw DPIRD observations.
It keeps exact published parameters, derived values, and chart digitisation in
separate fields and records the source document and figure for every series.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "Data_Sets" / "DPIRD" / "West_Australian_Dhufish"
SOURCE_ROOT = DATASET_ROOT / "Public_Sources"
EVIDENCE_ROOT = DATASET_ROOT / "Evidence"
OUTPUT_ROOT = DATASET_ROOT / "Omega_Ready"

ASSESSMENT_PDF = SOURCE_ROOT / "DPIRD_West_Coast_Demersal_2025_RAR2.pdf"

SOURCE_DOCUMENTS = (
    {
        "file": "DPIRD_West_Coast_Demersal_2025_RAR2.pdf",
        "title": "West Coast Demersal Scalefish Resource status assessment 2025",
        "url": "https://library.dpird.wa.gov.au/fish_rar/2/",
        "role": "Current assessment, public input summaries and model outputs",
    },
    {
        "file": "DPIRD_FOP151_2025_External_Review_Response.pdf",
        "title": "External review and departmental response (FOP 151)",
        "url": "https://library.dpird.wa.gov.au/fr_fop/101/",
        "role": "Bespoke/SS bridge configuration, review findings and sensitivities",
    },
    {
        "file": "DPIRD_West_Coast_Demersal_2021_FRR316.pdf",
        "title": "West Coast Demersal Scalefish Resource assessment 2021 (FRR 316)",
        "url": "https://library.dpird.wa.gov.au/fr_rr/133/",
        "role": "Earlier assessment and technical configuration",
    },
    {
        "file": "DPIRD_WCDSR_Synopsis_2024_FRR346.pdf",
        "title": "West Coast Demersal Scalefish Resource synopsis 2024 (FRR 346)",
        "url": "https://library.dpird.wa.gov.au/fr_rr/264/",
        "role": "Published life-history parameter table",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_evidence_pages() -> dict[str, Path]:
    """Render the four assessment figures at fixed, extraction-tested DPI."""
    if not ASSESSMENT_PDF.exists():
        raise FileNotFoundError(f"Missing official assessment PDF: {ASSESSMENT_PDF}")
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    document = fitz.open(ASSESSMENT_PDF)
    specifications = {
        "figure_3_15_retained_catch.png": (43, 180),
        "figure_3_17_cpue.png": (45, 220),
        "figure_3_19_age_composition.png": (48, 220),
        "figure_3_21_assessment_outputs.png": (52, 220),
    }
    rendered: dict[str, Path] = {}
    for name, (page_index, dpi) in specifications.items():
        target = EVIDENCE_ROOT / name
        pixmap = document[page_index].get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(target)
        rendered[name] = target
    document.close()
    return rendered


def _top_colour_y(
    pixels: np.ndarray,
    x: float,
    y_low: int,
    y_high: int,
    colours: set[tuple[int, int, int]],
    radius: int = 3,
) -> int | None:
    candidates: list[int] = []
    for x_value in range(round(x) - radius, round(x) + radius + 1):
        for y_value in range(y_low, y_high + 1):
            if tuple(int(v) for v in pixels[y_value, x_value]) in colours:
                candidates.append(y_value)
    return min(candidates) if candidates else None


def extract_catch(image_path: Path) -> pd.DataFrame:
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    # PyMuPDF and Poppler round a few vector fill channels one unit
    # differently. Both renderings are scientifically identical.
    area_colours = {
        (166, 206, 227), (166, 206, 226),
        (31, 120, 180),
        (178, 223, 138), (177, 223, 137),
        (51, 160, 44), (51, 159, 44),
    }
    catch_colours = area_colours | {(211, 211, 211), (210, 210, 210)}
    records: list[dict[str, float | int | str]] = []
    for year in range(1975, 2025):
        left_x = 303.5 + (year - 1980) * 9.35
        right_x = 879.0 + (year - 1980) * 9.35
        commercial_y = _top_colour_y(pixels, left_x, 190, 555, area_colours)
        charter_y = _top_colour_y(pixels, left_x, 695, 1004, catch_colours)
        total_y = _top_colour_y(pixels, right_x, 690, 1001, {(0, 0, 0)}, radius=2)
        if commercial_y is None or charter_y is None or total_y is None:
            raise RuntimeError(f"Could not digitise all catch series for {year}")
        commercial = max(0.0, (555.0 - commercial_y) / 1.165)
        charter = max(0.0, (1004.0 - charter_y) / 12.8)
        digitised_total_curve = max(0.0, (1004.0 - total_y) / 0.66)
        recreational = max(0.0, digitised_total_curve - commercial - charter)
        # The thin black total curve is partly obscured by stacked bars in a
        # few years (notably 2002-2004). A retained-catch total cannot be below
        # its independently digitised commercial and charter components, so
        # keep the raw curve value for audit and use the exact sector sum as
        # the internally coherent model input.
        total = commercial + charter + recreational
        discrepancy = total - digitised_total_curve
        records.append(
            {
                "year": year,
                "catch": total,
                "catch_total_curve_digitised": digitised_total_curve,
                "catch_total_reconciliation_t": discrepancy,
                "catch_commercial": commercial,
                "catch_charter": charter,
                "catch_recreational": recreational,
                "catch_evidence_class": (
                    "digitised_sector_sum_reconciled_where_total_curve_obscured"
                    if discrepancy > 0.5
                    else "digitised_from_published_figure"
                ),
                "catch_source": "RAR2 Figure 3.15",
            }
        )
    return pd.DataFrame(records)


def extract_catch_by_area(image_path: Path, annual_catch: pd.DataFrame) -> pd.DataFrame:
    """Digitise coloured area shares and reconcile them to sector totals.

    Black bar outlines consume a few pixels, so raw coloured heights are
    rescaled to the independently digitised sector total. Grey reconstructed
    bars do not publish an area split and are retained as ``unknown``.
    """

    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    colours = {
        "kalbarri": {(166, 206, 227), (166, 206, 226)},
        "mid_west": {(31, 120, 180)},
        "metropolitan": {(178, 223, 138), (177, 223, 137)},
        "south_west": {(51, 160, 44), (51, 159, 44)},
    }
    model_area = {"kalbarri": "north", "mid_west": "north", "metropolitan": "south", "south_west": "south"}
    panels = {
        "commercial": ("catch_commercial", 303.5, 190, 555, 1.165),
        "recreational": ("catch_recreational", 879.0, 190, 555, 1.27),
        "charter": ("catch_charter", 303.5, 695, 1004, 12.8),
    }
    rows: list[dict[str, object]] = []
    annual = annual_catch.set_index("year")
    for year in range(1975, 2025):
        for sector, (total_column, x_1980, y_low, y_high, scale) in panels.items():
            x = x_1980 + (year - 1980) * 9.35
            raw: dict[str, float] = {}
            for area, variants in colours.items():
                counts: list[int] = []
                for x_value in range(round(x) - 2, round(x) + 3):
                    counts.append(
                        sum(
                            tuple(int(v) for v in pixels[y_value, x_value]) in variants
                            for y_value in range(y_low, y_high + 1)
                        )
                    )
                raw[area] = float(np.median(counts)) / scale
            raw_total = sum(raw.values())
            sector_total = float(annual.loc[year, total_column])
            if raw_total <= 0.5:
                rows.append(
                    {
                        "year": year,
                        "sector": sector,
                        "management_area": "unknown",
                        "model_area": "unknown",
                        "retained_catch_t": sector_total,
                        "evidence_class": "digitised_reconstructed_total_area_split_not_published",
                        "source": "RAR2 Figure 3.15",
                    }
                )
                continue
            for area, raw_value in raw.items():
                rows.append(
                    {
                        "year": year,
                        "sector": sector,
                        "management_area": area,
                        "model_area": model_area[area],
                        "retained_catch_t": sector_total * raw_value / raw_total,
                        "evidence_class": "digitised_from_published_figure_reconciled_to_sector_total",
                        "source": "RAR2 Figure 3.15",
                    }
                )
    return pd.DataFrame(rows)


def _extract_marker_series(
    pixels: np.ndarray,
    *,
    panel_x_1985: float,
    year_start: int,
    year_end: int,
    y_low: int,
    y_high: int,
    y_zero: float,
    colour: tuple[int, int, int],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for year in range(year_start, year_end + 1):
        x = panel_x_1985 + (year - 1985) * 14.6
        candidates: list[int] = []
        for x_value in range(round(x) - 3, round(x) + 4):
            for y_value in range(y_low + 5, y_high - 5):
                if tuple(int(v) for v in pixels[y_value, x_value]) == colour:
                    candidates.append(y_value)
        if candidates:
            result[year] = max(0.0, (y_zero - float(np.median(candidates))) / 130.0)
    return result


def extract_cpue(image_path: Path) -> pd.DataFrame:
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    panels = {
        "north_dropline": (304.5, 269, 638, 583.5, 1985, 2007, 1985),
        "north_handline": (975.0, 269, 638, 583.5, 1988, 2007, 1988),
        "south_dropline": (304.5, 649, 1018, 962.5, 1985, 2007, 1985),
        "south_handline": (975.0, 649, 1018, 962.5, 1988, 2006, 1988),
    }
    series: dict[str, dict[int, float]] = {}
    for name, (x0, y_low, y_high, y_zero, monthly_start, monthly_end, _) in panels.items():
        series[f"index_{name}_monthly"] = _extract_marker_series(
            pixels,
            panel_x_1985=x0,
            year_start=monthly_start,
            year_end=monthly_end,
            y_low=y_low,
            y_high=y_high,
            y_zero=y_zero,
            colour=(169, 169, 169),
        )
        if name != "south_dropline":
            series[f"index_{name}_daily"] = _extract_marker_series(
                pixels,
                panel_x_1985=x0,
                year_start=2008,
                year_end=2024,
                y_low=y_low,
                y_high=y_high,
                y_zero=y_zero,
                colour=(0, 0, 0),
            )
    years = list(range(1975, 2025))
    output = pd.DataFrame({"year": years})
    for name, values in series.items():
        output[name] = output["year"].map(values)
    daily_columns = [
        "index_north_dropline_daily",
        "index_north_handline_daily",
        "index_south_handline_daily",
    ]
    daily = output[daily_columns]
    output["index"] = np.where(
        daily.notna().all(axis=1),
        np.exp(np.log(daily).mean(axis=1)),
        np.nan,
    )
    base = float(output.loc[output["year"] == 2008, "index"].iloc[0])
    output["index"] = output["index"] / base
    output["index_evidence_class"] = "digitised_from_published_figure"
    output["index_source"] = "RAR2 Figure 3.17"
    return output


def extract_age_composition(image_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    row_ticks = [
        (319.5, 362), (411, 453), (500.5, 544.5), (592, 636),
        (683, 725), (774, 816.5), (864, 908), (955, 999),
        (1046.5, 1089), (1138, 1180), (1227.5, 1271.5),
        (1319, 1363), (1410, 1452), (1501, 1543.5),
        (1591, 1635), (1682, 1724.5), (1773.5, 1816),
        (1865, 1907), (1954.5, 1998.5),
    ]
    sample_sizes = {
        "north": [484, 219, 201, 99, 463, 998, 844, 816, 1138, 908, 1119, 828, 984, 642, 233, 53, 17, 672, 564],
        "south": [98, 476, 517, 362, 535, 460, 420, 605, 694, 727, 772, 897, 1066, 875, 650, 181, 118, 905, 881],
    }
    area_axes = {"north": (293.0, 11.475), "south": (951.5, 11.525)}
    bar_colours = {(169, 169, 169), (139, 0, 0)}
    rows: list[dict[str, float | int | str]] = []
    for area, (x0, x_step) in area_axes.items():
        for row_index, year in enumerate(range(2003, 2022)):
            sample_size = sample_sizes[area][row_index]
            if sample_size <= 100:
                continue
            top, baseline = row_ticks[row_index]
            scale = (baseline - top) / 0.3
            proportions: list[float] = []
            for age in range(41):
                x = x0 + age * x_step
                candidates: list[int] = []
                for x_value in range(round(x) - 3, round(x) + 4):
                    for y_value in range(round(top) - 2, round(baseline) + 1):
                        if tuple(int(v) for v in pixels[y_value, x_value]) in bar_colours:
                            candidates.append(y_value)
                proportion = max(0.0, (baseline - min(candidates)) / scale) if candidates else 0.0
                proportions.append(proportion)
            total = sum(proportions)
            if total <= 0:
                raise RuntimeError(f"No age-composition bars detected for {area} {year}")
            proportions = [value / total for value in proportions]
            for age, proportion in enumerate(proportions):
                rows.append(
                    {
                        "year": year,
                        "area": area,
                        "sector": "all",
                        "age": age,
                        "proportion": proportion,
                        "sample_size": sample_size,
                        "evidence_class": "digitised_from_published_figure",
                        "source": "RAR2 Figure 3.19",
                    }
                )
    by_area = pd.DataFrame(rows)
    pooled_rows: list[dict[str, float | int | str]] = []
    for year, group in by_area.groupby("year"):
        weights = group[["area", "sample_size"]].drop_duplicates().set_index("area")["sample_size"]
        total_n = float(weights.sum())
        for age in range(31):
            if age < 30:
                age_group = group[group["age"] == age].groupby("area", as_index=True)["proportion"].sum()
            else:
                age_group = group[group["age"] >= 30].groupby("area", as_index=True)["proportion"].sum()
            value = sum(float(age_group.loc[area]) * float(weights.loc[area]) for area in age_group.index) / total_n
            pooled_rows.append(
                {
                    "year": int(year),
                    "sector": "all",
                    "age": age,
                    "proportion": value,
                    "sample_size": total_n,
                    "evidence_class": "derived_from_digitised_area_compositions",
                    "source": "RAR2 Figure 3.19; sample-size weighted North/South pool; ages 30+ pooled",
                }
            )
    return by_area, pd.DataFrame(pooled_rows)


def extract_published_assessment_outputs(image_path: Path) -> pd.DataFrame:
    """Digitise central trajectories from DPIRD RAR2 Figure 3.21."""

    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    north = (31, 120, 180)
    south = (105, 138, 105)

    def line_value(
        year: int,
        colour: tuple[int, int, int],
        y_low: int,
        y_high: int,
        y_zero: float,
        scale: float,
        radius: int = 3,
    ) -> float:
        x = 336.0 + (year - 1975) * (1188.0 / 49.0)
        candidates: list[int] = []
        for x_value in range(round(x) - radius, round(x) + radius + 1):
            for y_value in range(y_low, y_high + 1):
                if tuple(int(v) for v in pixels[y_value, x_value]) == colour:
                    candidates.append(y_value)
        return float((y_zero - np.median(candidates)) / scale) if candidates else np.nan

    rows: list[dict[str, object]] = []
    for year in range(1975, 2025):
        rows.append(
            {
                "year": year,
                "published_wcb_relative_female_spawning_biomass": line_value(year, (0, 0, 0), 245, 574, 574.0, 335.0, radius=2),
                "published_north_relative_female_spawning_biomass": line_value(year, north, 700, 1010, 1012.0, 335.0),
                "published_south_relative_female_spawning_biomass": line_value(year, south, 700, 1010, 1012.0, 335.0),
                "published_north_fishing_mortality": line_value(year, north, 1120, 1460, 1466.0, 710.0),
                "published_south_fishing_mortality": line_value(year, south, 1120, 1460, 1466.0, 710.0),
                "published_north_log_recruitment_deviation": line_value(year, north, 1550, 1830, 1688.0, 70.7),
                "published_south_log_recruitment_deviation": line_value(year, south, 1550, 1830, 1688.0, 70.7),
                "evidence_class": "digitised_central_trajectory_from_published_figure",
                "source": "RAR2 Figure 3.21",
            }
        )
    return pd.DataFrame(rows)


def parameter_register() -> pd.DataFrame:
    log_19 = float(np.log(19.0))
    rows = [
        ("species", "WA Dhufish", "text", "all", "published_exact", "FRR346 Table 4.3", "Species identity"),
        ("max_observed_age", 41, "years", "all", "published_exact", "FRR346 Table 4.3", "Observed maximum, not model plus group"),
        ("model_plus_group", 30, "years", "all", "published_exact", "FOP151 Table 4A.1", "Bespoke/bridge model 30+ group"),
        ("natural_mortality_M", 0.11, "year^-1", "all", "published_exact", "FRR346 Table 4.3; FOP151 Table 4A.1", "Fixed base value"),
        ("steepness_h", 0.75, "unitless", "all", "published_exact", "FOP151 Table 4A.1", "Fixed base value"),
        ("recruitment_sigma", 0.60, "log scale", "all", "published_exact", "FOP151 Table 4A.1", "Fixed base value"),
        ("growth_Linf", 983, "mm TL", "female", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("growth_k", 0.12, "year^-1", "female", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("growth_t0", 0.0, "years", "female", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("growth_Linf", 1119, "mm TL", "male", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("growth_k", 0.11, "year^-1", "male", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("growth_t0", 0.0, "years", "male", "published_exact", "FRR346 Table 4.3", "von Bertalanffy"),
        ("weight_length_a", 1.97e-5, "g/mm^b", "all", "published_exact", "FRR346 Table 4.3", "W(g)=a TL(mm)^b"),
        ("weight_length_a_omega", 1.97e-8, "kg/mm^b", "all", "derived_unit_conversion", "FRR346 Table 4.3", "Published coefficient converted grams to kilograms"),
        ("weight_length_b", 2.980, "unitless", "all", "published_exact", "FRR346 Table 4.3", "W=aL^b"),
        ("maturity_A50", 3.83, "years", "female", "published_exact", "FRR346 Table 4.3", "Age at 50% mature"),
        ("maturity_A95", 7.01, "years", "female", "published_exact", "FRR346 Table 4.3", "Age at 95% mature"),
        ("maturity_logistic_slope", (7.01 - 3.83) / log_19, "years", "female", "derived_parameterisation", "FRR346 Table 4.3", "Omega slope=(A95-A50)/ln(19)"),
        ("maturity_A50", 3.37, "years", "male", "published_exact", "FRR346 Table 4.3", "Age at 50% mature"),
        ("maturity_A95", 5.22, "years", "male", "published_exact", "FRR346 Table 4.3", "Age at 95% mature"),
        ("post_release_mortality", 0.50, "proportion", "all", "published_exact", "RAR2 model description; FOP151", "Base assumption; depth-specific uncertainty remains"),
        ("historical_retention_length", 500, "mm TL", "all", "published_rule", "FOP151 review response", "Historical minimum legal length; removed in 2023"),
        ("model_start_year", 1965, "year", "all", "published_exact", "FOP151 Table 4A.1", "Ten-year burn-in before catch series"),
        ("catch_start_year", 1975, "year", "all", "published_exact", "RAR2 Figure 3.15; FOP151", "Modelled retained catches"),
        ("initial_F_lower_bound", 0.03, "year^-1", "all", "published_exact", "FOP151 Table 4A.1", "Initial fishing mortality parameter bound"),
        ("initial_F_upper_bound", 0.07, "year^-1", "all", "published_exact", "FOP151 Table 4A.1", "Initial fishing mortality parameter bound"),
        ("selectivity_A50", np.nan, "years", "all", "not_publicly_available", "FOP151", "Selectivity form is described, exact fitted curve parameters are not public"),
        ("unfished_recruitment_R0", np.nan, "fish", "area", "not_publicly_available", "RAR2/FOP151", "Must be estimated; no public raw run file located"),
        ("initial_depletion", np.nan, "proportion", "area", "not_publicly_available", "RAR2/FOP151", "Do not substitute initial F bounds for depletion"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "units", "scope", "evidence_class", "source", "notes"])


def source_manifest() -> dict[str, object]:
    sources: list[dict[str, object]] = []
    for record in SOURCE_DOCUMENTS:
        path = SOURCE_ROOT / str(record["file"])
        if not path.exists():
            raise FileNotFoundError(f"Missing official source: {path}")
        with fitz.open(path) as document:
            pages = document.page_count
        sources.append(
            {
                **record,
                "local_path": str(path.relative_to(DATASET_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "pdf_pages": pages,
            }
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "WA Dhufish public-evidence reconstruction",
        "sources": sources,
        "raw_dpird_model_files_available": False,
        "raw_dpird_annual_input_tables_available": False,
        "warning": "Chart-digitised values are approximations and are never represented as raw DPIRD data.",
    }


def write_readme() -> None:
    text = """# WA Dhufish — public-evidence Omega dataset

This folder is a transparent reconstruction from official DPIRD publications. It is designed to test whether Omega can reproduce the broad behaviour of the published WA Dhufish assessment and to identify which assumptions deserve further testing.

It is **not** the full DPIRD assessment dataset. DPIRD's raw annual input tables, bespoke ADMB source/run files, fitted selectivity parameters, objective-function components, covariance output, and accepted/rejected run folders were not found in the public releases collected here.

## Evidence classes

- `published_exact`: a value printed in an official DPIRD table or model description.
- `published_rule`: a documented regulation or modelling rule.
- `derived_unit_conversion` / `derived_parameterisation`: a transparent mathematical conversion of an exact value.
- `digitised_from_published_figure`: an approximate value recovered from pixels in an official figure.
- `not_publicly_available`: required information that must not be invented.

## Omega-ready files

- `Omega_Ready/dpird_wa_dhufish_public_reconstruction.csv`: annual retained catch by sector and published CPUE series.
- `Omega_Ready/catch_by_area_sector.csv`: published coloured area shares, reconciled to the independently digitised sector totals; reconstructed grey bars retain an unknown area.
- `Omega_Ready/age_composition.csv`: sample-size-weighted North/South public age-composition reconstruction.
- `Omega_Ready/age_composition_by_area.csv`: the separate public North and South reconstructions.
- `Omega_Ready/published_assessment_outputs_digitised.csv`: digitised DPIRD central biomass, fishing-mortality and recruitment-deviation trajectories for benchmark comparison.
- `Omega_Ready/parameter_register.csv`: published, derived, and missing model parameters.
- `Omega_Ready/source_manifest.json`: official URLs and cryptographic hashes.

The original PDFs remain unchanged in `Public_Sources`. Figure images in `Evidence` are reproducible derivatives used only for digitisation.
"""
    (DATASET_ROOT / "README.md").write_text(text, encoding="utf-8")


def validate_outputs(
    main: pd.DataFrame,
    by_area: pd.DataFrame,
    pooled: pd.DataFrame,
    catch_by_area: pd.DataFrame | None = None,
    published_outputs: pd.DataFrame | None = None,
) -> list[str]:
    checks: list[str] = []
    if list(main["year"]) != list(range(1975, 2025)):
        raise AssertionError("Annual dataset must cover every year from 1975 through 2024")
    terminal = float(main.loc[main["year"] == 2024, "catch"].iloc[0])
    if abs(terminal - 137.0) > 5.0:
        raise AssertionError(f"Digitised 2024 retained catch {terminal:.1f} t is not close to published 137 t")
    checks.append(f"2024 digitised retained catch {terminal:.1f} t agrees within 5 t of published 137 t")
    first_index = float(main.loc[main["year"] == 2008, "index"].iloc[0])
    if abs(first_index - 1.0) > 1e-9:
        raise AssertionError("Primary composite index must be normalised to 1.0 in 2008")
    checks.append("Primary daily CPUE composite is normalised to 1.0 in 2008")
    for frame, group_cols, label in (
        (by_area, ["year", "area"], "area age compositions"),
        (pooled, ["year"], "pooled age compositions"),
    ):
        sums = frame.groupby(group_cols)["proportion"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-8):
            raise AssertionError(f"{label} do not all sum to one")
        checks.append(f"All {label} sum to 1.0")
    if catch_by_area is not None:
        sector_columns = {
            "commercial": "catch_commercial",
            "charter": "catch_charter",
            "recreational": "catch_recreational",
        }
        grouped = catch_by_area.groupby(["year", "sector"])["retained_catch_t"].sum()
        annual = main.set_index("year")
        for (year, sector), value in grouped.items():
            if not np.isclose(float(value), float(annual.loc[year, sector_columns[sector]]), atol=1e-8):
                raise AssertionError(f"Area catches do not reconcile for {sector} {year}")
        checks.append("All available area catch components reconcile to annual sector totals")
        sector_sum = annual[["catch_commercial", "catch_charter", "catch_recreational"]].sum(axis=1)
        if not np.allclose(sector_sum.to_numpy(dtype=float), annual["catch"].to_numpy(dtype=float), atol=1e-8):
            raise AssertionError("Annual retained-catch totals do not reconcile to sector components")
        checks.append("Every annual retained-catch total equals commercial + charter + recreational components")
    if published_outputs is not None:
        terminal = float(
            published_outputs.loc[
                published_outputs["year"] == 2024,
                "published_wcb_relative_female_spawning_biomass",
            ].iloc[0]
        )
        if abs(terminal - 0.15) > 0.01:
            raise AssertionError(f"Digitised terminal published depletion {terminal:.3f} is not close to 0.15")
        checks.append(f"Digitised Figure 3.21 terminal WCB relative spawning biomass is {terminal:.3f}, consistent with published 0.15")
    return checks


def build() -> dict[str, object]:
    rendered = render_evidence_pages()
    catch = extract_catch(rendered["figure_3_15_retained_catch.png"])
    catch_by_area = extract_catch_by_area(rendered["figure_3_15_retained_catch.png"], catch)
    cpue = extract_cpue(rendered["figure_3_17_cpue.png"])
    by_area, pooled = extract_age_composition(rendered["figure_3_19_age_composition.png"])
    published_outputs = extract_published_assessment_outputs(rendered["figure_3_21_assessment_outputs.png"])
    main = catch.merge(cpue, on="year", how="left")
    main["biomass"] = np.nan
    main["dataset_status"] = "public_evidence_reconstruction_not_raw_dpird_data"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    main.to_csv(OUTPUT_ROOT / "dpird_wa_dhufish_public_reconstruction.csv", index=False)
    by_area.to_csv(OUTPUT_ROOT / "age_composition_by_area.csv", index=False)
    pooled.to_csv(OUTPUT_ROOT / "age_composition.csv", index=False)
    catch_by_area.to_csv(OUTPUT_ROOT / "catch_by_area_sector.csv", index=False)
    published_outputs.to_csv(OUTPUT_ROOT / "published_assessment_outputs_digitised.csv", index=False)
    parameter_register().to_csv(OUTPUT_ROOT / "parameter_register.csv", index=False)
    manifest = source_manifest()
    (OUTPUT_ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checks = validate_outputs(main, by_area, pooled, catch_by_area, published_outputs)
    metadata = {
        "display_name": "WA Dhufish — DPIRD public-evidence reconstruction",
        "short_description": "Published parameters plus digitised official catch, CPUE and age-composition figures for transparent Omega stress testing.",
        "species": "Glaucosoma hebraicum",
        "source": "Western Australian Department of Primary Industries and Regional Development",
        "source_url": "https://library.dpird.wa.gov.au/fish_rar/2/",
        "model_type": "integrated age-structured public reconstruction",
        "difficulty": "advanced",
        "available_data": ["retained catch", "commercial CPUE", "age compositions", "published biological parameters"],
        "required_files": [
            "Omega_Ready/dpird_wa_dhufish_public_reconstruction.csv",
            "Omega_Ready/age_composition.csv",
            "Omega_Ready/parameter_register.csv",
        ],
        "supported_workflows": ["integrated assessment", "sensitivity analysis", "retrospective", "ASPM-style comparison"],
        "recommended_first_action": "Read README.md and review parameter_register.csv evidence_class before fitting.",
        "original_or_working_copy": "public reconstruction; original PDFs preserved",
        "raw_dpird_dataset": False,
        "validation_checks": checks,
    }
    (DATASET_ROOT / "omega_dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme()
    return {
        "output_root": str(OUTPUT_ROOT),
        "annual_rows": len(main),
        "area_age_rows": len(by_area),
        "pooled_age_rows": len(pooled),
        "area_catch_rows": len(catch_by_area),
        "published_output_rows": len(published_outputs),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate an existing build without replacing it")
    args = parser.parse_args()
    if args.check_only:
        main_frame = pd.read_csv(OUTPUT_ROOT / "dpird_wa_dhufish_public_reconstruction.csv")
        area_frame = pd.read_csv(OUTPUT_ROOT / "age_composition_by_area.csv")
        pooled_frame = pd.read_csv(OUTPUT_ROOT / "age_composition.csv")
        catch_area_frame = pd.read_csv(OUTPUT_ROOT / "catch_by_area_sector.csv")
        published_frame = pd.read_csv(OUTPUT_ROOT / "published_assessment_outputs_digitised.csv")
        result = {"checks": validate_outputs(main_frame, area_frame, pooled_frame, catch_area_frame, published_frame)}
    else:
        result = build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
