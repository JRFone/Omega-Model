from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SS3FileSet:
    starter: str = "starter.ss"
    data: str = "data.ss"
    control: str = "control.ss"
    forecast: str = "forecast.ss"


def strip_ss_comments(text: str) -> list[str]:
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            rows.append(line)
    return rows


def numeric_tokens(text: str) -> list[float]:
    values = []
    for line in strip_ss_comments(text):
        for token in line.replace(",", " ").split():
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def parse_report_sso(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "header"
    sections[current] = []
    section_pattern = re.compile(r"^[A-Z][A-Z0-9_ ]{3,}$")
    for line in lines:
        stripped = line.strip()
        if section_pattern.match(stripped) and len(stripped.split()) <= 8:
            current = stripped.replace(" ", "_")
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    time_series = []
    for key in ("ANNUAL_TIME_SERIES", "TIME_SERIES"):
        if key not in sections:
            continue
        table_lines = [line.strip() for line in sections[key] if line.strip()]
        if not table_lines:
            continue
        header_index = None
        for i, line in enumerate(table_lines):
            lower = line.lower()
            if ("year" in lower or re.search(r"\byr\b", lower)) and ("spawn" in lower or "bio" in lower):
                header_index = i
                break
        if header_index is None:
            continue
        header = table_lines[header_index].split()
        for line in table_lines[header_index + 1:]:
            tokens = line.split()
            if len(tokens) < len(header):
                continue
            row = {}
            for name, token in zip(header, tokens):
                try:
                    row[name] = float(token)
                except ValueError:
                    row[name] = token
            time_series.append(row)
        if time_series:
            break
    return {
        "sections": sorted(sections),
        "section_line_counts": {key: len(value) for key, value in sections.items()},
        "time_series": time_series,
        "raw_length": len(text),
    }


def export_minimal_ss3(
    output_dir: str | Path,
    years: Sequence[int],
    catches: Sequence[float],
    indices: Sequence[float] | None = None,
    max_age: int = 30,
    sexes: int = 2,
    areas: int = 1,
    seasons: int = 1,
    file_set: SS3FileSet | None = None,
) -> dict[str, str]:
    file_set = file_set or SS3FileSet()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    years = [int(value) for value in years]
    catches = [float(value) for value in catches]
    if len(years) != len(catches) or len(years) < 2:
        raise ValueError("SS3 export requires matching year and catch series with at least two years.")
    indices = [float(value) for value in indices] if indices is not None else []
    starter = f"""# Omega FISH generated SS3 starter file
{file_set.data}
{file_set.control}
0 # init values source
0 # run display detail
1 # detailed age-structured output
0 # write detailed output
0 # write parm trace
0 # cumulative report
0 # prior likelihood constant
1 # soft bounds
1 # data file check
999
"""
    data_lines = [
        "# Omega FISH generated SS3 data foundation",
        f"{years[0]} # start year",
        f"{years[-1]} # end year",
        f"{seasons} # seasons per year",
        "12 # months per season placeholder",
        "1 # spawning month",
        f"{sexes} # number of sexes",
        f"{max_age} # maximum age",
        f"{areas} # number of areas",
        "1 # number of fleets",
        "1 # catch units biomass",
        "0.05 # catch standard error",
        "# year season fleet catch catch_se",
    ]
    for year, catch in zip(years, catches):
        data_lines.append(f"{year} 1 1 {catch:.10g} 0.05")
    data_lines.append("-9999 0 0 0 0 # end catch")
    if indices:
        data_lines.extend(["1 # number of CPUE observations", "# year season index obs se"])
        for year, value in zip(years[: len(indices)], indices):
            if np.isfinite(value) and value > 0:
                data_lines.append(f"{year} 1 1 {value:.10g} 0.20")
        data_lines.append("-9999 0 0 0 0 # end CPUE")
    else:
        data_lines.append("0 # number of CPUE observations")
    data_lines.extend([
        "0 # discard fleets",
        "0 # mean body weight observations",
        "0 # length bins",
        "0 # length composition observations",
        "0 # age bins",
        "0 # age composition observations",
        "0 # mean size-at-age observations",
        "0 # environmental observations",
        "0 # generalized size composition methods",
        "999 # data file terminator",
    ])
    control = f"""# Omega FISH generated SS3 control foundation
# This is an interoperability starting point, not a completed stock-specific control file.
1 # empirical weight-at-age
0 # number of growth patterns minus one
1 # number of platoons
0 # recruitment distribution method
0 # recruitment area interaction
0 # number of recruitment cycles
0 # natural mortality setup placeholder
0 # growth setup placeholder
0 # maturity setup placeholder
0 # fecundity setup placeholder
0 # recruitment setup placeholder
0 # fishing mortality method placeholder
0 # selectivity setup placeholder
999 # control file terminator
"""
    forecast = f"""# Omega FISH generated SS3 forecast foundation
1 # benchmark type
2 # MSY method
0.40 # target depletion
0.10 # limit depletion
{years[-1] + 1} # forecast start year
20 # forecast years
0.45 # P-star
999
"""
    files = {
        file_set.starter: starter,
        file_set.data: "\n".join(data_lines) + "\n",
        file_set.control: control,
        file_set.forecast: forecast,
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    manifest = {
        "generated_by": "Omega FISH Model",
        "files": list(files),
        "years": [years[0], years[-1]],
        "max_age": max_age,
        "sexes": sexes,
        "areas": areas,
        "seasons": seasons,
        "warning": "Generated files are a syntactic foundation and require stock-specific SS3 control settings and validation before execution.",
    }
    (output / "omega_ss3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {name: str(output / name) for name in files} | {"manifest": str(output / "omega_ss3_manifest.json")}


def compare_time_series(
    omega_rows: Sequence[Mapping[str, Any]],
    ss3_rows: Sequence[Mapping[str, Any]],
    omega_year_key: str = "year",
    ss3_year_key: str = "Yr",
    variable_pairs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    variable_pairs = variable_pairs or {"spawning_biomass": "SpawnBio", "total_biomass": "Bio_all"}
    omega_by_year = {int(row[omega_year_key]): row for row in omega_rows if omega_year_key in row}
    ss3_by_year = {int(float(row[ss3_year_key])): row for row in ss3_rows if ss3_year_key in row}
    years = sorted(set(omega_by_year) & set(ss3_by_year))
    rows = []
    summary = {}
    for omega_name, ss3_name in variable_pairs.items():
        differences = []
        relative = []
        for year in years:
            if omega_name not in omega_by_year[year] or ss3_name not in ss3_by_year[year]:
                continue
            omega_value = float(omega_by_year[year][omega_name])
            ss3_value = float(ss3_by_year[year][ss3_name])
            difference = omega_value - ss3_value
            rel = difference / max(abs(ss3_value), 1e-12)
            differences.append(difference)
            relative.append(rel)
            rows.append({"year": year, "omega_variable": omega_name, "ss3_variable": ss3_name, "omega": omega_value, "ss3": ss3_value, "difference": difference, "relative_difference": rel})
        if differences:
            summary[omega_name] = {
                "points": len(differences),
                "mean_absolute_difference": float(np.mean(np.abs(differences))),
                "mean_absolute_relative_difference": float(np.mean(np.abs(relative))),
                "maximum_absolute_relative_difference": float(np.max(np.abs(relative))),
            }
    return {"years_compared": years, "rows": rows, "summary": summary}
