from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .provenance import SOURCE_DERIVED, provenance_from_file, provenance_from_text, stable_id, transformation_record


@dataclass(frozen=True)
class StockDataset:
    name: str
    frame: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_columns: list[str] = field(default_factory=list)
    index_columns: list[str] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        return [int(v) for v in self.frame["year"].tolist()]


def read_stock_csv(text: str | dict, name: str = "Uploaded stock") -> StockDataset:
    if isinstance(text, dict) and "value" in text:
        text = str(text["value"])
    provenance = provenance_from_text(name, str(text), notes="CSV supplied through the app/API. Original text hash is preserved.")
    df = pd.read_csv(StringIO(text.strip()))
    return normalise_frame(df, name=name, provenance=provenance)


def read_stock_file(path: str | Path) -> StockDataset:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    provenance = provenance_from_file(path, title=path.stem)
    return normalise_frame(df, name=path.stem, provenance=provenance)


def normalise_frame(df: pd.DataFrame, name: str = "Stock", provenance: dict[str, Any] | None = None) -> StockDataset:
    input_rows = int(len(df))
    raw_columns = [str(c) for c in df.columns]
    transformations: list[dict[str, Any]] = []
    warnings: list[str] = []
    aliases = {
        "year": {"year", "yr", "date"},
        "catch": {"catch", "landings", "retained_catch", "total_catch", "c"},
        "index": {"index", "cpue", "survey", "abundance", "standardised_cpue", "standardized_cpue"},
        "biomass": {"biomass", "spawner", "ssb", "spawning_biomass", "b"},
    }
    lookup = {str(c).strip().lower(): c for c in df.columns}
    mapped: dict[str, str] = {}
    for target, names in aliases.items():
        for candidate in names:
            if candidate in lookup:
                mapped[target] = lookup[candidate]
                break

    if "year" not in mapped or "catch" not in mapped:
        raise ValueError("Input needs at least year and catch columns. Optional columns: index, biomass.")

    transformations.append(
        transformation_record(
            "column_alias_mapping",
            input_rows,
            "Map source column names to the internal model-ready schema.",
            {"mapped_columns": mapped, "raw_columns": raw_columns},
        )
    )
    out = pd.DataFrame()
    for target, source in mapped.items():
        out[target] = pd.to_numeric(df[source], errors="coerce")
    extra_index_columns = _extra_index_columns(df, mapped)
    for output_name, source in extra_index_columns.items():
        out[output_name] = pd.to_numeric(df[source], errors="coerce")
    transformations.append(
        transformation_record(
            "numeric_conversion",
            input_rows,
            "Convert mapped columns to numeric values; invalid values become missing.",
            {"columns": list(mapped.keys()), "extra_index_columns": extra_index_columns},
        )
    )
    before_required = len(out)
    out = out.dropna(subset=["year", "catch"]).copy()
    dropped_required = before_required - len(out)
    if dropped_required:
        warnings.append(f"Dropped {dropped_required} rows missing required year or catch values.")
        transformations.append(
            transformation_record(
                "drop_missing_required_values",
                dropped_required,
                "Rows missing year or catch cannot be used by the biomass-dynamics model.",
            )
        )
    out["year"] = out["year"].astype(int)
    negative_catch = int((out["catch"] < 0).sum())
    out["catch"] = out["catch"].clip(lower=0)
    if negative_catch:
        warnings.append(f"Clipped {negative_catch} negative catch values to zero.")
        transformations.append(
            transformation_record(
                "clip_negative_catch",
                negative_catch,
                "Catch/removals cannot be negative in the current internal model.",
            )
        )
    for optional in ["index", "biomass"]:
        if optional not in out:
            out[optional] = float("nan")
            transformations.append(
                transformation_record(
                    "add_missing_optional_column",
                    len(out),
                    f"Optional column `{optional}` was absent; internal schema keeps it as missing.",
                    {"column": optional},
                )
            )
    for optional in ["index", "biomass", *[col for col in out.columns if col.startswith("index_")]]:
        invalid_optional = int(((out[optional] <= 0) & out[optional].notna()).sum())
        out[optional] = out[optional].where(out[optional] > 0)
        if invalid_optional:
            warnings.append(f"Set {invalid_optional} non-positive {optional} values to missing.")
            transformations.append(
                transformation_record(
                    "set_non_positive_optional_values_missing",
                    invalid_optional,
                    f"`{optional}` observations must be positive for lognormal fitting.",
                    {"column": optional},
                )
            )
    duplicate_years = int(out["year"].duplicated(keep="last").sum())
    out = out.sort_values("year").drop_duplicates("year", keep="last").reset_index(drop=True)
    transformations.append(
        transformation_record(
            "sort_and_deduplicate_year",
            duplicate_years,
            "Sort by year and keep the last row for duplicate years.",
        )
    )
    if duplicate_years:
        warnings.append(f"Removed {duplicate_years} duplicate year rows, keeping the last occurrence.")
    if len(out) < 5:
        raise ValueError("At least five annual rows are needed for a useful fit.")
    return StockDataset(
        name=name,
        frame=out,
        provenance=provenance or provenance_from_text(name, df.to_csv(index=False), notes="Dataset provenance was created during normalisation."),
        transformations=transformations,
        warnings=warnings,
        raw_columns=raw_columns,
        index_columns=[col for col in out.columns if col == "index" or str(col).startswith("index_")],
    )


def to_csv(dataset: StockDataset) -> str:
    return dataset.frame.to_csv(index=False)


def merge_series(series: Iterable[StockDataset], name: str = "Merged stock") -> StockDataset:
    frames = [s.frame for s in series]
    if not frames:
        raise ValueError("No series supplied.")
    df = pd.concat(frames, ignore_index=True).groupby("year", as_index=False).mean(numeric_only=True)
    provenance = {
        "evidence_id": stable_id("EV", name, [s.provenance.get("hash_sha256") for s in series]),
        "title": name,
        "source_type": SOURCE_DERIVED,
        "hash_sha256": None,
        "extraction_method": "mean_merge_of_stock_datasets",
        "verification_status": "derived",
        "notes": "Merged from existing StockDataset objects; source hashes are listed in transformation details.",
    }
    merged = normalise_frame(df, name=name, provenance=provenance)
    merged.transformations.append(
        transformation_record(
            "merge_series",
            len(df),
            "Average numeric values by year across input stock datasets.",
            {"source_hashes": [s.provenance.get("hash_sha256") for s in series]},
        )
    )
    return merged


def _extra_index_columns(df: pd.DataFrame, mapped: dict[str, str]) -> dict[str, str]:
    used = {str(value) for value in mapped.values()}
    extras: dict[str, str] = {}
    for col in df.columns:
        raw = str(col)
        key = raw.strip().lower()
        if raw in used or key in {"year", "yr", "date", "catch", "landings", "biomass", "spawner", "ssb", "b"}:
            continue
        is_index = (
            key.startswith("index_")
            or key.startswith("cpue_")
            or key.startswith("survey_")
            or key.endswith("_index")
            or key.endswith("_cpue")
            or "abundance_index" in key
        )
        if not is_index:
            continue
        safe = "index_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")
        if safe == "index_index":
            safe = "index_extra"
        count = 2
        base = safe
        while safe in extras:
            safe = f"{base}_{count}"
            count += 1
        extras[safe] = raw
    return extras
