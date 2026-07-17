from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


DATA_FILE_NAMES = (
    "model_ready_timeseries.csv",
    "model_ready_timeseries.xlsx",
    "data.csv",
    "timeseries.csv",
)


@dataclass(frozen=True)
class DatasetEntry:
    identifier: str
    display_name: str
    root: Path
    source: str = "Omega"
    difficulty: str = "Beginner"
    model_type: str = "Age structured"
    description: str = "Omega model dataset"
    data_types: tuple[str, ...] = field(default_factory=tuple)
    primary_file: Path | None = None
    age_composition: Path | None = None
    length_composition: Path | None = None
    metadata_path: Path | None = None
    original: bool = True
    recommended_tools: tuple[str, ...] = field(default_factory=tuple)
    expected_behavior: str = ""

    @property
    def coverage(self) -> str:
        if self.model_type.lower() == "stock synthesis" and (self.root / "starter.ss").exists():
            return "Full NOAA/SS3 model"
        required = {"catch", "CPUE/index", "biomass", "age", "length", "sector catch", "recruitment"}
        if self.primary_file is not None and self.age_composition is not None and self.length_composition is not None and required.issubset(set(self.data_types)):
            return "Full Omega dataset"
        return "Partial inputs"

    @property
    def workspace_coverage(self) -> str:
        if self.model_type.lower() == "stock synthesis":
            return "NOAA validation + Omega annual-data adapter"
        if self.coverage == "Full Omega dataset":
            return "All Omega data-driven workspaces"
        return "Depends on available inputs"

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("root", "primary_file", "age_composition", "length_composition", "metadata_path"):
            value = payload.get(key)
            payload[key] = str(value) if value is not None else None
        payload["data_types"] = list(self.data_types)
        payload["recommended_tools"] = list(self.recommended_tools)
        return payload


class DatasetLibrary:
    """Discovers Omega and NOAA datasets without mutating source folders."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def scan(self) -> list[DatasetEntry]:
        if not self.root.exists():
            return []
        entries: list[DatasetEntry] = []
        seen: set[Path] = set()
        for folder in self._candidate_folders():
            resolved = folder.resolve()
            if resolved in seen:
                continue
            entry = self._read_entry(folder)
            if entry is not None:
                seen.add(resolved)
                entries.append(entry)
        return sorted(entries, key=lambda item: (item.difficulty.lower(), item.source.lower(), item.display_name.lower()))

    def _candidate_folders(self) -> Iterable[Path]:
        metadata = sorted(self.root.rglob("omega_dataset.json"))
        for path in metadata:
            yield path.parent
        for path in sorted(self.root.rglob("model_ready_timeseries.csv")):
            yield path.parent
        for path in sorted(self.root.rglob("starter.ss")):
            yield path.parent

    def _read_entry(self, folder: Path) -> DatasetEntry | None:
        metadata_path = folder / "omega_dataset.json"
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}

        primary = self._find_primary(folder, metadata)
        starter = folder / "starter.ss"
        if primary is None and not starter.exists():
            return None
        age = self._optional_path(folder, metadata.get("age_composition"), "age_composition.csv")
        length = self._optional_path(folder, metadata.get("length_composition"), "length_composition.csv")
        types = list(metadata.get("data_types", [])) if isinstance(metadata.get("data_types"), list) else []
        if primary is not None:
            types.extend(self._infer_csv_types(primary))
        if age is not None:
            types.append("age")
        if length is not None:
            types.append("length")
        if starter.exists():
            types.append("SS3")
        relative = folder.relative_to(self.root).as_posix()
        source = str(metadata.get("source") or ("NOAA" if "noaa" in relative.lower() else "Omega"))
        return DatasetEntry(
            identifier=str(metadata.get("id") or relative.replace("/", "::")),
            display_name=str(metadata.get("display_name") or folder.name.replace("_", " ")),
            root=folder.resolve(),
            source=source,
            difficulty=str(metadata.get("difficulty") or self._difficulty_from_path(relative)),
            model_type=str(metadata.get("model_type") or ("Stock Synthesis" if starter.exists() else "Age structured")),
            description=str(metadata.get("description") or "Dataset discovered by Omega."),
            data_types=tuple(sorted(set(types), key=str.lower)),
            primary_file=primary.resolve() if primary is not None else None,
            age_composition=age.resolve() if age is not None else None,
            length_composition=length.resolve() if length is not None else None,
            metadata_path=metadata_path.resolve() if metadata_path.exists() else None,
            original=bool(metadata.get("original", True)),
            recommended_tools=tuple(str(value) for value in metadata.get("recommended_tools", []) if isinstance(value, str))
            if isinstance(metadata.get("recommended_tools"), list)
            else (),
            expected_behavior=str(metadata.get("expected_behavior") or ""),
        )

    @staticmethod
    def _optional_path(folder: Path, configured: object, fallback: str) -> Path | None:
        candidate = folder / str(configured) if configured else folder / fallback
        return candidate if candidate.exists() else None

    @staticmethod
    def _find_primary(folder: Path, metadata: dict[str, object]) -> Path | None:
        configured = metadata.get("primary_file")
        if configured:
            candidate = folder / str(configured)
            if candidate.exists():
                return candidate
        for name in DATA_FILE_NAMES:
            candidate = folder / name
            if candidate.exists():
                return candidate
        csv_files = [path for path in sorted(folder.glob("*.csv")) if "composition" not in path.name.lower()]
        return csv_files[0] if csv_files else None

    @staticmethod
    def _difficulty_from_path(relative: str) -> str:
        lowered = relative.lower()
        name = Path(relative).name.lower()
        if any(token in name for token in ("tag", "move", "area", "morph", "timevary", "hake", "vermillion", "platoon", "sablefish", "dogfish")):
            return "Advanced"
        if "advanced" in lowered:
            return "Advanced"
        if any(token in name for token in ("discard", "dm", "wtatage", "selex", "lorenzen")) or "intermediate" in lowered:
            return "Intermediate"
        return "Beginner"

    @staticmethod
    def _infer_csv_types(path: Path) -> list[str]:
        try:
            header = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0].lower()
        except (OSError, IndexError):
            return []
        types = []
        for name, markers in {
            "catch": ("catch",),
            "CPUE/index": ("cpue", "index"),
            "biomass": ("biomass",),
            "recruitment": ("recruit",),
        }.items():
            if any(marker in header for marker in markers):
                types.append(name)
        return types

    def write_catalogue(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([entry.as_json() for entry in self.scan()], indent=2), encoding="utf-8")
        return path


def materialize_omega_timeseries(entry: DatasetEntry, cache_root: str | Path) -> tuple[Path, str]:
    """Return an Omega-ready file, transparently adapting SS3 annual catch/index data when needed."""

    if entry.primary_file is not None:
        return entry.primary_file, "native Omega dataset"
    starter_path = entry.root / "starter.ss"
    if entry.model_type.lower() != "stock synthesis" or not starter_path.exists():
        raise ValueError(f"{entry.display_name} has no model-ready time-series input.")

    from stock_model.ss3_validation import load_model_file_set, parse_ss3_data

    _starter_text, data_text, _control_text, _forecast_text = load_model_file_set(entry.root)
    data = parse_ss3_data(data_text)
    catches: dict[int, float] = defaultdict(float)
    indices: dict[int, list[float]] = defaultdict(list)
    for observation in data.catches:
        if data.start_year <= observation.year <= data.end_year and math.isfinite(observation.catch) and observation.catch >= 0:
            catches[int(observation.year)] += float(observation.catch)
    for observation in data.indices:
        if data.start_year <= observation.year <= data.end_year and math.isfinite(observation.observation) and observation.observation > 0:
            indices[int(observation.year)].append(float(observation.observation))

    digest = hashlib.sha256(f"{entry.identifier}|{entry.root}".encode("utf-8")).hexdigest()[:16]
    output = Path(cache_root) / digest
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "model_ready_timeseries.csv"
    lines = ["year,catch,index,biomass,catch_commercial,catch_charter,catch_recreational,recruitment_multiplier"]
    for year in range(int(data.start_year), int(data.end_year) + 1):
        catch = catches.get(year, 0.0)
        values = indices.get(year, [])
        index = math.exp(sum(math.log(value) for value in values) / len(values)) if values else ""
        lines.append(f"{year},{catch:.12g},{index if index == '' else f'{index:.12g}'},,{catch:.12g},0,0,1")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "source_dataset": entry.display_name,
        "source_folder": str(entry.root),
        "adapter": "SS3 annual catch/index to Omega time series",
        "years": [data.start_year, data.end_year],
        "limitations": [
            "Catch observations are summed across SS3 fleets and assigned to Omega's commercial sector for compatibility.",
            "Positive abundance indices in a year are combined by geometric mean.",
            "Biomass, age composition, length composition, and SS3 structural settings are not invented or converted.",
            "Use NOAA / SS3 Validation for source-model structural and native SS3 checks.",
        ],
    }
    (output / "source_adapter.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path, "adapted SS3 annual catch/index data; biomass and compositions remain unavailable"
