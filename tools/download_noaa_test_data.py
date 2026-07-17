from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOAA_ROOT = ROOT / "Data_Sets" / "NOAA"
SOURCE_ROOT = NOAA_ROOT / "_sources"
MANIFEST_PATH = NOAA_ROOT / "NOAA_SOURCE_MANIFEST.json"
CATALOGUE_JSON = NOAA_ROOT / "NOAA_TEST_MODEL_CATALOGUE.json"
CATALOGUE_CSV = NOAA_ROOT / "NOAA_TEST_MODEL_CATALOGUE.csv"

REPOSITORIES = (
    {
        "owner": "nmfs-ost",
        "name": "ss3-test-models",
        "purpose": "Official Stock Synthesis regression and feature test models",
    },
    {
        "owner": "nmfs-ost",
        "name": "ss3-user-examples",
        "purpose": "Official Stock Synthesis example models for users",
    },
)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Omega-FISH-Model/1.5"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Omega-FISH-Model/1.5"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
        bundle.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one repository root in {archive.name}; found {len(roots)}")
    return roots[0]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_summary(root: Path) -> tuple[int, int, str | None]:
    files = [path for path in root.rglob("*") if path.is_file()]
    total_size = sum(path.stat().st_size for path in files)
    licence = next((path.name for path in files if path.name.lower() in {"license", "license.md", "license.txt", "licence", "licence.md"}), None)
    return len(files), total_size, licence


def refresh_repository(owner: str, name: str, purpose: str) -> dict[str, Any]:
    metadata = _request_json(f"https://api.github.com/repos/{owner}/{name}")
    branch = str(metadata["default_branch"])
    commit = _request_json(f"https://api.github.com/repos/{owner}/{name}/commits/{branch}")
    sha = str(commit["sha"])
    url = f"https://api.github.com/repos/{owner}/{name}/zipball/{sha}"
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"omega-{name}-") as temp_text:
        temp = Path(temp_text)
        archive = temp / f"{name}.zip"
        extracted = temp / "extracted"
        extracted.mkdir()
        _download(url, archive)
        source = _safe_extract(archive, extracted)
        staged = SOURCE_ROOT / f".{name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(source, staged)
        target = SOURCE_ROOT / name
        previous = SOURCE_ROOT / f".{name}.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if target.exists():
            target.replace(previous)
        staged.replace(target)
        if previous.exists():
            shutil.rmtree(previous)

    file_count, total_size, licence = _repository_summary(target)
    return {
        "repository": name,
        "url": f"https://github.com/{owner}/{name}",
        "purpose": purpose,
        "branch": branch,
        "commit_sha": sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "total_bytes": total_size,
        "licence_file": licence,
        "local_path": target.relative_to(ROOT).as_posix(),
    }


def _find_named(folder: Path, names: tuple[str, ...]) -> Path | None:
    lower = {path.name.lower(): path for path in folder.iterdir() if path.is_file()}
    return next((lower[name.lower()] for name in names if name.lower() in lower), None)


def _starter_inputs(starter: Path) -> tuple[Path | None, Path | None]:
    values: list[str] = []
    try:
        lines = starter.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except OSError:
        return None, None
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if content:
            values.append(content.split()[0])
        if len(values) >= 2:
            break
    data = starter.parent / values[0] if values else None
    control = starter.parent / values[1] if len(values) > 1 else None
    return (data if data and data.exists() else None, control if control and control.exists() else None)


def _difficulty(name: str) -> str:
    lowered = name.lower()
    if lowered in {"simple", "simple_nocpue"} or lowered.startswith("simple_") and "discard" not in lowered and "dm" not in lowered:
        return "Beginner"
    if any(token in lowered for token in ("tag", "move", "area", "morph", "timevary", "hake", "vermillion", "spatial")):
        return "Advanced"
    return "Intermediate"


def build_catalogue() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not SOURCE_ROOT.exists():
        return rows
    for starter in sorted(SOURCE_ROOT.rglob("starter.ss")):
        folder = starter.parent
        starter_data, starter_control = _starter_inputs(starter)
        data_file = starter_data or _find_named(folder, ("data.ss", "data_echo.ss", "data.ss_new"))
        control_file = starter_control or _find_named(folder, ("control.ss", "control.ss_new"))
        forecast = _find_named(folder, ("forecast.ss",))
        files = [path for path in folder.iterdir() if path.is_file()]
        relative = folder.relative_to(SOURCE_ROOT)
        name = folder.name
        lowered_names = " ".join(path.name.lower() for path in files)
        rows.append(
            {
                "model_name": name,
                "source_repository": relative.parts[0] if relative.parts else "",
                "relative_folder": relative.as_posix(),
                "difficulty": _difficulty(name),
                "starter_file": starter.name,
                "data_file": data_file.name if data_file else "",
                "control_file": control_file.name if control_file else "",
                "forecast_file": forecast.name if forecast else "",
                "file_count": len(files),
                "has_expected_report": any(path.name.lower() == "report.sso" for path in files),
                "has_tagging_hint": "tag" in lowered_names or "tag" in name.lower(),
                "has_discard_hint": "discard" in lowered_names or "discard" in name.lower(),
                "has_movement_hint": "move" in lowered_names or "move" in name.lower() or "area" in name.lower(),
                "complete_core_files": bool(data_file and control_file),
            }
        )
    NOAA_ROOT.mkdir(parents=True, exist_ok=True)
    CATALOGUE_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = list(rows[0]) if rows else ["model_name", "source_repository", "relative_folder"]
    with CATALOGUE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def check_only() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {"repositories": []}
    repositories = []
    for configured in REPOSITORIES:
        target = SOURCE_ROOT / configured["name"]
        file_count, total_size, licence = _repository_summary(target) if target.exists() else (0, 0, None)
        repositories.append(
            {
                "repository": configured["name"],
                "present": target.exists(),
                "file_count": file_count,
                "total_bytes": total_size,
                "licence_file": licence,
            }
        )
    models = build_catalogue()
    return {"status": "PASS" if all(item["present"] for item in repositories) else "INCOMPLETE", "repositories": repositories, "models": len(models), "manifest": manifest}


def refresh() -> dict[str, Any]:
    repositories = [refresh_repository(**configured) for configured in REPOSITORIES]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Official NOAA/NMFS scientific repositories are preserved as local validation snapshots. Their contents are provided as-is by their respective maintainers.",
        "repositories": repositories,
    }
    NOAA_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    models = build_catalogue()
    return {
        "status": "PASS",
        "repositories": len(repositories),
        "models": len(models),
        "files": sum(item["file_count"] for item in repositories),
        "bytes": sum(item["total_bytes"] for item in repositories),
        "manifest": str(MANIFEST_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and catalogue official NOAA Stock Synthesis test data.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--refresh", action="store_true", help="Download a fresh pinned snapshot of each official data repository")
    group.add_argument("--check-only", action="store_true", help="Check the existing local library without downloading")
    args = parser.parse_args()
    result = refresh() if args.refresh else check_only()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
