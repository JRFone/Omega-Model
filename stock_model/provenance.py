from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


APP_NAME = "Omega FISH Model"
APP_VERSION = "1.4.1"

SOURCE_INTERNAL = "internal_platform"
SOURCE_USER = "user_supplied"
SOURCE_DPIRD = "dpird_extracted"
SOURCE_STOCK_SYNTHESIS = "stock_synthesis_import"
SOURCE_RECONSTRUCTED = "published_reconstruction"
SOURCE_EXTERNAL = "external_reference"
SOURCE_DERIVED = "application_derived"
SOURCE_MODEL_ESTIMATE = "model_estimate"
SOURCE_ASSUMED = "assumed_unavailable"


@dataclass(frozen=True)
class ProvenanceRecord:
    evidence_id: str
    title: str
    source_type: str
    path: str | None = None
    hash_sha256: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_organization: str | None = None
    extraction_method: str | None = None
    verification_status: str = "unverified"
    access_status: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssumptionRecord:
    assumption_id: str
    assumption: str
    category: str
    reason: str
    evidence: str
    treatment: str
    sensitivity_tested: str
    materiality: str
    effect_direction: str
    status: str = "open"
    owner: str = "application"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if part is not None)
    return f"{prefix}-{sha256_text(raw)[:12].upper()}"


def classify_source(title: str = "", path: str | Path | None = None, declared: str | None = None) -> str:
    if declared:
        return declared
    text = f"{title} {path or ''}".lower()
    if "report.sso" in text or "stock synthesis" in text or "ss_companion" in text:
        return SOURCE_STOCK_SYNTHESIS
    if "dpird" in text or "fop" in text or "fisheries occasional publication" in text:
        return SOURCE_DPIRD
    if "data_sets" in text or "data_set_" in text:
        return SOURCE_USER
    if "\\data\\" in text or "/data/" in text or "testing" in text or "punt_rebuilder_sample" in text:
        return SOURCE_INTERNAL
    if "\\models\\" in text or "/models/" in text or "\\reports\\" in text or "/reports/" in text:
        return SOURCE_DERIVED
    if "base_software_archive" in text or "admb" in text or "r4ss" in text:
        return SOURCE_EXTERNAL
    return SOURCE_USER


def provenance_from_text(title: str, text: str, source_type: str | None = None, notes: str = "") -> dict[str, Any]:
    digest = sha256_text(text)
    resolved_type = classify_source(title=title, declared=source_type)
    return ProvenanceRecord(
        evidence_id=stable_id("EV", title, digest),
        title=title,
        source_type=resolved_type,
        hash_sha256=digest,
        extraction_method="pasted_csv_or_api_payload",
        verification_status="unverified",
        notes=notes,
    ).to_dict()


def provenance_from_file(path: str | Path, title: str | None = None, source_type: str | None = None, notes: str = "") -> dict[str, Any]:
    source_path = Path(path)
    digest = sha256_file(source_path)
    title_value = title or source_path.name
    resolved_type = classify_source(title=title_value, path=source_path, declared=source_type)
    organization = "DPIRD / public extraction" if resolved_type == SOURCE_DPIRD else None
    extraction = "direct_file_read"
    if resolved_type == SOURCE_DPIRD:
        extraction = "structured_extraction_from_public_material"
    elif resolved_type == SOURCE_STOCK_SYNTHESIS:
        extraction = "stock_synthesis_report_import"
    return ProvenanceRecord(
        evidence_id=stable_id("EV", source_path, digest),
        title=title_value,
        source_type=resolved_type,
        path=str(source_path),
        hash_sha256=digest,
        source_organization=organization,
        extraction_method=extraction,
        verification_status="partly_verified" if source_path.exists() else "unverified",
        access_status="available",
        notes=notes,
    ).to_dict()


def transformation_record(operation: str, affected_rows: int, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "transformation_id": stable_id("TR", operation, affected_rows, json.dumps(details or {}, sort_keys=True, default=str)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "affected_rows": int(affected_rows),
        "reason": reason,
        "details": details or {},
    }


def assumptions_from_settings(settings: Any, projection_settings: Any | None = None) -> list[dict[str, Any]]:
    data = _public_dict(settings)
    assumptions = [
        AssumptionRecord(
            "ASM-MODEL-FORM",
            f"Internal surplus-production model form is {data.get('model', 'unknown')}.",
            "structural",
            "Selected model form for this run.",
            "Application setting, not an external assessment conclusion.",
            "fixed",
            "yes",
            "high",
            "Can change biomass, depletion and MSY proxies.",
        ),
        AssumptionRecord(
            "ASM-R-PRIOR",
            f"Productivity r prior median is {data.get('r_prior_median')}; prior CV is {data.get('r_prior_cv')}.",
            "biological",
            "Slow-stock productivity prior used by the proxy model.",
            "Application setting unless supplied by the user or evidence register.",
            "prior",
            "yes",
            "high",
            "Lower productivity generally increases depletion/rebuilding risk.",
        ),
        AssumptionRecord(
            "ASM-OBS-CV",
            f"Observation CV is {data.get('obs_cv')}.",
            "statistical",
            "Controls expected observation noise for CPUE/biomass observations.",
            "Application setting.",
            "fixed",
            "yes",
            "medium",
            "Changes how strongly observations influence the fit.",
        ),
        AssumptionRecord(
            "ASM-PROCESS-CV",
            f"Process CV is {data.get('process_cv')}.",
            "statistical",
            "Controls stochastic process variation in projections.",
            "Application setting.",
            "fixed",
            "yes",
            "medium",
            "Changes projection uncertainty.",
        ),
        AssumptionRecord(
            "ASM-INDEX-WEIGHT",
            f"CPUE/index weight is {data.get('index_weight')}.",
            "data",
            "Controls influence of CPUE/index observations.",
            "Application setting.",
            "fixed",
            "yes",
            "high",
            "Higher values make CPUE/index more influential.",
        ),
        AssumptionRecord(
            "ASM-BIOMASS-WEIGHT",
            f"Biomass observation weight is {data.get('biomass_weight')}.",
            "data",
            "Controls influence of biomass observations.",
            "Application setting.",
            "fixed",
            "yes",
            "medium",
            "Higher values make biomass observations more influential.",
        ),
        AssumptionRecord(
            "ASM-CATCH-MULTIPLIER",
            f"Catch/removal multiplier is {data.get('catch_multiplier')}.",
            "data",
            "Applied to catch/removals before fitting.",
            "Application setting; should be linked to evidence when used for release mortality or hidden removals.",
            "fixed",
            "yes",
            "high",
            "Higher removals generally reduce fitted biomass or increase risk.",
        ),
    ]
    target = data.get("target_depletion")
    assumptions.append(
        AssumptionRecord(
            "ASM-TARGET-DEPLETION",
            "No target depletion constraint is applied." if target is None else f"Target depletion constraint is {target}.",
            "statistical",
            "Soft constraint used only when explicitly supplied.",
            "Application setting or external reference point if supplied.",
            "none" if target is None else "penalty",
            "yes",
            "high" if target is not None else "low",
            "Can force model toward an assumed or external status point.",
        )
    )
    if projection_settings is not None:
        projection = _public_dict(projection_settings)
        assumptions.append(
            AssumptionRecord(
                "ASM-PROJECTION-RULE",
                f"Projection strategy is {projection.get('strategy', 'unknown')}.",
                "management",
                "Selected scenario rule for future catch or fishing pressure.",
                "Application setting.",
                "scenario",
                "yes",
                "high",
                "Changes forecast biomass, catch and risk metrics.",
            )
        )
    return [assumption.to_dict() for assumption in assumptions]


def build_run_manifest(
    name: str,
    run_type: str,
    dataset: Any,
    settings: Any,
    projection_settings: Any | None = None,
    result: Any | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    dataset_provenance = getattr(dataset, "provenance", {}) or {}
    settings_dict = _public_dict(settings)
    projection_dict = _public_dict(projection_settings) if projection_settings is not None else {}
    best = getattr(result, "best", None) or {}
    run_id = stable_id(
        "RUN",
        name,
        run_type,
        dataset_provenance.get("hash_sha256"),
        json.dumps(settings_dict, sort_keys=True, default=str),
        json.dumps(projection_dict, sort_keys=True, default=str),
    )
    return {
        "run_id": run_id,
        "name": name,
        "run_type": run_type,
        "created": datetime.now().isoformat(timespec="seconds"),
        "application": APP_NAME,
        "application_version": APP_VERSION,
        "dataset_name": getattr(dataset, "name", name),
        "dataset_provenance": dataset_provenance,
        "source_data_hashes": [dataset_provenance.get("hash_sha256")] if dataset_provenance.get("hash_sha256") else [],
        "normalisation_log": getattr(dataset, "transformations", []),
        "model_settings": settings_dict,
        "projection_settings": projection_dict,
        "random_seed": settings_dict.get("seed"),
        "model_type": settings_dict.get("model"),
        "model_engine": "internal_proxy_biomass_dynamics",
        "external_model_rebuild": False,
        "best_estimates": best,
        "assumptions": assumptions_from_settings(settings, projection_settings),
        "warnings": warnings or [],
        "output_classification": {
            "fit_history": SOURCE_MODEL_ESTIMATE,
            "projection": SOURCE_DERIVED,
            "sensitivity": SOURCE_DERIVED,
            "optimization": SOURCE_DERIVED,
            "status_comparison": SOURCE_DPIRD,
            "stock_synthesis_import": SOURCE_STOCK_SYNTHESIS,
        },
    }


def _public_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return {}
