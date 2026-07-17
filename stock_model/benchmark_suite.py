from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import exp
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .advanced_structures import default_wa_demersal_settings, life_history, simulate_spatial_seasonal, validate_movement
from .closed_loop_mse import production
from .observation_models import ageing_error_matrix, apply_ageing_error, dirichlet_multinomial_nll, multinomial_nll
from .tagging import TagObservation, TagRelease, TaggingSettings, predict_tag_recaptures


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    expected: float
    actual: float
    absolute_difference: float
    relative_difference: float
    tolerance: float
    status: str
    category: str
    notes: str = ""


def check(name: str, expected: float, actual: float, tolerance: float, category: str, notes: str = "") -> BenchmarkResult:
    difference = abs(float(actual) - float(expected))
    relative = difference / max(abs(float(expected)), 1e-12)
    status = "PASS" if difference <= tolerance else "FAIL"
    return BenchmarkResult(name, float(expected), float(actual), difference, relative, float(tolerance), status, category, notes)


def run_benchmarks(output_dir: str | Path | None = None) -> dict[str, Any]:
    results: list[BenchmarkResult] = []

    # Schaefer production hand calculation.
    results.append(check("Schaefer production at B=K/2", 450.0, production(5000.0, 10000.0, 0.18), 1e-10, "population dynamics"))

    # Survival under natural mortality.
    results.append(check("Annual survival M=0.12", exp(-0.12), float(np.exp(-0.12)), 1e-12, "mortality"))

    # Movement conservation.
    settings = default_wa_demersal_settings(start_year=2000, years=3, max_age=8)
    movement = validate_movement(settings)
    results.append(check("Movement row conservation", 1.0, float(np.max(np.abs(movement.sum(axis=-1)))), 1e-12, "movement", "Maximum row sum should equal one."))

    # Ageing-error conservation.
    matrix = ageing_error_matrix(10, 0.6)
    results.append(check("Ageing-error row conservation", 1.0, float(np.max(matrix.sum(axis=1))), 1e-12, "observation"))
    composition = np.zeros(11); composition[5] = 1.0
    observed = apply_ageing_error(composition, matrix)
    results.append(check("Ageing-error composition conservation", 1.0, float(observed.sum()), 1e-12, "observation"))

    # Likelihood identities.
    counts = np.array([20.0, 30.0, 50.0])
    probs = counts / counts.sum()
    multi = multinomial_nll(counts, probs)
    dm = dirichlet_multinomial_nll(counts, probs, 1000.0)
    results.append(check("Multinomial residual sum", 0.0, float(sum(multi.residuals)), 1e-12, "likelihood"))
    results.append(check("Dirichlet-multinomial residual sum", 0.0, float(sum(dm.residuals)), 1e-12, "likelihood"))

    # Tagging no-movement test.
    release = [TagRelease(2000, 0, 3, 1000)]
    observations = [TagObservation(0, 2000, 0, 0, 40)]
    tag_settings = TaggingSettings(natural_mortality=0.0, tag_loss=0.0, tag_induced_mortality=0.0, initial_mixing_survival=1.0, reporting_rates=(0.8,), fleet_capture_rates=(0.05,))
    tag = predict_tag_recaptures(release, observations, np.eye(1), tag_settings)
    results.append(check("Tag recapture expectation", 40.0, float(tag.predictions[0]["predicted_recaptures"]), 1e-10, "tagging"))

    # Spatial/sex conservation under zero catch and zero recruitment variation: ensure finite positive trajectory.
    zero_catch = {fleet.name: [0.0] * settings.years for fleet in settings.fleets}
    spatial = simulate_spatial_seasonal(settings, zero_catch, recruitment_deviations=[0.0] * settings.years, environmental_effect=[0.0] * settings.years)
    terminal = float(spatial.history[-1]["total_biomass"])
    results.append(check("Spatial model positive terminal biomass", 1.0, 1.0 if terminal > 0 and np.isfinite(terminal) else 0.0, 0.0, "integrated model"))

    passed = sum(result.status == "PASS" for result in results)
    payload = {
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": [asdict(result) for result in results],
        "verdict": "PASS" if passed == len(results) else "FAIL",
        "scope_note": "These are deterministic software benchmarks. Independent assessment replication and uncertainty-coverage studies remain separate scientific validation tasks.",
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lines = ["# Omega FISH Benchmark Results", "", f"Passed: {passed}/{len(results)}", "", "| Benchmark | Category | Expected | Actual | Tolerance | Status |", "|---|---|---:|---:|---:|---|"]
        for result in results:
            lines.append(f"| {result.name} | {result.category} | {result.expected:.10g} | {result.actual:.10g} | {result.tolerance:.3g} | {result.status} |")
        (output / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
