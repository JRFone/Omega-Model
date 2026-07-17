from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import ModelSettings, _objective_breakdown, _simulate
from .native_backend import get_native_engine, native_status


@dataclass(frozen=True)
class NativeBenchmarkSettings:
    candidates: int = 10000
    years: int = 80
    repeats: int = 3
    seed: int = 130031


def run_native_benchmark(settings: NativeBenchmarkSettings | None = None) -> dict[str, Any]:
    settings = settings or NativeBenchmarkSettings()
    rng = np.random.default_rng(settings.seed)
    years = np.arange(1950, 1950 + settings.years, dtype=np.int32)
    catches = np.maximum(0.0, 350.0 + 140.0 * np.sin(np.linspace(0.0, 8.0, settings.years)))
    true_biomass = _simulate(years, catches, 12000.0, 0.24, 0.88, "schaefer")
    index = 0.015 * true_biomass * np.exp(rng.normal(0.0, 0.18, settings.years))
    biomass_observed = np.full(settings.years, np.nan)
    biomass_observed[::8] = true_biomass[::8] * np.exp(rng.normal(0.0, 0.12, len(true_biomass[::8])))
    model_settings = ModelSettings(model="schaefer")

    theta = np.column_stack(
        [
            rng.uniform(np.log(5000.0), np.log(25000.0), settings.candidates),
            rng.uniform(np.log(0.05), np.log(0.7), settings.candidates),
            rng.uniform(-2.5, 3.0, settings.candidates),
            rng.uniform(np.log(0.05), np.log(0.8), settings.candidates),
        ]
    )

    python_times: list[float] = []
    native_times: list[float] = []
    python_values = None
    native_values = None
    engine = get_native_engine()

    for _ in range(max(1, settings.repeats)):
        started = time.perf_counter()
        python_values = np.asarray(
            [_objective_breakdown(row, years, catches, index, biomass_observed, model_settings)[0] for row in theta],
            dtype=float,
        )
        python_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        native_values, _gradient, _backend = engine.batch_objective(
            theta, years, catches, index, biomass_observed, model_settings, gradients=False
        )
        native_times.append(time.perf_counter() - started)

    assert python_values is not None and native_values is not None
    finite = np.isfinite(python_values) & np.isfinite(native_values) & (python_values < 1e17) & (native_values < 1e17)
    max_abs = float(np.max(np.abs(python_values[finite] - native_values[finite]))) if finite.any() else float("nan")
    max_rel = float(np.max(np.abs(python_values[finite] - native_values[finite]) / np.maximum(1.0, np.abs(python_values[finite])))) if finite.any() else float("nan")
    python_median = float(np.median(python_times))
    native_median = float(np.median(native_times))
    speedup = python_median / native_median if native_median > 0 else float("inf")
    status = native_status()
    return {
        "benchmark": "production objective batch",
        "settings": asdict(settings),
        "native_status": status,
        "python_seconds": python_median,
        "native_seconds": native_median,
        "measured_speedup": float(speedup),
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "valid_comparisons": int(finite.sum()),
        "parity_pass": bool(max_rel <= 1e-10),
        "note": "Machine-specific benchmark. Speed varies with CPU, compiler, thread count and candidate count.",
    }


def write_native_benchmark(path: str | Path, settings: NativeBenchmarkSettings | None = None) -> dict[str, Any]:
    result = run_native_benchmark(settings)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = write_native_benchmark("reports/native_benchmark.json")
    print(json.dumps(result, indent=2, default=str))
