from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_VERSION = "1.4.1"
SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT


def _check_import(module: str) -> dict[str, Any]:
    try:
        loaded = importlib.import_module(module)
        return {
            "name": f"Import {module}",
            "status": "PASS",
            "detail": getattr(loaded, "__version__", "available"),
        }
    except Exception as exc:
        return {"name": f"Import {module}", "status": "FAIL", "detail": str(exc)}


def _run_unit_tests() -> dict[str, Any]:
    tests_dir = SOURCE_ROOT / "tests"
    if not tests_dir.exists():
        return {
            "name": "Combined unit tests",
            "status": "SKIP",
            "detail": "Tests are not bundled in this executable.",
            "tests_run": 0,
        }
    # Prefer pytest because the packaged suite contains both unittest classes and
    # lightweight pytest-style functions. Fall back to unittest where pytest is
    # unavailable, such as a minimal frozen runtime.
    try:
        test_env = os.environ.copy()
        # Prevent nested BLAS/OpenMP pools from oversubscribing the machine
        # while pytest runs independent numerical smoke tests. Production MSE
        # and profile workers still use their configured parallelism.
        test_env.setdefault("OMP_NUM_THREADS", "1")
        test_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        test_env.setdefault("MKL_NUM_THREADS", "1")
        test_env.setdefault("NUMEXPR_NUM_THREADS", "1")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(tests_dir)],
            cwd=str(SOURCE_ROOT),
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
            env=test_env,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode in {0, 1}:
            import re

            match = re.search(r"(\d+) passed", output)
            tests_run = int(match.group(1)) if match else 0
            return {
                "name": "Combined unit tests",
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "detail": output[-6000:],
                "tests_run": tests_run,
                "runner": "pytest",
                "failures": 0 if completed.returncode == 0 else None,
                "errors": 0 if completed.returncode == 0 else None,
                "skipped": 0,
            }
    except (OSError, subprocess.SubprocessError):
        pass

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return {
        "name": "Combined unit tests",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "detail": stream.getvalue()[-6000:],
        "tests_run": result.testsRun,
        "runner": "unittest fallback",
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }


def run_self_check(*, full_tests: bool = True, demo_years: int = 8) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "Python version",
            "status": "PASS" if sys.version_info >= (3, 11) else "FAIL",
            "detail": platform.python_version(),
        }
    )
    for module in ("numpy", "pandas", "scipy", "plotly", "tkinter"):
        checks.append(_check_import(module))

    try:
        from stock_model.native_backend import native_status

        native = native_status()
        checks.append(
            {
                "name": "C++ native numerical backend",
                "status": "PASS" if native.get("available") else "WARN",
                "detail": native.get("build_info") or native.get("reason"),
                **native,
            }
        )
    except Exception:
        checks.append({"name": "C++ native numerical backend", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        from stock_model.native_benchmark import NativeBenchmarkSettings, run_native_benchmark

        native_parity = run_native_benchmark(NativeBenchmarkSettings(candidates=24, years=16, repeats=1, seed=1303))
        checks.append(
            {
                "name": "Native/Python objective parity",
                "status": "PASS" if native_parity.get("parity_pass") else "FAIL",
                "detail": f"{native_parity.get('valid_comparisons')} comparisons; maximum relative difference {native_parity.get('max_relative_difference')}",
                "measured_speedup_small_smoke": native_parity.get("measured_speedup"),
            }
        )
    except Exception:
        checks.append({"name": "Native/Python objective parity", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        import omega_complete_app  # noqa: F401
        import integrated_assessment_app  # noqa: F401
        import quant_lab_app  # noqa: F401
        import noaa_validation_app  # noqa: F401
        import expert_workflow_app  # noqa: F401
        import chart_studio_app  # noqa: F401
        import priority_diagnostics_app  # noqa: F401
        import mse_truth_lab_app  # noqa: F401
        checks.append({"name": "Desktop interfaces", "status": "PASS", "detail": "All GUI modules imported, including Biomass Evidence & Advanced MSE, Expert Workflow, Priority Diagnostics and Interactive Chart Studio."})
    except Exception:
        checks.append({"name": "Desktop interfaces", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        from stock_model.benchmark_suite import run_benchmarks

        benchmark = run_benchmarks()
        summary = benchmark.get("summary", {})
        passed = int(summary.get("passed", 0))
        total = int(summary.get("total", 0))
        checks.append(
            {
                "name": "Deterministic benchmarks",
                "status": "PASS" if total > 0 and passed == total else "FAIL",
                "detail": f"{passed} of {total} passed",
                "passed": passed,
                "total": total,
            }
        )
    except Exception:
        checks.append({"name": "Deterministic benchmarks", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        from stock_model.ss3_validation import validate_model_directory

        fixture = SOURCE_ROOT / "validation_data" / "noaa_ss3" / "Simple"
        noaa = validate_model_directory(fixture, model_name="Simple")
        noaa_summary = noaa.summary
        checks.append(
            {
                "name": "Pinned NOAA SS3 Simple validation",
                "status": "PASS" if noaa_summary.get("validation_status") == "PASS" else "FAIL",
                "detail": f"{noaa_summary.get('checks_passed')} of {noaa_summary.get('checks_total')} checks passed; feature parity {noaa_summary.get('capabilities_at_parity')}/{noaa_summary.get('capabilities_total')}",
            }
        )
    except Exception:
        checks.append({"name": "Pinned NOAA SS3 Simple validation", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        import numpy as np
        import pandas as pd

        from stock_model.core import _simulate
        from stock_model.data_io import read_stock_csv
        from stock_model.state_space_biomass import StateSpaceBiomassSettings, fit_state_space_biomass

        years = np.arange(2000, 2012)
        catches = np.linspace(35.0, 80.0, len(years))
        truth = _simulate(years, catches, 4000.0, 0.22, 0.80, "schaefer")
        index = truth / 4000.0
        frame = pd.DataFrame({"year": years, "catch": catches, "index": index, "biomass": np.nan})
        dataset = read_stock_csv(frame.to_csv(index=False), "self-check state-space")
        state_result = fit_state_space_biomass(
            dataset,
            StateSpaceBiomassSettings(particles=64, candidates=8, seed=1404),
        )
        state_ok = bool(state_result.diagnostics.get("state_space")) and np.isfinite(state_result.best.get("terminal_biomass", np.nan))
        checks.append(
            {
                "name": "State-space biomass evidence engine",
                "status": "PASS" if state_ok else "FAIL",
                "detail": "Bootstrap particle filter completed with finite latent biomass and uncertainty output.",
            }
        )
    except Exception:
        checks.append({"name": "State-space biomass evidence engine", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        from stock_model.complete_assessment import run_complete_demo

        demo = run_complete_demo(years=max(5, int(demo_years)))
        grade = str(demo.get("reliability", {}).get("grade", ""))
        benchmark_summary = demo.get("benchmarks", {}).get("summary", {})
        demo_ok = bool(grade) and int(benchmark_summary.get("failed", 1)) == 0
        checks.append(
            {
                "name": "Integrated end-to-end demonstration",
                "status": "PASS" if demo_ok else "FAIL",
                "detail": f"Reliability example grade {grade}; benchmark failures {benchmark_summary.get('failed')}",
            }
        )
    except Exception:
        checks.append({"name": "Integrated end-to-end demonstration", "status": "FAIL", "detail": traceback.format_exc()})

    try:
        from tempfile import TemporaryDirectory
        from stock_model.interactive_charts import ChartProfile, InteractiveChartFactory, SeriesSpec

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "chart_smoke.html"
            factory = InteractiveChartFactory(ChartProfile(range_slider=False))
            figure = factory.time_series([SeriesSpec("Smoke", [1, 2, 3], [1.0, 1.5, 1.2])])
            factory.write_html(figure, output, include_plotlyjs=True)
            ok = output.exists() and output.stat().st_size > 1000
        checks.append({"name": "Interactive chart engine", "status": "PASS" if ok else "FAIL", "detail": "Offline Plotly chart generated and validated."})
    except Exception:
        checks.append({"name": "Interactive chart engine", "status": "FAIL", "detail": traceback.format_exc()})

    if full_tests:
        checks.append(_run_unit_tests())

    failures = [row for row in checks if row["status"] == "FAIL"]
    warnings = [row for row in checks if row["status"] in {"WARN", "SKIP"}]
    return {
        "application": "Omega FISH Model",
        "version": APP_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_status": "READY" if not failures else "NOT_READY",
        "scientific_status": "REQUIRES STOCK-SPECIFIC CALIBRATION AND INDEPENDENT REVIEW",
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "checks_failed": len(failures),
        "checks_warned_or_skipped": len(warnings),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Omega FISH software readiness self-check.")
    parser.add_argument("--quick", action="store_true", help="Skip the complete unit-test discovery run.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "self_check_latest.json")
    args = parser.parse_args()
    result = run_self_check(full_tests=not args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["software_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
