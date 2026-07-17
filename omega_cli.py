from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_model.benchmark_suite import run_benchmarks
from stock_model.closed_loop_mse import MSESettings, ManagementProcedure, OperatingModelSettings, generate_hcr_grid, run_closed_loop_mse
from stock_model.complete_assessment import run_complete_demo
from stock_model.core import ModelSettings, fit
from stock_model.data_io import read_stock_file
from stock_model.expert_workflow import ExpertWorkflowSettings, WorkflowOverride, run_expert_workflow
from stock_model.interactive_charts import ChartProfile, InteractiveChartFactory, SeriesSpec
from stock_model.likelihood_profiles import ProfileSettings, profile_likelihood
from stock_model.aspm_diagnostic import ASPMSettings, run_age_structured_aspm
from stock_model.interval_coverage import CoverageSettings, run_interval_coverage
from stock_model.native_backend import native_status
from stock_model.native_benchmark import NativeBenchmarkSettings, write_native_benchmark
from stock_model.ss3_validation import capability_matrix, competitive_scorecard, download_latest_ss3_executable, download_noaa_model, validate_model_directory, write_validation_report
from stock_model.biomass_truth_engine import BiomassTruthSettings, estimate_best_supported_biomass
from stock_model.experimental_diagnostics import ExperimentalDiagnosticSettings, run_experimental_diagnostics
from stock_model.advanced_mse import (
    AdvancedMSESettings,
    MSEAssessmentSettings,
    MSEManagementProcedure,
    MSEObservationSettings,
    default_operating_scenarios,
    generate_management_grid,
    run_advanced_mse,
)
from stock_model.age_structured import AgeFitSettings, AgeStructuredSettings, fit_age_structured, read_composition_file
from stock_model.truth_mse_charts import write_advanced_mse_dashboard, write_biomass_truth_dashboard, write_experimental_diagnostics_dashboard


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Omega FISH Model command-line interface")
    commands = root.add_subparsers(dest="command", required=True)
    complete = commands.add_parser("complete-demo", help="Run the cumulative Releases 4-11 demonstration")
    complete.add_argument("--years", type=int, default=12)
    complete.add_argument("--seed", type=int, default=45120)
    complete.add_argument("--output", type=Path, default=Path("reports/complete_demo"))
    benchmark = commands.add_parser("benchmarks", help="Run deterministic numerical benchmarks")
    benchmark.add_argument("--output", type=Path, default=Path("reports/benchmarks"))
    mse = commands.add_parser("mse", help="Run a default closed-loop MSE grid")
    mse.add_argument("--years", type=int, default=30)
    mse.add_argument("--simulations", type=int, default=500)
    mse.add_argument("--seed", type=int, default=60013)
    mse.add_argument("--output", type=Path, default=Path("reports/mse.json"))
    noaa = commands.add_parser("noaa-validate", help="Validate a NOAA or local Stock Synthesis model folder")
    noaa.add_argument("model_folder", type=Path, nargs="?", default=Path("validation_data/noaa_ss3/Simple"))
    noaa.add_argument("--model-name", default="Simple")
    noaa.add_argument("--ss3-executable", type=Path)
    noaa.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=Path("reports/noaa_validation"), help="Directory for the HTML and JSON validation reports")
    download = commands.add_parser("noaa-download", help="Download a pinned NOAA SS3 test model")
    download.add_argument("model_name")
    download.add_argument("--output", type=Path, default=Path("validation_cache/noaa_ss3"))
    ss3_download = commands.add_parser("ss3-download", help="Download the latest official NOAA/NMFS Stock Synthesis executable")
    ss3_download.add_argument("--platform", choices=("windows", "linux", "macos"), default="windows")
    ss3_download.add_argument("--output", type=Path, default=Path("tools/ss3"))
    matrix = commands.add_parser("capability-matrix", help="Print Omega capability parity against NOAA SS3 test models")
    matrix.add_argument("--model-name")
    commands.add_parser("competitive-scorecard", help="Print the evidence-gated better-than-SS development scorecard")
    expert = commands.add_parser("expert-workflow", help="Run the automatic or exploration expert diagnostic workflow")
    expert.add_argument("dataset", type=Path)
    expert.add_argument("--model", choices=("schaefer", "fox", "pella"), default="schaefer")
    expert.add_argument("--mode", choices=("automatic", "exploration"), default="automatic")
    expert.add_argument("--speed", choices=("quick", "standard", "deep"), default="quick")
    expert.add_argument("--skip", action="append", default=[], help="Exploration-mode step name to skip; may be repeated")
    expert.add_argument("--override-reason", default="CLI exploration override", help="Reason recorded for CLI exploration overrides")
    expert.add_argument("--workers", type=int, default=0, help="Parallel worker count; 0 selects a safe automatic value")
    expert.add_argument("--output", type=Path, default=Path("reports/expert_workflow.json"))
    native = commands.add_parser("native-status", help="Report the compiled C++ numerical backend status")
    native_build = commands.add_parser("native-build", help="Build and test the local C++ numerical backend")
    native_build.add_argument("--clean", action="store_true")
    native_build.add_argument("--no-openmp", action="store_true")
    native_benchmark = commands.add_parser("native-benchmark", help="Measure native/Python parity and machine-specific batch speed")
    native_benchmark.add_argument("--candidates", type=int, default=10000)
    native_benchmark.add_argument("--years", type=int, default=80)
    native_benchmark.add_argument("--repeats", type=int, default=3)
    native_benchmark.add_argument("--output", type=Path, default=Path("reports/native_benchmark.json"))
    profile = commands.add_parser("profile", help="Run a fully refitted production-model likelihood profile")
    profile.add_argument("dataset", type=Path)
    profile.add_argument("parameter", choices=("k", "r", "initial_depletion", "sigma"))
    profile.add_argument("--model", choices=("schaefer", "fox", "pella"), default="schaefer")
    profile.add_argument("--points", type=int, default=21)
    profile.add_argument("--multistarts", type=int, default=3)
    profile.add_argument("--workers", type=int, default=1)
    profile.add_argument("--output", type=Path, default=Path("reports/likelihood_profile.json"))
    aspm = commands.add_parser("aspm", help="Run the genuine age-structured ASPM and ASPM-R diagnostic")
    aspm.add_argument("dataset", type=Path)
    aspm.add_argument("--age-composition", type=Path)
    aspm.add_argument("--length-composition", type=Path)
    aspm.add_argument("--multistarts", type=int, default=4)
    aspm.add_argument("--output", type=Path, default=Path("reports/aspm_diagnostic.json"))
    coverage = commands.add_parser("coverage", help="Run formal known-truth interval-coverage testing")
    coverage.add_argument("dataset", type=Path)
    coverage.add_argument("--model", choices=("schaefer", "fox", "pella"), default="schaefer")
    coverage.add_argument("--replicates", type=int, default=100)
    coverage.add_argument("--method", action="append", choices=("hessian", "profile", "parametric_bootstrap"), default=[])
    coverage.add_argument("--workers", type=int, default=1)
    coverage.add_argument("--search-draws", type=int, default=160)
    coverage.add_argument("--native-threads-per-worker", type=int, default=1)
    coverage.add_argument("--no-time-series", action="store_true")
    coverage.add_argument("--output", type=Path, default=Path("reports/interval_coverage.json"))
    biomass_evidence = commands.add_parser("biomass-evidence", help="Estimate the best-supported biomass across model structures and index series")
    biomass_evidence.add_argument("dataset", type=Path)
    biomass_evidence.add_argument("--search-draws", type=int, default=500)
    biomass_evidence.add_argument("--samples", type=int, default=1200)
    biomass_evidence.add_argument("--holdout-years", type=int, default=4)
    biomass_evidence.add_argument("--output", type=Path, default=Path("reports/biomass_evidence.json"))
    biomass_evidence.add_argument("--dashboard", type=Path, default=Path("reports/biomass_evidence.html"))
    experimental = commands.add_parser("experimental-diagnostics", help="Run simple and complex experimental model diagnostics")
    experimental.add_argument("dataset", type=Path)
    experimental.add_argument("--search-draws", type=int, default=220)
    experimental.add_argument("--output", type=Path, default=Path("reports/experimental_diagnostics.json"))
    experimental.add_argument("--dashboard", type=Path, default=Path("reports/experimental_diagnostics.html"))
    advanced_mse = commands.add_parser("advanced-mse", help="Run separate-truth age-structured closed-loop management strategy evaluation")
    advanced_mse.add_argument("dataset", type=Path)
    advanced_mse.add_argument("--age-composition", type=Path)
    advanced_mse.add_argument("--years", type=int, default=20)
    advanced_mse.add_argument("--simulations", type=int, default=50, help="Simulations per operating scenario")
    advanced_mse.add_argument("--assessment-mode", choices=("fast_filter", "biomass_ensemble", "full_age_structured"), default="fast_filter")
    advanced_mse.add_argument("--workers", type=int, default=1)
    advanced_mse.add_argument("--full-grid", action="store_true")
    advanced_mse.add_argument("--output", type=Path, default=Path("reports/advanced_mse.json"))
    advanced_mse.add_argument("--dashboard", type=Path, default=Path("reports/advanced_mse.html"))
    chart = commands.add_parser("chart-demo", help="Generate an offline interactive chart demonstration")
    chart.add_argument("--output", type=Path, default=Path("reports/interactive_charts/chart_demo.html"))
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "complete-demo":
        payload = run_complete_demo(args.years, args.seed, args.output)
        print(json.dumps({"output": str(args.output), "reliability": payload["reliability"], "benchmarks": payload["benchmarks"]["summary"]}, indent=2))
        return
    if args.command == "benchmarks":
        payload = run_benchmarks(args.output)
        print(json.dumps(payload["summary"], indent=2))
        return
    if args.command == "mse":
        procedures = generate_hcr_grid((0.35, 0.40, 0.45), (0.10, 0.15), (0.10, 0.20))
        payload = run_closed_loop_mse(OperatingModelSettings(), procedures, MSESettings(args.years, args.simulations, args.seed))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "procedures": len(payload["summary"]), "pareto": len(payload["pareto_front"])}, indent=2))
        return
    if args.command == "noaa-validate":
        result = validate_model_directory(args.model_folder, model_name=args.model_name, native_executable=args.ss3_executable)
        outputs = write_validation_report(result, args.output_dir)
        print(json.dumps({"summary": result.summary, "outputs": outputs}, indent=2))
        return
    if args.command == "noaa-download":
        destination = args.output / args.model_name
        manifest = download_noaa_model(args.model_name, destination)
        print(json.dumps({"destination": str(destination), "files": len(manifest["downloaded_files"]), "commit": manifest["commit"]}, indent=2))
        return
    if args.command == "ss3-download":
        manifest = download_latest_ss3_executable(args.output, platform_name=args.platform)
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "capability-matrix":
        print(json.dumps(capability_matrix(args.model_name), indent=2))
        return
    if args.command == "competitive-scorecard":
        print(json.dumps(competitive_scorecard(), indent=2))
        return
    if args.command == "expert-workflow":
        dataset = read_stock_file(args.dataset)
        payload = run_expert_workflow(
            dataset,
            ModelSettings(model=args.model),
            ExpertWorkflowSettings(
                mode=args.mode,
                speed=args.speed,
                workers=args.workers,
                skipped_steps=tuple(args.skip),
                overrides=tuple(
                    WorkflowOverride("skip_step", step, args.override_reason, "CLI analyst")
                    for step in args.skip
                ),
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2, default=str))
        return
    if args.command == "native-status":
        print(json.dumps(native_status(), indent=2))
        return
    if args.command == "native-build":
        from build_native_backend import build

        result = build(clean=bool(args.clean), configuration="Release", openmp=not bool(args.no_openmp), tests=True)
        print(json.dumps(result, indent=2))
        return
    if args.command == "native-benchmark":
        result = write_native_benchmark(
            args.output,
            NativeBenchmarkSettings(candidates=args.candidates, years=args.years, repeats=args.repeats),
        )
        print(json.dumps({"output": str(args.output), "summary": result}, indent=2, default=str))
        return
    if args.command == "profile":
        dataset = read_stock_file(args.dataset)
        settings = ModelSettings(model=args.model)
        fitted = fit(dataset, settings)
        result = profile_likelihood(
            dataset,
            settings,
            fitted,
            args.parameter,
            ProfileSettings(points=args.points, multistarts=args.multistarts, workers=args.workers, use_cache=False),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2, default=str))
        return
    if args.command == "aspm":
        import pandas as pd

        dataset = read_stock_file(args.dataset)
        age_composition = pd.read_csv(args.age_composition) if args.age_composition else None
        length_composition = pd.read_csv(args.length_composition) if args.length_composition else None
        result = run_age_structured_aspm(
            dataset,
            age_composition=age_composition,
            length_composition=length_composition,
            settings=ASPMSettings(multistarts=args.multistarts),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2, default=str))
        return
    if args.command == "coverage":
        dataset = read_stock_file(args.dataset)
        settings = ModelSettings(model=args.model)
        truth_fit = fit(dataset, settings)
        methods = tuple(args.method) if args.method else ("hessian",)
        result = run_interval_coverage(
            dataset,
            settings,
            truth_fit,
            CoverageSettings(replicates=args.replicates, methods=methods, workers=args.workers, search_draws=args.search_draws, native_threads_per_worker=args.native_threads_per_worker, include_time_series=not args.no_time_series),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2, default=str))
        return
    if args.command == "biomass-evidence":
        dataset = read_stock_file(args.dataset)
        result = estimate_best_supported_biomass(
            dataset,
            BiomassTruthSettings(search_draws=args.search_draws, samples=args.samples, holdout_years=args.holdout_years),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        args.output.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        write_biomass_truth_dashboard(result, args.dashboard)
        print(json.dumps({"output": str(args.output), "dashboard": str(args.dashboard), "summary": result.summary}, indent=2, default=str))
        return
    if args.command == "experimental-diagnostics":
        dataset = read_stock_file(args.dataset)
        result = run_experimental_diagnostics(dataset, settings=ExperimentalDiagnosticSettings(search_draws=args.search_draws))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        write_experimental_diagnostics_dashboard(result, args.dashboard)
        print(json.dumps({"output": str(args.output), "dashboard": str(args.dashboard), "summary": result["summary"]}, indent=2, default=str))
        return
    if args.command == "advanced-mse":
        dataset = read_stock_file(args.dataset)
        age_composition = read_composition_file(args.age_composition) if args.age_composition else None
        base = fit_age_structured(
            dataset,
            AgeStructuredSettings(),
            AgeFitSettings(population=16, generations=6, local_rounds=2, seed=8841),
            age_composition=age_composition,
        )
        procedures = generate_management_grid() if args.full_grid else [
            MSEManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.20, fishing_fraction_of_fmsy=0.55, maximum_catch_change=0.10),
            MSEManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.75),
            MSEManagementProcedure("Yield focused", target_depletion=0.35, limit_depletion=0.10, fishing_fraction_of_fmsy=0.95, maximum_catch_change=0.30),
            MSEManagementProcedure("Seasonal closure", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, seasonal_closure_fraction=0.25),
            MSEManagementProcedure("Lower recreational effort", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, bag_limit_effort_multiplier=0.70),
        ]
        result = run_advanced_mse(
            base,
            procedures,
            scenarios=default_operating_scenarios(),
            observation=MSEObservationSettings(),
            assessment=MSEAssessmentSettings(mode=args.assessment_mode),
            settings=AdvancedMSESettings(years=args.years, simulations_per_scenario=args.simulations, workers=args.workers),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        write_advanced_mse_dashboard(result, args.dashboard)
        print(json.dumps({"output": str(args.output), "dashboard": str(args.dashboard), "summary": result["summary"]}, indent=2, default=str))
        return
    if args.command == "chart-demo":
        years = list(range(1980, 2026))
        values = [max(0.1, 0.95 - 0.014 * (year - 1980)) for year in years]
        figure = InteractiveChartFactory(ChartProfile()).time_series(
            [SeriesSpec("Omega demonstration", years, values, mode="lines+markers")],
            title="Omega FISH interactive chart demonstration",
            x_title="Year",
            y_title="Relative biomass",
        )
        output = InteractiveChartFactory(ChartProfile()).write_html(figure, args.output)
        print(json.dumps({"output": str(output)}, indent=2))
        return


if __name__ == "__main__":
    main()
