# Omega FISH Model 1.4.1

Omega FISH is a Windows desktop stock-assessment, diagnostic and management-strategy platform. Version 1.2 places eight workspaces behind one launcher:

1. **Integrated Assessment** — age-structured fitting, biology, fleets, selectivity, retention, discards, compositions and projections.
2. **Automatic Expert Workflow** — convergence, jitter, optimizers, profiles, retrospectives, hindcasts, ASPM, influence, recovery, coverage, ensembles, MSE and reliability grading.
3. **Interactive Chart Studio** — zoomable, editable, personalisable scientific charts and offline dashboards.
4. **Quant Lab** — global optimization, high-dimensional diagnostics, stress tests, model ensembles and risk frontiers.
5. **NOAA / SS3 Validation** — pinned NOAA test models, parser checks, native SS3 execution and capability tracking.
6. **Validation and MSE** — deterministic benchmarks, CPUE standardization, tagging demonstrations and closed-loop MSE.
7. **System Self-Check** — dependencies, interfaces, benchmarks, NOAA checks, chart generation and complete tests.
8. **Roadmap and Evidence** — mathematical specification, governance, limitations and validation documentation.

## One-click Windows setup

Double-click:

```text
SETUP_OMEGA_FISH.bat
```

Setup creates an isolated environment, installs dependencies, runs validation, builds the Windows executable, installs it for the current user and creates shortcuts. It stops before installation when validation fails.

## Starting from source

```text
Start_Omega_FISH_Model.bat
```

Direct workspace launchers include:

```text
Start Omega FISH Expert Workflow.bat
Start Omega FISH Interactive Charts.bat
Start Omega FISH NOAA Validation.bat
Start Omega FISH Quant Lab.bat
Start Omega FISH Integrated Assessment.bat
```

## Automatic versus exploration mode

Automatic mode runs every implemented expert diagnostic gate. Exploration mode permits alternative ideas and recorded skips. Omega does not block investigation, but it does not silently remove skipped or failed checks from the evidence report.

See `EXPERT_WORKFLOW.md`.

## Interactive charts

Charts support zoom, pan, selection, linked hover, range sliders, uncertainty overlays, editable titles, annotations, PNG export, personal profiles and offline dashboards. Long series use WebGL and display-only downsampling for responsive interaction.

See `INTERACTIVE_CHARTS.md`.

## Command-line examples

```text
python omega_cli.py expert-workflow Data_Sets/Data_set_Age_Structured_Demo/model_ready_timeseries.csv --mode automatic --speed quick --output reports/expert_workflow.json
python omega_cli.py chart-demo --output reports/interactive_charts/chart_demo.html
python omega_cli.py noaa-validate validation_data/noaa_ss3/Simple --model-name Simple --output-dir reports/noaa_validation
python omega_cli.py capability-matrix
python omega_cli.py competitive-scorecard
```

## Building a distributable installer

```text
BUILD_FINAL_WINDOWS_RELEASE.bat
```

The portable application is produced under `dist\Omega FISH Model`. When Inno Setup 6 is available, a conventional installer is also produced under `release`.

## Scientific boundary

Software readiness means the program builds and its implemented calculations pass the supplied tests. A management assessment still requires verified data, defensible assumptions, stock-specific configuration, full diagnostics, independent replication where possible and independent scientific review.

Omega 1.4 is designed to make those requirements visible and difficult to omit; it does not replace them.


## Omega 1.3 native engine and priority diagnostics

Use **Native Engine & Priority Diagnostics** from the launcher for compiled-engine status, refitted likelihood profiles, age-structured ASPM/ASPM-R and formal interval coverage. Run `BUILD_OMEGA_NATIVE_BACKEND.bat` after installing a Windows C++ build toolchain when setup could not create the native DLL. Omega retains a Python fallback, but the C++ engine is the intended high-performance path.

## Omega 1.4 biomass evidence and advanced MSE

The launcher now includes **Biomass Evidence & Advanced MSE**. This workspace automatically fits deterministic and state-space biomass models plus index variants, reports a best-supported evidence-weighted trajectory with an identifiability grade, runs a separate-truth age-structured closed-loop MSE, and applies both simple and experimental diagnostics. Quick, standard and formal presets remain editable. All runs preserve JSON evidence and offline interactive dashboards.

Real data do not provide an assumption-free known true biomass. Omega therefore separates the estimated biomass trajectory from its evidence-strength grade and warnings. Formal MSE configuration can reach a 10/10 completeness score, but that score is not independent scientific certification.


### Release 1.4 command-line examples

```text
python omega_cli.py biomass-evidence Data_Sets/Data_set_Age_Structured_Demo/model_ready_timeseries.csv --output reports/biomass_evidence.json --dashboard reports/biomass_evidence.html
python omega_cli.py experimental-diagnostics Data_Sets/Data_set_Age_Structured_Demo/model_ready_timeseries.csv --output reports/experimental_diagnostics.json --dashboard reports/experimental_diagnostics.html
python omega_cli.py advanced-mse Data_Sets/Data_set_Age_Structured_Demo/model_ready_timeseries.csv --age-composition Data_Sets/Data_set_Age_Structured_Demo/age_composition.csv --years 20 --simulations 50 --assessment-mode fast_filter --output reports/advanced_mse.json --dashboard reports/advanced_mse.html
```
