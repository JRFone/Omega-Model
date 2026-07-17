# Omega FISH Model 1.4.1

Omega FISH is a research and decision-support application for fish-stock assessment, quantitative diagnostics, biomass-evidence synthesis, management strategy evaluation (MSE), and structural comparison with NOAA Stock Synthesis 3 (SS3) inputs and outputs.

The repository is self-contained at its root. Runtime outputs are written below `reports/` and are not version-controlled.

## Requirements

- Windows 10 or later for the desktop workflow described here
- Python 3.11 or later (Python 3.12 is used in GitHub Actions)
- CMake 3.27 or later
- A C++17 compiler (Visual Studio 2022 Build Tools with the **Desktop development with C++** workload is recommended)
- Git, for source control only

## Install from the repository root

Open PowerShell in this directory and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements_runtime.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_build.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

If Python 3.12 is unavailable, use another installed Python version that satisfies `pyproject.toml` (`>=3.11`).

## Launch Omega

The exact root-level development launch command is:

```powershell
.\.venv\Scripts\python.exe omega_desktop.py
```

`Launch Omega FISH Model.bat` launches the same application and prefers the
local virtual environment. Run `CREATE_DESKTOP_SHORTCUT.bat` once to create the
**Omega FISH Model** Windows Desktop shortcut.

The default application is a single-window shell with Back, Forward, Home,
an adjustable and mouse-wheel-scrollable sidebar, a quick dataset dropdown,
main-content scrolling, dark/light/high-contrast themes, and optional detachable
workspaces. **Settings** is at the bottom of the sidebar and can move the sidebar
to the left, top, right, or bottom. **RESET DEFAULTS** restores the standard
realistic workload and display settings. Select
**Guided practice** for a required-click tour that highlights each real control,
or **WATCH A COMPLETE MODEL** for an automatic in-window demonstration
that loads the protected beginner dataset and completes a short, real
age-structured teaching fit. **FULL AUTO RUN** uses the active dataset and starts
the complete automatic expert workflow, running every check supported by its
available inputs while retaining warnings and failures.

Long-running fits, diagnostics, downloads, projections and optimizations show a
bottom processing bar. The sidebar **Error Log** keeps callback and workspace
failures visible. Right-clicking a chart, dataset table, result table, slider or
workspace background opens actions relevant to that part of the interface.
Integrated Assessment control groups are collapsible, and both the control
panel and main workspace support mouse-wheel scrolling.

Open **Visual Parameter Lab** to move carrying-capacity, growth, starting-biomass,
natural mortality, recruitment strength/variability, fishing pressure,
catchability and observation-error sliders. Its biomass chart updates
immediately, displays the selected uncertainty range, and overlays observed
biomass or a visibly rescaled abundance index when available. The lab is a
deterministic scenario explorer, not a fitted parameter estimate or accuracy
claim. Individual workspaces can still be detached or
opened directly, for example:

```powershell
.\.venv\Scripts\python.exe omega_desktop.py --mode integrated
.\.venv\Scripts\python.exe omega_desktop.py --mode parameters
.\.venv\Scripts\python.exe omega_desktop.py --mode truthmse
.\.venv\Scripts\python.exe omega_desktop.py --mode priority
.\.venv\Scripts\python.exe omega_desktop.py --mode noaa
.\.venv\Scripts\python.exe omega_desktop.py --mode charts
```

## Build and verify the native engine

```powershell
.\.venv\Scripts\python.exe build_native_backend.py --clean
.\.venv\Scripts\python.exe omega_cli.py native-status
```

The build uses `native/CMakeLists.txt`, runs the native CTest target, and copies the platform library into `stock_model/native_libs/`. The library and generated `native_build.json` are local build products and are ignored by Git.

## Run tests and self-checks

Run the full Python suite and software self-check:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe omega_self_check.py --output reports\self_check_latest.json
.\.venv\Scripts\python.exe -m stock_model.native_benchmark
.\.venv\Scripts\python.exe -m compileall omega_desktop.py omega_cli.py omega_self_check.py stock_model tests
```

Run native tests directly after building:

```powershell
.\.venv\Scripts\ctest.exe --test-dir build\native -C Release --output-on-failure
```

If `ctest.exe` is not present in `.venv\Scripts`, use the `ctest` executable reported by `Get-Command ctest`.

## NOAA Simple validation

The pinned fixture validation is deterministic and offline:

```powershell
.\.venv\Scripts\python.exe omega_cli.py noaa-validate validation_data\noaa_ss3\Simple --model-name Simple --output-dir reports\noaa_offline
```

This validates the supplied starter, data, and control files and exercises report generation. It is not the same as running the official SS3 executable or demonstrating full numerical equivalence. Native SS3 comparison requires a separately downloaded official executable:

```powershell
.\.venv\Scripts\python.exe omega_cli.py ss3-download --platform windows --output tools\ss3
```

Review [NOAA_VALIDATION.md](NOAA_VALIDATION.md) for the evidence levels and remaining parity gaps before interpreting a comparison.

## Dataset and NOAA test-model library

Omega discovers model inputs below `Data_Sets/` and displays them in the
in-application Dataset Library. Each Omega dataset may include an
`omega_dataset.json` file describing its source, difficulty, inputs, and
scientific purpose. Original inputs are not modified by the single-window
workflow.

Use the header dropdown for a quick selection or **BROWSE DATASETS** for the
searchable library and its difficulty and coverage filters. **Omega Complete
Demonstration / Diagnostics Reference (Known Simulation)** is the packaged
full-input dataset for all native Omega workspaces. In addition to annual,
age-composition and length-composition inputs, it includes chart-ready examples
for optimizer grids, retrospective peels, likelihood profiles, coverage and MSE.
It is synthetic software test data, not evidence about a real stock. NOAA/SS3
validation uses the separately packaged official NOAA Simple configuration.

The Dataset Library also includes the **WA Dhufish — DPIRD public-evidence
reconstruction**. It preserves the official publications and labels
vector-extracted or digitised series as approximations; it is not represented as
the unpublished raw DPIRD assessment dataset.

The **Quick model health** tab in Priority Diagnostics wraps long explanations
and presents the quick verdict, accuracy evidence, confounding risk, reason, and
next action in separate readable columns. “Accurate” is reserved for known-truth
recovery evidence; a successful software run alone is not described as accurate.

In **NOAA / SS3**, click **RUN NOAA DATA + COMPARE**. The **NOAA vs Omega** tab
shows each pinned NOAA reference answer beside Omega's result, difference,
tolerance, and verdict. Selecting an official SS3 executable additionally runs
NOAA's program; the table alone is not a claim of full SS3 numerical parity.

Download or refresh the current official NOAA/NMFS Stock Synthesis test models
and user examples from inside the Dataset Library, or run:

```powershell
.\.venv\Scripts\python.exe tools\download_noaa_test_data.py --refresh
```

The large snapshots are stored locally under `Data_Sets/NOAA/_sources/` and are
excluded from Git. `NOAA_SOURCE_MANIFEST.json` records the exact repository
commit SHAs, and the CSV/JSON catalogues describe the model folders discovered
in those snapshots.

## Repository structure

- `stock_model/` — assessment, diagnostics, MSE, chart, SS3, and native-loader modules
- `native/` — C++17 native engine, headers, CMake configuration, and native tests
- `tests/` — Python unit, integration, parity, validation, and release-readiness tests
- `Data_Sets/` — Omega demonstrations, local NOAA test models, and dataset metadata
- `ui/` — single-window shell support, dataset discovery, themes, and tutorials
- `tools/` — reproducible NOAA downloader and Windows shortcut utilities
- `validation_data/` — pinned NOAA/SS3 validation fixtures
- `assets/` — application icon and packaged assets
- `installer/` — Inno Setup source; not required for development installation
- `.github/workflows/` — root-relative CI validation workflows
- `omega_desktop.py` — desktop launcher
- `omega_cli.py` — command-line interface
- `omega_self_check.py` — integrated software-readiness checks

## Scientific limitations

Omega is research software, not a scientifically certified assessment system. Results require stock-specific calibration, sensitivity analysis, independent review, and scrutiny of data quality and structural assumptions.

The “best-supported biomass” result is an evidence-weighted estimate across the models and indices supplied to the application. It is **not assumption-free true biomass**. A high model weight, narrow uncertainty interval, deterministic benchmark pass, or agreement with a fixture does not prove that the model structure or observations represent the real stock.

The pinned NOAA Simple workflow verifies parsing, deterministic fixtures, comparison plumbing, and report generation. Unless a compatible official SS3 executable is run successfully, it does not include native SS3 execution. Even when native SS3 runs, agreement on selected outputs is not evidence of complete SS3 feature or numerical equivalence.

Advanced MSE, interval coverage, retrospective analysis, hindcasts, jitter/multistart diagnostics, and structural comparisons are scenario-dependent diagnostics. Small smoke-test workloads verify software execution only and must not be reported as formal scientific validation.

Additional limitations are documented in [KNOWN_LIMITATIONS_COMPLETE.md](KNOWN_LIMITATIONS_COMPLETE.md), [MODEL_GOVERNANCE.md](MODEL_GOVERNANCE.md), and [VALIDATION_PLAN_COMPLETE.md](VALIDATION_PLAN_COMPLETE.md).
