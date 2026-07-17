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

`Start_Omega_FISH_Model.bat` launches the same desktop application and prefers the local virtual environment. Individual workspaces can also be opened directly, for example:

```powershell
.\.venv\Scripts\python.exe omega_desktop.py --mode integrated
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

## Repository structure

- `stock_model/` — assessment, diagnostics, MSE, chart, SS3, and native-loader modules
- `native/` — C++17 native engine, headers, CMake configuration, and native tests
- `tests/` — Python unit, integration, parity, validation, and release-readiness tests
- `Data_Sets/` — demonstration datasets
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
