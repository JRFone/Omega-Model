# Omega FISH 1.3 Native Numerical Architecture

## Architecture

Omega uses a hybrid design:

- **Python**: desktop interface, workflow orchestration, data handling, evidence records and reporting.
- **Plotly**: interactive charts, zoom, pan, selection, overlays and offline HTML export.
- **C++17 native engine**: repeated population-dynamics and likelihood calculations.
- **OmegaDual forward-mode automatic differentiation**: exact first derivatives for the four production-model parameters implemented in the current native engine.
- **SciPy L-BFGS-B**: bounded refinement using native gradients where available.
- **OpenMP**: parallel batch objective calculations.
- **CMake**: reproducible cross-platform builds and native tests.

The native library exposes a stable C ABI and is loaded by `stock_model/native_backend.py` through `ctypes`. This keeps the desktop and Python workflow independent of compiler-specific C++ bindings. When the shared library is absent, Omega uses a tested Python fallback rather than refusing to run.

## Implemented native functions

- Schaefer, Fox and Pella-Tomlinson production calculations.
- Deterministic biomass trajectories.
- Complete production-model objective components.
- Analytic forward-mode gradients for transformed carrying capacity, productivity, initial depletion and observation error.
- Parallel scoring of candidate parameter sets.
- Configurable native thread count.
- Native/Python parity tests.

## Extension points

CMake options exist for CppAD, Ceres Solver, Ipopt, NLopt, Eigen and CUDA. These are **extension hooks**, not active production dependencies in release 1.3. The internal OmegaDual engine is used for the current four-parameter production model. More complex age-structured derivatives should move to a mature sparse automatic-differentiation framework after equivalence tests are established.

## Speed policy

Omega improves speed by combining algorithmic and implementation changes:

1. Score large candidate batches in compiled C++.
2. Parallelise independent candidates and diagnostic runs.
3. Use automatic gradients for local refinement.
4. Cache profile and repeated-analysis results by configuration hash.
5. Retain progressive output and background execution so interfaces remain responsive.
6. Reserve GPU work for genuinely large independent simulations rather than small sequential fits.

## Current boundary

The production-model objective is native. The full age-structured assessment, ASPM population engine, profile orchestration and interval-coverage orchestration still contain Python components. Release 1.3 establishes the native contract and priority diagnostics; it does not claim that the whole application has been migrated to C++.
