# Omega FISH Model — Quant Lab

The Quant Lab is a standalone Windows desktop interface for quantitative fish-stock analysis. It runs locally, uses the existing Omega FISH data format, and does not require a browser for its main interface.

## Start the desktop program

Double-click:

```text
Start Omega FISH Quant Lab.bat
```

The existing browser interface remains available from the Quant Lab toolbar.

## Main analysis modules

### Corrected production-model core

The updated core:

- uses analytic model-specific MSY, BMSY and FMSY for Schaefer, Fox and Pella-Tomlinson
- calculates every projection member's depletion using that member's own carrying capacity `K`
- no longer silently replaces the requested initial biomass with first-year catch
- exposes likelihood, prior and penalty components separately
- exposes initial-depletion and catch-to-capacity penalty controls
- labels candidate-fit spread as non-posterior uncertainty
- stores member-specific terminal biomass, depletion and reference points

### Global parameter optimisation

Available dependency-free optimisers:

- Differential Evolution
- continuous genetic optimisation
- covariance-adapting evolution strategy (`cma_es`)
- bounded Nelder-Mead multi-start (`nelder_mead`)
- random multi-start comparison
- local coordinate refinement after the global search

The default eight-dimensional search covers:

1. carrying capacity `K`
2. productivity `r`
3. initial depletion
4. observation error `sigma`
5. CPUE/index weight
6. biomass-observation weight
7. catch/removals multiplier
8. Pella-Tomlinson shape

The internal covariance-adaptation method follows the core CMA-ES concept but is not presented as a reference implementation of the external CMA-ES software.

### Five-optimizer agreement test

The same data and objective are solved independently using five algorithms. The output compares:

- objective value and objective delta
- terminal depletion
- `K`, `r`, initial depletion and error
- data weights and catch multiplier
- local-identifiability status
- range and coefficient of variation across algorithms

Material optimiser disagreement is evidence of a difficult, flat, multi-modal or incompletely searched objective surface.

### Eight-dimensional diagnostics

The interface provides:

- parallel-coordinate paths
- pairwise correlations
- rank correlations with objective and terminal depletion
- principal-component loadings
- finite-difference 8D Hessian/curvature matrix
- curvature eigenvalues and effective rank
- condition-number classification
- weak-direction parameter loadings
- one-dimensional profiles through every search dimension

Flat or negative-curvature directions indicate weak local identification, boundary effects or an imperfect optimum. They are not confidence intervals.

### Three-dimensional objective surface

The rotatable 3D view displays:

- `K`
- `r`
- objective-function delta

The surface is sliced through the other six values from the selected best eight-dimensional candidate.

### Cross-model ensemble and structural disagreement

The same dataset is fitted with:

- Schaefer
- Fox
- Pella-Tomlinson

The module reports model-specific fit and projection results, between-model ranges, and relative candidate weights based on objective delta. These are candidate-comparison weights, not Bayesian model probabilities.

### Rolling walk-forward validation

The model is repeatedly fitted using only earlier years and then predicts later observations. It reports:

- holdout CPUE/index log RMSE and bias
- holdout biomass relative RMSE and bias when biomass observations exist
- fold-by-fold predictions
- terminal-depletion instability across prediction origins

This is the fisheries equivalent of walk-forward backtesting. It tests prediction rather than only in-sample fit.

### Genetic HCR risk frontier

A multi-objective genetic search evaluates harvest-control-rule settings against:

- average catch
- probability of falling below the biomass limit
- catch volatility
- expected biological-limit shortfall

The output is a Pareto frontier. There is no single scientifically correct optimum without explicit management preferences.

### Projection risk analytics

Risk outputs include:

- cumulative and average catch
- catch volatility
- maximum and terminal probability below the limit
- expected shortfall below the biological limit
- maximum depletion drawdown
- downside depletion p10
- probability-based rebuilding year
- risk-adjusted yield index

The depletion p10 is a simulation quantile, not financial VaR. Expected shortfall is the average biological-limit deficit conditional on a simulated breach.

### Controlled stress tests

Scenarios include:

- catch under-reporting and over-reporting
- CPUE/index level bias
- CPUE hyperstability and hyperdepletion
- an index catchability regime shift
- missing index years
- index observation noise
- biomass-observation scaling

### Sensitivity and regime screens

The Quant Lab includes:

- Saltelli-style first-order and total-order projection sensitivity screening
- exploratory CPUE/index change-point screening

These tools identify sensitivity and possible structural changes. They do not establish the cause of a change.

## Export an analysis package

Use **Export results** in the desktop toolbar. The application writes:

- a self-contained HTML technical report
- complete JSON output
- CSV tables for optimiser candidates, risk frontier, stress tests, sensitivity, walk-forward validation, optimiser agreement and model ensemble

## Build a Windows executable

Double-click:

```text
build_quant_lab_exe.bat
```

The script creates:

```text
dist\Omega FISH Quant Lab\Omega FISH Quant Lab.exe
```

## Validation

Run:

```text
run_quant_tests.bat
```

or:

```powershell
python -m unittest discover -s tests -v
```

The current isolated suite covers:

- Schaefer, Fox and Pella-Tomlinson production and reference points
- objective-component accounting
- initial-state handling
- member-specific projection depletion
- model-specific HCR calculations
- Differential Evolution, genetic, covariance-adaptation, Nelder-Mead and random multi-start logic
- 8D local curvature and profile diagnostics
- Pareto-front non-domination
- stress testing and regime screening
- Saltelli-style sensitivity screening
- rolling walk-forward validation
- optimiser agreement
- cross-model ensemble calculations
- projection risk metrics
- HTML/JSON/CSV report generation

Passing software tests does not establish real-stock accuracy. Real-stock accuracy remains conditional on data quality, model structure, biological assumptions and complete uncertainty propagation.


## Function Release 3: Integrated Assessment Lab

The Quant Lab toolbar now opens a separate age-structured interface. It adds ages, growth, maturity, Beverton-Holt recruitment, sector selectivity, retention, post-release mortality, Baranov catch reconstruction, composition fitting, equilibrium reference points, stochastic projections and management strategy evaluation. See `INTEGRATED_ASSESSMENT.md`.
