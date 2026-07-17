# Function Release 3 — Integrated Assessment Foundation

- Added `stock_model/age_structured.py`.
- Added a dedicated Integrated Assessment desktop interface.
- Added age and plus-group population dynamics.
- Added growth, weight and maturity-at-age.
- Added Beverton-Holt recruitment and stochastic recruitment projections.
- Added commercial, charter and recreational fleets.
- Added sector selectivity, retention and discard/post-release mortality.
- Added Baranov catch reconstruction and sector dead-discard reporting.
- Added age and length composition inputs and likelihood components.
- Added equilibrium MSY, BMSY and FMSY calculations.
- Added fixed-catch, fixed-F and 40-10 HCR projections.
- Added management strategy evaluation and Pareto frontier.
- Added demonstration datasets, launchers, executable build scripts and five new tests.

# Quant Lab changelog

## 2026-07-15 — Cumulative Quant Lab Release 2

Added:

- covariance-adapting evolution optimiser
- bounded Nelder-Mead multi-start optimiser
- five-optimizer agreement diagnostics
- finite-difference 8D Hessian, eigenvalue and condition-number analysis
- one-dimensional profiles through all eight optimisation dimensions
- rolling walk-forward predictive validation
- Schaefer/Fox/Pella structural ensemble
- between-model disagreement diagnostics
- projection drawdown, downside and expected-shortfall analytics
- self-contained HTML, JSON and CSV report packages
- desktop tabs for optimiser agreement, model ensemble and walk-forward validation
- additional numerical tests

Release 2 is cumulative and includes all Release 1 mathematical corrections and desktop components.

## 1.2.0 — Expert workflow and interactive charts

- Added automatic and exploration diagnostic workflows.
- Added finite-difference gradients, component profiles, MASE, ASPM-style checks, data removal, composition reweighting, recovery, coverage and integrated MSE.
- Added interactive Plotly chart engine, Chart Studio, personal profiles, WebGL rendering and offline dashboards.
- Added CLI and GitHub Actions coverage for the new workspaces.

## 1.3.0 — Native engine and priority diagnostics

- Added compiled C++17 production-model engine, OpenMP batching and automatic gradients.
- Added fully refitted likelihood profiles, genuine age-structured ASPM and formal interval coverage.
- Added Native Engine & Priority Diagnostics desktop workspace and cross-platform CI.
