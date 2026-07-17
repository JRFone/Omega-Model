# Omega FISH 1.3 Priority Diagnostics

Development priority was frozen around three diagnostics before further feature expansion.

## 1. Fully refitted likelihood profiles

For every profile point Omega fixes the selected parameter and re-estimates all remaining active production-model parameters. The implementation includes multiple starts, neighbouring-point warm starts, parallel profile points, failed-point retention, component likelihoods, derived stock quantities, profile confidence thresholds, one-dimensional profiles and two-dimensional surfaces.

Supported profile parameters in release 1.3 are carrying capacity, productivity, initial depletion and observation error. Age-structured parameters such as natural mortality, steepness and selectivity remain future extensions.

## 2. Genuine age-structured ASPM and ASPM-R

The ASPM diagnostic retains age structure, natural mortality, growth, maturity, weight-at-age, selectivity, retention, release mortality, catches and stock-recruit assumptions. It removes composition likelihood information and tests whether catches plus abundance indices support the broad trajectory.

Variants include standard ASPM, ASPM-R with recruitment deviations, no-index ASPM, index-specific influence and comparison with the complete age-structured fit. A no-index result is explicitly labelled assumption-driven.

## 3. Formal known-truth interval coverage

Omega repeatedly simulates data from a known population, refits the model, calculates uncertainty intervals and checks whether each interval contains the truth. It records failed fits rather than silently removing them.

Implemented interval methods are Hessian/delta, refitted profile likelihood and parametric bootstrap. Outputs include empirical coverage, Wilson Monte Carlo intervals, bias, relative bias, RMSE, mean interval width, failures, time-series coverage and false stock-status classifications.

## Scientific interpretation

These tools expose evidence; they do not automatically establish that a model is correct. A profile can be flat because the parameter is weakly identified. ASPM disagreement identifies dependence on composition data or structure but does not by itself prove the full model wrong. Good nominal coverage under the same generating and estimation model does not guarantee robustness to misspecification.
