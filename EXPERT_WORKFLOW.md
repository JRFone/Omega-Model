# Omega FISH Automatic Expert Workflow

Omega 1.2 combines diagnostic and model-development functions that are commonly assembled from Stock Synthesis, r4ss, ss3diags, ss3sim, SSMSE and custom R scripts.

## Operating modes

### Automatic mode

Automatic mode runs every implemented diagnostic gate. A failed check does not disappear and the workflow continues so the analyst can see the complete evidence pattern. The final reliability grade incorporates the available convergence, consistency, predictive and conflict evidence.

### Exploration mode

Exploration mode allows alternative configurations, skipped steps and unconventional ideas. It does not prohibit investigation. Every skip or override is preserved in the exported evidence record with the reason supplied by the analyst.

This design separates two principles:

- exploration should remain open;
- management conclusions should not silently bypass failed or untested diagnostics.

## Automated workflow

The workspace currently runs:

1. Base assessment fit.
2. Residual diagnostics and residual heatmap.
3. Parameter-bound checks.
4. Finite-difference maximum-gradient check.
5. Jitter and random-multistart stability.
6. Independent optimizer agreement.
7. Local Hessian, weak-direction and likelihood-profile diagnostics.
8. Likelihood-component preference profiles.
9. Retrospective peels and Mohn's rho.
10. Walk-forward hindcasting and MASE.
11. ASPM-style catch-and-index diagnostic.
12. Data-removal influence analysis.
13. Index and biomass weighting comparison.
14. Age and length composition reweighting when composition files are present.
15. Data-conflict matrix.
16. Schaefer, Fox and Pella structural ensemble.
17. Known-truth simulation-recovery testing.
18. Candidate-interval coverage testing.
19. Closed-loop management strategy evaluation.
20. Automatic evidence-based reliability grade.

## Analysis depth

- **Quick** uses small repetition counts for interactive development.
- **Standard** increases optimizer, jitter, simulation and MSE effort.
- **Deep** is intended for overnight or high-performance runs.

Independent steps use parallel workers where practical. Results may be cached by dataset and configuration. The desktop remains responsive because the workflow runs in a background thread and reports the active step.

## Composition files

When the selected dataset is in a folder containing:

```text
age_composition.csv
length_composition.csv
```

Omega automatically adds composition reweighting runs. The files are not required for production-model diagnostics.

## Outputs

The workflow writes:

- complete JSON evidence;
- a diagnostic-gate table;
- reliability evidence;
- interactive charts and a tabbed HTML dashboard;
- override and skip records;
- simulation-recovery and MSE results.

## Scientific boundary

The workflow automates implemented diagnostics. It does not establish full SS3 numerical parity, validate a stock-specific dataset, replace independent fisheries-science review or certify a model for regulation.
