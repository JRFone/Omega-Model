# Omega FISH Model 1.4.1

## Biomass Evidence, Advanced MSE and Experimental Diagnostics

Release 1.4 adds three connected systems.

1. **Best-supported biomass synthesis** fits deterministic and state-space production structures plus index variants, evaluates holdout prediction, caps single-model dominance, propagates parameter/process uncertainty and grades whether absolute biomass is strongly or weakly identified.
2. **Advanced age-structured MSE** separates the operating truth from the assessment model, simulates catch, abundance indices and age compositions, applies data lags, sector allocations, closures, compliance and implementation error, and tests management procedures across several plausible biological truths.
3. **Experimental diagnostic triangulation** adds catchability/hyperstability tests, residual change points, nonlinear residual memory, spectral structure, Hessian sloppiness, posterior-predictive checks, data cloning and adversarial data perturbations.

The new **Biomass Evidence & Advanced MSE Lab** provides background execution, editable quick/standard/formal workloads, JSON evidence and offline interactive Plotly dashboards.

## Scientific boundary

Real data cannot reveal an assumption-free known true biomass. Omega reports a **best-supported evidence-weighted estimate** and a separate identifiability grade. In simulation, where truth is known, recovery and coverage can be measured directly.

A displayed MSE readiness score of 10/10 means the configured experiment includes all defined formal components. It does not replace stock-specific conditioning, independent implementation checks, scientific peer review or management governance.


## Release validation

- 62 Python unit and integration tests passed.
- 9 deterministic benchmarks passed.
- 31 pinned NOAA Simple checks passed.
- State-space particle-filter smoke test passed.
- Full self-check: 15 checks passed, 0 failed.
