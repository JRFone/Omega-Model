# Known Limitations — Omega FISH Model 1.4.1

- The internal spatial age-structured engine is a development implementation, not verified SS3 equivalence.
- SS3 export creates a starting file set, not a completed stock-specific control configuration.
- Random effects are not yet estimated through a Laplace approximation.
- Numerical derivatives can be unstable for poorly scaled or discontinuous objectives.
- The random-walk MCMC baseline does not include HMC/NUTS convergence diagnostics.
- CPUE standardisation uses a log-linear Gaussian model and does not yet implement full delta, Tweedie or spatiotemporal GLMM alternatives.
- Depth-specific post-release mortality uses configured depth mixtures unless raw release-depth data are supplied.
- Tagging currently uses a simplified recapture likelihood and annual transition matrix.
- The reliability grade covers implemented diagnostics only.
- Real-stock accuracy depends on complete, correctly standardised and independently verified data.
- Peer review, benchmark coverage and governance are processes, not software switches.

## NOAA / Stock Synthesis validation boundary

Omega 1.1 verifies selected parser and deterministic-equation results against a
pinned NOAA Simple test-model fixture. This is not full SS3 numerical equivalence.
Features shown as partial or not implemented in the parity matrix must not be
presented as complete. Native SS3 comparisons require the official executable on
the user's computer.

## Omega 1.2 expert-workflow and chart boundary

- The finite-difference gradient is not ADMB automatic differentiation and can be sensitive to scaling and step size.
- Fixed-other-parameter component profiles complement, but do not replace, fully refitted likelihood profiles.
- ASPM-style diagnostics on the production-model route are not identical to removing composition data from a complete SS3 age-structured assessment.
- Quick composition reweighting uses reduced optimizer effort for screening; standard or deep runs are required before scientific interpretation.
- Candidate-ensemble coverage is a diagnostic of Omega's current uncertainty approximation, not formal proof of nominal frequentist coverage.
- Parallel execution and caching improve workflow speed but do not change model mathematics.
- Interactive chart downsampling is display-only. Exported model data remain full resolution.
- Plotly HTML charts are evidence displays, not replacements for machine-readable result files.


## Release 1.3 boundaries

- The C++ engine currently covers production-model dynamics and likelihoods, not every age-structured calculation.
- OmegaDual provides tested first derivatives for four production parameters; it is not yet a replacement for mature sparse AD systems across the entire assessment.
- CppAD, Ceres, Ipopt, NLopt, Eigen and CUDA are configured as optional extension points but are not activated by default.
- ASPM is age structured, but numerical equivalence with ss3diags ASPM across the NOAA catalogue remains to be demonstrated.
- Formal coverage code is implemented; large 500–1,000 replicate stock-specific studies have not been run in this release environment.
- The Windows native DLL and desktop bundle must be built and tested on Windows or through the included GitHub Actions workflow.

## Release 1.4 boundaries

- The Biomass Evidence Engine estimates the best-supported biomass conditional on the supplied catches, indices, observations, priors and model structures. It cannot observe an assumption-free true biomass in a real fishery.
- The advanced MSE supports full age-structured reassessment, but formal runs with hundreds of simulations and many operating scenarios can be computationally expensive. Quick and standard modes are development tools, not management-grade simulation sizes.
- MSE readiness 10/10 means the configured experiment contains the defined components; it does not establish that the operating models cover every plausible truth or that the result has passed independent review.
- Experimental change-point, mutual-information, spectral, data-cloning and adversarial diagnostics identify patterns and sensitivity. They do not prove the cause of a model problem.
- The current advanced MSE is age structured and multi-sector, but it is not yet a complete multispecies ecosystem model or a fully explicit spatial fleet-behaviour simulator.

- The compact state-space biomass candidate uses a bootstrap particle filter and random candidate search. It is useful for structural triangulation but is not yet a sparse-Laplace random-effects implementation comparable with a mature TMB/ADMB state-space assessment.
- A narrow interval is not evidence of truth when structural alternatives, catch history, index standardisation or absolute scale are weakly identified.
