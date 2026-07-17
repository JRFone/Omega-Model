# Experimental Diagnostic Triangulation

Release 1.4 adds diagnostics intended to reveal patterns that standard residual and convergence checks can miss.

- catchability drift and hyperstability regression;
- residual mean change-point search using BIC;
- nonlinear lag dependence using mutual information and permutation testing;
- residual frequency-spectrum concentration;
- Hessian eigenvalue and parameter-sloppiness analysis;
- posterior-predictive-style residual checks;
- experimental data-cloning contraction checks;
- adversarial perturbation of catch, index trends, scale and terminal observations;
- simple lag-one and runs tests retained beside the complex methods.

These diagnostics are **hypothesis generators**. A flag identifies a pattern requiring targeted model and data investigation. It does not prove a biological mechanism or automatically invalidate an assessment.
