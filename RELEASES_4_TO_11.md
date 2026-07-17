# Omega FISH Model — Cumulative Development Releases 4–11

This cumulative build adds the functional foundations requested after Function Release 3. It deliberately prioritises capability over interface cleanup.

## Release 4 — Spatial and biological structure

- Separate female and male life histories.
- Multiple named areas.
- Seasonal population dynamics.
- Age-, sex-, season- and origin-specific movement matrices.
- Area-specific recruitment shares and productivity multipliers.
- Fecundity-weighted spawning output.
- Time blocks for fleet selectivity and retention.
- Depth-weighted post-release mortality by fleet.

## Release 5 — Composition likelihoods and ageing error

- Normal, lognormal and Student-t likelihoods.
- Multinomial, Dirichlet-multinomial and logistic-normal compositions.
- Explicit ageing-error matrices.
- Application of ageing error to predicted true ages.
- Francis-style composition weighting diagnostics.
- Dirichlet concentration profiling and approximate effective sample size.
- Fully itemised likelihood-component table.

## Release 6 — CPUE, catchability and changing fishing behaviour

- Log-linear CPUE standardisation.
- Year, vessel, area, month and continuous covariate effects.
- Ridge-stabilised coefficient estimation.
- Annual standardised index with approximate intervals.
- Catchability elasticity diagnostics.
- Hyperstability and hyperdepletion classification.
- Technology-trend term.
- Fleet time blocks for selectivity, retention and fishing power.
- Depth-specific release mortality.

## Release 7 — Estimation and uncertainty engine

- Parameter specification, bounds, transforms, fixed/estimated flags and phases.
- Gaussian priors.
- Bounded multi-start Nelder–Mead optimisation.
- Numerical gradients and Hessian.
- Covariance, standard errors and correlation matrix.
- Boundary and conditioning diagnostics.
- Re-optimised parameter profiles.
- Parametric-bootstrap framework.
- Baseline random-walk MCMC and posterior summaries.

## Release 8 — Diagnostics and reliability grading

- Data-conflict correlation matrix and conflict score.
- Residual lag-one correlation and runs test.
- Retrospective Mohn-style rho utility.
- Prior-versus-likelihood influence classification.
- Leave-one-dataset/year influence table.
- Automatic A–F reliability grade with transparent reasons.

## Release 9 — Closed-loop management strategy evaluation

- Operating model separated from assessment model.
- Observation error, process error, catch bias and implementation error.
- Assessment intervals and 40–10-style harvest control rules.
- Catch-change constraints, closures and P-star adjustment.
- Regime-switching productivity.
- Probability above target and limit.
- Catch, catch volatility, closure frequency, rebuilding time and economic metrics.
- Pareto frontier for catch, risk and stability.

## Release 10 — SS3 interoperability and benchmark reproduction

- Comment-safe SS3 text parsing utilities.
- `Report.sso` section and time-series extraction.
- Minimal starter/data/control/forecast file export.
- Omega-versus-SS3 time-series comparison.
- Deterministic benchmark suite with tolerance reporting.
- JSON and Markdown benchmark reports.

## Release 11 — Reproducibility and review package

- Complete Assessment desktop interface.
- Command-line interface.
- HTML, JSON and CSV exports.
- SS3 export manifest.
- Mathematical specification.
- Data dictionary.
- Model governance and validation plan.
- Known-limitations register.
- Automated cumulative test suite.

## Scientific status

These releases create functional development foundations. They do not by themselves make Omega a peer-reviewed replacement for Stock Synthesis. That status requires stock-specific configuration, benchmark replication, uncertainty-coverage testing and independent scientific review.
