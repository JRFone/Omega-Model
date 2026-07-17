# Omega FISH Model — Mathematical Specification

## State vector

The seasonal spatial engine stores numbers by sex, area and age:

\[
N_{s,a,x,t,q}
\]

where `s` is sex, `a` is area, `x` is age, `t` is year and `q` is season.

## Growth and weight

Length-at-age uses a von Bertalanffy curve:

\[
L_{s,x}=L_{\infty,s}\left(1-e^{-k_s(x-t_{0,s})}\right).
\]

Weight-at-age is:

\[
W_{s,x}=a_sL_{s,x}^{b_s}.
\]

Maturity and selectivity use logistic curves.

## Movement

At the beginning of each season:

\[
N'_{s,j,x}=\sum_i N_{s,i,x}P_{q,s,x,i,j}
\]

with every origin row of `P` constrained to sum to one.

## Recruitment

Expected recruitment follows Beverton–Holt:

\[
R_t=\frac{4hR_0S_t}{S_0(1-h)+S_t(5h-1)}.
\]

Recruitment deviations, autocorrelation and environmental effects are applied on the log scale with lognormal bias correction.

## Fishing and discards

Fleet encounter mortality is partitioned into retained mortality and dead-discard mortality. Dead discard mortality is calculated using the fleet’s depth-band mixture. Catch-at-age uses the Baranov equation:

\[
C_{x}=N_x\frac{F_x}{Z_x}\left(1-e^{-Z_x}\right).
\]

Annual target catches are reconstructed using bounded bisection on fleet fishing mortality.

## Observation likelihoods

The software implements normal, lognormal, Student-t, multinomial, Dirichlet-multinomial and logistic-normal likelihoods. All components are reported separately before weighting.

## CPUE standardisation

The current internal standardiser fits:

\[
\log(C/E)=\beta_0+\beta_{year}+\beta_{categorical}+\beta_{continuous}+\epsilon.
\]

Annual year effects are exponentiated and normalised to mean one.

## Inference

Parameters are transformed to an unconstrained scale. The current internal optimiser is bounded, multi-start Nelder–Mead. Gradients and Hessians use central finite differences. Covariance uses the generalised inverse Hessian where necessary.

## Closed-loop MSE

The operating model generates the true stock. Data are observed with error, an assessment estimates status, a harvest rule sets catch, implementation error changes realised catch and the cycle repeats.

## Limitations

The finite-difference and random-walk MCMC engines are development baselines. They are not substitutes for automatic differentiation, Laplace random-effects estimation or HMC/NUTS in large production assessments.


## Native differentiation and optimisation

For the production-model parameter vector `theta = (log K, log r, logit d0, log sigma)`, the C++ engine evaluates the same objective components as the Python reference implementation. Forward-mode dual numbers propagate first derivatives through population dynamics and likelihood calculations. Broad candidate scoring is parallelised; bounded local refinement uses L-BFGS-B with the native gradient when available. Numerical finite differences remain a diagnostic parity check and fallback rather than the preferred production gradient.
