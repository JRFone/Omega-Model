# Omega FISH Model Governance

## Run integrity

Every management-facing run should preserve:

- source-file hashes;
- data transformations;
- model version and commit;
- parameter values, bounds, priors and phases;
- random seeds;
- likelihood components and weights;
- convergence and Hessian results;
- diagnostics and sensitivity tests;
- exported results and report hashes.

## Change control

Model code, configuration, data and scientific decisions must be versioned separately. A change in data weighting, selectivity blocks, natural mortality, recruitment penalties or catch reconstruction must create a new run rather than overwrite an existing result.

## Review states

Suggested run states are `draft`, `diagnostic`, `rejected`, `candidate`, `reference` and `approved`. Approval should require a named reviewer and recorded reasons.

## Independent review

No automated grade replaces independent scientific review. Material management use should require review by analysts who did not configure the base model.
