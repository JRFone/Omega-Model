# Advanced Age-Structured Management Strategy Evaluation

The release 1.4 MSE uses a closed loop:

1. age-structured operating truth;
2. fishery and survey observation generation;
3. an imperfect and potentially misspecified assessment;
4. a management procedure;
5. implementation and compliance error;
6. realised fishing applied to the operating truth;
7. recruitment and population update;
8. repeated reassessment and decision cycles.

## Operating-model uncertainty

Default scenarios include natural-mortality uncertainty, poor recruitment, recruitment-regime shifts, CPUE hyperstability and catchability drift, release-mortality uncertainty and selectivity shifts.

## Management controls

Procedures can use biomass ramps, fixed F or fixed catch, sector allocations, catch-change limits, complete closure below a limit, seasonal and spatial closure multipliers, recreational bag-limit effort multipliers, effort controls, compliance and implementation error.

## Assessment options

- `fast_filter` for rapid exploration;
- `biomass_ensemble` for repeated evidence-weighted production-model reassessment;
- `full_age_structured` for the highest-standard but most computationally expensive closed-loop reassessment.

## Decision outputs

Omega reports safety, catch, catch stability, closure frequency, economic value, assessment bias/RMSE, false healthy and false overfished classifications, worst-case scenario performance, Pareto fronts, risk-adjusted utility and formal-readiness checks.

## Robust decision analysis

Release 1.4 also calculates scenario-specific utility, regret relative to the best procedure under each operating truth, maximum and weighted regret, scenario-win frequency, lower-tail utility, a minimax-regret procedure and the expected value of perfect information. These outputs distinguish a procedure that performs well on average from one that remains defensible under the worst plausible scenarios.
