# Biomass Evidence Engine

Omega does not select one biomass curve merely because it has the smallest in-sample likelihood. The engine:

- fits deterministic Schaefer, Fox and Pella–Tomlinson structures;
- fits a bootstrap-particle-filter state-space Schaefer candidate with latent annual biomass, process error and filtered uncertainty;
- fits a robust composite abundance index and individual index series;
- uses held-out years to measure prediction rather than fit alone;
- caps the maximum weight of a single candidate;
- propagates candidate, parameter and process uncertainty;
- reports annual biomass and depletion intervals;
- measures cross-model disagreement and index conflict;
- grades absolute-scale identifiability;
- retains the complete candidate weight table.

The output phrase **best-supported biomass estimate** is mandatory. The software must not relabel the estimate as observed truth when the input contains only catches and relative abundance indices.


## State-space candidate

The state-space candidate treats annual biomass as a latent process rather than forcing every year to follow a perfectly deterministic curve. A bootstrap particle filter propagates biomass through catch and production, updates particle weights using abundance indices and any direct biomass observations, and reports filtered annual intervals. It is included as an additional structural candidate in the evidence ensemble. It remains a compact implementation and does not replace a fully conditioned random-effects age-structured assessment.

## Automatic estimate versus truth

For simulated data, Omega can compare the estimate with known truth. For a real fishery, it reports the best-supported estimate, prediction performance, model disagreement, index conflict, interval width and an identifiability grade. No program can recover an assumption-free true biomass from incomplete fishery data.
