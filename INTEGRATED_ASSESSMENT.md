# Omega FISH Integrated Assessment Lab

## Purpose

This release adds an age-structured assessment foundation beside the existing production, state-space, delay-difference and Quant Lab tools. It is designed to make the biological and fishery processes explicit and inspectable.

## Implemented processes

- Ages from age 0 through a configurable plus group.
- Von Bertalanffy length-at-age.
- Weight-at-age and maturity-at-age.
- Female spawning biomass.
- Beverton-Holt stock-recruitment with steepness.
- Recruitment deviations and autocorrelation in projections.
- Commercial, charter and recreational sectors.
- Sector-specific selectivity and retention.
- Legal-size retention curves.
- Released fish and post-release/discard mortality.
- Baranov catch equations and annual fishing-mortality reconstruction.
- Survey selectivity and profiled catchability.
- Biomass and CPUE/index likelihoods.
- Age-composition and length-composition likelihood components.
- Differential Evolution estimation with local refinement.
- Equilibrium MSY, BMSY and FMSY grid calculation.
- Fixed-catch, fixed-F and 40-10 HCR projections.
- Recruitment and implementation uncertainty.
- Multi-strategy management strategy evaluation and Pareto front.
- CSV, JSON and HTML output packages.

## Starting the interface

Double-click:

```text
Start Omega FISH Integrated Assessment.bat
```

The Quant Lab toolbar also contains **Open Integrated Assessment Lab**.

## Time-series format

Minimum columns:

```csv
year,catch
2000,80
2001,85
```

Useful optional columns:

```text
index
biomass
catch_commercial
catch_charter
catch_recreational
recruitment_multiplier
recruitment_index
juvenile_index
```

The age-structured loader preserves those additional columns even though the general production-model loader only needs year, catch, index and biomass.

## Composition format

Age composition:

```csv
year,sector,age,proportion,sample_size
2000,all,3,0.14,120
```

Length composition:

```csv
year,sector,length_mm,proportion,sample_size
2000,all,500,0.22,120
```

Counts can be supplied instead of proportions; each year-sector group is normalised automatically.

## Demonstration files

```text
Data_Sets\Data_set_Age_Structured_Demo\model_ready_timeseries.csv
Data_Sets\Data_set_Age_Structured_Demo\age_composition.csv
Data_Sets\Data_set_Age_Structured_Demo\length_composition.csv
```

## Scope statement

This is a functional integrated-assessment foundation, not a claim of equivalence to a completed Stock Synthesis model. Future function releases can add sex structure, multiple areas, tagging, explicit ageing-error matrices, time-varying selectivity, environmental recruitment covariates and full state-space recruitment estimation.
