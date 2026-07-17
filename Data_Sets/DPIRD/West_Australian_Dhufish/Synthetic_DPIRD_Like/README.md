# WA dhufish DPIRD-like controlled synthetic dataset

This is the closest internally coherent Omega test dataset that can be built from the public material captured so far. It is **not DPIRD's raw assessment dataset**.

## What is public evidence

- Retained catch and sector histories digitised from RAR2 Figure 3.15.
- Commercial CPUE digitised from RAR2 Figure 3.17.
- North/South age-composition sampling schedule and sample sizes digitised from RAR2 Figure 3.19.
- Relative spawning biomass, fishing mortality and available recruitment-deviation points digitised from RAR2 Figure 3.21.
- Published biological parameters recorded in `parameter_register.csv`.

## What is synthetic

- A one-stock Omega operating truth calibrated to the digitised WCB relative-spawning-biomass trajectory.
- Synthetic CPUE observations generated from the operating truth at the captured CPUE sampling years.
- Synthetic age and length samples generated from the known operating truth using the captured age-sampling years and sample sizes.
- Calibrated R0 and working selectivity values. These are not DPIRD estimates.

## Which file to use

- `model_ready_timeseries_conditioned.csv`: exact annual synthetic recruitment multipliers are included. Use this first to test whether Omega can recover a known truth.
- `model_ready_timeseries_blind.csv`: recruitment truth is hidden. Use this to expose model-structure and identifiability limitations.
- Load `age_composition.csv` and `length_composition.csv` with either time series.
- `synthetic_truth.csv` is the answer key and must not be treated as an observation.

## Calibration result

- Target-trajectory RMSE: 0.0201 B/B0.
- Synthetic terminal depletion: 0.1492.
- Digitised DPIRD terminal depletion: 0.1493.

Every chart in `Charts` includes labelled X and Y axes. The original captured files are copied alongside the synthetic observations so each transformation can be audited.
