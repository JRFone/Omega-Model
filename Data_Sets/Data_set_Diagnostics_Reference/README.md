# Omega Complete Demonstration Dataset

This is the default full-coverage test dataset. It is a controlled synthetic stock, so expected behaviour can be checked without presenting invented values as a real fishery assessment.

## Included test data

- Annual catch, abundance index, biomass, uncertainty, fishing mortality, recruitment and natural mortality.
- Commercial, charter and recreational catch components.
- Age compositions and length compositions with sample sizes.
- Chart-ready residuals, optimizer runs, a two-parameter objective grid and likelihood components.
- Retrospective peel labels, observed/predicted hindcast values, coverage examples and MSE procedures.

`model_ready_timeseries.csv` is loaded by model-fitting workspaces. `age_composition.csv` and `length_composition.csv` support the age-structured assessment. `all_functions_chart_data.csv` is automatically loaded by Chart Studio so every chart type has a working example.

## Scientific boundary

These data verify software paths and demonstrate interpretation. They do not validate a real stock, establish model accuracy, or replace fishery-specific sampling and independent scientific review. NOAA/SS3 validation uses the separate official NOAA Simple model because a complete SS3 configuration requires its original starter, data and control files.
