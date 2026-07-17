# WA Dhufish — public-evidence Omega dataset

This folder is a transparent reconstruction from official DPIRD publications. It is designed to test whether Omega can reproduce the broad behaviour of the published WA Dhufish assessment and to identify which assumptions deserve further testing.

It is **not** the full DPIRD assessment dataset. DPIRD's raw annual input tables, bespoke ADMB source/run files, fitted selectivity parameters, objective-function components, covariance output, and accepted/rejected run folders were not found in the public releases collected here.

## Evidence classes

- `published_exact`: a value printed in an official DPIRD table or model description.
- `published_rule`: a documented regulation or modelling rule.
- `derived_unit_conversion` / `derived_parameterisation`: a transparent mathematical conversion of an exact value.
- `digitised_from_published_figure`: an approximate value recovered from pixels in an official figure.
- `vector_extracted_from_published_figure`: an approximate value recovered from the plotted vector path and calibrated against the published axes.
- `not_publicly_available`: required information that must not be invented.

## Omega-ready files

- `Omega_Ready/dpird_wa_dhufish_public_reconstruction.csv`: annual retained catch by sector and published CPUE series.
- `Omega_Ready/catch_by_area_sector.csv`: commercial, recreational and charter sector totals extracted from the published vector curves. Public area splits are deliberately marked `not_resolved` rather than invented.
- `Omega_Ready/age_composition.csv`: sample-size-weighted North/South public age-composition reconstruction.
- `Omega_Ready/age_composition_by_area.csv`: the separate public North and South reconstructions.
- `Omega_Ready/published_assessment_outputs_digitised.csv`: digitised DPIRD central biomass, fishing-mortality and recruitment-deviation trajectories for benchmark comparison.
- `Omega_Ready/parameter_register.csv`: published, derived, and missing model parameters.
- `Omega_Ready/source_manifest.json`: official URLs and cryptographic hashes.

The original PDFs remain unchanged in `Public_Sources`. Figure images in `Evidence` are reproducible derivatives used only for digitisation.
