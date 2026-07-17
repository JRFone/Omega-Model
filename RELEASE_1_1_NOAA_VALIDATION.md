# Omega FISH Model 1.1.0

## Main addition

A new NOAA / Stock Synthesis Validation Lab establishes a reproducible route
from official SS3 regression-test models to Omega parser tests, deterministic
mathematical checks, native SS3 execution and transparent capability-gap
reporting.

## Added

- official NOAA SS3 model catalog;
- pinned source commit and source manifests;
- internet downloader for complete NOAA model folders;
- starter, data and control file parsers;
- native SS3 temporary-run support;
- parser for model structure, catches, indices and control parameters;
- L1/L2 von Bertalanffy conversion;
- SS3 5–95% logistic selectivity conversion;
- length-based maturity support in Omega biological engines;
- offline NOAA Simple regression fixture;
- 31 deterministic NOAA Simple checks;
- feature-parity matrix against ten NOAA test configurations;
- HTML and JSON validation reports;
- dedicated validation desktop interface;
- command-line download and validation commands;
- NOAA validation in the main software self-check;
- expanded combined test suite.

## Interface changes

- redesigned launcher with six workspaces;
- dedicated NOAA validation dashboard;
- summary metric cards;
- pass/fail/parity chart;
- source, structure, check, parity and native-run tabs;
- direct report export;
- pinned-source and claim-boundary information displayed in the interface.

## Scientific status

This release improves verification and transparency. It does not establish full
Stock Synthesis replacement status. Native SS3 output replication and broad
simulation-recovery performance remain required.
