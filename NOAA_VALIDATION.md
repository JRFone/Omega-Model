# NOAA / Stock Synthesis Validation Lab

## Purpose

Omega FISH 1.1 adds a dedicated validation layer against the official NOAA
Stock Synthesis software-testing repository:

- Repository: https://github.com/nmfs-ost/ss3-test-models
- Pinned validation commit: `3d1f9c0aad7e439a73bd807b02d0ffe4d7b3b944`
- Stock Synthesis source: https://github.com/nmfs-ost/ss3-source-code

The NOAA repository describes its models as regression-test configurations and
warns that the included data may have been altered for testing. Omega therefore
uses them only for software validation, not as evidence about any stock.

## What is now verified offline

The embedded `Simple` fixture verifies:

- starter-file parsing;
- start and end years;
- seasons, sexes, ages, areas, fleets and surveys;
- annual catch and abundance-index extraction;
- natural-mortality parameter extraction;
- SS3 L1/L2 von Bertalanffy conversion to Linf and t0;
- reconstruction of the two reference lengths;
- weight-at-length calculations;
- Beverton-Holt recruitment at unfished spawning biomass;
- recruitment-deviation reference-vector extraction;
- fishing-mortality reference-vector extraction;
- feature-parity reporting.

Passing these checks establishes parser correctness and selected deterministic
mathematical agreement. It does not establish complete numerical equivalence to
Stock Synthesis.

## Downloading complete NOAA test models

Open **NOAA / SS3 Validation** from the Omega launcher and choose a model. The
program queries the official GitHub repository, downloads every file in that
model folder, stores SHA-256 hashes, and writes a source manifest.

Command-line equivalent:

```text
omega-fish noaa-download Simple --output validation_cache/noaa_ss3
```

## Running native Stock Synthesis

Select an SS3 executable in the validation interface. Omega copies the chosen
model and executable to a temporary directory, runs SS3 without modifying the
source model, captures stdout/stderr and warnings, and parses `Report.sso` when
created.

Command-line equivalent:

```text
omega-fish noaa-validate validation_cache/noaa_ss3/Simple \
  --model-name Simple \
  --ss3-executable C:\path\to\ss3.exe \
  --output reports/noaa_validation
```

## Current capability gaps

The parity matrix deliberately marks functions as `partial` or
`not_implemented`. Major remaining gaps include:

- cubic-spline selectivity;
- double-normal selectivity;
- full generalized size-composition likelihoods;
- complete empirical weight-at-age handling;
- complete growth-cessation models;
- exact native SS3 input/output equivalence;
- automatic differentiation and production-grade random-effects integration;
- large-scale simulation-estimation coverage studies.

Omega should only be described as exceeding Stock Synthesis in an area after a
published benchmark demonstrates that result. Interface quality, auditability,
provenance, automated diagnostics and integrated validation are valid comparison
areas; full mathematical superiority is not yet established.
