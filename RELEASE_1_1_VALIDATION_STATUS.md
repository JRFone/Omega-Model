# Omega FISH Model 1.1.0 — NOAA / SS3 Validation Status

## Completed in this release

- Pinned official NOAA `ss3-test-models` support at commit `3d1f9c0aad7e439a73bd807b02d0ffe4d7b3b944`.
- Embedded offline `Simple` model fixture with recorded source hashes.
- Parsers for `starter.ss`, `data.ss`, `control.ss`, and native `Report.sso` hand-off.
- Deterministic checks for growth conversion, weight-at-length, selectivity, maturity, recruitment, catches, fleets, indices, and model structure.
- Official NOAA model downloader and latest SS3 executable downloader.
- Isolated native SS3 execution in a temporary working directory.
- HTML and JSON comparison reports.
- Feature-parity matrix covering core, advanced, and stress-test NOAA configurations.
- Evidence-gated better-than-SS development scorecard.
- New NOAA / Stock Synthesis Validation desktop workspace.
- GitHub Actions workflow for Linux unit tests and Windows native SS3 comparison.

## Results in the packaged environment

- Complete unit and integration suite: **41 passed**.
- Deterministic benchmark suite: **9 of 9 passed**.
- Embedded NOAA Simple validation: **31 of 31 checks passed**.
- NOAA Simple declared feature parity: **5 of 6**; estimated-growth parity remains partial.
- Omega self-check: **READY**, 9 checks passed, 0 failed.
- End-to-end demonstration: passed.

## Claim boundary

The embedded NOAA validation proves that the files are parsed consistently and that selected deterministic equations reproduce expected values. It does not establish full numerical equivalence with Stock Synthesis.

The included Windows GitHub Actions job downloads the latest official NOAA/NMFS SS3 executable, downloads the pinned NOAA Simple model, runs SS3, and saves the native comparison report. That job must run successfully after the release is pushed to GitHub before native equivalence is recorded as passed.

## Better-than-SS objective

Omega is already designed to provide integrated provenance, assumption tracking, optimizer agreement, walk-forward validation, data-conflict analysis, reliability grading, and closed-loop MSE in one application. The project will only claim an overall advantage after it passes the wider NOAA model catalog, uncertainty-coverage studies, controlled runtime comparisons, and independent scientific review.
