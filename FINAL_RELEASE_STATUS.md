# Omega FISH 1.4.1 release status

## Software distribution

Status: **READY FOR CUMULATIVE UPDATE PACKAGING AND WINDOWS BUILD**

Implemented:

- unified eight-workspace launcher;
- Integrated Assessment;
- Quant Lab;
- NOAA / SS3 Validation Lab;
- Validation and MSE;
- Automatic Expert Workflow;
- Interactive Chart Studio;
- saved personal chart profiles;
- offline interactive dashboards;
- source launchers and Windows build definitions;
- environment setup, self-check and GitHub Actions validation.

## Automated validation at development stage

- cumulative tests: 62 passed;
- deterministic benchmarks: 9 passed in the existing benchmark suite;
- pinned NOAA Simple validation: available;
- interactive chart engine: offline generation tested;
- GUI modules: import checks included;
- expert workflow: automatic and exploration functions exercised.

The final package contains a test report generated after packaging and revalidation.

## External work software cannot self-certify

- independent code audit;
- independent fisheries-science peer review;
- stock-specific data verification;
- full native SS3 numerical parity across the NOAA catalogue;
- formal uncertainty coverage for final production configurations;
- regulatory or legal acceptance.


## Omega 1.3 status

The source release contains a compiled C++ production-model engine, native automatic gradients, parallel candidate evaluation and complete implementations of the three current priority diagnostic systems. The Linux native library was built and tested in the development environment. Windows build scripts and CI are included, but a Windows DLL was not executed in the Linux development environment.

## Release 1.4 addition

Release 1.4 adds evidence-weighted biomass synthesis, a separate-truth age-structured MSE, automatic management-procedure comparison, formal configuration readiness gates, and experimental diagnostic triangulation. The software is ready for cumulative update packaging and continued validation. It is not independently certified for regulatory use.


## Release 1.4 validation result

- Python tests: 62 passed, 0 failed.
- Full self-check: 15 checks passed, 0 failed.
- State-space biomass filter: passed.
- Advanced MSE closed-loop smoke tests: passed.
- Experimental diagnostics and offline dashboards: passed.
- Scientific status: stock-specific calibration and independent review still required.
