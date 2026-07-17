# Omega FISH Interactive Chart Studio

Omega 1.2 adds a reusable Plotly-based chart engine and desktop Chart Studio.

## Interaction

Every supported chart can provide:

- mouse-wheel zoom;
- box zoom, pan, lasso and box selection;
- double-click axis reset;
- unified hover and linked crosshairs;
- clickable legends to hide, isolate or restore series;
- year-range sliders;
- editable chart titles, axis titles and annotations;
- line, rectangle, circle and free-path annotations;
- PNG export from the chart modebar;
- responsive resizing;
- offline HTML output.

## Personalisation

Named chart profiles save:

- theme;
- font family and sizes;
- line and marker sizes;
- grid, legend and range-slider settings;
- hover and crosshair behaviour;
- paper and plot backgrounds;
- colour palette;
- chart height;
- downsampling threshold;
- export scale.

Profiles are stored under the current user's `.omega_fish/chart_profiles.json`. They affect display only and do not alter model inputs, weights or results.

## Supported scientific charts

The chart engine includes dedicated builders for:

- time-series overlays and uncertainty bands;
- residual heatmaps;
- jitter and multistart distributions;
- optimizer agreement;
- likelihood-component conflict;
- likelihood profiles;
- retrospective peels;
- hindcast prediction and MASE;
- structural ensemble fan charts;
- closed-loop MSE trade-off frontiers;
- interval-coverage plots.

## Performance

Long time series use Plotly WebGL traces. Series above the selected threshold are displayed using Largest-Triangle-Three-Buckets downsampling, which preserves endpoints, peaks and turning points. The underlying model data are not modified.

The dashboard stores one local Plotly JavaScript bundle beside the HTML report, so multiple charts remain offline without embedding the library repeatedly.

## Starting the interface

Use either the main launcher or:

```text
Start Omega FISH Interactive Charts.bat
```

The Expert Workflow workspace automatically builds a multi-panel interactive dashboard using the selected personal chart profile.
