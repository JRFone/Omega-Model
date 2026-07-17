# Omega FISH Complete Data Dictionary

## Time-series fields

| Field | Meaning | Unit/format |
|---|---|---|
| `year` | Calendar or assessment year | Integer |
| `catch` | Total retained catch | Biomass |
| `catch_commercial` | Commercial retained catch | Biomass |
| `catch_charter` | Charter retained catch | Biomass |
| `catch_recreational` | Private recreational retained catch | Biomass |
| `index` | Standardised abundance index | Positive relative index |
| `biomass` | Independent biomass observation | Biomass |
| `recruitment_multiplier` | Externally supplied recruitment multiplier | Positive ratio |

## Raw CPUE fields

| Field | Meaning |
|---|---|
| `catch` | Catch for the record |
| `effort` | Effort for the record |
| `year` | Fishing year |
| `vessel` | Confidential vessel effect identifier |
| `skipper` | Optional confidential skipper identifier |
| `area` | Spatial stratum |
| `month` | Seasonal stratum |
| `depth` | Fishing depth |
| `technology_year` | Optional technology trend covariate |

Raw vessel and skipper identifiers should remain protected. Reports should expose annual indices and model coefficients, not identifiable source records.

## Composition fields

Age or length composition tables should contain year, fleet/sector, area, sex, bin, count or proportion, and input sample size. The selected likelihood and weighting method must be recorded with the run.

## Tagging fields

Tag-release records include release year, area, age, sex and number released. Recapture records include release identifier, recapture year, fleet, area and observed recaptures.

## Movement

Movement arrays have dimensions:

`season × sex × age × origin area × destination area`

Every origin row must be non-negative and sum to one.
