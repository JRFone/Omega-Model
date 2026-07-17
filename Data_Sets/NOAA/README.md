# NOAA Stock Synthesis test-data library

Omega downloads the official NOAA/NMFS `ss3-test-models` and
`ss3-user-examples` repositories into `Data_Sets/NOAA/_sources` for local use.
The downloaded snapshots are intentionally excluded from Git because they are
large and can be recreated exactly from the recorded commit SHAs.

Refresh the library:

```powershell
.\.venv\Scripts\python.exe tools\download_noaa_test_data.py --refresh
```

Check the installed library without downloading:

```powershell
.\.venv\Scripts\python.exe tools\download_noaa_test_data.py --check-only
```

`NOAA_SOURCE_MANIFEST.json` records the repository URLs, branches, exact commit
SHAs, timestamps, file counts, sizes, and detected licence files. The catalogue
files list each discovered model folder. Omega treats the source snapshot as
read-only and creates model working files elsewhere.
