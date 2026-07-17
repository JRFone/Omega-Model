# Omega FISH embedded extract of the official NOAA Simple SS3 test model.
# Source: nmfs-ost/ss3-test-models, commit 3d1f9c0aad7e439a73bd807b02d0ffe4d7b3b944
# Original data.ss blob SHA: ef3ec428959ddd8cf688690cbdd9f52dc48c07b7
# This extract contains the complete core header, fleet, catch, and CPUE sections used by Omega validation.
#V3.30.25.00;_safe;_compile_date:_Jun 30 2026;_Stock_Synthesis_by_Richard_Methot_(NOAA)_using_ADMB_13.2
1971 #_StartYr
2001 #_EndYr
1 #_Nseas
12 #_months/season
2 #_Nsubseasons
1 #_spawn_month
2 #_Nsexes
40 #_Nages
1 #_Nareas
3 #_Nfleets
#_fleet_type fishery_timing area catch_units need_catch_mult fleetname
1 -1 1 1 0 FISHERY
3 1 1 2 0 SURVEY1
3 1 1 2 0 SURVEY2
#_Catch data: year, seas, fleet, catch, catch_se
-999 1 1 0 0.01
1971 1 1 0 0.01
1972 1 1 200 0.01
1973 1 1 1000 0.01
1974 1 1 1000 0.01
1975 1 1 2000 0.01
1976 1 1 3000 0.01
1977 1 1 4000 0.01
1978 1 1 5000 0.01
1979 1 1 6000 0.01
1980 1 1 8000 0.01
1981 1 1 10000 0.01
1982 1 1 10000 0.01
1983 1 1 10000 0.01
1984 1 1 10000 0.01
1985 1 1 10000 0.01
1986 1 1 10000 0.01
1987 1 1 10000 0.01
1988 1 1 9000 0.01
1989 1 1 8000 0.01
1990 1 1 7000 0.01
1991 1 1 6000 0.01
1992 1 1 4000 0.01
1993 1 1 4000 0.01
1994 1 1 4000 0.01
1995 1 1 4000 0.01
1996 1 1 4000 0.01
1997 1 1 3000 0.01
1998 1 1 3000 0.01
1999 1 1 3000 0.01
2000 1 1 3000 0.01
2001 1 1 3000 0.01
-9999 0 0 0 0
#_CPUE_and_surveyabundance_and_index_observations
#_fleet units errtype SD_report
1 1 0 0 # FISHERY
2 1 0 1 # SURVEY1
3 0 0 0 # SURVEY2
#_year month fleet obs stderr
1977 7 2 339689 0.3
1980 7 2 193353 0.3
1983 7 2 151984 0.3
1986 7 2 55221.8 0.3
1989 7 2 59232.3 0.3
1992 7 2 31137.5 0.3
1995 7 2 35845.4 0.3
1998 7 2 27492.6 0.3
2001 7 2 37338.3 0.3
1990 7 3 5.19333 0.7
1991 7 3 1.1784 0.7
1992 7 3 5.94383 0.7
1993 7 3 0.770106 0.7
1994 7 3 16.318 0.7
1995 7 3 1.36339 0.7
1996 7 3 4.76482 0.7
1997 7 3 51.0707 0.7
1998 7 3 1.36095 0.7
1999 7 3 0.862531 0.7
2000 7 3 5.97125 0.7
2001 7 3 1.69379 0.7
-9999 1 1 1 1
0 #_N_fleets_with_discard
1 # use length composition data
# age composition section is present in the complete NOAA file
