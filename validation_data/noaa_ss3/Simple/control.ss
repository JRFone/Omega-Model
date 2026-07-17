# Omega FISH embedded extract of the official NOAA Simple SS3 test model.
# Source: nmfs-ost/ss3-test-models, commit 3d1f9c0aad7e439a73bd807b02d0ffe4d7b3b944
# Original control.ss blob SHA: bd0cff158a0b0c52406fbbeef9bcfff03132b93e
0 #_natM_type
1 # GrowthModel
0 #_Age(post-settlement) for L1
25 #_Age(post-settlement) for L2
1 #_maturity_option
1 #_fecundity_at_length option
#_ LO HI INIT PRIOR PR_SD PR_type PHASE env_var use_dev dev_minyr dev_maxyr dev_PH Block Block_Fxn
0.05 0.15 0.1 0.1 0.8 0 -3 0 0 0 0 0 0 0 # NatM_uniform_Fem_GP_1
10 45 21.6591 36 10 6 2 0 0 0 0 0 0 0 # L_at_Amin_Fem_GP_1
40 90 71.654 70 10 6 4 0 0 0 0 0 0 0 # L_at_Amax_Fem_GP_1
0.05 0.25 0.14724 0.15 0.8 6 4 0 0 0 0 0 0 0 # VonBert_K_Fem_GP_1
0.05 0.25 0.1 0.1 0.8 0 -3 0 0 0 0 0 0 0 # CV_young_Fem_GP_1
0.05 0.25 0.1 0.1 0.8 0 -3 0 0 0 0 0 0 0 # CV_old_Fem_GP_1
-3 3 2.44e-06 2.44e-06 0.8 0 -3 0 0 0 0 0 0 0 # Wtlen_1_Fem_GP_1
-3 4 3.34694 3.34694 0.8 0 -3 0 0 0 0 0 0 0 # Wtlen_2_Fem_GP_1
50 60 55 55 0.8 0 -3 0 0 0 0 0 0 0 # Mat50%_Fem_GP_1
-3 3 -0.25 -0.25 0.8 0 -3 0 0 0 0 0 0 0 # Mat_slope_Fem_GP_1
0.05 0.15 0.1 0.1 0.8 0 -3 0 0 0 0 0 0 0 # NatM_uniform_Mal_GP_1
0 45 0 36 10 0 -3 0 0 0 0 0 0 0 # L_at_Amin_Mal_GP_1
40 90 69.5399 70 10 6 4 0 0 0 0 0 0 0 # L_at_Amax_Mal_GP_1
0.05 0.25 0.163476 0.15 0.8 6 4 0 0 0 0 0 0 0 # VonBert_K_Mal_GP_1
-3 3 2.44e-06 2.44e-06 0.8 0 -3 0 0 0 0 0 0 0 # Wtlen_1_Mal_GP_1
-3 4 3.34694 3.34694 0.8 0 -3 0 0 0 0 0 0 0 # Wtlen_2_Mal_GP_1
3 #_Spawner-Recruitment
3 31 8.81206 10.3 10 0 1 0 0 0 0 0 0 0 # SR_LN(R0)
0.2 1 0.573835 0.7 0.05 1 4 0 0 0 0 0 0 0 # SR_BH_steep
0 2 0.6 0.8 0.8 0 -4 0 0 0 0 0 0 0 # SR_sigmaR
-5 5 0 0 1 0 -4 0 0 0 0 0 0 0 # SR_regime
0 0 0 0 0 0 -99 0 0 0 0 0 0 0 # SR_autocorr
2 #do_recdev
1971 # first year of main recr_devs
2001 # last year of main recr_devs
# all recruitment deviations
# 1971R 1972R 1973R 1974R 1975R 1976R 1977R 1978R 1979R 1980R 1981R 1982R 1983R 1984R 1985R 1986R 1987R 1988R 1989R 1990R 1991R 1992R 1993R 1994R 1995R 1996R 1997R 1998R 1999R 2000R 2001R
# 0.134968 -0.0507585 0.0972615 -0.164709 0.0443744 0.710789 -0.00439021 0.0113464 0.266822 0.186805 0.100984 -0.208826 -0.418054 -0.284269 0.420545 0.586619 0.262575 0.204289 -0.321237 0.664807 -0.607033 -0.193662 -0.7506 0.446329 -0.524044 0.539563 1.20007 -0.459983 -0.567356 0.263829 -0.216963
0.3 # F ballpark value
-2001 # F ballpark year
3 # F_Method
2.95 # max F
4 # N iterations
# F rates by fleet x season
#_year: 1971 1972 1973 1974 1975 1976 1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001
# FISHERY 0 0.00211813 0.0106457 0.0107404 0.0217792 0.0334421 0.046095 0.0601249 0.0759325 0.10803 0.147259 0.162937 0.181299 0.203352 0.230858 0.266731 0.315247 0.338831 0.3551 0.356617 0.339428 0.2384 0.243238 0.251023 0.263884 0.283723 0.227427 0.23848 0.247863 0.25268 0.253549
