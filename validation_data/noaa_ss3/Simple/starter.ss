#V3.30.25.00;_safe;_compile_date:_Jun 30 2026;_Stock_Synthesis_by_Richard_Methot_(NOAA)_using_ADMB_13.2
#_Stock_Synthesis_is_a_work_of_the_U.S._Government_and_is_not_subject_to_copyright_protection_in_the_United_States.
#_Foreign_copyrights_may_apply._See_copyright.txt_for_more_information.
#_User_support_available_at:_https://groups.google.com/g/ss3-forum_and_NMFS.Stock.Synthesis@noaa.gov
#_User_info_available_at:_https://nmfs-ost.github.io/ss3-website/
#_Source_code_at:_https://github.com/nmfs-ost/ss3-source-code

#C starter comment here
data.ss #_datfile
control.ss #_ctlfile
0 #_init_values_src:  0 (use init values in control file); 1 (use ss3.par)
0 #_screen_display:  0 (minimal); 1 (one line per iter); 2 (each logL)
1 #_report_table_selection:  0 (minimal); 1 (all tables)
0 #_checkup
0 #_parmtrace
0 #_cumreport
1 #_prior_like
1 #_soft_bounds
1 #_N_bootstraps
10 #_last_estimation_phase
0 #_MCMCburn
1 #_MCMCthin
0 # jitter_fraction
1969 #_minyr_sdreport
2011 #_maxyr_sdreport
0 #_N_STD_yrs
0.0001 #_converge_criterion
0 #_retro_yr
1 #_min_age_summary_bio
2 #_depl_basis
1 #_depl_denom_frac
4 #_SPR_basis
3 # F_std_units
0 # F_std_basis
0 #_MCMC_output_detail
0 #_deprecated
-1 #_seed
0 #_Compatibility
3.30 #_final
