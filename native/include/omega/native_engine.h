#pragma once

#include <stddef.h>

#if defined(_WIN32)
  #if defined(OMEGA_NATIVE_BUILD)
    #define OMEGA_API __declspec(dllexport)
  #else
    #define OMEGA_API __declspec(dllimport)
  #endif
#else
  #define OMEGA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum OmegaProductionModel {
    OMEGA_SCHAEFER = 0,
    OMEGA_FOX = 1,
    OMEGA_PELLA = 2
};

enum OmegaObjectiveComponent {
    OMEGA_INDEX_LIKELIHOOD = 0,
    OMEGA_BIOMASS_LIKELIHOOD = 1,
    OMEGA_TERMINAL_DEPLETION = 2,
    OMEGA_OBSERVATION_PRIOR = 3,
    OMEGA_PRODUCTIVITY_PRIOR = 4,
    OMEGA_INITIAL_DEPLETION_PRIOR = 5,
    OMEGA_CATCH_CAPACITY_PENALTY = 6,
    OMEGA_COMPONENT_COUNT = 7
};

typedef struct OmegaProductionSettings {
    int model;
    double pella_shape;
    double target_depletion;
    int use_target_depletion;
    double target_depletion_cv;
    double r_prior_median;
    double r_prior_cv;
    double obs_cv;
    double index_weight;
    double biomass_weight;
    double initial_depletion_prior_mean;
    double initial_depletion_prior_sd;
    double initial_depletion_prior_weight;
    double catch_to_capacity_penalty_weight;
    double observation_prior_log_sd;
} OmegaProductionSettings;

OMEGA_API int omega_engine_abi_version(void);
OMEGA_API const char* omega_engine_build_info(void);
OMEGA_API int omega_engine_has_openmp(void);
OMEGA_API int omega_engine_max_threads(void);
OMEGA_API void omega_engine_set_threads(int threads);

OMEGA_API double omega_production(
    double biomass,
    double k,
    double r,
    int model,
    double pella_shape
);

OMEGA_API int omega_simulate_production(
    const double* catches,
    int n,
    double k,
    double r,
    double initial_depletion,
    int model,
    double pella_shape,
    double* biomass_out
);

/*
 * theta is [log_k, log_r, logit_initial_depletion, log_sigma].
 * Missing observations must be passed as NaN. years may be NULL because the
 * current deterministic engine only needs the observation order.
 */
OMEGA_API int omega_production_objective_gradient(
    const double* theta,
    const int* years,
    const double* catches,
    const double* index,
    const double* biomass_observed,
    int n,
    const OmegaProductionSettings* settings,
    double* objective_out,
    double* gradient_out,
    double* biomass_prediction_out,
    double* components_out
);

OMEGA_API int omega_production_batch_objective(
    const double* theta_matrix,
    int candidates,
    const int* years,
    const double* catches,
    const double* index,
    const double* biomass_observed,
    int n,
    const OmegaProductionSettings* settings,
    double* objectives_out,
    double* gradients_out
);

#ifdef __cplusplus
}
#endif
