#include "omega/native_engine.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    assert(omega_engine_abi_version() == 3);
    assert(std::abs(omega_production(500.0, 1000.0, 0.2, OMEGA_SCHAEFER, 1.35) - 50.0) < 1e-10);

    const int n = 6;
    const int years[n] = {2000, 2001, 2002, 2003, 2004, 2005};
    const double catches[n] = {20, 25, 30, 35, 30, 25};
    const double index[n] = {800, 790, 760, 720, 700, 690};
    const double observed_biomass[n] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
    const double theta[4] = {std::log(1000.0), std::log(0.2), std::log(0.8 / 0.2), std::log(0.2)};
    OmegaProductionSettings settings{};
    settings.model = OMEGA_SCHAEFER;
    settings.pella_shape = 1.35;
    settings.target_depletion_cv = 0.25;
    settings.r_prior_median = 0.18;
    settings.r_prior_cv = 0.75;
    settings.obs_cv = 0.22;
    settings.index_weight = 1.0;
    settings.biomass_weight = 1.0;
    settings.initial_depletion_prior_mean = 0.85;
    settings.initial_depletion_prior_sd = 0.30;
    settings.initial_depletion_prior_weight = 0.04;
    settings.catch_to_capacity_penalty_weight = 0.08;
    settings.observation_prior_log_sd = 0.75;

    double objective = 0.0;
    double gradient[4]{};
    double biomass[n]{};
    double components[OMEGA_COMPONENT_COUNT]{};
    assert(omega_production_objective_gradient(theta, years, catches, index, observed_biomass, n, &settings, &objective, gradient, biomass, components) == 0);
    assert(std::isfinite(objective));
    for (double value : gradient) assert(std::isfinite(value));
    for (double value : biomass) assert(std::isfinite(value) && value > 0.0);

    std::cout << omega_engine_build_info() << "\n";
    std::cout << "objective=" << objective << " gradient=" << gradient[0] << "," << gradient[1] << "," << gradient[2] << "," << gradient[3] << "\n";
    return 0;
}
